import logging

import incense
import pytorch_lightning as pl
import sacred
import torch
from pytorch_lightning import callbacks as pl_cb
from sacred.observers import MongoObserver

from common.data import LARaDataModule, OpportunityDataModule, Pamap2DataModule
from common.helper import (GlobalPlaceholder, QConfigFactory, QuantizationMode,
                           QuantizationModeMapping, applyConversionAfterModeMapping,
                           applyQuantizationModeMapping, bestExperimentWF1, checkpointsById,
                           getRunCheckpointDirectory, parseMongoConfig)
from common.model import CNNIMU
from common.pl_components import (MonitorAcc, MonitorBatchTime, MonitorWF1, SacredLogger)

ex = sacred.Experiment(name='QAT_CNN-IMU')
ex.observers.append(MongoObserver(**parseMongoConfig('./config.ini')))
loader = incense.ExperimentLoader(
    **parseMongoConfig('./config.ini', adapt='IncenseExperimentLoader'))
logger = logging.getLogger(__name__)


def bestRunIdByDataset(dataset: str) -> int:
  if dataset == 'lara':
    return bestExperimentWF1(loader,
                             experiment_name='CNN-IMU_LARa',
                             min_id=None,
                             max_id=None,
                             my_meta={
                                 'runner': 'cnn_imu_rerun_best.py',
                                 'flags': 'disabled_puppet'
                             })[1]
  elif dataset == 'opportunity':
    return bestExperimentWF1(loader,
                             experiment_name='CNN-IMU_Opportunity-Locomotion',
                             min_id=None,
                             max_id=None,
                             my_meta={
                                 'runner': 'cnn_imu_rerun_best.py',
                                 'flags': 'disabled_puppet'
                             })[1]
  elif dataset == 'pamap2':
    return bestExperimentWF1(loader,
                             experiment_name='CNN-IMU_Pamap2(activity_labels)',
                             min_id=None,
                             max_id=None,
                             my_meta={
                                 'runner': 'cnn_imu_rerun_best.py',
                                 'flags': 'disabled_puppet'
                             })[1]
  else:
    raise ValueError(f'No such Dataset {dataset}')


def buildQuantizationModeMapping(weight_observer, weight_observer_args, activation_observer,
                                 activation_observer_args, imu_input_quantization,
                                 imu_pipeline_quantization, imu_pipeline_fc_quantization,
                                 fc_quantization, output_layer_quantization):
  qat_qconfig_factory = QConfigFactory(
      GlobalPlaceholder(activation_observer, **activation_observer_args),
      GlobalPlaceholder(weight_observer, **weight_observer_args))

  if imu_input_quantization == 'qat':
    q_block0 = QuantizationMode.qat(False,
                                    True,
                                    operator_fuse_list=[['conv1', 'relu1'], ['conv2', 'relu2']],
                                    qconfig_factory=qat_qconfig_factory)
  elif imu_input_quantization == 'none':
    q_block0 = QuantizationMode.none(False, False, qconfig_factory=qat_qconfig_factory)
  else:
    raise ValueError(f'Unknown {imu_input_quantization=}')

  if imu_pipeline_quantization == 'qat':
    q_blockn = QuantizationMode.qat(imu_input_quantization == 'qat',
                                    True,
                                    operator_fuse_list=[['conv1', 'relu1'], ['conv2', 'relu2']],
                                    qconfig_factory=qat_qconfig_factory)
  elif imu_pipeline_quantization == 'none':
    q_blockn = QuantizationMode.none(imu_input_quantization == 'qat',
                                     False,
                                     qconfig_factory=qat_qconfig_factory)
  else:
    raise ValueError(f'Unknown {imu_pipeline_quantization=}')

  if imu_pipeline_fc_quantization == 'qat':
    q_pfc = QuantizationMode.qat(imu_pipeline_quantization == 'qat',
                                 True,
                                 qconfig_factory=qat_qconfig_factory,
                                 operator_fuse_list=['1', '2'])
  elif imu_pipeline_fc_quantization == 'none':
    q_pfc = QuantizationMode.none(imu_pipeline_quantization == 'qat',
                                  False,
                                  qconfig_factory=qat_qconfig_factory)
  else:
    raise ValueError(f'Unknown {imu_pipeline_fc_quantization=}')

  q_fc = QuantizationMode.fuse_only(operator_fuse_list=['1', '2'])
  if fc_quantization == 'qat':
    q_fc_0 = QuantizationMode.qat(imu_pipeline_fc_quantization == 'qat',
                                  True, [],
                                  qconfig_factory=qat_qconfig_factory)
  elif fc_quantization == 'none':
    q_fc_0 = QuantizationMode.none(imu_pipeline_fc_quantization == 'qat',
                                   False,
                                   qconfig_factory=qat_qconfig_factory)
  else:
    raise ValueError(f'Unknown {fc_quantization=}')

  if output_layer_quantization == 'qat':
    q_fc_1 = QuantizationMode.qat(fc_quantization == 'qat',
                                  False, [],
                                  qconfig_factory=qat_qconfig_factory)
  elif output_layer_quantization == 'none':
    q_fc_1 = QuantizationMode.none(fc_quantization == 'qat',
                                   False,
                                   qconfig_factory=qat_qconfig_factory)
  else:
    raise ValueError(f'Unknown {output_layer_quantization=}')

  q_mode_map = QuantizationModeMapping()
  q_mode_map.addRegexMapping(r'pipelines.\d*.blocks.0$', q_block0)
  q_mode_map.addRegexMapping(r'pipelines.\d*.blocks.\d*$', q_blockn)
  q_mode_map.addRegexMapping(r'pipe_fc.\d*$', q_pfc)
  q_mode_map.addNameMapping(r'fc', q_fc)
  q_mode_map.addNameMapping('fc.1', q_fc_0)
  q_mode_map.addNameMapping('fc.4', q_fc_1)

  return q_mode_map


@ex.config
def defaultConfig():
  use_dataset = 'lara'
  trained_model_run_id = bestRunIdByDataset(use_dataset)
  backend = 'fbgemm'
  batch_size = 128
  max_epochs = 10
  validation_interval = 0.3
  loss_patience = 10
  optimizer = 'Adam'
  extra_hyper_params = {}
  restore_optimizer = True
  if optimizer == 'Adam':
    extra_hyper_params['optimizer'] = 'Adam'
    extra_hyper_params['lr'] = 1e-3
    extra_hyper_params['betas'] = (0.9, 0.999)
    extra_hyper_params['weight_decay'] = 0
  elif optimizer == 'RMSProp':
    extra_hyper_params['optimizer'] = 'RMSProp'
    extra_hyper_params['lr'] = 1e-2
    extra_hyper_params['alpha'] = 0.99
    extra_hyper_params['weight_decay'] = 0.95
    extra_hyper_params['momentum'] = 0

  n_bits = 7
  activation_observer = 'torch.ao.quantization.MovingAverageMinMaxObserver'  # Cannot be 'PerChannel'
  activation_qscheme = 'torch.per_tensor_affine'
  activation_observer_args = {
      'dtype': GlobalPlaceholder('torch.quint8'),  # activation dtype must be quint8
      'quant_min': 0,
      'quant_max': 2**n_bits - 1,
      'qscheme': GlobalPlaceholder(activation_qscheme)
  }

  weight_range = 'full'  # full, uint or symmetric
  weight_observer = 'torch.ao.quantization.PerChannelMinMaxObserver'
  weight_qscheme = f'torch.{"per_channel" if "PerChannel" in weight_observer else "per_tensor"}_{"symmetric" if weight_range == "symmetric" else "affine"}'
  weight_observer_args = {
      'dtype':
          GlobalPlaceholder('torch.qint8'),  # weight dtype must be qint8
      'quant_min':
          0 if weight_range == 'uint' else
          -2**(n_bits - 1) if weight_range == 'full' else -2**(n_bits - 1) + 1,
      'quant_max':
          2**(n_bits - 1) - 1,
      'qscheme':
          GlobalPlaceholder(weight_qscheme)
  }

  imu_input_quantization = 'qat'
  imu_pipeline_quantization = 'qat'
  imu_pipeline_fc_quantization = 'qat'
  fc_quantization = 'qat'
  output_layer_quantization = 'qat'

  quantization_mode_mapping = buildQuantizationModeMapping(
      weight_observer, weight_observer_args, activation_observer, activation_observer_args,
      imu_input_quantization, imu_pipeline_quantization, imu_pipeline_fc_quantization,
      fc_quantization, output_layer_quantization)


@ex.automain
def main(use_dataset, backend, batch_size, max_epochs, trained_model_run_id, loss_patience,
         validation_interval, quantization_mode_mapping, restore_optimizer, extra_hyper_params,
         _run) -> None:
  base_cfg = loader.find_by_id(trained_model_run_id).to_dict()['config']
  torch.backends.quantized.engine = backend

  # load training data
  if use_dataset == 'lara':
    data_module = LARaDataModule(batch_size=batch_size,
                                 window=base_cfg['window'],
                                 stride=base_cfg['stride'],
                                 sample_frequency=base_cfg['sample_frequency'])
  elif use_dataset == 'pamap2':
    data_module = Pamap2DataModule(batch_size=batch_size,
                                   window=base_cfg['window'],
                                   stride=base_cfg['stride'],
                                   sample_frequency=base_cfg['sample_frequency'],
                                   use_transient_class=base_cfg['use_transient_class'])
  elif use_dataset == 'opportunity':
    data_module = OpportunityDataModule(batch_size=batch_size,
                                        window=base_cfg['window'],
                                        stride=base_cfg['stride'],
                                        blank_invalid_columns=base_cfg['blank_invalid_columns'])
  else:
    raise ValueError(f'Unexpected dataset {use_dataset}')

  # Load the floating point model
  logger.info(f'Loading checkpoint of run {trained_model_run_id}')
  ckpt = checkpointsById(root='./logs/checkpoints', run_id=trained_model_run_id)['best_wf1']
  fp32_model = CNNIMU.load_from_checkpoint(checkpoint_path=ckpt)

  # Prepare for QAT
  setattr(fp32_model, 'quantization_mapping', quantization_mode_mapping)
  qat_model = applyQuantizationModeMapping(module=fp32_model,
                                           quantization_mode_mapping=quantization_mode_mapping,
                                           inplace=False)
  # Update optimizer config in qat_model
  qat_model.extra_hyper_params.update(extra_hyper_params)
  if restore_optimizer:
    qat_model.optimizer_restore = torch.load(ckpt)['optimizer_states'][0]

  # Run QAT
  best_wf1_ckpt = pl_cb.ModelCheckpoint(
      dirpath=getRunCheckpointDirectory(root='logs/checkpoints', _run=_run),
      filename=
      'BEST_WF1-e={epoch}-s={step}-loss={validation/loss}-acc={validation/acc}-wf1={validation/wf1}',
      auto_insert_metric_name=False,
      monitor='validation/wf1',
      mode='max',
  )
  trainer = pl.Trainer(logger=SacredLogger(_run=_run),
                       enable_checkpointing=True,
                       accelerator='cpu',
                       max_epochs=max_epochs,
                       log_every_n_steps=1,
                       callbacks=[
                           MonitorAcc(),
                           MonitorWF1(),
                           MonitorBatchTime(),
                           pl_cb.EarlyStopping(monitor='validation/loss',
                                               mode='min',
                                               min_delta=0,
                                               patience=loss_patience),
                           pl_cb.LearningRateMonitor(),
                           pl_cb.DeviceStatsMonitor(), best_wf1_ckpt
                       ],
                       val_check_interval=validation_interval)
  trainer.validate(model=qat_model, datamodule=data_module)  # Zero fit validation
  trainer.fit(model=qat_model, datamodule=data_module)

  # Convert to quantized
  qat_model = CNNIMU.load_from_checkpoint(best_wf1_ckpt.best_model_path)
  qat_model.trainer = None
  qat_model.eval()
  q_model = applyConversionAfterModeMapping(module=qat_model, inplace=False)

  # Test and save qmodel
  trainer.test(model=q_model, datamodule=data_module)
  trainer.save_checkpoint(filepath=getRunCheckpointDirectory(root='logs/checkpoints',
                                                             _run=_run).joinpath('model.ckpt'),
                          weights_only=True)
