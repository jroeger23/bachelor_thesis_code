import logging
from typing import Optional

import incense
import pytorch_lightning as pl
import sacred
import torch
import torch.ao.quantization as tq
from sacred.observers import MongoObserver

from common.data import LARaDataModule, OpportunityDataModule, Pamap2DataModule
from common.helper import (GlobalPlaceholder, QConfigFactory, QuantizationMode,
                           QuantizationModeMapping, applyConversionAfterModeMapping,
                           applyQuantizationModeMapping, bestExperimentWF1, checkpointsById,
                           getRunCheckpointDirectory, parseMongoConfig)
from common.model import CNNIMU
from common.pl_components import (MonitorAcc, MonitorBatchTime, MonitorWF1, SacredLogger)

logger = logging.getLogger(__name__)

ex = sacred.Experiment(name='PTQ_CNN-IMU')
ex.observers.append(MongoObserver(**parseMongoConfig(file='./config.ini')))
loader = incense.ExperimentLoader(
    **parseMongoConfig(file='./config.ini', adapt='IncenseExperimentLoader'))


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
  static_qconfig_factory = QConfigFactory(
      GlobalPlaceholder(activation_observer, **activation_observer_args),
      GlobalPlaceholder(weight_observer, **weight_observer_args))

  dynamic_qconfig_factory = QConfigFactory(
      GlobalPlaceholder(tq.PlaceholderObserver,
                        dtype=GlobalPlaceholder('torch.quint8'),
                        quant_min=0,
                        quant_max=127,
                        is_dynamic=True), GlobalPlaceholder(weight_observer,
                                                            **weight_observer_args))
  if imu_input_quantization == 'static':
    q_block0 = QuantizationMode.ptq(False,
                                    True,
                                    operator_fuse_list=[['conv1', 'relu1'], ['conv2', 'relu2']],
                                    qconfig_factory=static_qconfig_factory)
  elif imu_input_quantization == 'dynamic':
    q_block0 = QuantizationMode.ptdq(False,
                                     False,
                                     operator_fuse_list=[['conv1', 'relu1'], ['conv2', 'relu2']],
                                     qconfig_factory=dynamic_qconfig_factory)
  elif imu_input_quantization == 'none':
    q_block0 = QuantizationMode.none(False, False, qconfig_factory=static_qconfig_factory)
  else:
    raise ValueError(f'Unknown {imu_input_quantization=}')

  if imu_pipeline_quantization == 'static':
    q_blockn = QuantizationMode.ptq(imu_input_quantization == 'static',
                                    True,
                                    operator_fuse_list=[['conv1', 'relu1'], ['conv2', 'relu2']],
                                    qconfig_factory=static_qconfig_factory)
  elif imu_pipeline_quantization == 'dynamic':
    q_blockn = QuantizationMode.ptdq(imu_input_quantization == 'static',
                                     False,
                                     operator_fuse_list=[['conv1', 'relu1'], ['conv2', 'relu2']],
                                     qconfig_factory=dynamic_qconfig_factory)
  elif imu_pipeline_quantization == 'none':
    q_blockn = QuantizationMode.none(imu_input_quantization == 'static',
                                     False,
                                     qconfig_factory=static_qconfig_factory)
  else:
    raise ValueError(f'Unknown {imu_pipeline_quantization=}')

  if imu_pipeline_fc_quantization == 'static':
    q_pfc = QuantizationMode.ptq(imu_pipeline_quantization == 'static',
                                 True,
                                 qconfig_factory=static_qconfig_factory,
                                 operator_fuse_list=['1', '2'])
  elif imu_pipeline_fc_quantization == 'dynamic':
    q_pfc = QuantizationMode.ptdq(imu_pipeline_quantization == 'static',
                                  False,
                                  qconfig_factory=dynamic_qconfig_factory,
                                  operator_fuse_list=['1', '2'])
  elif imu_pipeline_fc_quantization == 'none':
    q_pfc = QuantizationMode.none(imu_pipeline_quantization == 'static',
                                  False,
                                  qconfig_factory=static_qconfig_factory)
  else:
    raise ValueError(f'Unknown {imu_pipeline_fc_quantization=}')

  q_fc = QuantizationMode.fuse_only(operator_fuse_list=['1', '2'])
  if fc_quantization == 'static':
    q_fc_0 = QuantizationMode.ptq(imu_pipeline_fc_quantization == 'static',
                                  True, [],
                                  qconfig_factory=static_qconfig_factory)
  elif fc_quantization == 'dynamic':
    q_fc_0 = QuantizationMode.ptdq(imu_pipeline_fc_quantization == 'static',
                                   False, [],
                                   qconfig_factory=dynamic_qconfig_factory)
  elif fc_quantization == 'none':
    q_fc_0 = QuantizationMode.none(imu_pipeline_fc_quantization == 'static',
                                   False,
                                   qconfig_factory=static_qconfig_factory)
  else:
    raise ValueError(f'Unknown {fc_quantization=}')

  if output_layer_quantization == 'static':
    q_fc_1 = QuantizationMode.ptq(fc_quantization == 'static',
                                  False, [],
                                  qconfig_factory=static_qconfig_factory)
  elif output_layer_quantization == 'dynamic':
    q_fc_1 = QuantizationMode.ptdq(fc_quantization == 'static',
                                   False, [],
                                   qconfig_factory=dynamic_qconfig_factory)
  elif output_layer_quantization == 'none':
    q_fc_1 = QuantizationMode.none(fc_quantization == 'static',
                                   False,
                                   qconfig_factory=static_qconfig_factory)
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
def defautltConfig():
  use_dataset = 'lara'
  trained_model_run_id = bestRunIdByDataset(use_dataset)
  backend = 'fbgemm'
  batch_size = 128
  limit_calibration_set = None
  n_bits = 7
  activation_observer = 'torch.ao.quantization.HistogramObserver'  # Cannot be 'PerChannel'
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

  imu_input_quantization = 'static'
  imu_pipeline_quantization = 'static'
  imu_pipeline_fc_quantization = 'static'
  fc_quantization = 'static'
  output_layer_quantization = 'static'

  quantization_mode_mapping = buildQuantizationModeMapping(
      weight_observer, weight_observer_args, activation_observer, activation_observer_args,
      imu_input_quantization, imu_pipeline_quantization, imu_pipeline_fc_quantization,
      fc_quantization, output_layer_quantization)


@ex.automain
def main(use_dataset: str, trained_model_run_id: int, backend: str, batch_size: int,
         limit_calibration_set: Optional[float], quantization_mode_mapping: QuantizationModeMapping,
         _run):
  base_cfg = loader.find_by_id(trained_model_run_id).to_dict()['config']
  torch.backends.quantized.engine = backend

  # load datamodule
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
  data_module.setup('fit')
  data_module.setup('test')

  # Load checkpoint
  logger.info(f'Loading checkpoint of run {trained_model_run_id}')
  ckpt = checkpointsById(root='./logs/checkpoints', run_id=trained_model_run_id)['best_wf1']
  fp32_model = CNNIMU.load_from_checkpoint(checkpoint_path=ckpt)
  fp32_model.eval()

  # Convert to observed
  logger.info('Converting to observed model')
  fp32_model.eval()
  fp32_model.storeQuantizationModeMapping(quantization_mode_mapping)
  prepared_model = applyQuantizationModeMapping(fp32_model,
                                                quantization_mode_mapping,
                                                inplace=False)
  logger.info(f'Prepared:\n{prepared_model}')

  # Gather calibration data
  logger.info('Gathering activation statistics on train dataset')
  trainer_statistics = pl.Trainer(logger=SacredLogger(_run),
                                  enable_checkpointing=False,
                                  limit_test_batches=limit_calibration_set,
                                  callbacks=[MonitorBatchTime(on_test='calibration/batch_time')],
                                  accelerator='auto')
  trainer_statistics.test(model=prepared_model, dataloaders=data_module.train_dataloader())

  # Convert to quantized
  prepared_model.eval()
  prepared_model.trainer = None
  q_model = applyConversionAfterModeMapping(prepared_model, inplace=False)
  logger.info(f'Quantized:\n{q_model}')

  # Test the quantized model
  trainer_eval = pl.Trainer(logger=SacredLogger(_run),
                            enable_checkpointing=False,
                            accelerator='cpu',
                            callbacks=[MonitorAcc(), MonitorBatchTime(),
                                       MonitorWF1()])
  logger.info('Testing quantized model')
  trainer_eval.test(model=q_model, dataloaders=data_module.test_dataloader())

  # Save quantized model
  ckpt_path = getRunCheckpointDirectory(root='./logs/checkpoints', _run=_run).joinpath('model.ckpt')
  trainer_eval.save_checkpoint(filepath=ckpt_path, weights_only=True)
