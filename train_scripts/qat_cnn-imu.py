import logging

import incense
import pytorch_lightning as pl
import sacred
import torch
import torch.ao.quantization as tq
from pytorch_lightning import callbacks as pl_cb
from sacred.observers import MongoObserver

from common.data import LARaDataModule, OpportunityDataModule, Pamap2DataModule
from common.helper import (GlobalPlaceholder, QConfigFactory, checkpointsById,
                           getRunCheckpointDirectory, parseMongoConfig)
from common.model import CNNIMU
from common.pl_components import (MonitorAcc, MonitorBatchTime, MonitorWF1, SacredLogger)

ex = sacred.Experiment(name='QAT_CNN-IMU')
ex.observers.append(MongoObserver(**parseMongoConfig('./config.ini')))
loader = incense.ExperimentLoader(
    **parseMongoConfig('./config.ini', adapt='IncenseExperimentLoader'))
logger = logging.getLogger(__name__)


def bestRunIdByDataset(dataset: str) -> int:
  lara_query = {
      '$and': [
          {
              'experiment.name': 'CNN-IMU_LARa'
          },
          {
              '_id': {
                  '$gte': 215
              }
          },
          {
              'status': 'COMPLETED'
          },
      ]
  }
  opportunity_query = {
      '$and': [
          {
              'experiment.name': 'CNN-IMU_Opportunity-Locomotion'
          },
          {
              '_id': {
                  '$gte': 173
              }
          },
          {
              'status': 'COMPLETED'
          },
      ]
  }
  pamap2_query = {
      '$and': [
          {
              'experiment.name': 'CNN-IMU_Pamap2(activity_labels)'
          },
          {
              '_id': {
                  '$gte': 183
              }
          },
          {
              'status': 'COMPLETED'
          },
      ]
  }
  ex_wf1 = lambda e: e.metrics['test/wf1'].max()

  if dataset == 'lara':
    ex = max(loader.find(lara_query), key=ex_wf1)
  elif dataset == 'opportunity':
    ex = max(loader.find(opportunity_query), key=ex_wf1)
  elif dataset == 'pamap2':
    ex = max(loader.find(pamap2_query), key=ex_wf1)
  else:
    raise ValueError(f'Unexpected dataset {dataset}')

  return ex.to_dict()['_id']


@ex.config
def defaultConfig():
  use_dataset = 'lara'
  base_experiment_id = bestRunIdByDataset(use_dataset)
  backend = 'fbgemm'
  batch_size = 128
  max_epochs = 10
  validation_interval = 0.3
  loss_patience = 10
  optimizer = 'Adam'
  extra_hyper_params = {}
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


@ex.automain
def main(use_dataset, backend, batch_size, max_epochs, base_experiment_id, loss_patience,
         validation_interval, activation_observer, activation_observer_args, weight_observer,
         weight_observer_args, extra_hyper_params, _run) -> None:
  base_cfg = loader.find_by_id(base_experiment_id).to_dict()['config']
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
  logger.info(f'Loading checkpoint of run {base_experiment_id}')
  ckpt = checkpointsById(root='./logs/checkpoints', run_id=base_experiment_id)['best_wf1']
  fp32_model = CNNIMU.load_from_checkpoint(checkpoint_path=ckpt)

  # Prepare for QAT
  fp32_model.fuse_modules()
  fp32_model.qconfig_factory = QConfigFactory(
      GlobalPlaceholder(tq.FakeQuantize,
                        observer=GlobalPlaceholder(activation_observer,
                                                   **activation_observer_args)),
      GlobalPlaceholder(tq.FakeQuantize,
                        observer=GlobalPlaceholder(weight_observer, **weight_observer_args)))
  fp32_model.qconfig = fp32_model.qconfig_factory.getQConfig()
  qat_model = tq.prepare_qat(model=fp32_model.train(), inplace=False)
  qat_model.quantization_state = 'QAT_TRAIN'

  # Update optimizer config in qat_model
  qat_model.extra_hyper_params.update(extra_hyper_params)

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
  q_model = tq.convert(module=qat_model, inplace=False)
  q_model.quantization_state = 'QAT_DONE'

  # Test and save qmodel
  trainer.test(model=q_model, datamodule=data_module)
  trainer.save_checkpoint(filepath=getRunCheckpointDirectory(root='logs/checkpoints',
                                                             _run=_run).joinpath('qmodel.ckpt'),
                          weights_only=True)
