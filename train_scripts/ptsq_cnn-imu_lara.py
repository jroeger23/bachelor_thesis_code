import logging
from typing import Any, Dict

import incense
import pytorch_lightning as pl
import sacred
import torch
import torch.ao.quantization as tq
from pytorch_model_summary import summary
from sacred.observers import MongoObserver

from common.data import LARaDataModule
from common.helper import (GlobalPlaceholder, QConfigFactory, checkpointsById,
                           getRunCheckpointDirectory, parseMongoConfig)
from common.model import CNNIMU
from common.pl_components import (MonitorAcc, MonitorBatchTime, MonitorWF1, SacredLogger)

logger = logging.getLogger(__name__)

ex = sacred.Experiment(name='PTSQ_CNN-IMU_LARa')
ex.observers.append(MongoObserver.create(**parseMongoConfig('./config.ini')))


def bestRunId() -> int:
  loader = incense.ExperimentLoader(
      **parseMongoConfig('./config.ini', adapt='IncenseExperimentLoader'))
  query = {
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
  experiments = loader.find(query)

  best_wf1 = max(experiments, key=lambda e: e.metrics['test/wf1'].max())
  return best_wf1.to_dict()['_id']


@ex.config
def defautltConfig():
  trained_model_run_id = bestRunId()
  backend = 'fbgemm'
  batch_size = 32
  activation_observer = 'torch.ao.quantization.PlaceholderObserver'
  activation_qscheme = 'torch.per_tensor_affine'
  activation_observer_args = {
      'dtype': GlobalPlaceholder('torch.quint8'),
      'quant_min': 0,
      'quant_max': 2**7 - 1,
      'qscheme': GlobalPlaceholder(activation_qscheme)
  }
  weight_observer = 'torch.ao.quantization.PerChannelMinMaxObserver'
  weight_qscheme = 'torch.per_tensor_symmetric'
  weight_observer_args = {
      'dtype': GlobalPlaceholder('torch.qint8'),
      'quant_min': -2**6,
      'quant_max': 2**6 - 1,
      'qscheme': GlobalPlaceholder(weight_qscheme)
  }


@ex.automain
def main(trained_model_run_id: int, backend: str, batch_size: int, activation_observer: str,
         activation_observer_args: Dict[str, Any], weight_observer: str,
         weight_observer_args: Dict[str, Any], _run):
  torch.backends.quantized.engine = backend

  # Data module
  data_module = LARaDataModule(batch_size=batch_size, window=100, stride=12, sample_frequency=30)
  data_module.setup('fit')
  data_module.setup('test')

  # Load checkpoint
  logger.info(f'Loading checkpoint of run {trained_model_run_id}')
  ckpt = checkpointsById(root='./logs/checkpoints', run_id=trained_model_run_id)['best_wf1']
  fp32_model = CNNIMU.load_from_checkpoint(checkpoint_path=ckpt)
  fp32_model.eval()

  # Prepare for quantization
  fp32_model.qconfig_factory = QConfigFactory(
      GlobalPlaceholder(activation_observer, **activation_observer_args),
      GlobalPlaceholder(weight_observer, **weight_observer_args))
  setattr(fp32_model, 'qconfig', fp32_model.qconfig_factory.getQConfig())
  logger.info(summary(fp32_model, data_module.test_set[0][0]))
  logger.info(f'QConfig: {fp32_model.qconfig}')
  fp32_model.fuse_modules()
  tq.prepare(model=fp32_model, inplace=True)

  # Gather calibration data
  logger.info('Gathering activation statistics on train dataset')
  trainer_statistics = pl.Trainer(logger=False,
                                  enable_checkpointing=False,
                                  limit_test_batches=0.05,
                                  accelerator='cpu')
  trainer_statistics.test(model=fp32_model, dataloaders=data_module.train_dataloader())

  # Convert to quantized
  logger.info('Converting to quantized model')
  q_model = tq.convert(module=fp32_model)
  logger.info(summary(q_model, data_module.test_set[0][0]))

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
  trainer_eval.save_checkpoint(filepath=ckpt_path)

  # Try to load the model
  logger.info('Testing loaded quantized model')
  loaded_model = CNNIMU.load_from_checkpoint(ckpt_path)
  trainer_eval.test(model=loaded_model, dataloaders=data_module.test_dataloader())
