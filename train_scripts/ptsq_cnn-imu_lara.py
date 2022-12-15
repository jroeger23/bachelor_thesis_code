import logging

import incense
import pytorch_lightning as pl
import sacred
import torch.ao.quantization as tq
from pytorch_model_summary import summary
from sacred.observers import MongoObserver

from common.data import LARaDataModule
from common.helper import checkpointsById, parseMongoConfig
from common.model import CNNIMU
from common.pl_components import MonitorAcc, MonitorBatchTime, MonitorWF1

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


@ex.automain
def main(trained_model_run_id: int, backend: str, batch_size: int):
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
  setattr(fp32_model, 'qconfig', tq.get_default_qconfig(backend=backend))
  logger.info(summary(fp32_model, data_module.test_set[0][0]))
  logger.info(f'QConfig: {fp32_model.qconfig}')
  fp32_model.fuse_modules()
  tq.prepare(model=fp32_model, inplace=True)

  # Gather calibration data
  logger.info('Gathering activation statistics on train dataset')
  trainer_statistics = pl.Trainer(logger=False, enable_checkpointing=False, accelerator='cpu')
  trainer_statistics.test(model=fp32_model, dataloaders=data_module.train_dataloader())

  # Convert to int8
  logger.info('Converting to int8')
  int8_model = tq.convert(module=fp32_model)
  logger.info(summary(int8_model, data_module.test_set[0][0]))

  # Test the quantized model
  trainer_eval = pl.Trainer(logger=False,
                            enable_checkpointing=False,
                            accelerator='cpu',
                            callbacks=[MonitorAcc(), MonitorBatchTime(),
                                       MonitorWF1()])
  logger.info('Testing int8 model')
  trainer_eval.test(model=int8_model, dataloaders=data_module.test_dataloader())
