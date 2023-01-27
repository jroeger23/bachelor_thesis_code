import incense
from pathlib import Path
from cnn_imu_lara import ex as lara_experiment
from cnn_imu_opportunity_locomotion import ex as opportunity_experiment
from cnn_imu_pamap2 import ex as pamap2_experiment

from common.helper import parseMongoConfig

loader = incense.ExperimentLoader(
    **parseMongoConfig(file='./config.ini', adapt='IncenseExperimentLoader'))


def bestExperimentByDataset(dataset: str):
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

  return ex


bs_128 = {'batch_size': 128}
meta = {
    'my_meta': {
        'runner': Path(__file__).name,
        'flags': ['disabled_puppet'],
        'fixed_batch_size': 128
    }
}

lara_experiment.run(config_updates=bestExperimentByDataset('lara').to_dict()['config'] | bs_128,
                    meta_info=meta)
opportunity_experiment.run(
    config_updates=bestExperimentByDataset('opportunity').to_dict()['config'] | bs_128,
    meta_info=meta)
pamap2_experiment.run(config_updates=bestExperimentByDataset('pamap2').to_dict()['config'] | bs_128,
                      meta_info=meta)
