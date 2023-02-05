from typing import Union, Optional, Tuple
from pathlib import Path
from incense import ExperimentLoader
from incense.experiment import Experiment
from common.helper import parseMongoConfig
import logging

logger = logging.getLogger(__name__)

def bestExperimentWF1(loader: Union[str, Path, ExperimentLoader],
                      experiment_name: str,
                      min_id: Optional[int] = None,
                      max_id: Optional[int] = None,
                      hostname: Optional[str] = None,
                      my_meta: Optional[dict] = None) -> Tuple[Experiment, int]:
  if isinstance(loader, str) or isinstance(loader, Path):
    loader = ExperimentLoader(**parseMongoConfig(file=loader, adapt='IncenseExperimentLoader'))

  id_query = {
      '_id': ({} if min_id is None else {
          '$gte': min_id
      }) | ({} if max_id is None else {
          '$lte': max_id
      })
  } if min_id is not None or max_id is not None else None

  my_meta_query = {
      '$and': [{
          f'meta.my_meta.{k}': v
      } for k, v in my_meta.items()]
  } if my_meta is not None else None

  hostname_query = {'host.hostname': hostname} if hostname is not None else None

  query = {
      '$and': [
          {
              'experiment.name': experiment_name
          },
          {
              'status': 'COMPLETED'
          },
      ] + ([id_query] if id_query is not None else []) +
              ([my_meta_query] if my_meta_query is not None else []) +
              ([hostname_query] if hostname_query is not None else [])
  }
  logger.debug(f'bestExperimentWF1 - query {query}')

  experiments = loader.find(query)

  assert len(experiments) != 0, "Expected to find Experiments"

  best = max(experiments, key=lambda e: e.metrics['test/wf1'].max())

  return best, best.to_dict()['_id']
