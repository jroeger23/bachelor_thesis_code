from incense.experiment import Experiment
from typing import Iterable, List, Set


def metricBest(exs: Iterable[Experiment], metric: str, mode: str = 'max') -> Experiment:
  if mode == 'max':
    return max(exs, key=lambda e: e.metrics[metric].max())
  elif mode == 'min':
    return min(exs, key=lambda e: e.metrics[metric].min())
  else:
    raise ValueError(f'Unknown {mode=}')


def metricTopK(exs: Iterable[Experiment],
               metric: str,
               k: int,
               mode: str = 'max',
               fold: str = 'auto') -> List[Experiment]:
  if fold == 'auto':
    fold = mode

  if fold == 'max':
    key_fn = lambda e: e.metrics[metric].max()
  elif fold == 'min':
    key_fn = lambda e: e.metrics[metric].min()
  elif fold == 'median':
    key_fn = lambda e: e.metrics[metric].median()
  elif fold == 'mean':
    key_fn = lambda e: e.metrics[metric].mean()
  else:
    raise ValueError(f'Unknown {fold=}')

  if mode == 'max':
    ex_sorted = sorted(exs, key=key_fn, reverse=True)
  elif mode == 'min':
    ex_sorted = sorted(exs, key=key_fn, reverse=False)
  else:
    raise ValueError(f'Unknown {mode=}')

  return ex_sorted[:k] if k > 0 else ex_sorted


def configSubset(ex: Experiment, keys: Iterable[str]) -> dict:
  return {k: ex.to_dict()['config'][k] for k in keys}