from typing import Tuple, Callable, Any, Dict, List, Iterable
from collections import defaultdict


def groupByConfigKeys(cfg_keys: Tuple[Any, ...], to_cfg: Callable[[Any], Dict[Any, Any]],
                      items: Iterable[Any]) -> Dict[Tuple[Any, ...], List[Any]]:
  groups = defaultdict(list)

  for item in items:
    cfg = to_cfg(item)
    cfg_key = tuple([cfg[key] for key in cfg_keys])
    groups[cfg_key].append(item)

  return groups
