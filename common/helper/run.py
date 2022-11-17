from sacred.run import Run
from pathlib import Path
from typing import Union


def getRunCheckpointDirectory(root: Union[str, Path], _run: Run) -> Path:
  r_path = root if isinstance(root, Path) else Path(root)
  run_dir = f'{_run.experiment_info["name"]}-{_run._id}'

  return r_path.joinpath(run_dir)