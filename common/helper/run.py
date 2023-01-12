from sacred.run import Run
from pathlib import Path
from typing import Union, Dict
from os.path import basename


def getRunCheckpointDirectory(root: Union[str, Path], _run: Run) -> Path:
  r_path = root if isinstance(root, Path) else Path(root)
  run_dir = f'{_run.experiment_info["name"]}-{_run._id}'

  return r_path.joinpath(run_dir)


def checkpointsById(root: Union[str, Path], run_id: int) -> Dict[str, Path]:
  """Find checpoints in a root directory by run id
  Layout
  root/
    *-{run_id}/
      BEST_ACC*.ckpt
      BEST_LOSS*.ckpt
      BEST_WF1*.ckpt

  Args:
      root (Union[str, Path]): the root dir to search in
      run_id (int): the id of the run "root/*-{run_id}/*.ckpt"

  Returns:
      Dict[str, Path]: return dict with keys 'best_acc', 'best_loss', 'best_wf1'
  """
  root_path = root if isinstance(root, Path) else Path(root)

  run_path = Path(next(r for r in root_path.iterdir() if r.name.endswith(f'-{run_id}')))

  ret = {}

  for ckpt in run_path.iterdir():
    if basename(ckpt).startswith('BEST_ACC'):
      ret['best_acc'] = ckpt
    elif basename(ckpt).startswith('BEST_LOSS'):
      ret['best_loss'] = ckpt
    elif basename(ckpt).startswith('BEST_WF1'):
      ret['best_wf1'] = ckpt
    elif basename(ckpt) == 'model.ckpt':
      ret['model'] = ckpt

  return ret