from pytorch_lightning import loggers as pl_log
from sacred.run import Run
from typing import Optional
from datetime import datetime


class SacredLogger(pl_log.Logger):
  """A Sacred Logger /[T]/
  """

  def __init__(self, _run: Run, experiment_version: Optional[str] = None):
    """Create a new Sacred Logger using a sacred.run.Run object

    Args:
        _run (Run): the _run object
        experiment_version (Optional[str], optional): Experiment Version string, if none use the current timestamp. Defaults to None.
    """
    self.experiment_name = _run.experiment_info['name']
    self.experiment_version = experiment_version or str(datetime.now().timestamp())
    self._run = _run

  @property
  def name(self) -> str:
    return self.experiment_name

  @property
  def version(self):
    return self.experiment_version

  def log_metrics(self, metrics, step=None):
    for m, v in metrics.items():
      self._run.log_scalar(metric_name=m, value=v, step=step)

  def log_hyperparams(self, params, *args, **kwargs):
    # This was already done by sacred
    pass
