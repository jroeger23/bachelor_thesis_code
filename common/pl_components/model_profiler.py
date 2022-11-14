from pytorch_lightning import profiler as pl_p
from typing import Optional, Union
from pathlib import Path
from time import perf_counter
from datetime import datetime
from si_prefix import si_format
import torch


def actionIsLightningModule(action: str) -> bool:
  """ Decide, if action belongs to a lightning module

  Args:
      action (str): action name

  Returns:
      bool: does action start with "[LightningModule]"
  """
  return action.startswith('[LightningModule]')


def actionMethod(action: str) -> str:
  """Return the medhod of an action name (string after last '.')

  Args:
      action (str): the action name

  Returns:
      str: the action method name
  """
  return action.split('.')[-1]


class Timer:
  """A Performance timer
  """

  def __init__(self) -> None:
    """A Performance timer
    """
    self.start_time = None

  def start(self) -> None:
    """Start the timer

    Raises:
        ValueError: if it was already started before the next stop() call
    """
    if not self.start_time is None:
      raise ValueError('start() already called')
    self.start_time = perf_counter()

  def stop(self) -> float:
    """Stop the timer

    Raises:
        ValueError: if start() was not called before

    Returns:
        float: the time between start and stop in seconds
    """
    if self.start_time is None:
      raise ValueError('No prior call to start()')

    total = perf_counter() - self.start_time
    self.start_time = None
    return total


class ModelProfiler(pl_p.Profiler):
  """Profile a Models fitting and test times
  """

  def __init__(self,
               dirpath: Optional[Union[str, Path]] = None,
               filename: Optional[str] = None) -> None:
    """Profile a Models fitting and test times

    Args:
        dirpath (Optional[Union[str, Path]], optional): log directory path. Defaults to None.
        filename (Optional[str], optional): log file name. Defaults to None.
    """
    super().__init__(dirpath, filename)

    self.start_time = datetime.now()
    self.train_times = []
    self.test_times = []
    self.train_batch_timer = Timer()
    self.test_batch_timer = Timer()

  def start(self, action_name: str) -> None:
    if not actionIsLightningModule(action_name):
      return
    method = actionMethod(action_name)

    if method == 'on_train_batch_end':
      self.train_times.append(self.train_batch_timer.stop())
    elif method == 'on_test_batch_end':
      self.test_times.append(self.test_batch_timer.stop())

  def stop(self, action_name: str) -> None:
    if not actionIsLightningModule(action_name):
      return
    method = actionMethod(action_name)

    if method == 'on_train_batch_start':
      self.train_batch_timer.start()
    elif method == 'on_test_batch_start':
      self.test_batch_timer.start()

  def summary(self) -> str:
    train_times = torch.tensor(self.train_times)
    test_times = torch.tensor(self.test_times)

    fmt = '{value}{prefix}s'

    def summaryQ(data: torch.Tensor, quartiles: torch.Tensor) -> str:
      s = ""
      for v, q in zip(data.quantile(q=quartiles), quartiles):
        s += f'  - Q{q:.02f} = {si_format(v, 2, fmt)}\n'

      return s

    quantiles = torch.tensor([0.25, 0.5, 0.75])
    summary = f'\n{"-"*80}\nModelProfiler ({self.start_time})\n{"-"*20}\n'
    if len(self.train_times) != 0:
      summary += f'Total Train Time:         {si_format(train_times.sum(), 2, fmt)}\n'
      summary += f'Average Batch Train Time: {si_format(train_times.mean(), 2, fmt)}\n'
      summary += f'Train Batch Quantiles:\n{summaryQ(train_times, quantiles)}\n'
    if len(self.test_times) != 0:
      summary += f'Total Test Time:          {si_format(test_times.sum(), 2, fmt)}\n'
      summary += f'Average Batch Test Time:  {si_format(test_times.mean(), 2, fmt)}\n'
      summary += f'Test Batch Quantiles:\n{summaryQ(test_times, quantiles)}\n'

    return summary

  def teardown(self, stage: Optional[str] = None) -> None:
    self.train_times = []
    self.test_times = []
    return super().teardown(stage)