import pytorch_lightning as pl
import typing as t
import torch
import logging
from common.exception import MisconfigurationError

logger = logging.getLevelName(__name__)


class GenericResultMonitor(pl.Callback):
  """A generic result monitor.
  It expects to find "validation_probs", "validation_labels", "test_probs" and "test_labels" members
  in pl_module, depending on the configuration.
  """

  def __init__(self, on_validation: t.Optional[str], on_test: t.Optional[str],
               metric: t.Callable[[t.List[torch.Tensor], t.List[torch.Tensor]], float]):
    """Create new GenericResultMonitor

    Args:
        on_validation (t.Optional[str]): the log name for the validation metric (None means don't validation metric)
        on_test (t.Optional[str]): the log name for the test metric (None means don't test metric)
        metric (t.Callable[[t.List[torch.Tensor], t.List[torch.Tensor]], float]): the metric to use
    """
    self.on_validation = on_validation
    self.on_test = on_test
    self.metric = metric

  def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    setattr(pl_module, 'validation_probs', [])
    setattr(pl_module, 'validation_labels', [])

  def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    if self.on_validation is None:
      return

    try:
      probs = getattr(pl_module, 'validation_probs')
      labels = getattr(pl_module, 'validation_labels')
    except AttributeError as e:
      msg = f'pl_module is expected to have validation_probs and validation_labels attributes set.({e})'
      logger.error(msg)
      raise MisconfigurationError(msg)

    pl_module.log(self.on_validation, self.metric(probs, labels))

  def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    setattr(pl_module, 'test_probs', [])
    setattr(pl_module, 'test_labels', [])

  def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    if self.on_test is None:
      return

    try:
      probs = getattr(pl_module, 'test_probs')
      labels = getattr(pl_module, 'test_labels')
    except KeyError as e:
      msg = f'pl_module is expected to have test_probs and test_labels attributes set.({e})'
      logger.error(msg)
      raise MisconfigurationError(msg)

    pl_module.log(self.on_test, self.metric(probs, labels))