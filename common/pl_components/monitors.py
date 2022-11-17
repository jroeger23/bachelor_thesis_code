import pytorch_lightning as pl
import typing as t
import torch
import logging
from common.helper.metrics import wF1Score
from common.pl_components.generic_result_monitor import GenericResultMonitor
from common.pl_components.model_profiler import Timer

logger = logging.getLogger(__name__)


def getWF1(probs: t.List[torch.Tensor], labels: t.List[torch.Tensor]) -> float:
  probs_cpu = torch.row_stack(probs).detach().cpu()
  labels_cpu = torch.concat(labels).detach().cpu()
  n_classes = probs_cpu.shape[1]

  preds = probs_cpu.argmax(dim=1)
  preds_onehot = torch.eye(n_classes)[preds]
  return wF1Score(labels_cpu, preds_onehot)


def getAcc(probs: t.List[torch.Tensor], labels: t.List[torch.Tensor]) -> float:
  probs_device = torch.row_stack(probs)
  labels_device = torch.concat(labels)

  preds = probs_device.argmax(dim=1)
  hits = (preds == labels_device).type(torch.float).sum().item()
  return hits / len(preds)


class MonitorWF1(GenericResultMonitor):
  """Monitor the weighted F1 score
  """

  def __init__(self,
               on_validation: t.Optional[str] = "validation/wf1",
               on_test: t.Optional[str] = "test/wf1"):
    """Weighted F1 monitor for test/val 

    Args:
        on_validation (t.Optional[str], optional): validation metric log name. Defaults to "validation/wf1".
        on_test (t.Optional[str], optional): test metric log name. Defaults to "test/wf1".
    """

    super().__init__(on_validation=on_validation, on_test=on_test, metric=getWF1)


class MonitorAcc(GenericResultMonitor):
  """Monitor the overall accuracy
  """

  def __init__(self,
               on_validation: t.Optional[str] = "validation/acc",
               on_test: t.Optional[str] = "test/acc"):
    """Accuracy monitor for test/val 

    Args:
        on_validation (t.Optional[str], optional): validation metric log name. Defaults to "validation/wf1".
        on_test (t.Optional[str], optional): test metric log name. Defaults to "test/wf1".
    """

    super().__init__(on_validation=on_validation, on_test=on_test, metric=getAcc)


class MonitorBatchTime(pl.Callback):
  """Monitor all batch times

  Args:
      pl (_type_): _description_
  """

  def __init__(self):
    self.train_batch_times = []
    self.validation_batch_times = []
    self.test_batch_times = []
    self.train_timer = Timer()
    self.validation_timer = Timer()
    self.test_timer = Timer()

  def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch,
                           batch_idx) -> None:
    self.train_timer.start()

  def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch,
                         batch_idx) -> None:
    time = self.train_timer.stop()
    pl_module.log(name='train/batch_time', value=time)
    self.train_batch_times.append(time)

  def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch,
                                batch_idx, dataloader_idx) -> None:
    self.validation_timer.start()

  def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs,
                              batch, batch_idx, dataloader_idx) -> None:
    time = self.validation_timer.stop()
    pl_module.log(name='validation/batch_time', value=time)
    self.validation_batch_times.append(time)

  def on_test_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch,
                          batch_idx, dataloader_idx) -> None:
    self.test_timer.start()

  def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch,
                        batch_idx, dataloader_idx) -> None:
    time = self.test_timer.stop()
    pl_module.log(name='test/batch_time', value=time)
    self.test_batch_times.append(time)
