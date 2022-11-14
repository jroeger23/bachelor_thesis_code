import pytorch_lightning as pl
import typing as t
import torch
import logging
from common.metrics import wF1Score
from common.pl_components.generic_result_monitor import GenericResultMonitor

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