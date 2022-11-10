import matplotlib.pyplot as plt
import torch
import typing as t


def classProbs(batch_labels: torch.Tensor,
               batch_probs: torch.Tensor) -> t.List[t.Tuple[torch.Tensor, torch.Tensor]]:
  """Get a list of binary labels and probabilies. Each list entry is for one class

  Args:
      batch_labels (torch.Tensor): the labels of a batch (index or one-hot)
      batch_probs (torch.Tensor): the batch probabilities for each class

  Raises:
      ValueError: invalid tensor shapes

  Returns:
      t.List[t.Tuple[torch.Tensor, torch.Tensor]]: binary label and probabilities per class
  """
  n_classes = batch_probs.shape[1]

  if batch_labels.ndim == 1:
    labels = torch.eye(n_classes)[batch_labels]
  elif batch_labels.ndim == 2:
    labels = batch_labels
  else:
    raise ValueError(f'batch_labels is expected to be index or one-hot encoded')

  class_probs = [(labels[:, c], batch_probs[:, c]) for c in range(n_classes)]

  return class_probs


def auroc(binary_batch_labels: torch.Tensor,
          batch_probs: torch.Tensor,
          sample_points: int = 20,
          mode: str = 'integral') -> float:
  """Calculate the area under the roc curve

  Args:
      binary_batch_labels (torch.Tensor): a batch of binary labels
      batch_probs (torch.Tensor): a batch of predictions
      sample_points (int, optional): the granularity for the prediction thresholds. Defaults to 20.
      mode (str, optional): how to get the area. Defaults to 'integral'.

  Raises:
      ValueError: if the mode is unknown

  Returns:
      float: the area under the roc curve
  """
  result = 0

  if mode == 'integral':
    tprs, fprs = rocCurve(binary_batch_labels, batch_probs, sample_points)
    for k in range(len(tprs) - 1):
      result += (tprs[k + 1] + tprs[k]) * (fprs[k + 1] - fprs[k])
    result /= 2
  else:
    raise ValueError(f'Invalid mode argument value "{mode}"')

  return result


def rocCurve(binary_batch_labels: torch.Tensor,
             batch_probs: torch.Tensor,
             sample_points: int = 20) -> t.Tuple[t.List[float], t.List[float]]:
  """Get TP and FP rates

  Args:
      binary_batch_labels (torch.Tensor): a batch of binary labels
      batch_probs (torch.Tensor): a batch of predictions
      sample_points (int, optional): the granularity for the prediction thresholds. Defaults to 20.

  Returns:
      t.Tuple[t.List[float], t.List[float]]: a list of TP and of FP rates
  """
  tprs = []
  fprs = []

  for t in torch.linspace(start=1 + (1 / sample_points), end=0, steps=sample_points):
    pred = torch.zeros(len(binary_batch_labels))
    pred[batch_probs >= t] = 1

    tp = torch.logical_and(pred == binary_batch_labels,
                           binary_batch_labels == 1).type(torch.float).sum()
    tn = torch.logical_and(pred == binary_batch_labels,
                           binary_batch_labels == 0).type(torch.float).sum()
    fp = torch.logical_and(pred != binary_batch_labels,
                           binary_batch_labels == 0).type(torch.float).sum()
    fn = torch.logical_and(pred != binary_batch_labels,
                           binary_batch_labels == 1).type(torch.float).sum()

    tprs.append(tp / (tp + fn))
    fprs.append(fp / (fp + tn))

  return tprs, fprs


def rocFigure(binary_batch_labels: torch.Tensor,
              batch_probs: torch.Tensor,
              sample_points: int = 20):
  """Plot a roc curve

  Args:
      binary_batch_labels (torch.Tensor): a batch of binary labels
      batch_probs (torch.Tensor): a batch of predictions
      sample_points (int, optional): the granularity for the prediction thresholds. Defaults to 20.

  Returns:
      _type_: _description_
  """
  tprs, fprs = rocCurve(binary_batch_labels, batch_probs, sample_points)

  fig, ax = plt.subplots()
  ax.plot(fprs, tprs)
  ax.set_xlabel("False Positive Rate")
  ax.set_ylabel("True Positive Rate")

  return fig