from random import sample

import matplotlib.pyplot as plt
import torch


def classProbs(y, probs):
  n_classes = probs.shape[1]
  y = torch.eye(n_classes)[y]

  class_probs = []

  for c in range(n_classes):
    class_probs.append((y[:, c], probs[:, c]))

  return class_probs


def auroc(y, probs, sample_points=20, mode='integral'):
  result = 0

  if mode == 'integral':
    tprs, fprs = rocCurve(y, probs, sample_points)
    for k in range(len(tprs) - 1):
      result += (tprs[k + 1] + tprs[k]) * (fprs[k + 1] - fprs[k])
    result /= 2
  else:
    raise ValueError(f'Invalid mode argument value "{mode}"')

  return result


def rocCurve(y, probs, sample_points=20):
  tprs = []
  fprs = []

  for t in torch.linspace(start=1 + (1 / sample_points), end=0, steps=sample_points):
    y_pred = torch.zeros(len(y))
    y_pred[probs >= t] = 1

    tp = torch.logical_and(y_pred == y, y == 1).type(torch.float).sum()
    tn = torch.logical_and(y_pred == y, y == 0).type(torch.float).sum()
    fp = torch.logical_and(y_pred != y, y == 0).type(torch.float).sum()
    fn = torch.logical_and(y_pred != y, y == 1).type(torch.float).sum()

    tprs.append(tp / (tp + fn))
    fprs.append(fp / (fp + tn))

  return tprs, fprs


def rocFigure(y, probs, sample_points=20):
  tprs, fprs = rocCurve(y, probs, sample_points)

  fig, ax = plt.subplots()
  ax.plot(fprs, tprs)
  ax.set_xlabel("False Positive Rate")
  ax.set_ylabel("True Positive Rate")

  return fig