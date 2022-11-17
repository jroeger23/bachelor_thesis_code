from .metrics import f1Score, getPR, toBinary, wF1Score
import torch
from pytest import approx


def test_f1Score():
  pass


def test_getPR():
  tensor = torch.Tensor
  assert approx(getPR(tensor([0, 0, 0, 1]), tensor([0, 0, 0, 1])), rel=1e-2) == (1, 1)
  assert approx(getPR(tensor([0, 0, 1, 0]), tensor([0, 0, 0, 0])), rel=1e-2) == (0, 0)
  assert approx(getPR(tensor([0, 0, 0, 1]), tensor([1, 1, 1, 0])), rel=1e-2) == (0, 0)
  assert approx(getPR(tensor([1, 1, 1, 1]), tensor([0, 0, 1, 1])), rel=1e-2) == (1, 0.5)
  assert approx(getPR(tensor([1, 1, 1, 0]), tensor([0, 0, 1, 1])), rel=1e-2) == (0.5, 0.33)
  assert approx(getPR(tensor([1, 0, 0, 0]), tensor([1, 1, 1, 1])), rel=1e-2) == (0.25, 1)


def test_toBinary():
  labels = torch.Tensor([
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
  ])

  labels2 = torch.Tensor([0, 1, 2, 3]).type(torch.int64)

  probs = torch.Tensor([
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
  ])

  res = toBinary(labels, probs)
  res2 = toBinary(labels2, probs)
  assert res[0][0].equal(torch.Tensor([1, 0, 0, 0]))
  assert res[1][0].equal(torch.Tensor([0, 1, 0, 0]))
  assert res[2][0].equal(torch.Tensor([0, 0, 1, 0]))
  assert res[3][0].equal(torch.Tensor([0, 0, 0, 1]))
  assert res[4][0].equal(torch.Tensor([0, 0, 0, 0]))

  assert res[0][1].equal(torch.Tensor([0, 0, 0, 0]))
  assert res[1][1].equal(torch.Tensor([1, 0, 0, 0]))
  assert res[2][1].equal(torch.Tensor([0, 1, 0, 0]))
  assert res[3][1].equal(torch.Tensor([0, 0, 1, 0]))
  assert res[4][1].equal(torch.Tensor([0, 0, 0, 1]))

  assert res2[0][0].equal(torch.Tensor([1, 0, 0, 0]))
  assert res2[1][0].equal(torch.Tensor([0, 1, 0, 0]))
  assert res2[2][0].equal(torch.Tensor([0, 0, 1, 0]))
  assert res2[3][0].equal(torch.Tensor([0, 0, 0, 1]))
  assert res2[4][0].equal(torch.Tensor([0, 0, 0, 0]))

  assert res2[0][1].equal(torch.Tensor([0, 0, 0, 0]))
  assert res2[1][1].equal(torch.Tensor([1, 0, 0, 0]))
  assert res2[2][1].equal(torch.Tensor([0, 1, 0, 0]))
  assert res2[3][1].equal(torch.Tensor([0, 0, 1, 0]))
  assert res2[4][1].equal(torch.Tensor([0, 0, 0, 1]))
