import torch

from .pamap2 import Pamap2IMUView, Pamap2SplitIMUView, Pamap2View


def test_pamap2ViewOrder():
  view = Pamap2View(entries=Pamap2View.allEntries())

  test = torch.Tensor(range(52 * 10)).reshape((10, 52))

  batch, _ = view(test, torch.Tensor())

  assert batch.equal(test)


def test_pamap2IMUViewOrder():
  view1 = Pamap2IMUView(locations=Pamap2IMUView.allLocations(), with_heart_rate=True)
  view2 = Pamap2IMUView(locations=Pamap2IMUView.allLocations(), with_heart_rate=False)

  test = torch.Tensor(range(52 * 10)).reshape((10, 52))

  batch1, _ = view1(test, torch.Tensor())
  batch2, _ = view2(test, torch.Tensor())

  assert batch1.equal(test)
  assert batch2.equal(test[:, 1:])


def test_pamap2SplitIMUViewOrder():
  view = Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations())

  test = torch.Tensor(range(52 * 10)).reshape((10, 52))

  batches, _ = view(test, torch.Tensor())

  assert batches[0].equal(test[:, 1:18])
  assert batches[1].equal(test[:, 18:35])
  assert batches[2].equal(test[:, 35:52])