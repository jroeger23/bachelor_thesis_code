from .common import SegmentedDataset, majority_label

import torch


def test_SegmentedDataset():
  data = torch.Tensor(range(400)).reshape((40,5,2))
  labels = torch.Tensor(range(200)).reshape((40,5))

  dataset = SegmentedDataset(tensor=data, labels=labels, window=10, stride=2)

  assert len(dataset) == 16
  for i in range(len(dataset)):
    segment = dataset[i][0]
    assert torch.equal(segment, data[2*i:2*i + 10])


def test_majority_label():
  data = torch.Tensor([[1, 2, 1, 4, 5],
                       [1, 3, 1, 4, 2],
                       [2, 2, 3, 4, 3],
                       [2, 3, 1, 4, 4],
                       [2, 3, 1, 4, 3],
                       ])

  ret = majority_label(data)

  assert torch.equal(ret, torch.Tensor([2,3,1,4,3]))