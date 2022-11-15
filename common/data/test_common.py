import torch
import pytest

from .common import SegmentedDataset, majority_label, parse_tensor_line, describeLabels


def test_prase_tensor_line():
  data1 = [
      ('0,1,2,3,4,5,6', torch.Tensor([0, 1, 2, 3, 4, 5, 6])),
      ('6,5,4,3,2,1,0', torch.Tensor([6, 5, 4, 3, 2, 1, 0])),
      ('6,5,4,3,2,10,0', torch.Tensor([6, 5, 4, 3, 2, 10, 0])),
  ]
  data2 = [
      ('0 1    2 3 4 5    6', torch.Tensor([0, 1, 2, 3, 4, 5, 6])),
      ('6 5    4 3   2  1 0', torch.Tensor([6, 5, 4, 3, 2, 1, 0])),
      ('0 1    23 4 5   8 6', torch.Tensor([0, 1, 23, 4, 5, 8, 6])),
  ]
  data3 = [
      ('0.1.2.3.4.5.6', torch.Tensor([2, 3, 4, 5, 6])),
      ('6.5.4.3.2.1.0', torch.Tensor([4, 3, 2, 1, 0])),
      ('6.5.4.3.2.10.0', torch.Tensor([4, 3, 2, 10, 0])),
  ]

  for i, o in data1:
    assert parse_tensor_line(line=i, n_cols=7, sep_re=',').equal(o)

  for i, o in data2:
    assert parse_tensor_line(line=i, n_cols=7, sep_re='\\s+').equal(o)

  for i, o in data3:
    assert parse_tensor_line(line=i, n_cols=5, sep_re='\\.', drop_n=2).equal(o)


def test_describeLabels():
  labels_map = {
      0: 'zero',
      1: 'one',
      2: 'two',
      3: 'three',
      4: 'four',
  }

  data = [
      (torch.Tensor([0, 4, 1, 3, 2]), ['zero', 'four', 'one', 'three', 'two']),
      ([0, 4, 1, 3, 2], ['zero', 'four', 'one', 'three', 'two']),
      (4, 'four'),
  ]

  for i, o in data:
    assert describeLabels(labels_map, i) == o

  pytest.raises(ValueError, describeLabels, labels_map, torch.Tensor([[1, 4], [2, 3]]))


def test_SegmentedDataset():
  data = torch.Tensor(range(400)).reshape((40, 5, 2))
  labels = torch.Tensor(range(200)).reshape((40, 5))

  dataset = SegmentedDataset(tensor=data, labels=labels, window=10, stride=2)

  assert len(dataset) == 16
  for i in range(len(dataset)):
    segment = dataset[i][0]
    assert torch.equal(segment, data[2 * i:2 * i + 10])


def test_majority_label():
  data = torch.Tensor([
      [1, 2, 1, 4, 5],
      [1, 3, 1, 4, 2],
      [2, 2, 3, 4, 3],
      [2, 3, 1, 4, 4],
      [2, 3, 1, 4, 3],
  ])

  ret = majority_label(data)

  assert torch.equal(ret, torch.Tensor([2, 3, 1, 4, 3]))
