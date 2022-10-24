import logging
import typing as t
from enum import Enum
from itertools import chain
from sys import getsizeof

import torch
from si_prefix import si_format
from torch.utils.data import ConcatDataset, Dataset

from .common import SegmentedDataset, load_cached_dat

logger = logging.getLogger(__name__)

def split_data_record(raw : torch.Tensor):
  timestamp = raw[:, 0]
  label = raw[:, 1]
  tensor = raw[:, 2:]

  return timestamp, label, tensor

view_indices = {
  'heart_rate' : [0],
  'imu_hand' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
  'imu_chest' : [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34],
  'imu_ankle' : [35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51],
}

labels_map = {
  0 : 'other (transient activities)',
  1 : 'lying',
  2 : 'sitting',
  3 : 'standing',
  4 : 'walking',
  5 : 'running',
  6 : 'cycling',
  7 : 'Nordic walking',
  9 : 'watching TV',
  10 : 'computer work',
  11 : 'car driving',
  12 : 'ascending stairs',
  13 : 'descending stairs',
  16 : 'vacuum cleaning',
  17 : 'ironing',
  18 : 'folding laundry',
  19 : 'house cleaning',
  20 : 'playing soccer',
  24 : 'rope jumping',
}

class Pamap2View():
  def __init__(self, entries : t.List[str]) -> None:
    self.indices = []
    for e in entries:
      self.indices.extend(view_indices[e])

  def __call__(self, batch : torch.Tensor):
    batch = torch.atleast_2d(batch)
    return batch[:, self.indices]

  def describeLabels(labels) -> t.List[str]:
    if labels is torch.Tensor:
      if len(labels) == 1:
        return labels_map[int(l.item())]
      else:
        ret = []
        for l in labels:
          ret.append(labels_map[int(l.item())])
        return ret
    else:
      return labels_map[int(labels)]




  def getEntries() -> t.List[str]:
    return view_indices.keys()

class Pamap2Options(Enum):
  SUBJECT1 = 1
  SUBJECT2 = 2
  SUBJECT3 = 3
  SUBJECT4 = 4
  SUBJECT5 = 5
  SUBJECT6 = 6
  SUBJECT7 = 7
  SUBJECT8 = 8
  SUBJECT9 = 9
  ALL_SUBJECTS = 10
  OPTIONAL1 = 201
  OPTIONAL5 = 205
  OPTIONAL6 = 206
  OPTIONAL8 = 208
  OPTIONAL9 = 209
  ALL_OPTIONAL = 210
  FULL = 300


class Pamap2(Dataset):
  def __init__(self, root : str,
               window : int = 24,
               stride : int = 12,
               transform = None,
               opts : t.Iterable[Pamap2Options] = []):
    self.root = root
    self.transform = None
    
    logger.info(f'Loading Pamap2 Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {list(map(lambda o: o.name, opts))}')

    subjects = []
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT1 in opts: subjects.append('1')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT2 in opts: subjects.append('2')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT3 in opts: subjects.append('3')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT4 in opts: subjects.append('4')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT5 in opts: subjects.append('5')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT6 in opts: subjects.append('6')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT7 in opts: subjects.append('7')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT8 in opts: subjects.append('8')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT9 in opts: subjects.append('9')

    optional = []
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL1 in opts: optional.append('1')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL5 in opts: optional.append('5')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL6 in opts: optional.append('6')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL8 in opts: optional.append('8')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL9 in opts: optional.append('9')

    subjects = map(lambda p: f'Protocol/subject10{p}', subjects)
    optional = map(lambda p: f'Optional/subject10{p}', optional)

    data = []
    memory = 0

    for name in chain(subjects, optional):
      raw = load_cached_dat(root=self.root, name=name, logger=logger)
      memory += getsizeof(raw.storage())
      _, labels, tensor = split_data_record(raw)
      data.append(SegmentedDataset(tensor=tensor, labels=labels, window=window, stride=stride))
    
    self.data = ConcatDataset(data)

    logger.info(f'Pamap2 Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B')


  def __getitem__(self, index):
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)
