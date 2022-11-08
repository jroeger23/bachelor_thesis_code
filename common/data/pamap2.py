import logging
import os
import typing as t
from enum import Enum
from itertools import chain, product
from sys import getsizeof

import torch
from si_prefix import si_format
from torch.utils.data import ConcatDataset, Dataset

from .common import SegmentedDataset, Transform, ensure_download_zip, load_cached_dat

logger = logging.getLogger(__name__)

DOWNLOAD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip'
SHA512_HEX = '14e96ccbc985abab0df796a09521506001eee02cf43e85fdc3dcadfca6ecd3f7b457daf5ce35a3b63251f746619b036d18582a7cee35505501a526af1bb397fd'


def split_data_record(raw: torch.Tensor):
  timestamp = raw[:, 0]
  label = raw[:, 1]
  tensor = raw[:, 2:]

  return timestamp, label, tensor


view_indices = {
    'heart_rate': [0],
    'imu_hand': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'imu_chest': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
    'imu_ankle': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}

view_indices = {
    'heart_rate': 0,
    'imu_h_temp': 1,
    'imu_h_acc1_x': 2,
    'imu_h_acc1_y': 3,
    'imu_h_acc1_z': 4,
    'imu_h_acc2_x': 5,
    'imu_h_acc2_y': 6,
    'imu_h_acc2_z': 7,
    'imu_h_gyro_x': 8,
    'imu_h_gyro_y': 9,
    'imu_h_gyro_z': 10,
    'imu_h_magn_x': 11,
    'imu_h_magn_y': 12,
    'imu_h_magn_z': 13,
    'imu_h_orient1': 14,
    'imu_h_orient2': 15,
    'imu_h_orient3': 16,
    'imu_h_orient4': 17,
    'imu_c_temp': 18,
    'imu_c_acc1_x': 19,
    'imu_c_acc1_y': 20,
    'imu_c_acc1_z': 21,
    'imu_c_acc2_x': 22,
    'imu_c_acc2_y': 23,
    'imu_c_acc2_z': 24,
    'imu_c_gyro_x': 25,
    'imu_c_gyro_y': 26,
    'imu_c_gyro_z': 27,
    'imu_c_magn_x': 28,
    'imu_c_magn_y': 29,
    'imu_c_magn_z': 30,
    'imu_c_orient1': 31,
    'imu_c_orient2': 32,
    'imu_c_orient3': 33,
    'imu_c_orient4': 34,
    'imu_a_temp': 35,
    'imu_a_acc1_x': 36,
    'imu_a_acc1_y': 37,
    'imu_a_acc1_z': 38,
    'imu_a_acc2_x': 39,
    'imu_a_acc2_y': 40,
    'imu_a_acc2_z': 41,
    'imu_a_gyro_x': 42,
    'imu_a_gyro_y': 43,
    'imu_a_gyro_z': 44,
    'imu_a_magn_x': 45,
    'imu_a_magn_y': 46,
    'imu_a_magn_z': 47,
    'imu_a_orient1': 48,
    'imu_a_orient2': 49,
    'imu_a_orient3': 50,
    'imu_a_orient4': 51,
}

labels_map = {
    0: 'other (transient activities)',
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'Nordic walking',
    9: 'watching TV',
    10: 'computer work',
    11: 'car driving',
    12: 'ascending stairs',
    13: 'descending stairs',
    16: 'vacuum cleaning',
    17: 'ironing',
    18: 'folding laundry',
    19: 'house cleaning',
    20: 'playing soccer',
    24: 'rope jumping',
}


def describePamap2Labels(labels) -> t.List[str]:
  if type(labels) is torch.Tensor:
    if len(labels) == 1:
      return labels_map[int(labels.item())]
    elif labels.shape[1] == 1:
      return [labels_map[int(l.item())] for l in labels]
    else:
      raise ValueError('Cannot describe multi dimensional labels')
  elif type(labels) is t.List:
    return [labels_map[int(l)] for l in labels]
  else:
    return labels_map[int(labels)]


class Pamap2View(Transform):

  def __init__(self, entries: t.List[str]) -> None:
    self.entries = entries
    self.indices = [view_indices[e] for e in entries]

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.atleast_2d(batch)
    return batch[:, self.indices], labels

  def __str__(self) -> str:
    return f'Pamap2View({self.entries})'

  def entries() -> t.List[str]:
    return list(view_indices.keys())


class Pamap2IMUView(Transform):

  def __init__(self, locations: t.List[str], with_heart_rate: bool = True) -> None:
    self.locations = locations
    self.with_heart_rate = with_heart_rate
    suffixes = [
        '_temp', '_acc1_x', '_acc1_y', '_acc1_z', '_acc2_x', '_acc2_y', '_acc2_z', '_gyro_x',
        '_gyro_y', '_gyro_z', '_magn_x', '_magn_y', '_magn_z', '_orient1', '_orient2', '_orient3',
        '_orient4'
    ]
    entries = [l + s for l, s in product(locations, suffixes)]
    if with_heart_rate:
      entries = ['heart_rate'] + entries
    self.view = Pamap2View(entries=entries)

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    return self.view(batch, labels)

  def __str__(self) -> str:
    return f'Pamap2IMUView({self.locations}, with_heart_rate={self.with_heart_rate})'

  def locations() -> t.List[str]:
    return ['imu_h', 'imu_c', 'imu_a']


class Pamap2SplitIMUView(Transform):

  def __init__(self, locations: t.List[str]):
    self.locations = locations
    self.views = [Pamap2IMUView(locations=[l], with_heart_rate=False) for l in locations]

  def __call__(self, batch: torch.Tensor, labels: torch.Tensor):
    return [v(batch, torch.Tensor())[0] for v in self.views], labels

  def __str__(self) -> str:
    return f'Pamap2SplitView({self.locations})'

  def locations() -> t.List[str]:
    return Pamap2IMUView.locations()


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

  def __init__(self,
               root: str = './data',
               window: int = 24,
               stride: int = 12,
               transform=None,
               download: bool = True,
               opts: t.Iterable[Pamap2Options] = []):
    self.dataset_name = 'pamap2'
    self.zip_dirs = ['PAMAP2_Dataset/Protocol/', 'PAMAP2_Dataset/Optional/']
    self.root = os.path.join(root, self.dataset_name, 'PAMAP2_Dataset')
    self.transform = None

    if download:
      ensure_download_zip(url=DOWNLOAD_URL,
                          dataset_name=self.dataset_name,
                          root=root,
                          zip_dirs=self.zip_dirs,
                          sha512_hex=SHA512_HEX)

    logger.info(f'Loading Pamap2 Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {list(map(lambda o: o.name, opts))}')

    subjects = []
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT1 in opts:
      subjects.append('1')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT2 in opts:
      subjects.append('2')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT3 in opts:
      subjects.append('3')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT4 in opts:
      subjects.append('4')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT5 in opts:
      subjects.append('5')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT6 in opts:
      subjects.append('6')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT7 in opts:
      subjects.append('7')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT8 in opts:
      subjects.append('8')
    if Pamap2Options.ALL_SUBJECTS in opts or Pamap2Options.FULL in opts or Pamap2Options.SUBJECT9 in opts:
      subjects.append('9')

    optional = []
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL1 in opts:
      optional.append('1')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL5 in opts:
      optional.append('5')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL6 in opts:
      optional.append('6')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL8 in opts:
      optional.append('8')
    if Pamap2Options.ALL_OPTIONAL in opts or Pamap2Options.FULL in opts or Pamap2Options.OPTIONAL9 in opts:
      optional.append('9')

    subjects = map(lambda p: f'Protocol/subject10{p}', subjects)
    optional = map(lambda p: f'Optional/subject10{p}', optional)

    data = []
    memory = 0

    for name in chain(subjects, optional):
      raw = load_cached_dat(root=self.root, name=name, logger=logger)
      memory += getsizeof(raw.storage())
      _, labels, tensor = split_data_record(raw)
      if not transform is None:
        tensor, labels = transform(tensor, labels)
      data.append(SegmentedDataset(tensor=tensor, labels=labels, window=window, stride=stride))

    self.data = ConcatDataset(data)

    logger.info(
        f'Pamap2 Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B'
    )

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)
