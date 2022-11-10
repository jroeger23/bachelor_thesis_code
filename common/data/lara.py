import logging
import os
import typing as t
from enum import Enum
from itertools import product
from sys import getsizeof

import torch
from si_prefix import si_format
from torch.utils.data import ConcatDataset, Dataset

from .common import (SegmentedDataset, Transform, View, ensure_download_zip, load_cached_csv)

logger = logging.getLogger(__name__)

DOWNLOAD_URL = 'https://zenodo.org/record/3862782/files/IMU data.zip'
SHA512_HEX = '3609fc9f610aec8aa01b6aa0c4c63927ccb309b68a89dab625e1673dd9724fe8da94a2c42466ee4976ebd96c72b9ef337cdc837fbc944b69172e76d7887ca868'

labels_map = {
    0: 'standing',
    1: 'walking',
    2: 'cart',
    3: 'handling (upwards)',
    4: 'handling (centered)',
    5: 'handling (downwards)',
    6: 'synchronization',
    7: 'none',
}

data_view_indices = {
    'Time': 0,
    'LA_AccelerometerX': 1,
    'LA_AccelerometerY': 2,
    'LA_AccelerometerZ': 3,
    'LA_GyroscopeX': 4,
    'LA_GyroscopeY': 5,
    'LA_GyroscopeZ': 6,
    'LL_AccelerometerX': 7,
    'LL_AccelerometerY': 8,
    'LL_AccelerometerZ': 9,
    'LL_GyroscopeX': 10,
    'LL_GyroscopeY': 11,
    'LL_GyroscopeZ': 12,
    'N_AccelerometerX': 13,
    'N_AccelerometerY': 14,
    'N_AccelerometerZ': 15,
    'N_GyroscopeX': 16,
    'N_GyroscopeY': 17,
    'N_GyroscopeZ': 18,
    'RA_AccelerometerX': 19,
    'RA_AccelerometerY': 20,
    'RA_AccelerometerZ': 21,
    'RA_GyroscopeX': 22,
    'RA_GyroscopeY': 23,
    'RA_GyroscopeZ': 24,
    'RL_AccelerometerX': 25,
    'RL_AccelerometerY': 26,
    'RL_AccelerometerZ': 27,
    'RL_GyroscopeX': 28,
    'RL_GyroscopeY': 29,
    'RL_GyroscopeZ': 30,
}

labels_view_indices = {
    'Class': 0,
    'I-A_GaitCycle': 2,
    'I-B_Step': 3,
    'I-C_StandingStill': 3,
    'II-A_Upwards': 4,
    'II-B_Centred': 5,
    'II-C_Downwards': 6,
    'II-D_NoIntentionalMotion': 7,
    'II-E_TorsoRotation': 8,
    'III-A_Right': 9,
    'III-B_Left': 10,
    'III-C_NoArms': 11,
    'IV-A_BulkyUnit': 12,
    'IV-B_HandyUnit': 13,
    'IV-C_UtilityAux': 14,
    'IV-D_Cart': 15,
    'IV-E_Computer': 16,
    'IV-F_NoItem': 17,
    'V-A_None': 18,
    'VI-A_Error': 19,
}


def describeLARaLabels(labels) -> t.List[str]:
  if type(labels) is torch.Tensor:
    if len(labels) == 1:
      return [labels_map[int(labels.item())]]
    elif labels.shape[1] == 1:
      return [labels_map[int(l.item())] for l in labels]
    else:
      raise ValueError('Cannot describe multi dimensional labels')
  elif type(labels) is t.List:
    return [labels_map[int(l)] for l in labels]
  else:
    return [labels_map[int(labels)]]


class LARaLabelsView(View):

  def __init__(self, entries: t.List[str]) -> None:
    self.entries = entries
    self.indices = [labels_view_indices[e] for e in entries]

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.atleast_2d(labels)
    return batch, labels[:, self.indices]

  def __str__(self) -> str:
    return f'LARaLabelsView({self.entries})'


class LARaDataView(View):

  def __init__(self, entries: t.List[str]) -> None:
    self.entries = entries
    self.indices = [data_view_indices[e] - 1 for e in entries]  # adjust for trimmed time

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.atleast_2d(batch)
    return batch[:, self.indices], labels

  @staticmethod
  def allEntries() -> t.List[str]:
    return list(data_view_indices.keys())

  def __str__(self) -> str:
    return f'LARaDataView({self.entries})'


class LARaClassLabelView(View):

  def __init__(self):
    self.view = LARaLabelsView(entries=['Class'])

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    return self.view(batch, labels)

  @staticmethod
  def allEntries() -> t.List[str]:
    return list(labels_view_indices.keys())

  def __str__(self) -> str:
    return f'LARaClassLabelView'


class LARaIMUView(View):

  def __init__(self, locations: t.List[str]):
    self.locations = locations
    suffixes = [
        '_AccelerometerX', '_AccelerometerY', '_AccelerometerZ', '_GyroscopeX', '_GyroscopeY',
        '_GyroscopeZ'
    ]
    entries = [l + s for l, s in product(locations, suffixes)]
    self.view = LARaDataView(entries=entries)

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    return self.view(batch, labels)

  def __str__(self) -> str:
    return f'LARaIMUView({self.locations})'


class LARaOptions(Enum):
  ALL_SUBJECTS = 100
  SUBJECT07 = 107
  SUBJECT08 = 108
  SUBJECT09 = 109
  SUBJECT10 = 110
  SUBJECT11 = 111
  SUBJECT12 = 112
  SUBJECT13 = 113
  SUBJECT14 = 114
  SCENARIO1 = 201
  SCENARIO2 = 202
  SCENARIO3 = 203
  ALL_RUNS = 300
  RUN01 = 301
  RUN02 = 302
  RUN03 = 303
  RUN04 = 304
  RUN05 = 305
  RUN06 = 306
  RUN07 = 307
  RUN08 = 308
  RUN09 = 309
  RUN10 = 310
  RUN11 = 311
  RUN12 = 312
  RUN13 = 313
  RUN14 = 314
  RUN15 = 315
  RUN16 = 316
  RUN17 = 317
  RUN18 = 318
  RUN19 = 319
  RUN20 = 320
  RUN21 = 321
  RUN22 = 322
  RUN23 = 323
  RUN24 = 324
  RUN25 = 325
  RUN26 = 326
  RUN27 = 327
  RUN28 = 328
  RUN29 = 329
  RUN30 = 330


class LARa(Dataset):

  def __init__(self,
               root: str = './data',
               window: int = 24,
               stride: int = 12,
               transform: t.Optional[Transform] = None,
               view: t.Optional[View] = None,
               download: bool = True,
               opts: t.Iterable[LARaOptions] = []):
    self.dataset_name = 'LARa'
    self.zip_dirs = ['IMU data/']
    self.root = os.path.join(root, self.dataset_name, 'IMU data')
    self.view = view

    if download:
      ensure_download_zip(url=DOWNLOAD_URL,
                          dataset_name=self.dataset_name,
                          root=root,
                          zip_dirs=self.zip_dirs,
                          sha512_hex=SHA512_HEX)

    logger.info(f'Loading LARa Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {list(map(lambda o: o.name, opts))}')
    logger.info(f'  - Transform {str(transform)}',)
    logger.info(f'  - View {str(view)}',)

    subjects = []
    if LARaOptions.SUBJECT07 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S07')
    if LARaOptions.SUBJECT08 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S08')
    if LARaOptions.SUBJECT09 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S09')
    if LARaOptions.SUBJECT10 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S10')
    if LARaOptions.SUBJECT11 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S11')
    if LARaOptions.SUBJECT12 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S12')
    if LARaOptions.SUBJECT13 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S13')
    if LARaOptions.SUBJECT14 in opts or LARaOptions.ALL_SUBJECTS in opts:
      subjects.append('S14')

    runs = []
    if LARaOptions.RUN01 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO1 in opts:
      runs.append('L01_SUBJECT_R01')
    if LARaOptions.RUN02 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO1 in opts:
      runs.append('L01_SUBJECT_R02')
    if LARaOptions.RUN03 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R03')
    if LARaOptions.RUN04 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R04')
    if LARaOptions.RUN05 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R05')
    if LARaOptions.RUN06 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R06')
    if LARaOptions.RUN07 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R07')
    if LARaOptions.RUN10 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R10')
    if LARaOptions.RUN11 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R11')
    if LARaOptions.RUN12 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R12')
    if LARaOptions.RUN13 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R13')
    if LARaOptions.RUN14 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R14')
    if LARaOptions.RUN15 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R15')
    if LARaOptions.RUN16 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts:
      runs.append('L02_SUBJECT_R16')
    if LARaOptions.RUN17 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R17')
    if LARaOptions.RUN18 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R18')
    if LARaOptions.RUN19 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R19')
    if LARaOptions.RUN20 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R20')
    if LARaOptions.RUN21 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R21')
    if LARaOptions.RUN22 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R22')
    if LARaOptions.RUN23 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R23')
    if LARaOptions.RUN24 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R24')
    if LARaOptions.RUN25 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R25')
    if LARaOptions.RUN26 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R26')
    if LARaOptions.RUN27 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R27')
    if LARaOptions.RUN28 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R28')
    if LARaOptions.RUN29 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R29')
    if LARaOptions.RUN30 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts:
      runs.append('L03_SUBJECT_R30')

    memory = 0
    data = []

    for subject, run in product(subjects, runs):
      run_fix = run.replace('SUBJECT', subject)
      name = os.path.join(subject, run_fix)

      # skip non-existing
      csv_path = os.path.join(self.root, f'{name}.csv')
      if not os.path.exists(csv_path):
        logger.debug(f'Skipping {name}')
        continue

      _, tensor = load_cached_csv(root=self.root, name=name, drop_n=1, logger=logger)
      _, labels = load_cached_csv(root=self.root, name=f'{name}_labels', logger=logger)
      memory += getsizeof(tensor.storage())
      memory += getsizeof(labels.storage())
      if transform is not None:
        tensor, labels = transform(tensor, labels)
      data.append(SegmentedDataset(tensor=tensor, labels=labels, window=window, stride=stride))

    self.data = ConcatDataset(data)

    logger.info(
        f'LARa Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B'
    )

  def __getitem__(self, index):
    return self.view(*self.data[index]) if self.view is not None else self.data[index]

  def __len__(self) -> int:
    return len(self.data)
