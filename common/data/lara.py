import logging
import os
import typing as t
from enum import Enum
from itertools import product
from si_prefix import si_format
from torch.utils.data import ConcatDataset, Dataset
from sys import getsizeof

from .common import SegmentedDataset, ensure_download_zip, load_cached_csv

logger = logging.getLogger(__name__)

DOWNLOAD_URL = 'https://zenodo.org/record/3862782/files/IMU data.zip'
SHA512_HEX = '3609fc9f610aec8aa01b6aa0c4c63927ccb309b68a89dab625e1673dd9724fe8da94a2c42466ee4976ebd96c72b9ef337cdc837fbc944b69172e76d7887ca868'

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
  def __init__(self, root : str = './data',
               window : int = 24,
               stride : int = 12,
               transform = None,
               download : bool = True,
               opts : t.Iterable[LARaOptions] = []):
    self.dataset_name = 'LARa'  
    self.zip_dirs = [ 'IMU data/' ]
    self.root = os.path.join(root, self.dataset_name, 'IMU data')
    self.transform = None


    if download:
      ensure_download_zip(url=DOWNLOAD_URL, dataset_name=self.dataset_name, root=root, zip_dirs=self.zip_dirs, sha512_hex=SHA512_HEX)
    
    logger.info(f'Loading LARa Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {list(map(lambda o: o.name, opts))}')

    subjects = []
    if LARaOptions.SUBJECT07 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S07')
    if LARaOptions.SUBJECT08 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S08')
    if LARaOptions.SUBJECT09 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S09')
    if LARaOptions.SUBJECT10 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S10')
    if LARaOptions.SUBJECT11 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S11')
    if LARaOptions.SUBJECT12 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S12')
    if LARaOptions.SUBJECT13 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S13')
    if LARaOptions.SUBJECT14 in opts or LARaOptions.ALL_SUBJECTS in opts: subjects.append('S14')

    runs = []
    if LARaOptions.RUN01 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO1 in opts: runs.append('L01_SUBJECT_R01')
    if LARaOptions.RUN02 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO1 in opts: runs.append('L01_SUBJECT_R02')
    if LARaOptions.RUN03 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R03')
    if LARaOptions.RUN04 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R04')
    if LARaOptions.RUN05 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R05')
    if LARaOptions.RUN06 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R06')
    if LARaOptions.RUN07 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R07')
    if LARaOptions.RUN10 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R10')
    if LARaOptions.RUN11 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R11')
    if LARaOptions.RUN12 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R12')
    if LARaOptions.RUN13 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R13')
    if LARaOptions.RUN14 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R14')
    if LARaOptions.RUN15 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R15')
    if LARaOptions.RUN16 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO2 in opts: runs.append('L02_SUBJECT_R16')
    if LARaOptions.RUN17 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R17')
    if LARaOptions.RUN18 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R18')
    if LARaOptions.RUN19 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R19')
    if LARaOptions.RUN20 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R20')
    if LARaOptions.RUN21 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R21')
    if LARaOptions.RUN22 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R22')
    if LARaOptions.RUN23 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R23')
    if LARaOptions.RUN24 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R24')
    if LARaOptions.RUN25 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R25')
    if LARaOptions.RUN26 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R26')
    if LARaOptions.RUN27 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R27')
    if LARaOptions.RUN28 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R28')
    if LARaOptions.RUN29 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R29')
    if LARaOptions.RUN30 in opts or LARaOptions.ALL_RUNS in opts or LARaOptions.SCENARIO3 in opts: runs.append('L03_SUBJECT_R30')

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
      data.append(SegmentedDataset(tensor=tensor, labels=labels, window=window, stride=stride))

    self.data = ConcatDataset(data)

    logger.info(f'LARa Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B')

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)
