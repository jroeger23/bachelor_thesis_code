from genericpath import getsize
import logging
import os
import typing as t
from enum import Enum
from itertools import product
from sys import getsizeof

import torch
from si_prefix import si_format
from torch.utils.data import Dataset, ConcatDataset

from .common import SegmentedDataset, load_cached_dat

logger = logging.getLogger(__name__)


def parse_column_description_line(columns : t.MutableMapping[int, str], line : str):
  if line.isspace():
    return

  if not line.startswith("Column:"):
    return

  ix_s = line.find(' ') + 1
  ix_e = line.find(' ', ix_s + 1)

  ix = int(line[ix_s:ix_e])

  dsc_s = line.find(' ', ix_e) + 1
  dsc_e = line.find(';', dsc_s + 1)

  dsc = line[dsc_s:dsc_e]

  columns[ix] = dsc

def parse_columns_file(path : str) -> t.Mapping[int, str]:
  columns = {}

  with open(file=path, mode='r') as f:
    for line in f.readlines():
      parse_column_description_line(columns=columns, line=line)

  return columns

def split_data_record(tensor):
  time = tensor[:,0]
  data = tensor[:, 1:243]
  labels = tensor[:, 243:250]

  return time,data,labels


class OpportunityOptions(Enum):
  SUBJECT1 = 1
  SUBJECT2 = 2
  SUBJECT3 = 3
  SUBJECT4 = 4
  ALL_SUBJECTS = 5
  ADL1 = 6
  ADL2 = 7
  ADL3 = 8
  ADL4 = 9
  ADL5 = 10
  ALL_ADL = 11
  DRILL = 12
  LOCOMOTION = 13
  NO_OBJECTS = 14

class Opportunity(Dataset):
  def __init__(self, root : str,
               window : int = 24,
               stride : int = 12,
               transform = None,
               opts : t.Iterable[OpportunityOptions] = []):
    self.root = root
    self.transform = None
    
    logger.info(f'Loading Opportunity Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {opts}')

    self.columns = parse_columns_file(os.path.join(self.root, 'column_names.txt'))

    suffixes = []
    if OpportunityOptions.ADL1 in opts or OpportunityOptions.ALL_ADL in opts: suffixes.append('-ADL1')
    if OpportunityOptions.ADL2 in opts or OpportunityOptions.ALL_ADL in opts: suffixes.append('-ADL2')
    if OpportunityOptions.ADL3 in opts or OpportunityOptions.ALL_ADL in opts: suffixes.append('-ADL3')
    if OpportunityOptions.ADL4 in opts or OpportunityOptions.ALL_ADL in opts: suffixes.append('-ADL4')
    if OpportunityOptions.ADL5 in opts or OpportunityOptions.ALL_ADL in opts: suffixes.append('-ADL5')
    if OpportunityOptions.DRILL in opts or OpportunityOptions.ALL_ADL in opts: suffixes.append('-Drill')

    prefixes = []
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT1 in opts: prefixes.append('S1')
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT2 in opts: prefixes.append('S2')
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT3 in opts: prefixes.append('S3')
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT4 in opts: prefixes.append('S4')

    data = []
    memory = 0
    for prefix, suffix in product(prefixes, suffixes):
      raw = load_cached_dat(root=self.root, name=prefix+suffix, logger=logger)
      memory += getsizeof(raw.storage())
      _, tensor, labels = split_data_record(raw)
      data.append(SegmentedDataset(tensor=tensor, labels=labels, window=24, stride=12))

    self.data = ConcatDataset(data)

    logger.info(f'Opportunity Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B')


  def __getitem__(self, index):
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)
