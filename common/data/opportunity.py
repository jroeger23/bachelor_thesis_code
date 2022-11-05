import logging
import os
import typing as t
from enum import Enum
from itertools import product
from sys import getsizeof

import torch
from si_prefix import si_format
from torch.utils.data import Dataset, ConcatDataset

from .common import SegmentedDataset, load_cached_dat, ensure_download_zip

logger = logging.getLogger(__name__)

download_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'

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

view_indices = {
  'Accelerometer RKN^' : [0,1,2],
  'Accelerometer HIP' : [3,4,5],
  'Accelerometer LUA^' : [6,7,8],
  'Accelerometer RUA_' : [9,10,11],
  'Accelerometer LH' : [12,13,14],
  'Accelerometer BACK' : [15,16,17],
  'Accelerometer RKN_' : [18,19,20],
  'Accelerometer RWR' : [21,22,23],
  'Accelerometer RUA^' : [24,25,26],
  'Accelerometer LUA_' : [27,28,29],
  'Accelerometer LWR' : [30,31,32],
  'Accelerometer RH' : [33,34,35],
  'InertialMeasurementUnit BACK' : [36,37,38,39,40,41,42,34,44,45,46,47,48],
  'InertialMeasurementUnit RUA' : [49,50,51,52,53,54,55,56,57,58,59,60,61],
  'InertialMeasurementUnit RLA' : [62,63,64,65,66,67,68,69,70,71,72,73,74],
  'InertialMeasurementUnit LUA' : [75,76,77,78,79,80,81,82,83,84,85,60,61],
  'InertialMeasurementUnit LLA' : [88,89,90,91,92,93,94,95,96,97,98,99,100],
  'InertialMeasurementUnit L-SHOE' : [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116],
  'InertialMeasurementUnit R-SHOE' : [117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132],
  'Accelerometer CUP' : [133,134,135,136,137],
  'Accelerometer SALAMI' : [138,139,140,141,142],
  'Accelerometer WATER' : [143,144,145,146,147],
  'Accelerometer CHEESE' : [148,149,150,151,152],
  'Accelerometer BREAD' : [153,154,155,156,157],
  'Accelerometer KNIFE1' : [158,159,160,161,162],
  'Accelerometer MILK' : [163,164,165,166,167],
  'Accelerometer SPOON' : [168,169,170,171,172],
  'Accelerometer SUGAR' : [173,174,175,176,177],
  'Accelerometer KNIFE2' : [178,179,180,181,182],
  'Accelerometer PLATE' : [183,184,185,186,187],
  'Accelerometer GLASS' : [188,189,190,191,192],
  'REED SWITCH DISHWASHER S1' : [193],
  'REED SWITCH FRIDGE S3' : [194],
  'REED SWITCH FRIDGE S2' : [195],
  'REED SWITCH FRIDGE S1' : [196],
  'REED SWITCH MIDDLEDRAWER S1' : [197],
  'REED SWITCH MIDDLEDRAWER S2' : [198],
  'REED SWITCH MIDDLEDRAWER S3' : [199],
  'REED SWITCH LOWERDRAWER S3' : [200],
  'REED SWITCH LOWERDRAWER S2' : [201],
  'REED SWITCH UPPERDRAWER' : [202],
  'REED SWITCH DISHWASHER S3' : [203],
  'REED SWITCH LOWERDRAWER S1' : [204],
  'REED SWITCH DISHWASHER S2' : [205],
  'REED SWITCH DISHWASHER S2' : [205],
  'Accelerometer DOOR1' : [206,207,208],
  'Accelerometer LAZYCHAIR' : [209,210,211],
  'Accelerometer DOOR2' : [212,213,214],
  'Accelerometer DISHWASHER' : [215,216,217],
  'Accelerometer UPPERDRAWER' : [218,219,220],
  'Accelerometer LOWERDRAWER' : [221,222,223],
  'Accelerometer MIDDLEDRAWER' : [224,225,226],
  'Accelerometer FRIDGE' : [227,228,229],
  'LOCATION TAG1': [230,231,232],
  'LOCATION TAG2': [233,234,235],
  'LOCATION TAG3': [236,237,238],
  'LOCATION TAG4': [239,240,241],
  'Locomotion' : [0],
  'HL_Activity': [1],
  'LL_Left_Arm' : [2],
  'LL_Left_Arm_Object' : [3],
  'LL_Right_Arm' : [4],
  'LL_Right_Arm_Object' : [5],
  'ML_Both_Arms' : [6],
}

labels_map = {
  1 : 'Stand',
  2 : 'Walk',
  4 : 'Sit',
  5 : 'Lie',
  201 : 'LL_Left_Arm unlock',
  202 : 'LL_Left_Arm stir',
  203 : 'LL_Left_Arm lock',
  204 : 'LL_Left_Arm close',
  205 : 'LL_Left_Arm reach',
  206 : 'LL_Left_Arm open',
  207 : 'LL_Left_Arm sip',
  208 : 'LL_Left_Arm clean',
  209 : 'LL_Left_Arm bite',
  210 : 'LL_Left_Arm cut',
  211 : 'LL_Left_Arm spread',
  212 : 'LL_Left_Arm release',
  213 : 'LL_Left_Arm move',
  401 : 'LL_Right_Arm unlock',
  402 : 'LL_Right_Arm stir',
  403 : 'LL_Right_Arm lock',
  404 : 'LL_Right_Arm close',
  405 : 'LL_Right_Arm reach',
  406 : 'LL_Right_Arm open',
  407 : 'LL_Right_Arm sip',
  408 : 'LL_Right_Arm clean',
  409 : 'LL_Right_Arm bite',
  410 : 'LL_Right_Arm cut',
  411 : 'LL_Right_Arm spread',
  412 : 'LL_Right_Arm release',
  413 : 'LL_Right_Arm move',
  301 : 'LL_Left_Arm_Object Bottle',
  302 : 'LL_Left_Arm_Object Salami',
  303 : 'LL_Left_Arm_Object Bread',
  304 : 'LL_Left_Arm_Object Sugar',
  305 : 'LL_Left_Arm_Object Dishwasher',
  306 : 'LL_Left_Arm_Object Switch',
  307 : 'LL_Left_Arm_Object Milk',
  308 : 'LL_Left_Arm_Object Drawer3 (lower)',
  309 : 'LL_Left_Arm_Object Spoon',
  310 : 'LL_Left_Arm_Object Knife cheese',
  311 : 'LL_Left_Arm_Object Drawer2 (middle)',
  312 : 'LL_Left_Arm_Object Table',
  313 : 'LL_Left_Arm_Object Glass',
  314 : 'LL_Left_Arm_Object Cheese',
  315 : 'LL_Left_Arm_Object Chair',
  316 : 'LL_Left_Arm_Object Door1',
  317 : 'LL_Left_Arm_Object Door2',
  318 : 'LL_Left_Arm_Object Plate',
  319 : 'LL_Left_Arm_Object Drawer1 (top)',
  320 : 'LL_Left_Arm_Object Fridge',
  321 : 'LL_Left_Arm_Object Cup',
  322 : 'LL_Left_Arm_Object Knife salami',
  323 : 'LL_Left_Arm_Object Lazychair',
  501 : 'LL_Right_Arm_Object Bottle',
  502 : 'LL_Right_Arm_Object Salami',
  503 : 'LL_Right_Arm_Object Bread',
  504 : 'LL_Right_Arm_Object Sugar',
  505 : 'LL_Right_Arm_Object Dishwasher',
  506 : 'LL_Right_Arm_Object Switch',
  507 : 'LL_Right_Arm_Object Milk',
  508 : 'LL_Right_Arm_Object Drawer3 (lower)',
  509 : 'LL_Right_Arm_Object Spoon',
  510 : 'LL_Right_Arm_Object Knife cheese',
  511 : 'LL_Right_Arm_Object Drawer2 (middle)',
  512 : 'LL_Right_Arm_Object Table',
  513 : 'LL_Right_Arm_Object Glass',
  514 : 'LL_Right_Arm_Object Cheese',
  515 : 'LL_Right_Arm_Object Chair',
  516 : 'LL_Right_Arm_Object Door1',
  517 : 'LL_Right_Arm_Object Door2',
  518 : 'LL_Right_Arm_Object Plate',
  519 : 'LL_Right_Arm_Object Drawer1 (top)',
  520 : 'LL_Right_Arm_Object Fridge',
  521 : 'LL_Right_Arm_Object Cup',
  522 : 'LL_Right_Arm_Object Knife salami',
  523 : 'LL_Right_Arm_Object Lazychair',
  406516 : 'ML_Both_Arms Open Door 1',
  406517 : 'ML_Both_Arms Open Door 2',
  404516 : 'ML_Both_Arms Close Door 1',
  404517 : 'ML_Both_Arms Close Door 2',
  406520 : 'ML_Both_Arms Open Fridge',
  404520 : 'ML_Both_Arms Close Fridge',
  406505 : 'ML_Both_Arms Open Dishwasher',
  404505 : 'ML_Both_Arms Close Dishwasher',
  406519 : 'ML_Both_Arms Open Drawer 1',
  404519 : 'ML_Both_Arms Close Drawer 1',
  406511 : 'ML_Both_Arms Open Drawer 2',
  404511 : 'ML_Both_Arms Close Drawer 2',
  406508 : 'ML_Both_Arms Open Drawer 3',
  404508 : 'ML_Both_Arms Close Drawer 3',
  408512 : 'ML_Both_Arms Clean Table',
  407521 : 'ML_Both_Arms Drink from Cup',
  405506 : 'ML_Both_Arms Toggle Switch',
  101 : 'HL_Activity Relaxing',
  102 : 'HL_Activity Coffee time',
  103 : 'HL_Activity Early morning',
  104 : 'HL_Activity Cleanup',
  105 : 'HL_Activity Sandwich time',
}

class OpportunityView():
  def __init__(self, entries : t.List[str]) -> None:
    self.indices = []
    for e in entries:
      self.indices.extend(view_indices[e])

  def __call__(self, batch : torch.Tensor):
    batch = torch.atleast_2d(batch)
    return batch[:, self.indices]

  def describeLabels(labels : torch.Tensor) -> t.List[str]:
    ret = []
    for l in labels:
      ret.append(labels_map[int(l.item())] if l != 0 else "<null>")

    return ret

  def getEntries() -> t.List[str]:
    return view_indices.keys()

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

class Opportunity(Dataset):
  def __init__(self, root : str,
               window : int = 24,
               stride : int = 12,
               transform = None,
               download : bool = True,
               opts : t.Iterable[OpportunityOptions] = []):

    self.dataset_name = 'opportunity'  
    self.zip_dir = 'OpportunityUCIDataset/dataset'
    self.root = os.path.join(root, self.dataset_name, self.zip_dir)
    self.transform = None


    if download:
      ensure_download_zip(url=download_url, dataset_name=self.dataset_name, root=root, zip_dirs=[self.zip_dir+'/'])

    logger.info(f'Loading Opportunity Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {list(map(lambda o: o.name, opts))}')

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
      data.append(SegmentedDataset(tensor=tensor, labels=labels, window=window, stride=stride))

    self.data = ConcatDataset(data)

    logger.info(f'Opportunity Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B')


  def __getitem__(self, index):
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)
