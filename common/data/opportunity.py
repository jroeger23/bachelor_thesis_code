import logging
import os
import typing as t
from enum import Enum
from itertools import product
from sys import getsizeof

import torch
from si_prefix import si_format
from torch.utils.data import ConcatDataset, Dataset

from .common import SegmentedDataset, ensure_download_zip, load_cached_dat, Transform, View, describeLabels

from functools import wraps

logger = logging.getLogger(__name__)

DOWNLOAD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
SHA512_HEX = 'adf10dab0e44eb5a3c49d48f92acc8c1c14b48d143aecaa37398c9d7a8d728a2ff5fcaa49ee24ca3276156bbac46d41ce9764567afbf48b16d53959e6d5e1cf2'


def parse_column_description_line(columns: t.MutableMapping[int, str], line: str):
  """ Parse Opportunity description line

  Args:
      columns (t.MutableMapping[int, str]): the column mapping from index -> name to be written to
      line (str): the line to parse
  """
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


def parse_columns_file(path: str) -> t.Mapping[int, str]:
  """Parse a file of Opportunity description columns

  Args:
      path (str): file path

  Returns:
      t.Mapping[int, str]: the index to name mapping
  """
  columns = {}

  with open(file=path, mode='r') as f:
    for line in f.readlines():
      parse_column_description_line(columns=columns, line=line)

  return columns


def split_data_record(tensor):
  time = tensor[:, 0]
  data = tensor[:, 1:243]
  labels = tensor[:, 243:250]

  return time, data, labels


view_indices = {
    'Accelerometer RKN^': [0, 1, 2],
    'Accelerometer HIP': [3, 4, 5],
    'Accelerometer LUA^': [6, 7, 8],
    'Accelerometer RUA_': [9, 10, 11],
    'Accelerometer LH': [12, 13, 14],
    'Accelerometer BACK': [15, 16, 17],
    'Accelerometer RKN_': [18, 19, 20],
    'Accelerometer RWR': [21, 22, 23],
    'Accelerometer RUA^': [24, 25, 26],
    'Accelerometer LUA_': [27, 28, 29],
    'Accelerometer LWR': [30, 31, 32],
    'Accelerometer RH': [33, 34, 35],
    'InertialMeasurementUnit BACK': [36, 37, 38, 39, 40, 41, 42, 34, 44, 45, 46, 47, 48],
    'InertialMeasurementUnit RUA': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    'InertialMeasurementUnit RLA': [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
    'InertialMeasurementUnit LUA': [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 60, 61],
    'InertialMeasurementUnit LLA': [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    'InertialMeasurementUnit L-SHOE': [
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116
    ],
    'InertialMeasurementUnit R-SHOE': [
        117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132
    ],
    'Accelerometer CUP': [133, 134, 135, 136, 137],
    'Accelerometer SALAMI': [138, 139, 140, 141, 142],
    'Accelerometer WATER': [143, 144, 145, 146, 147],
    'Accelerometer CHEESE': [148, 149, 150, 151, 152],
    'Accelerometer BREAD': [153, 154, 155, 156, 157],
    'Accelerometer KNIFE1': [158, 159, 160, 161, 162],
    'Accelerometer MILK': [163, 164, 165, 166, 167],
    'Accelerometer SPOON': [168, 169, 170, 171, 172],
    'Accelerometer SUGAR': [173, 174, 175, 176, 177],
    'Accelerometer KNIFE2': [178, 179, 180, 181, 182],
    'Accelerometer PLATE': [183, 184, 185, 186, 187],
    'Accelerometer GLASS': [188, 189, 190, 191, 192],
    'REED SWITCH DISHWASHER S1': [193],
    'REED SWITCH FRIDGE S3': [194],
    'REED SWITCH FRIDGE S2': [195],
    'REED SWITCH FRIDGE S1': [196],
    'REED SWITCH MIDDLEDRAWER S1': [197],
    'REED SWITCH MIDDLEDRAWER S2': [198],
    'REED SWITCH MIDDLEDRAWER S3': [199],
    'REED SWITCH LOWERDRAWER S3': [200],
    'REED SWITCH LOWERDRAWER S2': [201],
    'REED SWITCH UPPERDRAWER': [202],
    'REED SWITCH DISHWASHER S3': [203],
    'REED SWITCH LOWERDRAWER S1': [204],
    'REED SWITCH DISHWASHER S2': [205],
    'REED SWITCH DISHWASHER S2': [205],
    'Accelerometer DOOR1': [206, 207, 208],
    'Accelerometer LAZYCHAIR': [209, 210, 211],
    'Accelerometer DOOR2': [212, 213, 214],
    'Accelerometer DISHWASHER': [215, 216, 217],
    'Accelerometer UPPERDRAWER': [218, 219, 220],
    'Accelerometer LOWERDRAWER': [221, 222, 223],
    'Accelerometer MIDDLEDRAWER': [224, 225, 226],
    'Accelerometer FRIDGE': [227, 228, 229],
    'LOCATION TAG1': [230, 231, 232],
    'LOCATION TAG2': [233, 234, 235],
    'LOCATION TAG3': [236, 237, 238],
    'LOCATION TAG4': [239, 240, 241],
}

label_view_indices = {
    'Locomotion': 0,
    'HL_Activity': 1,
    'LL_Left_Arm': 2,
    'LL_Left_Arm_Object': 3,
    'LL_Right_Arm': 4,
    'LL_Right_Arm_Object': 5,
    'ML_Both_Arms': 6,
}

labels_map = {
    0: '<none>',
    1: 'Stand',
    2: 'Walk',
    4: 'Sit',
    5: 'Lie',
    201: 'LL_Left_Arm unlock',
    202: 'LL_Left_Arm stir',
    203: 'LL_Left_Arm lock',
    204: 'LL_Left_Arm close',
    205: 'LL_Left_Arm reach',
    206: 'LL_Left_Arm open',
    207: 'LL_Left_Arm sip',
    208: 'LL_Left_Arm clean',
    209: 'LL_Left_Arm bite',
    210: 'LL_Left_Arm cut',
    211: 'LL_Left_Arm spread',
    212: 'LL_Left_Arm release',
    213: 'LL_Left_Arm move',
    401: 'LL_Right_Arm unlock',
    402: 'LL_Right_Arm stir',
    403: 'LL_Right_Arm lock',
    404: 'LL_Right_Arm close',
    405: 'LL_Right_Arm reach',
    406: 'LL_Right_Arm open',
    407: 'LL_Right_Arm sip',
    408: 'LL_Right_Arm clean',
    409: 'LL_Right_Arm bite',
    410: 'LL_Right_Arm cut',
    411: 'LL_Right_Arm spread',
    412: 'LL_Right_Arm release',
    413: 'LL_Right_Arm move',
    301: 'LL_Left_Arm_Object Bottle',
    302: 'LL_Left_Arm_Object Salami',
    303: 'LL_Left_Arm_Object Bread',
    304: 'LL_Left_Arm_Object Sugar',
    305: 'LL_Left_Arm_Object Dishwasher',
    306: 'LL_Left_Arm_Object Switch',
    307: 'LL_Left_Arm_Object Milk',
    308: 'LL_Left_Arm_Object Drawer3 (lower)',
    309: 'LL_Left_Arm_Object Spoon',
    310: 'LL_Left_Arm_Object Knife cheese',
    311: 'LL_Left_Arm_Object Drawer2 (middle)',
    312: 'LL_Left_Arm_Object Table',
    313: 'LL_Left_Arm_Object Glass',
    314: 'LL_Left_Arm_Object Cheese',
    315: 'LL_Left_Arm_Object Chair',
    316: 'LL_Left_Arm_Object Door1',
    317: 'LL_Left_Arm_Object Door2',
    318: 'LL_Left_Arm_Object Plate',
    319: 'LL_Left_Arm_Object Drawer1 (top)',
    320: 'LL_Left_Arm_Object Fridge',
    321: 'LL_Left_Arm_Object Cup',
    322: 'LL_Left_Arm_Object Knife salami',
    323: 'LL_Left_Arm_Object Lazychair',
    501: 'LL_Right_Arm_Object Bottle',
    502: 'LL_Right_Arm_Object Salami',
    503: 'LL_Right_Arm_Object Bread',
    504: 'LL_Right_Arm_Object Sugar',
    505: 'LL_Right_Arm_Object Dishwasher',
    506: 'LL_Right_Arm_Object Switch',
    507: 'LL_Right_Arm_Object Milk',
    508: 'LL_Right_Arm_Object Drawer3 (lower)',
    509: 'LL_Right_Arm_Object Spoon',
    510: 'LL_Right_Arm_Object Knife cheese',
    511: 'LL_Right_Arm_Object Drawer2 (middle)',
    512: 'LL_Right_Arm_Object Table',
    513: 'LL_Right_Arm_Object Glass',
    514: 'LL_Right_Arm_Object Cheese',
    515: 'LL_Right_Arm_Object Chair',
    516: 'LL_Right_Arm_Object Door1',
    517: 'LL_Right_Arm_Object Door2',
    518: 'LL_Right_Arm_Object Plate',
    519: 'LL_Right_Arm_Object Drawer1 (top)',
    520: 'LL_Right_Arm_Object Fridge',
    521: 'LL_Right_Arm_Object Cup',
    522: 'LL_Right_Arm_Object Knife salami',
    523: 'LL_Right_Arm_Object Lazychair',
    406516: 'ML_Both_Arms Open Door 1',
    406517: 'ML_Both_Arms Open Door 2',
    404516: 'ML_Both_Arms Close Door 1',
    404517: 'ML_Both_Arms Close Door 2',
    406520: 'ML_Both_Arms Open Fridge',
    404520: 'ML_Both_Arms Close Fridge',
    406505: 'ML_Both_Arms Open Dishwasher',
    404505: 'ML_Both_Arms Close Dishwasher',
    406519: 'ML_Both_Arms Open Drawer 1',
    404519: 'ML_Both_Arms Close Drawer 1',
    406511: 'ML_Both_Arms Open Drawer 2',
    404511: 'ML_Both_Arms Close Drawer 2',
    406508: 'ML_Both_Arms Open Drawer 3',
    404508: 'ML_Both_Arms Close Drawer 3',
    408512: 'ML_Both_Arms Clean Table',
    407521: 'ML_Both_Arms Drink from Cup',
    405506: 'ML_Both_Arms Toggle Switch',
    101: 'HL_Activity Relaxing',
    102: 'HL_Activity Coffee time',
    103: 'HL_Activity Early morning',
    104: 'HL_Activity Cleanup',
    105: 'HL_Activity Sandwich time',
}


@wraps(describeLabels)
def describeOpportunityLabels(
    labels: t.Union[int, t.Iterable[int], torch.Tensor]) -> t.Union[str, t.List[str]]:
  return describeLabels(labels_map=labels_map, labels=labels)


def allOpportunityLabels() -> t.Mapping[int, str]:
  return labels_map


class OpportunityLabelView(View):

  def __init__(self, entries: t.List[str]) -> None:
    """OpportunityLabelView  filtes a label tensor to only contain certain columns

    Args:
        entries (t.List[str]): column names to keep
    """
    self.entries = entries
    self.indices = [label_view_indices[e] for e in entries]

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply Label View

    Args:
        sample (torch.Tensor): sample to view (untouched)
        labels (torch.Tensor): label to view

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: batch, label filtered by entries
    """
    labels = torch.atleast_2d(labels)
    return sample, labels[:, self.indices].squeeze()

  @staticmethod
  def allEntries() -> t.List[str]:
    return list(label_view_indices.keys())

  def __str__(self) -> str:
    return f'OpportunityLabelView {self.entries}'


class OpportunitySplitLabelView(View):

  def __init__(self, entries: t.List[str]) -> None:
    """OpportunityLabelView spllits a labels tensor into a list of 1D label tensors

    Args:
        entries (t.List[str]): column names to keep
    """
    self.entries = entries
    self.views = [OpportunityLabelView([e]) for e in entries]

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
    """Apply Label Split View

    Args:
        sample (torch.Tensor): batch to view (untouched)
        labels (torch.Tensor): label to view

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: sample, label list by entries
    """
    label_split = [v(sample, labels)[1] for v in self.views]
    return sample, label_split

  @staticmethod
  def allEntries() -> t.List[str]:
    return list(label_view_indices.keys())

  def __str__(self) -> str:
    return f'OpportunitySplitLabelView {self.entries}'


class OpportunityLocomotionLabelView(View):

  def __init__(self) -> None:
    """Create a new Opportunity Locomotion label view
    """
    self.view = OpportunityLabelView(['Locomotion'])

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply the view

    Args:
        sample (torch.Tensor): the sample (untouched)
        labels (torch.Tensor): the labels to view

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: sample, labels (locomotion)
    """
    return self.view(sample, labels)

  def __str__(self) -> str:
    return 'OpportunityLocomotionLabelView'


class OpportunitySensorUnitView(View):

  def __init__(self, entries: t.List[str]) -> None:
    self.entries = entries
    self.indices = []
    for e in entries:
      self.indices.extend(view_indices[e])

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply Sensor Unit View

    Args:
        sample (torch.Tensor): sample to view
        labels (torch.Tensor): label to view

    Returns:
        
    """
    sample = torch.atleast_2d(sample)
    return sample[:, self.indices].squeeze(), labels

  @staticmethod
  def allEntries() -> t.List[str]:
    return list(view_indices.keys())

  def __str__(self) -> str:
    return f'OpportunitySensorUnitView {self.entries}'


class OpportunitySplitSensorUnitsView(View):
  """A view to split a raw Opportunity data tensor into a list of sensor unit tensors
  """

  def __init__(self, entries: t.List[str]) -> None:
    """A view to split a raw Opportunity data tensor into a list of sensor unit tensors

    Args:
        entries (t.List[str]): the sensor unit entries to use (__call__ return order)
    """
    self.entries = entries
    self.views = [OpportunitySensorUnitView([e]) for e in entries]

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[t.List[torch.Tensor], torch.Tensor]:
    """Apply Split View

    Args:
        sample (torch.Tensor): the raw sample tensor to split
        labels (torch.Tensor): the labels (untouched)

    Returns:
        t.Tuple[t.List[torch.Tensor], torch.Tensor]: a list of sensor unit batch tensors, labels
    """
    sensor_units = [v(sample, labels)[0] for v in self.views]
    return sensor_units, labels

  @staticmethod
  def allEntries() -> t.List[str]:
    """Get all possible entries

    Returns:
        t.List[str]: the entries
    """
    return list(view_indices.keys())

  def __str__(self) -> str:
    return f'OpportunitySplitSensorUnitsView {self.entries}'


HUMAN_SENSOR_UNITS_ENTRIES = [
    'Accelerometer RKN^', 'Accelerometer HIP', 'Accelerometer LUA^', 'Accelerometer RUA_',
    'Accelerometer LH', 'Accelerometer BACK', 'Accelerometer RKN_', 'Accelerometer RWR',
    'Accelerometer RUA^', 'Accelerometer LUA_', 'Accelerometer LWR', 'Accelerometer RH',
    'InertialMeasurementUnit BACK', 'InertialMeasurementUnit RUA', 'InertialMeasurementUnit RLA',
    'InertialMeasurementUnit LUA', 'InertialMeasurementUnit LLA', 'InertialMeasurementUnit L-SHOE',
    'InertialMeasurementUnit R-SHOE'
]


class OpportunityHumanSensorUnitsView(View):
  """A view to split a raw Opportunity data tensor into a list of body worn sensor unit tensors
  """

  def __init__(self) -> None:
    """A view to split a raw Opportunity data tensor into a list of body worn sensor unit tensors
    """
    self.view = OpportunitySplitSensorUnitsView(HUMAN_SENSOR_UNITS_ENTRIES)

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[t.List[torch.Tensor], torch.Tensor]:
    """Apply Split View

    Args:
        sample (torch.Tensor): the raw sample tensor to split
        labels (torch.Tensor): the labels (untouched)

    Returns:
        t.Tuple[t.List[torch.Tensor], torch.Tensor]: a list of sensor unit batch tensors, labels
    """
    return self.view(sample, labels)

  def __str__(self) -> str:
    return f'OpportunityHumanSensorUnitsView'


class OpportunityLocomotionLabelAdjustMissing3(Transform):

  def __init__(self) -> None:
    """Create a view to fix the missing 3 entry in the opportunity labels
    """
    self.view = OpportunityLocomotionLabelView()

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply the view

    Args:
        sample (torch.Tensor): untouched
        labels (torch.Tensor): labels to fix

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: the labels with locomotion 4->3 and 5->4
    """
    labels_view = labels[:, label_view_indices['Locomotion']]
    labels_view[labels_view == 4] = 3
    labels_view[labels_view == 5] = 4

    return sample, labels

  def __str__(self) -> str:
    return 'OpportunityLocomotionLabelAdjustMissing3'


class OpportunityRemoveHumanSensorUnitNaNRows(Transform):

  def __init__(self) -> None:
    """Remove all rows from the sample, where any of the human sensors has a NaN value
    """
    self.view = OpportunitySensorUnitView(HUMAN_SENSOR_UNITS_ENTRIES)

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply the transformation

    Args:
        sample (torch.Tensor): the sample 
        label (torch.Tensor): the labels

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: sample(without NaN), label
    """
    sample_view, _ = self.view(sample, label)

    keep_cond = sample_view.isnan().any(dim=1, keepdim=False).logical_not()

    return sample[keep_cond], label[keep_cond]

  def __str__(self) -> str:
    return 'OpportunityRemoveHumanSensorUnitNaNRows'


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
  DEFAULT_TRAIN = 101
  DEFAULT_VALIDATION = 102
  DEFAULT_TEST = 103


class Opportunity(Dataset):

  def __init__(self,
               root: str = './data',
               window: int = 24,
               stride: int = 12,
               static_transform: t.Optional[Transform] = None,
               dynamic_transform: t.Optional[Transform] = None,
               view: t.Optional[View] = None,
               download: bool = True,
               opts: t.Iterable[OpportunityOptions] = []):

    self.dataset_name = 'opportunity'
    self.zip_dir = 'OpportunityUCIDataset/dataset'
    self.root = os.path.join(root, self.dataset_name, self.zip_dir)
    self.dynamic_transform = dynamic_transform
    self.view = view

    if download:
      ensure_download_zip(url=DOWNLOAD_URL,
                          dataset_name=self.dataset_name,
                          root=root,
                          zip_dirs=[self.zip_dir + '/'],
                          sha512_hex=SHA512_HEX)

    logger.info(f'Loading Opportunity Dataset...')
    logger.info(f'  - Segmentation (w={window}, s={stride})')
    logger.info(f'  - Subsets {list(map(lambda o: o.name, opts))}')
    logger.info(f'  - Static Transform: {str(static_transform)}')
    logger.info(f'  - Dynamic Transform: {str(dynamic_transform)}')
    logger.info(f'  - View: {str(view)}')

    self.columns = parse_columns_file(os.path.join(self.root, 'column_names.txt'))

    suffixes = []
    if OpportunityOptions.ADL1 in opts or OpportunityOptions.ALL_ADL in opts:
      suffixes.append('-ADL1')
    if OpportunityOptions.ADL2 in opts or OpportunityOptions.ALL_ADL in opts:
      suffixes.append('-ADL2')
    if OpportunityOptions.ADL3 in opts or OpportunityOptions.ALL_ADL in opts:
      suffixes.append('-ADL3')
    if OpportunityOptions.ADL4 in opts or OpportunityOptions.ALL_ADL in opts:
      suffixes.append('-ADL4')
    if OpportunityOptions.ADL5 in opts or OpportunityOptions.ALL_ADL in opts:
      suffixes.append('-ADL5')
    if OpportunityOptions.DRILL in opts:
      suffixes.append('-Drill')

    prefixes = []
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT1 in opts:
      prefixes.append('S1')
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT2 in opts:
      prefixes.append('S2')
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT3 in opts:
      prefixes.append('S3')
    if OpportunityOptions.ALL_SUBJECTS in opts or OpportunityOptions.SUBJECT4 in opts:
      prefixes.append('S4')

    if OpportunityOptions.DEFAULT_TRAIN in opts:
      runs = [
          'S1-ADL1', 'S1-ADL2', 'S1-ADL3', 'S1-ADL4', 'S1-ADL5', 'S1-Drill', 'S2-ADL1', 'S2-ADL2',
          'S2-Drill', 'S3-ADL1', 'S3-ADL2', 'S3-Drill', 'S4-ADL1', 'S4-ADL2', 'S4-ADL3', 'S4-ADL4',
          'S4-ADL5', 'S4-Drill'
      ]
    elif OpportunityOptions.DEFAULT_VALIDATION in opts:
      runs = ['S2-ADL3', 'S3-ADL3']
    elif OpportunityOptions.DEFAULT_TEST in opts:
      runs = ['S2-ADL4', 'S2-ADL5', 'S3-ADL4', 'S3-ADL5']
    else:
      runs = [prefix + suffix for prefix, suffix in product(prefixes, suffixes)]

    data = []
    memory = 0
    for run in runs:
      raw = load_cached_dat(root=self.root, name=run, logger=logger)
      memory += getsizeof(raw.storage())
      _, tensor, labels = split_data_record(raw)
      if static_transform is not None:
        tensor, labels = static_transform(tensor, labels)

      if len(tensor) != 0 and len(labels) != 0:
        data.append(SegmentedDataset(tensor=tensor, labels=labels, window=window, stride=stride))
      else:
        logger.warn(f'Run {run} is empty after static_transform')

    self.data = ConcatDataset(data)

    logger.info(
        f'Opportunity Dataset loaded. {len(self)} segments with shape: {tuple(self.data[0][0].shape)}. Memory: {si_format(memory)}B'
    )

  def __getitem__(self, index):
    item = self.data[index]
    if self.dynamic_transform is not None:
      item = self.dynamic_transform(*item)
    if self.view is not None:
      item = self.view(*item)
    return item

  def __len__(self) -> int:
    return len(self.data)
