import hashlib
import logging
import os
import re
import typing as t
from zipfile import ZipFile

import requests
import torch
import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def parse_tensor_line(line: str,
                      n_cols: int,
                      sep_re: str = ' ',
                      dtype=torch.float32,
                      drop_n: int = 0) -> torch.Tensor:
  """Parse a string of numbers into a tensor

  Args:
      line (str): the line to parse
      n_cols (int): the number of tensor entries (without the dropped entries)
      sep_re (str, optional): the number separator regex. Defaults to ' '.
      dtype (_type_, optional): the tensor datatype. Defaults to torch.float32.
      drop_n (int, optional): drop the first n entries. Defaults to 0.

  Returns:
      torch.Tensor: shape(n_cols)
  """
  ret = torch.zeros(n_cols, dtype=dtype)

  for ix, word in enumerate(re.split(pattern=sep_re, string=line)[drop_n:]):
    ret[ix] = float(word)

  return ret


def parse_dat(path: str, dtype=torch.float32) -> torch.Tensor:
  """Parse a .dat file with space separated columns and linebreak separated rows

  Args:
      path (str): the .dat file path
      dtype (_type_, optional): the tensor datatype. Defaults to torch.float32.

  Returns:
      torch.Tensor: shape(.dat rows, .dat columns)
  """
  with open(file=path, mode='r') as f:
    lines = f.readlines()
    n_cols = len(lines[0].split(' '))

    ret = torch.zeros(len(lines), n_cols, dtype=dtype)

    for ix, line in tqdm.tqdm(iterable=enumerate(lines),
                              desc='Parsing .dat',
                              unit='line',
                              total=len(ret),
                              leave=False):
      ret[ix] = parse_tensor_line(line=line, n_cols=n_cols, dtype=dtype)

    return ret


def load_cached_dat(root: str, name: str, dtype=torch.float32, logger=logger) -> torch.Tensor:
  """Load a tensor from a dat file. The parsed tensor will be cached in root/name.torch

  Args:
      root (str): the root-directory
      name (str): the name of the file (relative to root) to load (without .dat extension)
      dtype (_type_, optional): tensor datatype. Defaults to torch.float32.
      logger (_type_, optional): logger to use. Defaults to logger.

  Returns:
      torch.Tensor: the loaded tensor
  """
  dat_file = os.path.join(root, f'{name}.dat')
  torch_file = os.path.join(root, f'{name}.torch')
  logger.debug(f'Loading {root}: {name}')

  if os.path.exists(torch_file):
    with open(torch_file, 'rb') as f:
      return torch.load(f)

  logger.info(f'Did not find {name} cache. Parsing {dat_file}')

  tensor = parse_dat(path=dat_file, dtype=dtype)

  torch.save(tensor, torch_file)

  return tensor


CSV_SEP_REGEX = '\\s*,\\s*'


def parse_csv_header(path: str) -> t.List[str]:
  """Parse a csv header

  Args:
      path (str): the csv file

  Returns:
      t.List[str]: the list of entries in the header
  """
  with open(path, 'r') as f:
    return re.split(pattern=CSV_SEP_REGEX, string=f.readline())


def parse_csv(path: str,
              dtype=torch.float32,
              drop_n: int = 0) -> t.Tuple[t.List[str], torch.Tensor]:
  """Parse a csv file

  Args:
      path (str): the csv file to parse
      dtype (_type_, optional): the tensor datatype. Defaults to torch.float32.
      drop_n (int, optional): drop the first n columns. Defaults to 0.

  Returns:
      t.Tuple[t.List[str], torch.Tensor]: csv_header, data tensor
  """
  with open(file=path, mode='r') as f:
    lines = f.readlines()
    header = re.split(pattern=CSV_SEP_REGEX, string=lines[0])
    header = header[drop_n:]
    n_cols = len(header)

    ret = torch.zeros(len(lines) - 1, n_cols, dtype=dtype)

    for ix, line in tqdm.tqdm(iterable=enumerate(lines[1:]),
                              desc='Parsing .csv',
                              unit='line',
                              total=len(ret),
                              leave=False):
      ret[ix] = parse_tensor_line(line=line,
                                  n_cols=n_cols,
                                  sep_re=CSV_SEP_REGEX,
                                  dtype=dtype,
                                  drop_n=drop_n)

    return header, ret


def load_cached_csv(root: str,
                    name: str,
                    dtype=torch.float32,
                    drop_n: int = 0,
                    logger=logger) -> t.Tuple[t.List[str], torch.Tensor]:
  """Load a csv file. Will be cached in root/name.torch

  Args:
      root (str): the root directory
      name (str): the filename (relative to root, without .csv extension)
      dtype (_type_, optional): the datatype to use. Defaults to torch.float32.
      drop_n (int, optional): drop the first n columns. Defaults to 0.
      logger (_type_, optional): the logger to use. Defaults to logger.

  Returns:
      t.Tuple[t.List[str], torch.Tensor]: csv_header, csv data
  """
  csv_file = os.path.join(root, f'{name}.csv')
  torch_file = os.path.join(root, f'{name}.torch')
  logger.debug(f'Loading {root}: {name}')

  if os.path.exists(torch_file):
    with open(torch_file, 'rb') as f:
      return parse_csv_header(csv_file)[drop_n:], torch.load(f)

  logger.info(f'Did not find {name} cache. Parsing {csv_file}')

  header, tensor = parse_csv(path=csv_file, dtype=dtype, drop_n=drop_n)

  torch.save(tensor, torch_file)

  return header, tensor


def majority_label(labels: torch.Tensor) -> torch.Tensor:
  """A function to choose the most occuring lable of a label sequence

  Args:
      labels (torch.Tensor): the label sequence (sequence_lenth x label_depth)

  Returns:
      torch.Tensor: the most occuring label (1 x label_depth)
  """
  return labels.mode(dim=0)[0]


def sha512file(path: str) -> str:
  logger.debug(f'Computing SHA512 for "{path}"')
  with open(path, 'rb') as f:
    sha512 = hashlib.sha512()
    buffer = bytearray(1024000)
    buffer_view = memoryview(buffer)
    while n := f.readinto(buffer_view):
      sha512.update(buffer_view[:n])

    return sha512.hexdigest()


def ensure_download_file(url: str, file: str, sha512_hex: t.Optional[str] = None):
  """Make sure a file is downloaded and valid

  Args:
      url (str): download url
      file (str): file name at the local fs
      sha512_hex (t.Optional[str], optional): hash for integrity check. Defaults to None.

  Raises:
      RuntimeError: if the hash does not match after an immediate download
  """
  # check file
  if os.path.exists(file):
    if sha512_hex is None:
      return
    elif sha512file(file) != sha512_hex:
      logger.warn(f'"{file}" is present, but corrupt. Redownloading from "{url}"...')
    else:
      return
  else:
    logger.info(f'"{file}" not present, downloading from "{url}"...')

  # Download file
  response = requests.get(url=url, verify=True, stream=True)
  chunk_size = 4096
  length = int(response.headers.get('content-length') or '0')
  sha512 = hashlib.sha512()
  with open(file, 'wb') as f:
    with tqdm.tqdm(desc=os.path.basename(file),
                   unit='B',
                   unit_scale=True,
                   unit_divisor=1024,
                   total=length) as progress:
      for d in response.iter_content(chunk_size=chunk_size):
        if sha512_hex is not None:
          sha512.update(d)
        f.write(d)
        progress.update(n=chunk_size)

      if not sha512_hex is None and sha512.hexdigest() != sha512_hex:
        raise RuntimeError(f'"{file} does not match the given hash.')


def ensure_download_zip(url: str,
                        root: str,
                        dataset_name: str,
                        zip_dirs: t.List[str] = [],
                        sha512_hex: t.Optional[str] = None):
  """Ensure a zip file is present and valid

  Args:
      url (str): download url
      root (str): root-directory
      dataset_name (str): name of the dataset (<root>/<dataset_name>.zip)
      zip_dirs (t.List[str], optional): the directorys to extract from the zip. Defaults to [].
      sha512_hex (t.Optional[str], optional): hash of the zip file. Defaults to None.
  """
  dataset_direcoty = os.path.join(root, dataset_name)
  zip_path = dataset_direcoty + '.zip'
  if os.path.exists(dataset_direcoty):
    # OK
    logger.debug(f'Dataset at "{dataset_direcoty}" is present.')
    return

  logger.debug(f'Ensuring root path \"{root}\"')
  os.makedirs(name=root, exist_ok=True)

  # Download zip if not present
  ensure_download_file(url=url, file=zip_path, sha512_hex=sha512_hex)

  logger.info(f'Unzipping "{zip_path}"...')
  with ZipFile(zip_path, 'r') as zf:
    for d in zip_dirs:
      zf.extract(member=d, path=dataset_direcoty)
    else:
      zf.extractall(path=dataset_direcoty)
  logger.info(f'Done!')


def describeLabels(
    labels_map: t.Mapping[int, str], labels: t.Union[torch.Tensor, int,
                                                     t.Iterable[int]]) -> t.Union[str, t.List[str]]:
  """Describe a collection of labels (to string)

  Args:
      labels_map (t.Mapping[int, str]): the label_ix to name mapping
      labels (t.Union[torch.Tensor, int, t.Iterable[int]]): the collection of labels (tensor must be one dimensional)

  Raises:
      ValueError: If labels is a multi dimensional tensor
      TypeError: If labels is not tensor, int or List[int]

  Returns:
      t.Union[str, t.List[str]]: a single label or a list of labels
  """
  if type(labels) is torch.Tensor:
    if len(labels) == 1:
      return labels_map[int(labels.item())]
    elif labels.shape[1] == 1:
      return [labels_map[int(l.item())] for l in labels]
    else:
      raise ValueError('Cannot describe multi dimensional labels')
  elif type(labels) is t.List:
    return [labels_map[int(l)] for l in labels]
  elif type(labels) is int:
    return labels_map[labels]
  else:
    raise TypeError(f'Requires tensor, int or List[int] not {type(labels)}')


class View():
  """The View interface
  """

  def __call__(self, batch: torch.Tensor, labels: torch.Tensor) -> t.Any:
    """Apply a view

    Args:
        batch (torch.Tensor): batch to view
        labels (torch.Tensor): label to view

    Raises:
        NotImplementedError: Interface

    Returns:
        t.Any: the viewed data
    """
    raise NotImplementedError()

  def __str__(self) -> str:
    """Describe this view

    Raises:
        NotImplementedError: interface

    Returns:
        str: the description
    """
    raise NotImplementedError()


class Transform():
  """The Transform interface
  """

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply this transformation

    Args:
        batch (torch.Tensor): the batch to transform
        labels (torch.Tensor): the label to transform

    Raises:
        NotImplementedError: interface

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: (transformed batch, transformed labels)
    """
    raise NotImplementedError()

  def __str__(self) -> str:
    """Describe this transform

    Raises:
        NotImplementedError: interface

    Returns:
        str: description 
    """
    raise NotImplementedError()


class ComposeTransforms(Transform):
  """Composes multiple transforms into one
  """

  def __init__(self, transforms: t.List[Transform]):
    """Create a new ComposeTransforms Transform

    Args:
        transforms (t.List[Transform]): the list of transforms to apply (in order)
    """
    self.transforms = transforms

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply all transformations

    Args:
        batch (torch.Tensor): the batch to transform
        labels (torch.Tensor): the labels to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: (transformed batch, transformed labels)
    """
    for t in self.transforms:
      batch, labels = t(batch, labels)

    return batch, labels

  def __str__(self) -> str:
    """Describe this transform

    Returns:
        str: description
    """
    s = "ComposeTransforms: \n\t[\n"
    for t in self.transforms:
      s += f'\t  -> {str(t)}\n'

    return s + "\t]"


class CombineViews(View):
  """Combine two Views
  """

  def __init__(self, batch_view: View, labels_view: View):
    """New CombineViews

    Args:
        batch_view (View): the batch view
        labels_view (View): the labels view
    """
    self.batch_view = batch_view
    self.labels_view = labels_view

  def __call__(self, batch: torch.Tensor, labels: torch.Tensor) -> t.Any:
    """Apply both views

    Args:
        batch (torch.Tensor): the batch to view
        labels (torch.Tensor): the labels to view

    Returns:
        t.Any: view output
    """
    return self.labels_view(*self.batch_view(batch, labels))

  def __str__(self) -> str:
    return f'CombineViews \n\t[\n\t  Batch:  {str(self.batch_view)}\n\t  Labels: {str(self.labels_view)}\n\t]'


class NaNToConstTransform(Transform):
  """A Transformation, which replaces all NaN values with a constant
  """

  def __init__(self, batch_constant=0, label_constant=0):
    """Choose the constants for batch and label

    Args:
        batch_constant (int, optional): batch constant . Defaults to 0.
        label_constant (int, optional): labels constant . Defaults to 0.
    """
    self.batch_constant = batch_constant
    self.label_constant = label_constant

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply NaN to const transformation

    Args:
        batch (torch.Tensor): the batches to transform
        labels (torch.Tensor): the labels to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: nan replaced inputs
    """
    return torch.nan_to_num(input=batch,
                            nan=self.batch_constant), torch.nan_to_num(input=labels,
                                                                       nan=self.label_constant)

  def __str__(self) -> str:
    return f'NaNToConst (batch_nan={self.batch_constant}, label_nan={self.label_constant})'


class LabelDtypeTransform(Transform):
  """Transform the datatype of labels
  """

  def __init__(self, dtype: torch.dtype):
    """Choose the dtype

    Args:
        dtype (torch.dtype): new labels dtype
    """
    self.dtype = dtype

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply dtype transformation

    Args:
        batch (torch.Tensor): unchanged
        labels (torch.Tensor): the labels to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: batch, labels(as dtype)
    """
    return batch, labels.type(dtype=self.dtype)

  def __str__(self) -> str:
    return f'LabelDtypeTransform {self.dtype}'


class ResampleTransform(Transform):
  """A transformation, which is resampling batches and labels. Meant to used for not yet segmented data
  """

  def __init__(self, freq_in: int, freq_out: int):
    """Set Resampling parameters

    Args:
        freq_in (int): input frequency
        freq_out (int): output frequency
    """
    self.freq_in = freq_in
    self.freq_out = freq_out

  def __call__(self, batch: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply re-sampling to a sequence of data

    Args:
        batch (torch.Tensor): the sequence of datapoints (N x D)
        labels (torch.Tensor): the sequence of labels (N x L)

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: resampled batch (N' x D), resampled labels (N' x L)
    """
    ratio = self.freq_out / self.freq_in
    fb = tuple([ratio] + [1 for _ in range(batch.ndim - 1)])
    fl = tuple([ratio] + [1 for _ in range(labels.ndim - 1)])
    batch = torch.nn.functional.interpolate(input=batch.unsqueeze(0).unsqueeze(0), scale_factor=fb)
    labels = torch.nn.functional.interpolate(input=labels.unsqueeze(0).unsqueeze(0),
                                             scale_factor=fl)

    return batch.squeeze(), labels.squeeze()

  def __str__(self) -> str:
    return f'DownsampleTransform (f_in={self.freq_in}, f_out={self.freq_out})'


class SegmentedDataset(Dataset):
  """A Dataset wrapper that segments the underlying dataset
  """

  def __init__(self,
               tensor: torch.Tensor,
               labels: torch.Tensor,
               window: int,
               stride: int,
               label_fn=majority_label):
    """Define the dataset

    Args:
        tensor (torch.Tensor): the data sequence to segment
        labels (torch.Tensor): the labels sequence to segment
        window (int): segment size
        stride (int): step size
        label_fn (_type_, optional): label reducing function. Defaults to majority_label.
    """
    self.data = tensor
    self.labels = labels
    self.window = window
    self.stride = stride
    self.label_fn = label_fn

  def __getitem__(self, index):
    start = index * self.stride
    end = start + self.window
    return self.data[start:end], self.label_fn(self.labels[start:end])

  def __len__(self) -> int:
    return ((len(self.data) - self.window) // self.stride) + 1
