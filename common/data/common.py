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
    labels = labels.squeeze()
    if labels.ndim == 1:
      if len(labels) == 1:
        return labels_map[int(labels.item())]
      else:
        return [labels_map[int(l.item())] for l in labels]
    else:
      raise ValueError('Cannot describe multi dimensional labels')
  elif type(labels) is list:
    return [labels_map[int(l)] for l in labels]
  elif type(labels) is int:
    return labels_map[labels]
  else:
    raise TypeError(f'Requires tensor, int or List[int] not {type(labels)}')


class View():
  """The View interface
  """

  def __call__(self, sample: torch.Tensor, label: torch.Tensor) -> t.Any:
    """Apply a view

    Args:
        sample (torch.Tensor): sample to view
        label (torch.Tensor): label to view

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

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply this transformation

    Args:
        sample (torch.Tensor): the sample to transform
        label (torch.Tensor): the label to transform

    Raises:
        NotImplementedError: interface

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: (transformed sample, transformed label)
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

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply all transformations

    Args:
        sample (torch.Tensor): the sample to transform
        label (torch.Tensor): the label to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: (transformed batch, transformed labels)
    """
    for t in self.transforms:
      # only apply transformations, as long as there is still data
      if len(sample) != 0:
        sample, label = t(sample, label)

    return sample, label

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

  def __init__(self, sample_view: View, label_view: View):
    """New CombineViews

    Args:
        sample_view (View): the sample view
        label_view (View): the label view
    """
    self.sample_view = sample_view
    self.label_view = label_view

  def __call__(self, sample: torch.Tensor, lable: torch.Tensor) -> t.Any:
    """Apply both views

    Args:
        sample (torch.Tensor): the sample to view
        label (torch.Tensor): the label to view

    Returns:
        t.Any: view output
    """
    return self.label_view(*self.sample_view(sample, lable))

  def __str__(self) -> str:
    return f'CombineViews \n\t[\n\t  Sample:  {str(self.sample_view)}\n\t  Label: {str(self.label_view)}\n\t]'


class NaNToConstTransform(Transform):
  """A Transformation, which replaces all NaN values with a constant
  """

  def __init__(self, sample_constant=0, label_constant=0):
    """Choose the constants for batch and label

    Args:
        sample_constant (int, optional): sample constant . Defaults to 0.
        label_constant (int, optional): labels constant . Defaults to 0.
    """
    self.sample_constant = sample_constant
    self.label_constant = label_constant

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply NaN to const transformation

    Args:
        sample (torch.Tensor): the sample to transform
        label (torch.Tensor): the label to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: nan replaced inputs
    """
    return torch.nan_to_num(input=sample,
                            nan=self.sample_constant), torch.nan_to_num(input=label,
                                                                        nan=self.label_constant)

  def __str__(self) -> str:
    return f'NaNToConst (sample_nan={self.sample_constant}, label_nan={self.label_constant})'


class RemoveNanRows(Transform):

  def __init__(self) -> None:
    """Remove all nan rows from a sample/label pair
    """
    super().__init__()

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply remove nan rows

    Args:
        sample (torch.Tensor): the sample to filter
        label (torch.Tensor): the labels to filter

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: sample (no nan values), label (no nan values)
    """
    keep_cond_sample = torch.any(sample.isnan(), dim=1, keepdim=False).logical_not()

    if label.ndim == 1:
      keep_cond_label = label.isnan().logical_not()
    else:
      keep_cond_label = torch.any(label.isnan(), dim=1, keepdim=False).logical_not()

    keep_cond = torch.logical_and(keep_cond_sample, keep_cond_label)

    return sample[keep_cond], label[keep_cond]

  def __str__(self) -> str:
    return 'RemoveNanRows()'


class LabelDtypeTransform(Transform):
  """Transform the datatype of labels
  """

  def __init__(self, dtype: torch.dtype):
    """Choose the dtype

    Args:
        dtype (torch.dtype): new labels dtype
    """
    self.dtype = dtype

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply dtype transformation

    Args:
        sample (torch.Tensor): unchanged
        label (torch.Tensor): the label to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: sample, label(as dtype)
    """
    return sample, label.type(dtype=self.dtype)

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

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply re-sampling to a sequence of data

    Args:
        sample (torch.Tensor): the sequence of datapoints (N x D)
        label (torch.Tensor): the sequence of labels (N x L)

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: resampled batch (N' x D), resampled labels (N' x L)
    """
    ratio = self.freq_out / self.freq_in
    fb = tuple([ratio] + [1 for _ in range(sample.ndim - 1)])
    fl = tuple([ratio] + [1 for _ in range(labels.ndim - 1)])
    sample = torch.nn.functional.interpolate(input=sample.unsqueeze(0).unsqueeze(0),
                                             scale_factor=fb)
    labels = torch.nn.functional.interpolate(input=labels.unsqueeze(0).unsqueeze(0),
                                             scale_factor=fl)

    return sample.squeeze(), labels.squeeze()

  def __str__(self) -> str:
    return f'DownsampleTransform (f_in={self.freq_in}, f_out={self.freq_out})'


class BatchAdditiveGaussianNoise(Transform):

  def __init__(self, mu: float = 0, sigma: float = 1) -> None:
    """Add normal distributed noise to the batch

    Args:
        mu (float, optional): mean. Defaults to 0.
        sigma (float, optional): standard deviation. Defaults to 1.
    """
    self.mu = mu
    self.sigma = sigma
    self.normal = torch.distributions.Normal(loc=self.mu, scale=self.sigma)

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply gaussian noise to the sample

    Args:
        sample (torch.Tensor): the sample to transform
        label (torch.Tensor): the label (untouched)

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: sample + noise, label
    """
    noise = self.normal.sample(sample_shape=sample.shape)

    return sample + noise, labels

  def __str__(self) -> str:
    return f'BatchAdditiveGaussianNoise(mu={self.mu}, sigma={self.sigma})'


class RangeNormalize(Transform):

  def __init__(self, range_min: float = 0, range_max: float = 1, dim: int = 0) -> None:
    """Range normalize a sample along a dimension

    Args:
        range_min (float, optional): desired minimum sample value. Defaults to 0.
        range_max (float, optional): desired maximum sample value. Defaults to 1.
        dim (int, optional): the dimension, along which the normalization takes place. Defaults to 0.
    """
    self.range_min = range_min
    self.range_max = range_max
    self.dim = dim

  def __call__(self, sample: torch.Tensor,
               labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply sample range normalization

    Args:
        sample (torch.Tensor): the sample to transform
        label (torch.Tensor): the label to transform (untouched)

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: range normalied sample, labels
    """
    sample -= sample.amin(dim=self.dim, keepdim=True)  # Align zero
    sample /= sample.amax(dim=self.dim, keepdim=True)  # Unit size
    sample *= (self.range_max - self.range_min)  # Scale
    sample += self.range_min  # Shift zero

    return sample, labels

  def __str__(self) -> str:
    return f'RangeNormalize(range_min={self.range_min}, range_max={self.range_max}, dim={self.dim})'


class MeanVarianceNormalize(Transform):

  def __init__(self, mean: float = 0, variance: float = 1, dim: int = 0) -> None:
    """Mean/Varaince normalize a sample

    Args:
        mean (float, optional): target mean. Defaults to 0.
        variance (float, optional): target variance. Defaults to 1.
        dim (int, optional): the dimension, along which the normalization will be calculated. Defaults to 0.
    """
    self.mean = mean
    self.variance = variance
    self.dim = dim

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply the mean/variance normalization

    Args:
        sample (torch.Tensor): the sample to normalize
        label (torch.Tensor): the label (untouched)

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: normalized sample, label
    """

    sample -= sample.mean(dim=self.dim, keepdim=True)  # Zero mean
    std_dev = sample.std(dim=self.dim, keepdim=True)
    std_dev[std_dev == 0.0] = 1.0  # Ensure numerical stability
    sample /= std_dev  # Unit variance
    sample *= self.variance**0.5  # Scale to new variance
    sample += self.mean  # Shift to new mean

    return sample, label

  def __str__(self) -> str:
    return f'MeanVarianceNormalize(mean={self.mean}, variance={self.variance}, dim={self.dim})'


class ClipSampleRange(Transform):

  def __init__(self, range_min: float = 0, range_max: float = 1) -> None:
    self.range_min = range_min
    self.range_max = range_max

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    sample[sample < self.range_min] = self.range_min
    sample[sample > self.range_max] = self.range_max

    return sample, label

  def __str__(self) -> str:
    return f'ClipSampleRange(range_min={self.range_min}, range_max={self.range_max}'


class BlankInvalidColumns(Transform):

  def __init__(self,
               sample_thresh: t.Optional[float],
               label_thresh: t.Optional[float],
               sample_const: float = 0,
               label_const: float = 0) -> None:
    """Blank entire columns, with a certain theshold of NaN values

    Args:
        sample_thresh (t.Optional[float]): sample thresh [0..1] if None, dont blank
        label_thresh (t.Optional[float]): label thresh [0..1] if None, dont blank
        sample_const (float, optional): the sample constant to use for blanking. Defaults to 0.
        label_const (float, optional): the label constant to use for blanking. Defaults to 0.
    """
    self.sample_thresh = sample_thresh
    self.label_thresh = label_thresh
    self.sample_const = sample_const
    self.label_const = label_const

  def __call__(self, sample: torch.Tensor,
               label: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply the transformation

    Args:
        sample (torch.Tensor): the sample to transform
        label (torch.Tensor): the label to transform

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: (blanked sample, blanked label)
    """
    if self.sample_thresh is not None:
      sample_rel_nan = sample.isnan().type(torch.float).sum(dim=0, keepdim=False) / sample.size()[0]
      sample_blank_cond = sample_rel_nan >= self.sample_thresh
      sample[:, sample_blank_cond] = self.sample_const
    if self.label_thresh is not None:
      label_rel_nan = label.isnan().type(torch.float).sum(dim=0, keepdim=False) / label.size()[0]
      label_blank_cond = label_rel_nan >= self.label_thresh
      label[:, label_blank_cond] = self.label_const

    return sample, label

  def __str__(self) -> str:
    return f'BlankInvalidColumns(sample_thresh={self.sample_thresh}, label_thresh={self.label_thresh}, sample_const={self.label_const}, label_const={self.label_const})'


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
