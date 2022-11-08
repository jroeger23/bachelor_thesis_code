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
                      drop_n: int = 0):
  ret = torch.zeros(n_cols, dtype=dtype)

  for ix, word in enumerate(re.split(pattern=sep_re, string=line)[drop_n:]):
    ret[ix] = float(word)

  return ret


def parse_dat(path: str, dtype=torch.float32):
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


def load_cached_dat(root: str, name: str, dtype=torch.float32, logger=logger):
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


CSV_SEP_REGEX = '\s*,\s*'


def parse_csv_header(path: str) -> t.List[str]:
  with open(path, 'r') as f:
    return re.split(pattern=CSV_SEP_REGEX, string=f.readline())


def parse_csv(path: str,
              dtype=torch.float32,
              drop_n: int = 0) -> t.Tuple[t.List[str], torch.Tensor]:
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


def majority_label(labels: torch.Tensor):
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


def ensure_download_file(url: str, file: str, sha512_hex: str = None):
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
  length = int(response.headers.get('content-length'))
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
                        zip_dirs: t.List[int] = [],
                        sha512_hex: str = None):
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


class Transform():

  def __call__(self, batch: torch.Tensor, labels: torch.Tensor):
    raise NotImplementedError()

  def __str__(self) -> str:
    raise NotImplementedError()


class ComposeTransforms(Transform):

  def __init__(self, *transforms: t.Tuple[Transform]):
    self.transforms = transforms

  def __call__(self, batch: torch.Tensor, labels: torch.Tensor):
    for t in self.transforms:
      batch, labels = t(batch, labels)

    return batch, labels

  def __str__(self) -> str:
    s = "ComposeTransforms: \n\t[\n"
    for t in self.transforms:
      s += f'\t  -> {str(t)}\n'

    return s + "\t]"


class SegmentedDataset(Dataset):

  def __init__(self, tensor, labels, window: int, stride: int, label_fn=majority_label):
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
