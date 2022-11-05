import torch
import os
import logging
import tqdm
import requests
from zipfile import ZipFile
import typing as t


from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def parse_tensor_line(line : str, n_cols : int, dtype=torch.float32):
  ret = torch.zeros(n_cols, dtype=dtype)

  for ix, word in enumerate(line.split(sep=' ')):
    ret[ix] = float(word)

  return ret

def parse_dat(path : str, dtype = torch.float32):
  with open(file=path, mode='r') as f:
    lines = f.readlines()
    n_cols = len(lines[0].split(' '))
    
    ret = torch.zeros(len(lines), n_cols, dtype=dtype)

    for ix, line in tqdm.tqdm(iterable=enumerate(lines), desc='Parsing .dat', unit='line', total=len(ret), leave=False):
      ret[ix] = parse_tensor_line(line=line, n_cols=n_cols, dtype=dtype)

    return ret

def load_cached_dat(root : str, name : str, dtype=torch.float32, logger=logger):
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

def majority_label(labels : torch.Tensor):
  return labels.mode(dim=0)[0]


def ensure_download_file(url : str, file : str):
  # Download file if not present
  if not os.path.exists(file):
    logger.info(f'"{file}" not present, downloading from "{url}"...')
    response = requests.get(url=url, verify=True, stream=True)
    chunk_size = 4096
    length = int(response.headers.get('content-length'))
    with open(file, 'wb') as f:
      with tqdm.tqdm(desc=os.path.basename(file), unit='B', unit_scale=True, unit_divisor=1024, total=length) as progress:
        for d in response.iter_content(chunk_size=chunk_size):
          f.write(d)
          progress.update(n=chunk_size)


def ensure_download_zip(url : str, root : str, dataset_name : str, zip_dirs : t.List[int] = []):
  dataset_direcoty = os.path.join(root, dataset_name)
  zip_path = dataset_direcoty + '.zip'
  if os.path.exists(dataset_direcoty):
    # OK
    logger.debug(f'Dataset at "{dataset_direcoty}" is present.')
    return

  logger.debug(f'Ensuring root path \"{root}\"')
  os.makedirs(name=root, exist_ok=True)

  # Download zip if not present
  ensure_download_file(url=url, file=zip_path)

  logger.info(f'Unzipping "{zip_path}"...')
  with ZipFile(zip_path, 'r') as zf:
    for d in zip_dirs:
      zf.extract(member=d, path=dataset_direcoty)
    else:
      zf.extractall(path=dataset_direcoty)
  logger.info(f'Done!')



class SegmentedDataset(Dataset):
  def __init__(self, tensor, labels, window : int, stride : int, label_fn = majority_label):
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
