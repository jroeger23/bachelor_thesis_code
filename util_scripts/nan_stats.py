#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import numpy as np


def nan_loc(line: str) -> np.ndarray:
  occurances = [1 if col == 'NaN' else 0 for col in line.split(sep=' ')]
  return np.array(occurances, dtype=np.float32)


def count_nan_loc(path: Path) -> np.ndarray:
  with path.open(mode='r') as f:
    locs_per_line = [nan_loc(l) for l in f.readlines()]
    return np.row_stack(locs_per_line)


def print_file_nan_stats(path: Path) -> None:
  nan_loc = count_nan_loc(path)
  stat = nan_loc.sum(axis=0).squeeze()
  stat /= len(nan_loc)
  stat *= 100

  top = np.argsort(a=stat)[-7:]

  print(f'{path}:\t', end='')
  for ix in top:
    print(f' C{ix:03d}={float(stat[ix]):03.01f}%', end='')
  print('', flush=True)


def main():
  directory = Path(sys.argv[1] if len(sys.argv) >= 2 else os.getcwd())

  for path in directory.glob('**/*.dat'):
    print_file_nan_stats(path)


if __name__ == '__main__':
  main()
