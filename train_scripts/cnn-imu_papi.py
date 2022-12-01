import sys

import torch

from common.model import CNNIMU


def main():
  model = CNNIMU.load_from_checkpoint(sys.argv[1])

  batch = [
      torch.rand(size=(1, model.hparams['sample_length'], imu_size))
      for imu_size in model.hparams['imu_sizes']
  ]

  print(model.performanceStatistics(batch))


if __name__ == '__main__':
  main()