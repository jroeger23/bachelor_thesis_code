import sys

import torch

from common.model import CNNIMU
from ptflops import get_model_complexity_info


def main():
  model = CNNIMU.load_from_checkpoint(sys.argv[1])
  model.eval()

  def batch_ctor(*_):
    return {
        'imu_x': [
            torch.rand(size=(1, model.hparams['sample_length'], imu_size))
            for imu_size in model.hparams['imu_sizes']
        ]
    }

  macs, params = get_model_complexity_info(model=model,
                                           input_res=(0,),
                                           input_constructor=batch_ctor,
                                           as_strings=True,
                                           print_per_layer_stat=True,
                                           verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
  main()