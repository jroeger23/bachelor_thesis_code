import argparse
from pathlib import Path

from ptq_cnn_imu import ex as ptq_experiment

N_BITS = {
    ('n_bits', 7),
    ('n_bits', 6),
    ('n_bits', 5),
    ('n_bits', 4),
    ('n_bits', 3),
    ('n_bits', 2),
}

USE_DATASET = {
    ('use_dataset', 'lara'),
    ('use_dataset', 'opportunity'),
    ('use_dataset', 'pamap2'),
}

WEIGHT_RANGE = {
    ('weight_range', 'symmetric'),
    ('weight_range', 'full'),
}

IMU_INPUT_QUANTIZATION = [
    {
        'imu_input_quantization': 'static',
    },
    {
        'imu_input_quantization': 'none',
    },
]

IMU_PIPELINE_QUANTIZATION = [
    {
        'imu_pipeline_quantization': 'static',
    },
    {
        'imu_pipeline_quantization': 'none',
    },
]

IMU_PIPELINE_FC_QUANTIZATION = [
    {
        'imu_fc_quantization': 'static',
    },
    {
        'imu_fc_quantization': 'none',
    },
]

FC_QUANTIZATION = [
    {
        'fc_quantization': 'static',
    },
    {
        'fc_quantization': 'none',
    },
]

OUTPUT_LAYER_QUANTIZATION = [
    {
        'output_layer_quantization': 'static',
    },
    {
        'output_layer_quantization': 'none',
    },
]

QUANTIZATION_MODE = [
    {
        'imu_input_quantization': 'static',
        'imu_pipeline_quantization': 'static',
        'imu_pipeline_fc_quantization': 'static',
        'fc_quantization': 'static',
        'output_layer_quantization': 'static',
    },
]

BASE = {
    'activation_observer': 'torch.ao.quantization.MovingAverageMinMaxObserver',
    'weight_observer': 'torch.ao.quantization.PerChannelMinMaxObserver',
}

meta = {
    'my_meta': {
        'runner': Path(__file__).name,
        'version': 0,
    }
}


def allConfigs():
  variable = [
      dict([b, d]) | BASE | imu_input_quantization | imu_pipeline_quantization |
      imu_pipeline_fc_quantization | fc_quantization | output_layer_quantization for b in N_BITS
      for d in USE_DATASET for imu_input_quantization in IMU_INPUT_QUANTIZATION
      for imu_pipeline_quantization in IMU_PIPELINE_QUANTIZATION
      for imu_pipeline_fc_quantization in IMU_PIPELINE_FC_QUANTIZATION
      for fc_quantization in FC_QUANTIZATION
      for output_layer_quantization in OUTPUT_LAYER_QUANTIZATION
  ]

  return variable


parser = argparse.ArgumentParser(description='Run relevant observer configuration for ptq_cnn_imu')
parser.add_argument('--dry_run', '-d', action='store_true')
args = parser.parse_args()

configs = allConfigs()

if args.dry_run:
  for x in configs:
    print(x)

  print(f'Would run {len(configs)} configurations')
else:
  for x in configs:
    ptq_experiment.run(config_updates=x, meta_info=meta)
