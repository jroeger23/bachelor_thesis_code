import argparse
from pathlib import Path

from qat_cnn_imu import ex as qat_experiment

N_BITS = {
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

QUANTIZATION_MODE = [
    {
        'imu_input_quantization': 'qat',
        'imu_pipeline_quantization': 'qat',
        'imu_pipeline_fc_quantization': 'qat',
        'fc_quantization': 'qat',
        'output_layer_quantization': 'qat',
    },
    {
        'imu_input_quantization': 'none',
        'imu_pipeline_quantization': 'qat',
        'imu_pipeline_fc_quantization': 'qat',
        'fc_quantization': 'qat',
        'output_layer_quantization': 'none',
    },
    {
        'imu_input_quantization': 'qat',
        'imu_pipeline_quantization': 'qat',
        'imu_pipeline_fc_quantization': 'qat',
        'fc_quantization': 'none',
        'output_layer_quantization': 'none',
    },
    {
        'imu_input_quantization': 'qat',
        'imu_pipeline_quantization': 'qat',
        'imu_pipeline_fc_quantization': 'none',
        'fc_quantization': 'none',
        'output_layer_quantization': 'none',
    },
    {
        'imu_input_quantization': 'none',
        'imu_pipeline_quantization': 'none',
        'imu_pipeline_fc_quantization': 'qat',
        'fc_quantization': 'qat',
        'output_layer_quantization': 'qat',
    },
    {
        'imu_input_quantization': 'none',
        'imu_pipeline_quantization': 'none',
        'imu_pipeline_fc_quantization': 'qat',
        'fc_quantization': 'qat',
        'output_layer_quantization': 'none',
    },
]
QUANTIZATION_MODE_V2 = [
    {
        'imu_input_quantization': 'qat',
        'imu_pipeline_quantization': 'qat',
        'imu_pipeline_fc_quantization': 'qat',
        'fc_quantization': 'qat',
        'output_layer_quantization': 'none',
    },
]

QUANTIZATION_MODE_V3 = QUANTIZATION_MODE + QUANTIZATION_MODE_V2

BASE = {
    'activation_observer': 'torch.ao.quantization.MovingAverageMinMaxObserver',
    'weight_observer': 'torch.ao.quantization.PerChannelMinMaxObserver',
}


def allConfigs(use_v2: bool, use_v3: bool):
  variable = [
      dict([b, d]) | BASE | qmode for b in N_BITS for d in USE_DATASET for qmode in (
          QUANTIZATION_MODE_V2 if use_v2 else QUANTIZATION_MODE_V3 if use_v3 else QUANTIZATION_MODE)
  ]

  return variable


parser = argparse.ArgumentParser(description='Run relevant mode configurations for qat_cnn_imu')
parser.add_argument('--dry_run', '-d', action='store_true')
parser.add_argument('--use_v2', action='store_true')
parser.add_argument('--use_v3', action='store_true')
args = parser.parse_args()

configs = allConfigs(args.use_v2, args.use_v3)

meta = {
    'my_meta': {
        'runner': Path(__file__).name,
        'version': 2 if args.use_v2 else 3 if args.use_v3 else 0,
    }
}

if args.dry_run:
  for x in configs:
    print(x)

  print(f'Would run {len(configs)} configurations')
else:
  for x in configs:
    qat_experiment.run(config_updates=x, meta_info=meta)
