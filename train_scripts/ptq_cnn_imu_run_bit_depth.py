import argparse
from pathlib import Path

from ptq_cnn_imu import ex as ptq_experiment

ACTIVATION_OBSERVER = [
    ('activation_observer', 'torch.ao.quantization.HistogramObserver'),
    ('activation_observer', 'torch.ao.quantization.MovingAverageMinMaxObserver'),
]

WEIGHT_OBSERVER = {
    ('weight_observer', 'torch.ao.quantization.MinMaxObserver'),
    ('weight_observer', 'torch.ao.quantization.PerChannelMinMaxObserver'),
}

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

QUANTIZATION_MODES = [{
    'imu_input_quantization': 'none',
    'imu_pipeline_quantization': 'static',
    'imu_pipeline_fc_quantization': 'static',
    'fc_quantization': 'static',
    'output_layer_quantization': 'none',
}, {
    'imu_input_quantization': 'static',
    'imu_pipeline_quantization': 'static',
    'imu_pipeline_fc_quantization': 'static',
    'fc_quantization': 'static',
    'output_layer_quantization': 'static',
}]

BASE = {
    'weight_range': 'full',
}

meta = {
    'my_meta': {
        'runner': Path(__file__).name,
        'version': 0,
    }
}


def allConfigs():
  variable = [
      dict([ao, wo, b, d]) | qmode | BASE for ao in ACTIVATION_OBSERVER for wo in WEIGHT_OBSERVER
      for qmode in QUANTIZATION_MODES for b in N_BITS for d in USE_DATASET
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
