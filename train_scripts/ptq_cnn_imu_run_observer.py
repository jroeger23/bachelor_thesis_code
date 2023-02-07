from ptq_cnn_imu import ex as ptq_experiment
import argparse

ACTIVATION_OBSERVER = [
    ('activation_observer', 'torch.ao.quantization.HistogramObserver'),
    ('activation_observer', 'torch.ao.quantization.MinMaxObserver'),
    ('activation_observer', 'torch.ao.quantization.MovingAverageMinMaxObserver'),
]

WEIGHT_OBSERVER = {
    ('weight_observer', 'torch.ao.quantization.MinMaxObserver'),
    ('weight_observer', 'torch.ao.quantization.PerChannelMinMaxObserver'),
}
WEIGHT_RANGE = {
    ('weight_range', 'full'),
    ('weight_range', 'symmetric'),
    ('weight_range', 'uint'),
}

USE_DATASET = {
    ('use_dataset', 'lara'),
    ('use_dataset', 'opportunity'),
    ('use_dataset', 'pamap2'),
}

BASE = {
    'imu_input_quantization': 'static',
    'imu_pipeline_quantization': 'static',
    'imu_pipeline_fc_quantization': 'static',
    'fc_quantization': 'static',
    'output_layer_quantization': 'static',
}


def allConfigs():
  variable = [
      dict([ao, wo, wr, d]) | BASE for ao in ACTIVATION_OBSERVER for wo in WEIGHT_OBSERVER
      for wr in WEIGHT_RANGE for d in USE_DATASET
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
    ptq_experiment.run(config_updates=x)
