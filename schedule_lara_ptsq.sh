#!/bin/bash

n_bits="7,6,5,4,3"
activation_observer="torch.ao.quantization.HistogramObserver,torch.ao.quantization.MinMaxObserver,torch.ao.quantization.PerChannelMinMaxObserver,torch.ao.quantization.MovingAverageMinMaxObserver,torch.ao.quantization.MovingAveragePerChannelMinMaxObserver"
weight_observer="torch.ao.quantization.MinMaxObserver,torch.ao.quantization.PerChannelMinMaxObserver"
weight_range="full,symmetric,uint"
limit_calibration_set="None,0.1,0.01"


experiment='train_scripts/ptsq_cnn-imu_lara.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "n_bits={$n_bits}\ activation_observer={$activation_observer}\ weight_observer={$weight_observer}\ weight_range={$weight_range}\ limit_calibration_set={$limit_calibration_set}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done