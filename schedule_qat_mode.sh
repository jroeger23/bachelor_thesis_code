#!/bin/bash

n_bits="7,"
activation_observer="torch.ao.quantization.HistogramObserver,torch.ao.quantization.MinMaxObserver,torch.ao.quantization.MovingAverageMinMaxObserver"
weight_observer="torch.ao.quantization.MinMaxObserver,torch.ao.quantization.PerChannelMinMaxObserver"
weight_range="full,"
dataset="pamap2,lara,opportunity"


experiment="train_scripts/qat_cnn-imu.py"

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "n_bits={$n_bits}\ activation_observer={$activation_observer}\ weight_observer={$weight_observer}\ weight_range={$weight_range}\ dataset={$dataset}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done