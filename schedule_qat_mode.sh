#!/bin/bash

activation_observer="torch.ao.quantization.HistogramObserver,torch.ao.quantization.MinMaxObserver,torch.ao.quantization.MovingAverageMinMaxObserver"
weight_observer="torch.ao.quantization.MinMaxObserver,torch.ao.quantization.PerChannelMinMaxObserver"
use_dataset="pamap2,lara,opportunity"


experiment="train_scripts/qat_cnn-imu.py"

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "n_bits=7\ activation_observer={$activation_observer}\ weight_observer={$weight_observer}\ weight_range=full\ use_dataset={$use_dataset}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done