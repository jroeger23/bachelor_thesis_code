#!/bin/bash

n_bits="7,6,5,4,3,2,1"
weight_range="full,symmetric,uint"
use_dataset="pamap2,lara,opportunity"


experiment='train_scripts/qat_cnn-imu.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "n_bits={$n_bits}\ activation_observer=torch.ao.quantization.MovingAverageMinMaxObserver\ weight_observer=torch.ao.quantization.PerChannelMinMaxObserver\ weight_range={$weight_range}\ use_dataset={$use_dataset}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done