#!/bin/bash

stride="12,33"
batch_size="64,256"
cnn_imu_blocks="2,3"
cnn_imu_fc_features="256,512"
use_transient_class="True,False"

experiment='cnn-imu_pamap2.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "use_transient_class={$use_transient_class}\ cnn_imu_blocks={$cnn_imu_blocks}\ cnn_imu_fc_features={$cnn_imu_fc_features}\ batch_size={$batch_size}\ stride={$stride}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done