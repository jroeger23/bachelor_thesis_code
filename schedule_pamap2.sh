#!/bin/bash

stride="10,12,20"
batch_size="64,256"
cnn_imu_blocks="2,3"
cnn_imu_fc_features="256,512"
use_transient_class="True,False"

experiment='cnn-imu_pamap2.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "use_transient_class={$use_transient_class}\ cnn_imu_blocks={$cnn_imu_blocks}\ batch_size={$batch_size}\ stride={$stride}"
}

IFS=','
for opt in $(generate_options); do 
  python $experiment $opt
done