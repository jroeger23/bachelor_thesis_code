#!/bin/bash

stride="12,33"
batch_size="64,256"
cnn_imu_blocks="2,3"
cnn_imu_fc_features="256,512"

experiment='train_scripts/cnn_imu_pamap2_all.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "use_transient_class=False\ cnn_imu_blocks={$cnn_imu_blocks}\ cnn_imu_fc_features={$cnn_imu_fc_features}\ batch_size={$batch_size}\ stride={$stride}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done