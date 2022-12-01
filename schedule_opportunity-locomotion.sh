#!/bin/bash

batch_size="64,256"
cnn_imu_blocks="2,3"
cnn_imu_fc_features="256,512"
optimizer="RMSProp,Adam"

experiment='train_scripts/cnn-imu_opportunity-locomotion.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "cnn_imu_blocks={$cnn_imu_blocks}\ cnn_imu_fc_features={$cnn_imu_fc_features}\ batch_size={$batch_size}\ optimizer={$optimizer}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done