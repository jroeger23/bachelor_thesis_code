#!/bin/bash

batch_size="64,256"
cnn_imu_blocks="2,3"
cnn_imu_fc_features="256,512"

experiment='train_scripts/cnn_imu_opportunity_locomotion.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "cnn_imu_blocks={$cnn_imu_blocks}\ cnn_imu_fc_features={$cnn_imu_fc_features}\ batch_size={$batch_size}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done