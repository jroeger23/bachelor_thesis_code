#!/bin/bash

stride="12,33,50"
batch_size="64,256,512"
cnn_imu_blocks="2,3"
cnn_imu_fc_features="256,512"
lr="1e-3,1e-4"

experiment='cnn-imu_lara.py'

generate_options() {
  local IFS=' '
  eval printf 'with\ %s,' "cnn_imu_blocks={$cnn_imu_blocks}\ cnn_imu_fc_features={$cnn_imu_fc_features}\ batch_size={$batch_size}\ stride={$stride}\ lr={$lr}"
}

IFS=','
for opt in $(generate_options); do 
  IFS=' '
  python $experiment $opt
done