#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpuid> [size=128]"
    exit
fi

gpuid=$1
size=128
model_dir=./model_$size

CUDA_VISIBLE_DEVICES=$gpuid python translate_amr.py --decode --train_dir $model_dir --size $size --use_lstm

