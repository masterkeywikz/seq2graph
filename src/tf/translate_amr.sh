#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpuid> [size=128]"
    exit
fi

gpuid=$1

size=128

if [ $# -eq 2 ]; then
    size=$2
fi

model_dir=./model_$size

CUDA_VISIBLE_DEVICES=$gpuid python translate_amr.py --size 128 --train_dir $model_dir --use_lstm --output_keep_prob 0.5

