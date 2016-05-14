#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpuid> [size=256]"
    exit
fi

gpuid=$1

size=256

if [ $# -eq 2 ]; then
    size=$2
fi

model_dir=./model_$size

CUDA_VISIBLE_DEVICES=$gpuid python translate_amr.py --size 256 --train_dir $model_dir  --src_vocab_size 19000 --dst_vocab_size 8965 --src_fn toks_thred50_70 --dst_fn amrseq_thred50_70 --use_lstm

