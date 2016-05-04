#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpuid>"
    exit
fi

gpuid=$1

CUDA_VISIBLE_DEVICES=$gpuid python translate_amr.py --decode

