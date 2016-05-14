#!/usr/bin/bash
#TRAIN_DIR=../train_linearize
#mkdir -p $TRAIN_DIR
#python categorize_amr.py --run_dir $TRAIN_DIR --data_dir ../train

DEV_DIR=../dev_linearize
mkdir -p $DEV_DIR
python categorize_amr.py --run_dir $DEV_DIR --data_dir ../dev
#./train/aligned_amr_nosharp
