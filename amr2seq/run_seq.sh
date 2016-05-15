#!/usr/bin/bash
#If sequence to AMR
#python amr2seq.py --version 2.0 --amr2seq --train_data_dir ../train --data_dir ../dev

#If AMR to sequence
python amr2seq.py --data_dir ../dev --amrseq_file ../dev/dev.parsed.1.0.amr --seq2amr --version 2.0
