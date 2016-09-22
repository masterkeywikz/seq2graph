#!/bin/sh
#If sequence to AMR
#python amr2seq.py --version 2.0 --amr2seq --train_data_dir ../train --data_dir ../dev

INPUT=$1
# clear up the parsed file
#tail -n +1 $INPUT | sed 's/> //g' > $INPUT.tmp
cat $INPUT > $INPUT.tmp

#If AMR to sequence
python amr2seq.py --data_dir ../dev --amrseq_file $INPUT.tmp --seq2amr --version 2.2

#rm $INPUT.tmp
