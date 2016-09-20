#!/bin/sh

cat ../../dev/linearized_tokseq | python3.5 translate.py --decode > ../seq2graph.decode.1.1.log &

cp ../seq2graph.decode.1.1.log ../data/dev.decode.1.1.amrseq

cd ../../amr2seq

./run_seq.sh ../tfamr/data/dev.decode.1.1.amrseq

#python amr2seq.py --data_dir ../dev --amrseq_file ../tfamr/data/dev.decode.1.1.amrseq --seq2amr --version 1.1