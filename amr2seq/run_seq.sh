#!/usr/bin/bash
#If sequence to AMR
AMRseqfile=$1
resultfile=$2
python amr2seq.py --amrseq_file $AMRseqfile --amr_result_file $resultfile --seq2amr

#If AMR to sequence
#resultfile=$1
#python amr2seq.py --seq_result_file $resultfile --amr2seq
