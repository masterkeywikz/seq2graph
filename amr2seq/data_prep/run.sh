#!/bin/bash
#python ./categorize_amr.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel
#python ./categorize_amr.py --data_dir ../../dev --use_lemma --run_dir run_dir --stats_dir stats --use_stats --map_file ./run_dir/train_map
python ./categorize_amr.py --data_dir ../../eval --use_lemma --run_dir run_dir --stats_dir stats --use_stats --map_file ./run_dir/train_map
