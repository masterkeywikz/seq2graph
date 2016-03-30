DEV_DIR=dev_linearize
mkdir -p $DEV_DIR
#python categorize_amr.py --run_dir run_linearize --data_dir train
python categorize_amr.py --run_dir $DEV_DIR --data_dir ../dev
