DEV_DIR=train_extract
mkdir -p $DEV_DIR
python fragment_forest.py --run_dir $DEV_DIR --data_dir ../dev
