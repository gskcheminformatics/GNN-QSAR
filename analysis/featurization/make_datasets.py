import os
import argparse
import pandas as pd
from make_data.dataset import create_preprocessed_data
import cProfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--inp_path", type=str)
    parser.add_argument("--process_dir_path", type=str)
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.isdir(os.path.join(args.log_dir, 'dataset')):
        os.makedirs(os.path.join(args.log_dir, 'dataset'))

    os.makedirs(os.path.join(args.process_dir_path, 'processed'))
    create_preprocessed_data(input_tsv_file=args.inp_path, processed_dir=os.path.join(args.process_dir_path, 'processed'), compute_all_features=True, cx_pK=True, log_dir=args.log_dir, time_features=False)
