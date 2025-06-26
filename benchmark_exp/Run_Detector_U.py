# TSB_AD/Run_Detector_U.py

import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import *
from TSB_AD.utils.slidingWindows import find_length_rank

# Seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def process_single_file(filename, args, optimal_hp):
    """
    Processes a single file: runs the anomaly detector, saves the score,
    and returns the evaluation metrics as a list.
    This function is designed to be called in parallel.
    """
    target_dir = os.path.join(args.score_dir, args.AD_Name)
    
    # Skip if score already exists
    if os.path.exists(os.path.join(target_dir, filename.split('.')[0] + '.npy')):
        return None

    try:
        print(f"Processing: {filename} by {args.AD_Name}")
        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        
        start_time = time.time()

        # Route to the correct model execution function
        if args.AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(args.AD_Name, data, **optimal_hp)
        else:
            raise Exception(f"{args.AD_Name} is not defined in Unsupervise_AD_Pool")
        
        run_time = time.time() - start_time

        if not isinstance(output, np.ndarray):
            logging.error(f'At {filename}: ' + str(output))
            return None

        logging.info(f'Success at {filename} | Time: {run_time:.3f}s | Length: {len(label)}')
        np.save(os.path.join(target_dir, filename.split('.')[0] + '.npy'), output)

        if args.save:
            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
            metrics_list = list(evaluation_result.values())
            metrics_list.insert(0, run_time)
            metrics_list.insert(0, filename)
            return metrics_list
        return None

    except Exception as e:
        logging.error(f"Failed to process file {filename}. Error: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--file_list', type=str, default='Datasets/File_List/TSB-AD-U.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--AD_Name', type=str, default='TSPulse2')
    args = parser.parse_args()

    # Setup directories and logging
    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    print("CUDA available: ", torch.cuda.is_available())
    print("cuDNN version: ", torch.backends.cudnn.version())

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    print(f'Optimal Hyperparameters for {args.AD_Name}: {Optimal_Det_HP}')

    # --- Parallel Execution ---
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(process_single_file)(filename, args, Optimal_Det_HP) 
        for filename in tqdm(file_list, desc=f"Running {args.AD_Name} on Univariate data")
    )
    
    # --- Aggregate and Save Results ---
    # Filter out None values from failed or skipped files
    write_csv = [res for res in results if res is not None]

    if args.save and write_csv:
        col_w = ['AUC-PR','AUC-ROC','VUS-PR','VUS-ROC','Standard-F1','PA-F1','Event-based-F1','R-based-F1','Affiliation-F']
        col_w.insert(0, 'Time')
        col_w.insert(0, 'file')
        w_csv = pd.DataFrame(write_csv, columns=col_w)
        w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False, float_format="%.5f")
        print(f"Results saved to {args.save_dir}/{args.AD_Name}.csv")
    else:
        print("No new results to save.") 