# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License
# Modified by AI Assistant for resumable HP tuning

import argparse
import ast  # To convert string representation of dict back to dict
import itertools
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.HP_list import Multi_algo_HP_dict  # Assumes your HP dict is here
from TSB_AD.model_wrapper import *
from TSB_AD.utils.slidingWindows import find_length_rank

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    # ArgumentParser
    parser = argparse.ArgumentParser(
        description='Resumable HP Tuning - Find Best Avg VUS-PR')
    parser.add_argument('--dataset_dir', type=str,
                        default='Datasets/TSB-AD-M/')
    parser.add_argument('--file_list', type=str,
                        default='Datasets/File_List/TSB-AD-M-Tuning.csv')
    parser.add_argument('--save_dir', type=str,
                        default='eval/HP_tuning/multi/')
    parser.add_argument('--AD_Name', type=str, default='Custom_AD')
    args = parser.parse_args()

    # --- Create Save Directory ---
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Define Progress File Path ---
    progress_file_path = os.path.join(
        args.save_dir, f'{args.AD_Name}_all_runs.csv')

    # --- Load File List ---
    try:
        file_list = pd.read_csv(args.file_list)[
            'file_name'].dropna().unique().tolist()
        if not file_list:
            raise ValueError("File list is empty.")
        print(f"Loaded {len(file_list)} unique files for tuning.")
    except Exception as e:
        print(f"Error loading file list '{args.file_list}': {e}")
        exit()

    # --- Get Hyperparameters ---
    try:
        if args.AD_Name == 'Custom_AD':
            Det_HP = {  # Performance-focused HPs
                'win_size': [50, 100, 150], 'lr': [0.001, 0.0005, 0.0002],
                'n_hidden': [32, 64, 128], 'n_latent': [8, 16, 32],
                'rnn_layers': [1, 2, 3], 'nf_layers': [0, 1, 3],
                'beta': [0.1, 0.01, 0.001], 'patience': [10, 20],
                'epochs': [100, 150, 200]
            }
            print(f"Using Performance-Focused HPs for {args.AD_Name}")
        else:
            Det_HP = Multi_algo_HP_dict[args.AD_Name]
            print(f"Using HPs from Multi_algo_HP_dict for {args.AD_Name}")
    except KeyError:
        print(f"Error: Hyperparameters for '{args.AD_Name}' not found.")
        exit()
    except Exception as e:
        print(f"Error accessing hyperparameters: {e}")
        exit()

    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total hyperparameter combinations to test: {len(combinations)}")

    # --- Load Existing Progress ---
    write_csv = []  # Stores results as list of lists
    completed_runs = set()  # Stores tuples of (filename, hp_string)
    if os.path.exists(progress_file_path):
        print(f"Loading existing progress from: {progress_file_path}")
        try:
            # Read existing results, ensure HP is treated as string
            loaded_df = pd.read_csv(progress_file_path, dtype={'HP': str})
            # Populate write_csv from the loaded DataFrame
            write_csv = loaded_df.values.tolist()
            # Populate the set of completed runs for quick lookup
            if 'file' in loaded_df.columns and 'HP' in loaded_df.columns:
                completed_runs = set(zip(loaded_df['file'], loaded_df['HP']))
                print(f"Resuming. Found {len(completed_runs)} completed runs.")
            else:
                print(
                    "Warning: Progress file missing 'file' or 'HP' column. Starting fresh.")
                write_csv = []  # Reset if format is wrong
                completed_runs = set()
        except pd.errors.EmptyDataError:
            print("Progress file is empty. Starting fresh.")
        except Exception as e:
            print(f"Error loading progress file: {e}. Starting fresh.")
            write_csv = []
            completed_runs = set()

    metric_keys_cache = None  # To store metric keys header

    # --- Original Nested Loop Structure ---
    hp_processed_count = 0  # Count how many new HPs are processed in this session
    newly_completed_runs_session = 0  # Count newly completed runs in this session

    for params in combinations:
        hp_string = str(params)  # String representation for lookup/storage
        hp_needs_processing = False  # Flag to check if any file needs processing for this HP

        # --- Check if any file *needs* processing for this HP before loading data ---
        for filename_check in file_list:
            if (filename_check, hp_string) not in completed_runs:
                hp_needs_processing = True
                break  # Found at least one file needing processing

        if not hp_needs_processing:
            print(f"Skipping HP (already completed for all files): {params}")
            continue  # Move to the next HP combination

        # --- If we need to process this HP set, print and proceed ---
        hp_processed_count += 1
        print(f"\n[{hp_processed_count}] Testing HP: {params}")

        for filename in file_list:
            # --- Check if this specific run is already done ---
            if (filename, hp_string) in completed_runs:
                # print(f"  Skipping completed run: {filename}") # Can be verbose
                continue  # Skip to next file

            # --- If not completed, process it ---
            newly_completed_runs_session += 1
            print('  Processing: {}'.format(filename))
            file_path = os.path.join(args.dataset_dir, filename)
            list_w = []  # Results for this specific run

            try:  # File loading/prep try-block
                df = pd.read_csv(file_path).dropna()
                if df.empty or 'Label' not in df.columns or df.shape[1] < 2:
                    print(f"    Warning: Skipping invalid file {filename}.")
                    # Log failure implicitly by not adding to write_csv / completed_runs
                    continue
                data = df.iloc[:, 0:-1].values.astype(float)
                label = df['Label'].astype(int).to_numpy()

                # Determine Train/Test Split (with fallback)
                train_index_str = filename.split('.')[0].split('_')[-3]
                try:
                    train_index = int(train_index_str)
                    if not (0 < train_index < len(data)):
                        train_index = max(1, int(0.3 * len(data)))
                except ValueError:
                    train_index = max(1, int(0.3 * len(data)))
                data_train = data[:train_index, :]

                # Estimate sliding window
                slidingWindow = find_length_rank(
                    data[:, 0].reshape(-1, 1), rank=1) if data.shape[1] > 0 else 10
                slidingWindow = max(2, slidingWindow)

            except Exception as file_load_e:
                print(
                    f"    Error loading or preparing file {filename}: {file_load_e}")
                continue  # Skip this file

            # --- Model run and metric calculation try-block ---
            try:
                output = None
                if args.AD_Name in Semisupervise_AD_Pool:
                    # Assuming returns score array
                    output = run_Semisupervise_AD(
                        args.AD_Name, data_train, data, **params)
                elif args.AD_Name in Unsupervise_AD_Pool:
                    output = run_Unsupervise_AD(args.AD_Name, data, **params)
                else:
                    print(
                        f"    Warning: {args.AD_Name} not in known pools. Skipping run.")
                    list_w = [0]*9  # Default error values

                # Get Metrics
                if output is not None and isinstance(output, np.ndarray) and output.ndim == 1 and len(output) == len(label):
                    evaluation_result = get_metrics(
                        output, label, slidingWindow=slidingWindow)
                    if metric_keys_cache is None:
                        metric_keys_cache = list(evaluation_result.keys())
                    list_w = list(evaluation_result.values())
                else:
                    print(f"    Warning: Invalid score output. Assigning zero metrics.")
                    if metric_keys_cache:
                        list_w = [0] * len(metric_keys_cache)
                    else:
                        list_w = [0]*9

            except Exception as run_error:
                print(
                    f"    Error during model run or metric calculation: {run_error}")
                # traceback.print_exc()
                if metric_keys_cache:
                    list_w = [0] * len(metric_keys_cache)
                else:
                    list_w = [0]*9

            # --- Append results for this run ---
            list_w.insert(0, hp_string)  # Store HP string
            list_w.insert(0, filename)
            write_csv.append(list_w)

            # --- Add to completed set and save progress ---
            completed_runs.add((filename, hp_string))
            try:
                # Define column names (use cache or fallback)
                if metric_keys_cache:
                    col_names = ['file', 'HP'] + metric_keys_cache
                else:
                    col_names = ['file', 'HP'] + \
                        [f'Metric_{k}' for k in range(len(list_w)-2)]

                # Create DataFrame from potentially updated write_csv list
                temp_df_to_save = pd.DataFrame(write_csv, columns=col_names)
                # Overwrite progress file
                temp_df_to_save.to_csv(progress_file_path, index=False)
                # print(f"    Progress saved. Total completed runs: {len(completed_runs)}") # Can be verbose
            except Exception as save_e:
                print(
                    f"    Warning: Could not save progress after {filename}: {save_e}")

    # --- End of Loops ---
    print(f"\n--- Run Complete ---")
    print(
        f"Processed {newly_completed_runs_session} new file/HP combinations in this session.")

    # --- Post-Processing to Find Best HP (Same as before) ---
    if not write_csv:
        print("Error: No results available.")
        exit()

    # Define column names again for final DF creation (in case cache wasn't set)
    if metric_keys_cache:
        col_w = ['file', 'HP'] + metric_keys_cache
    else:
        print("Warning: Metric keys were not cached. Inferring from result length.")
        num_metrics = len(write_csv[0]) - 2
        col_w = ['file', 'HP'] + [f'Metric_{k}' for k in range(num_metrics)]
        if 'VUS-PR' not in col_w[2:] and num_metrics >= 9:
            metric_keys_cache = ['AUC-ROC', 'AUC-PR', 'F1', 'Precision',
                                 'Recall', 'VUS-ROC', 'VUS-PR', 'PA-F1', 'Standard-F1']
            col_w = ['file', 'HP'] + metric_keys_cache
            print("Assumed default metric order.")
        elif 'VUS-PR' not in col_w[2:]:
            print("Error: Cannot identify VUS-PR column. Cannot determine best HP.")
            exit()

    results_df = pd.DataFrame(write_csv, columns=col_w)

    # Calculate average VUS-PR
    results_df['VUS-PR'] = pd.to_numeric(results_df['VUS-PR'], errors='coerce')
    avg_scores = results_df.groupby('HP')['VUS-PR'].mean()

    if avg_scores.empty:
        print("Error: Could not calculate average scores.")
        best_params_str = "N/A"
        best_avg_vus_pr = -float('inf')
    else:
        best_params_str = avg_scores.idxmax()
        best_avg_vus_pr = avg_scores.max()

    # --- Report Best ---
    print("\n--- Best Hyperparameter Results ---")
    print(f"Best Average VUS-PR Score: {best_avg_vus_pr:.6f}")
    print(f"Best Hyperparameters (string representation): {best_params_str}")

    # Convert back to dict
    try:
        best_params_dict = ast.literal_eval(best_params_str)
        print(f"\nBest Hyperparameters (parsed):")
        for key, value in best_params_dict.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nWarning: Could not parse best HP string: {e}")
        best_params_dict = best_params_str

    # --- Save Best Parameters ---
    best_hp_save_path = os.path.join(
        args.save_dir, f'{args.AD_Name}_best_hp.txt')
    try:
        with open(best_hp_save_path, 'w') as f:
            f.write(f"Algorithm: {args.AD_Name}\n")
            f.write(f"Best Average VUS-PR: {best_avg_vus_pr:.6f}\n")
            f.write("Best Hyperparameters:\n")
            if isinstance(best_params_dict, dict):
                for key, value in best_params_dict.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {best_params_str}\n")
        print(f"Best hyperparameters saved to {best_hp_save_path}")
    except Exception as e:
        print(f"Error saving best hyperparameters: {e}")

    print(f"\nTotal Execution Time: {time.time() - Start_T:.2f} seconds")
