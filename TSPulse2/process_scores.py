import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import sys
import warnings
import argparse
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

def evaluate_single_algorithm(algorithm_name, data_mode):
    """
    Evaluates anomaly scores for a single algorithm.
    It now correctly constructs paths based on the new, organized structure.
    """
    print("\n" + "=" * 60)
    print(f"Starting evaluation for: '{algorithm_name}' on '{data_mode}' data.")
    print("=" * 60)

    # --- 1. Determine paths based on arguments ---
    if data_mode.startswith("uni"):
        dataset_base_dir = os.path.join(PROJECT_ROOT, "Datasets", "TSB-AD-U")
        file_list_suffix = "U-Tuning.csv" if "tuning" in data_mode else "U-Eva.csv"
    elif data_mode.startswith("multi"):
        dataset_base_dir = os.path.join(PROJECT_ROOT, "Datasets", "TSB-AD-M")
        file_list_suffix = "M-Tuning.csv" if "tuning" in data_mode else "M-Eva.csv"
    else:
        print(f"Error: Unknown data mode '{data_mode}'. Exiting.")
        return

    file_list_path = os.path.join(PROJECT_ROOT, "Datasets", "File_List", f"TSB-AD-{file_list_suffix}")
    
    # This is the crucial part: paths are now organized by algorithm
    score_dir = os.path.join(PROJECT_ROOT, "eval", "score", data_mode, algorithm_name)
    save_dir = os.path.join(PROJECT_ROOT, "eval", "metrics", data_mode)
    save_path = os.path.join(save_dir, f"{algorithm_name}.csv")

    if not os.path.isdir(score_dir):
        print(f"Error: Score directory not found at '{score_dir}'. Please ensure benchmark jobs have run and saved scores here.")
        return

    # --- 2. Load file list and check for existing results ---
    try:
        all_files_df = pd.read_csv(file_list_path)
        all_possible_files = all_files_df["file_name"].tolist()
    except FileNotFoundError:
        print(f"Error: Main file list not found at '{file_list_path}'.")
        return
        
    evaluated_files = set()
    if os.path.exists(save_path):
        try:
            # Always strip .csv when checking, just to be safe
            evaluated_files = set(pd.read_csv(save_path)["file"].str.replace(r'\.csv$', '', regex=True))
            print(f"Found {len(evaluated_files)} already evaluated files in {save_path}.")
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            evaluated_files = set()

    files_to_process = [f for f in all_possible_files if os.path.splitext(f)[0] not in evaluated_files]

    if not files_to_process:
        print("All files already evaluated. Nothing to do.")
        return

    print(f"Total files: {len(all_possible_files)}, To process: {len(files_to_process)}")

    # --- 3. Process remaining files ---
    new_results = []
    pbar = tqdm(files_to_process, desc=f"Evaluating {algorithm_name}")
    for csv_filename in pbar:
        basename = os.path.splitext(csv_filename)[0]
        pbar.set_postfix_str(basename)

        # Find a matching score file, trying exact match first, then a pattern.
        score_path = None
        exact_score_path = os.path.join(score_dir, f"{basename}.npy")
        if os.path.exists(exact_score_path):
            score_path = exact_score_path
        else:
            # Use sorted glob to ensure deterministic behavior if multiple files are found
            pattern_paths = sorted(glob.glob(os.path.join(score_dir, f"{basename}-*.npy")))
            if pattern_paths:
                if len(pattern_paths) > 1:
                    warnings.warn(f"Multiple score files found for '{basename}'. Using '{os.path.basename(pattern_paths[0])}'.")
                score_path = pattern_paths[0]

        # If no score file was found, we cannot proceed with this CSV.
        if not score_path:
            continue
        
        try:
            data_path = os.path.join(dataset_base_dir, csv_filename)
            df = pd.read_csv(data_path).dropna()
            labels = df.iloc[:, -1].values.astype(int)
            data = df.iloc[:, 0:-1].values.astype(float)
            anomaly_scores = np.load(score_path)

            data_for_window = data[:, 0] if data.ndim > 1 else data
            slidingWindow = find_length_rank(data_for_window.reshape(-1, 1), rank=1)

            metrics_dict = get_metrics(anomaly_scores, labels, slidingWindow=slidingWindow)
            # --- FIX: ALWAYS save the clean basename ---
            metrics_dict["file"] = f"{basename}.csv"
            new_results.append(metrics_dict)
        except FileNotFoundError:
            # This will catch if the data_path is missing, as we already found the score_path
            continue
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    # --- 4. Combine and save ---
    if not new_results:
        print("No new results generated.")
        return

    new_results_df = pd.DataFrame(new_results)
    
    if os.path.exists(save_path) and not pd.read_csv(save_path).empty:
        final_df = pd.concat([pd.read_csv(save_path), new_results_df], ignore_index=True)
    else:
        final_df = new_results_df

    final_df.drop_duplicates(subset=["file"], keep="last", inplace=True)
    cols = ["file"] + [col for col in final_df.columns if col != "file"]
    final_df = final_df[cols]

    os.makedirs(save_dir, exist_ok=True)
    final_df.to_csv(save_path, index=False)
    print(f"Successfully saved updated results to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection scores.")
    parser.add_argument("algorithm", type=str, help="Algorithm name (e.g., TSPulse_ZS_ensemble).")
    parser.add_argument("data_mode", type=str, choices=["multi", "multi-tuning", "uni", "uni-tuning"], help="Data mode.")
    args = parser.parse_args()

    evaluate_single_algorithm(args.algorithm, args.data_mode)

if __name__ == "__main__":
    main()
