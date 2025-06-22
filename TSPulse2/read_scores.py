import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import argparse

# Suppress UndefinedMetricWarning from sklearn.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- FIX 1: Dynamically determine the project's root directory ---
# This script is in TSPulse2/, so we go up one level ('..') to find the project root.
# This makes all subsequent paths relative and the script portable.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to the Python path to allow importing from TSB_AD
sys.path.insert(0, PROJECT_ROOT)
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank


def evaluate_single_algorithm(score_dir):
    """
    Evaluates the anomaly scores for a single algorithm, skipping files
    that have already been evaluated if a results file exists.

    Args:
        score_dir (str): The full path to the directory containing the .npy score files.
    """
    path_parts = score_dir.strip(os.path.sep).split(os.path.sep)
    ad_name = path_parts[-1]
    data_type = path_parts[-2]

    print("\n" + "=" * 60)
    print(f"Starting evaluation for algorithm: '{ad_name}' on '{data_type}' data.")
    print("=" * 60)

    # --- FIX: Build all paths by joining them with the PROJECT_ROOT ---
    # This logic is now more robust to correctly identify the data type.
    if data_type.startswith("uni"):
        dataset_dir = os.path.join(PROJECT_ROOT, "Datasets", "TSB-AD-U")
        file_list_name = f"TSB-AD-U-{'Tuning' if 'tuning' in data_type else 'Eva'}.csv"
        file_list_path = os.path.join(
            PROJECT_ROOT, "Datasets", "File_List", file_list_name
        )
        save_dir = os.path.join(PROJECT_ROOT, "eval", "metrics", data_type)
    elif data_type.startswith("multi"):
        dataset_dir = os.path.join(PROJECT_ROOT, "Datasets", "TSB-AD-M")
        file_list_name = f"TSB-AD-M-{'Tuning' if 'tuning' in data_type else 'Eva'}.csv"
        file_list_path = os.path.join(
            PROJECT_ROOT, "Datasets", "File_List", file_list_name
        )
        save_dir = os.path.join(PROJECT_ROOT, "eval", "metrics", data_type)
    else:
        print(f"Warning: Unknown data type '{data_type}' in path. Skipping.")
        return

    # --- 2. Load the list of all possible files and check for existing results ---
    try:
        all_files_df = pd.read_csv(file_list_path)
        all_possible_files = all_files_df["file_name"].tolist()
    except FileNotFoundError:
        print(f"Error: Main file list not found at '{file_list_path}'. Skipping.")
        return

    save_path = os.path.join(save_dir, f"{ad_name}.csv")
    evaluated_files = set()
    existing_results_df = None

    if os.path.exists(save_path):
        print(
            f"Found existing results file: {save_path}. Loading to skip completed files."
        )
        try:
            existing_results_df = pd.read_csv(save_path)
            # Check for the column containing filenames
            if "file" in existing_results_df.columns:
                # Strip .csv extension for comparison, as we compare against basenames
                evaluated_files = set(
                    existing_results_df["file"]
                    .astype(str)
                    .str.replace(r"\.csv$", "", regex=True)
                )
                print(f"Found {len(evaluated_files)} already evaluated files.")
            elif not existing_results_df.empty:
                print(
                    "Warning: Could not find a 'file' column in existing results. Re-evaluating all files."
                )
        except pd.errors.EmptyDataError:
            print("Warning: Existing results file is empty. Will start from scratch.")
            existing_results_df = None
        except Exception as e:
            print(
                f"An error occurred while reading the existing results file: {e}. Starting from scratch."
            )
            existing_results_df = None
            evaluated_files = set()

    # Determine which files still need to be processed
    files_to_process = [
        f for f in all_possible_files if os.path.splitext(f)[0] not in evaluated_files
    ]

    if not files_to_process:
        print(
            "All files have already been evaluated for this algorithm. Nothing to do."
        )
        return

    print(
        f"Total files: {len(all_possible_files)}, Already evaluated: {len(evaluated_files)}, To process: {len(files_to_process)}"
    )

    new_results = []
    # --- 3. Loop through each file, load data/scores, and evaluate ---
    pbar = tqdm(files_to_process, desc=f"Evaluating {ad_name}")
    for csv_filename in pbar:
        basename = os.path.splitext(csv_filename)[0]
        pbar.set_postfix_str(basename)
        data_path = os.path.join(dataset_dir, csv_filename)
        score_path = os.path.join(score_dir, f"{basename}.npy")

        try:
            df = pd.read_csv(data_path).dropna()
            labels = df.iloc[:, -1].values.astype(int)
            data = df.iloc[:, 0:-1].values.astype(float)
            anomaly_scores = np.load(score_path)

            data_for_window = data[:, 0] if data.ndim > 1 else data
            slidingWindow = find_length_rank(data_for_window.reshape(-1, 1), rank=1)

            metrics_dict = get_metrics(
                anomaly_scores, labels, slidingWindow=slidingWindow
            )
            metrics_dict["file"] = basename
            new_results.append(metrics_dict)

        except FileNotFoundError:
            print(
                f"Warning: Skipping '{basename}'. Could not find data '{data_path}' or score '{score_path}'."
            )
            continue
        except Exception as e:
            print(f"Error processing file {basename}: {e}")
            failed_result = {"file": basename, "F1": 0, "AUC-ROC": 0, "AUC-PR": 0}
            new_results.append(failed_result)

    # --- 4. Combine and save results ---
    if not new_results:
        print("No new results were generated.")
        return

    new_results_df = pd.DataFrame(new_results)
    if existing_results_df is not None:
        final_df = pd.concat(
            [existing_results_df, new_results_df], ignore_index=True
        ).drop_duplicates(subset=["file"], keep="last")
    else:
        final_df = new_results_df

    # Reorder columns to have 'file' first
    cols = ["file"] + [col for col in final_df.columns if col != "file"]
    final_df = final_df[cols]

    os.makedirs(save_dir, exist_ok=True)
    final_df.to_csv(save_path, index=False)
    print(f"Successfully saved updated results to {save_path}")


def main():
    """
    Runs evaluation for a specific algorithm and data type provided as arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection scores for a given algorithm and data type."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        help="The name of the algorithm to evaluate (e.g., 'TSPulse_ZS_ensemble').",
    )
    parser.add_argument(
        "data",
        type=str,
        choices=["multi", "multi-tuning", "uni", "uni-tuning"],
        help="The type of data the algorithm was run on.",
    )

    args = parser.parse_args()

    # --- FIX 3: Construct the score directory path using the PROJECT_ROOT ---
    score_dir = os.path.join(PROJECT_ROOT, "eval", "score", args.data, args.algorithm)

    if not os.path.isdir(score_dir):
        print(
            f"Error: Score directory not found at '{score_dir}'. Please check algorithm and data names."
        )
        return

    # Run the evaluation for the specified directory
    evaluate_single_algorithm(score_dir)

    print("\n" + "=" * 60)
    print("Evaluation finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
