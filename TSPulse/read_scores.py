import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank


# --- Configuration ---
# Name of the algorithm/method whose scores you want to evaluate
AD_NAME = "TSPulse_ZS_ensemble"

# Base directory paths
DATASET_DIR = "/workspaces/TSB-AD/Datasets/TSB-AD-U/"
SCORE_DIR = f"/workspaces/TSB-AD/eval/score/uni/{AD_NAME}/"
SAVE_DIR = "/workspaces/TSB-AD/eval/metrics/uni/"
FILE_LIST_PATH = "/workspaces/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv"


def evaluate_scores():
    """
    Reads anomaly scores from .npy files, compares them against ground truth labels
    from corresponding .csv files, and saves the evaluation metrics to a CSV.
    """
    # --- 1. Load the list of files to be evaluated ---
    try:
        file_list_df = pd.read_csv(FILE_LIST_PATH)
        files_to_evaluate = file_list_df["file_name"].tolist()
    except FileNotFoundError:
        print(f"Error: Evaluation file list not found at '{FILE_LIST_PATH}'")
        return

    print(
        f"Found {len(files_to_evaluate)} files to evaluate for algorithm: '{AD_NAME}'"
    )

    # This list will store the results for each file
    all_results = []

    # --- 2. Loop through each file, load data/scores, and evaluate ---
    for csv_filename in tqdm(files_to_evaluate, desc=f"Evaluating {AD_NAME}"):
        basename = os.path.splitext(csv_filename)[0]

        data_path = os.path.join(DATASET_DIR, csv_filename)
        score_path = os.path.join(SCORE_DIR, f"{basename}.npy")

        try:
            # --- Load Ground Truth Data and Labels ---
            df = pd.read_csv(data_path).dropna()
            # The last column is the label
            labels = df.iloc[:, -1].values.astype(int)
            # All other columns are data (needed for sliding window calculation)
            data = df.iloc[:, 0:-1].values.astype(float)

            # --- Load Anomaly Scores ---
            anomaly_scores = np.load(score_path)

            # --- Calculate the sliding window size (replicating the original logic) ---
            # This is a specific requirement for the TSB-AD evaluation framework
            slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

            # --- Get Metrics ---
            # This is the core evaluation function from the reference script
            metrics_dict = get_metrics(
                anomaly_scores, labels, slidingWindow=slidingWindow
            )

            # --- Store the result for this file ---
            # Add the filename to the dictionary of metrics
            metrics_dict["file_name"] = basename
            all_results.append(metrics_dict)

        except FileNotFoundError:
            print(
                f"Warning: Skipping '{basename}'. Could not find data file '{data_path}' or score file '{score_path}'."
            )
            continue
        except Exception as e:
            print(f"Error processing file {basename}: {e}")
            # Create a dummy result to indicate failure for this file
            failed_result = {"file_name": basename, "F1": 0, "AUC-ROC": 0, "AUC-PR": 0}
            all_results.append(failed_result)

    # --- 3. Save the final results to a CSV file ---
    if not all_results:
        print("No results were generated. Exiting.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(all_results)

    # Reorder columns to have 'file_name' first
    cols = ["file_name"] + [col for col in results_df.columns if col != "file_name"]
    results_df = results_df[cols]

    # Ensure the save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{AD_NAME}_metrics.csv")

    results_df.to_csv(save_path, index=False)

    print("\n" + "=" * 50)
    print("Evaluation complete.")
    print(f"Results for '{AD_NAME}' saved to: {save_path}")
    print("=" * 50)


# --- Run the main evaluation function ---
if __name__ == "__main__":
    evaluate_scores()
