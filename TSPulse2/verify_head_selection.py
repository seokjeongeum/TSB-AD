import pandas as pd
import numpy as np
import os
import glob
import re

# --- Configuration ---

# This script assumes it's located in a subdirectory of the main project.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories containing the benchmark results
EVAL_METRICS_DIRS = [
    os.path.join(PROJECT_ROOT, "eval", "metrics", "uni"),
    os.path.join(PROJECT_ROOT, "eval", "metrics", "multi"),
]
TUNING_METRICS_DIRS = [
    os.path.join(PROJECT_ROOT, "eval", "metrics", "uni-tuning"),
    os.path.join(PROJECT_ROOT, "eval", "metrics", "multi-tuning"),
]

# The two CSVs provided, which state the best head chosen on the tuning set.
# Assumes these are in the same directory as the script.
SCRIPT_DIR = os.path.dirname(__file__)
PROVIDED_BEST_HEADS_FILES = [
    os.path.join(SCRIPT_DIR, "TSPulse_Output_Selection_Univariate.csv"),
    os.path.join(SCRIPT_DIR, "TSPulse_Output_Selection_Multivariate.csv"),
]

METRIC_TO_COMPARE = "VUS-PR"


def map_algo_to_head_name(algo_name):
    """Maps the TSPulse algorithm/file name to the simplified head name from the paper."""
    if "ensemble" in algo_name:
        return "Headensemble"
    elif "fft" in algo_name:
        return "Headfft"
    elif "future" in algo_name:
        return "Headfuture"
    elif "time" in algo_name:
        return "Headtime"
    return "Unknown"


def extract_dataset_name_from_file(filename):
    """Helper function to get a dataset group name from a raw filename."""
    match = re.search(r"^\d+_(.*?)_id_", filename)
    return match.group(1) if match else "Unknown"


def load_all_results(metrics_dirs):
    """Loads all TSPulse* result CSVs from a list of directories into a single DataFrame."""
    all_results = []
    print(f"Loading results from: {', '.join([os.path.basename(d) for d in metrics_dirs])}")
    
    for dir_path in metrics_dirs:
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found, skipping: {dir_path}")
            continue

        # --- FIX: Determine 'uni' or 'multi' robustly from the directory name ---
        dir_name = os.path.basename(dir_path)
        if dir_name.startswith("uni"):
            data_type = "uni"
        elif dir_name.startswith("multi"):
            data_type = "multi"
        else:
            continue  # Skip if it's not a uni or multi directory

        tspulse_files = glob.glob(os.path.join(dir_path, "TSPulse_ZS*.csv"))
        for file_path in tspulse_files:
            try:
                algo_name = os.path.basename(file_path).replace(".csv", "")
                df = pd.read_csv(file_path)
                
                if "file" in df.columns and METRIC_TO_COMPARE in df.columns:
                    df = df[["file", METRIC_TO_COMPARE]]
                    # Standardize to the head name (e.g., Headensemble, Headfft)
                    df["Head"] = map_algo_to_head_name(algo_name)
                    df["DataType"] = data_type  # Add the column
                    all_results.append(df)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                # It's okay to skip empty or missing files
                continue
            except Exception as e:
                print(f"Could not process file {file_path}: {e}")

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def get_best_head_per_file(df):
    """Pivots the dataframe and finds the head with the max score for each file."""
    if df.empty or "file" not in df.columns or "Head" not in df.columns:
        return pd.Series(dtype=str)
        
    # Pivot to get files as rows, heads as columns, and metric as values
    pivot = pd.pivot_table(
        df,
        values=METRIC_TO_COMPARE,
        index="file",
        columns="Head",
        aggfunc="first", # No aggregation needed as we are on a per-file basis
    )
    # Find the column name (Head) with the maximum value for each row (file)
    return pivot.idxmax(axis=1)


def get_best_head_per_dataset(df):
    """
    Calculates the mean performance for each head within each dataset group
    and returns the best performing head for each dataset.
    """
    if df.empty:
        return pd.Series(dtype=str)

    # Calculate the mean score for each head within each dataset
    mean_scores = df.groupby(["Dataset", "Head"])[METRIC_TO_COMPARE].mean().reset_index()

    # Find the index of the max score for each dataset
    best_indices = mean_scores.groupby("Dataset")[METRIC_TO_COMPARE].idxmax()

    # Select the best rows and create a Series mapping Dataset -> Best Head
    best_heads_df = mean_scores.loc[best_indices].set_index("Dataset")

    return best_heads_df["Head"]


def task_one_verify_with_provided_best_heads(eval_results_df, provided_best_heads_df):
    """
    Task 1: Uses the provided CSVs to check if the tuning-set choice was best on the eval set.
    """
    print("\n" + "=" * 80)
    print("TASK 1: Verifying Alignment Using Provided 'Best Head' Files from the Paper")
    print("=" * 80)

    if provided_best_heads_df.empty:
        print("Provided best heads data is empty. Skipping Task 1.")
        return

    # Merge the paper's best head choice with our actual evaluation results
    # The 'Dataset' column is now pre-calculated on eval_results_df
    merged_df = pd.merge(
        eval_results_df, provided_best_heads_df, on="Dataset", how="inner"
    )

    if merged_df.empty:
        print("Could not merge any data. Please check dataset names.")
        return

    # Find the actual best performing head from our evaluation results
    actual_best_heads = get_best_head_per_file(merged_df)
    actual_best_heads.name = "ActualBestHead_Eval"

    # Join this back to compare
    final_comparison = (
        merged_df[["file", "Dataset", "Best_TSPulse_Output"]]
        .drop_duplicates()
        .set_index("file")
    )
    final_comparison = final_comparison.join(actual_best_heads)
    final_comparison.dropna(inplace=True)

    if final_comparison.empty:
        print("No comparable files found after processing. Check file names and data.")
        return

    # Perform the alignment check
    final_comparison["Is_Aligned"] = (
        final_comparison["Best_TSPulse_Output"]
        == final_comparison["ActualBestHead_Eval"]
    )

    alignment_count = final_comparison["Is_Aligned"].sum()
    total_files = len(final_comparison)
    alignment_percentage = (
        (alignment_count / total_files) * 100 if total_files > 0 else 0
    )

    print(
        f"Comparison based on {total_files} unique files present in both evaluation results and provided lists."
    )
    print(f"Alignment Count: {alignment_count} / {total_files}")
    print(f"Alignment Percentage: {alignment_percentage:.2f}%")
    print(
        "\nThis means the 'best head' selected on the tuning set (from the paper) was also the best head on the evaluation set for this many files."
    )

    # Display misaligned files for debugging
    misaligned_files = final_comparison[~final_comparison["Is_Aligned"]]
    if not misaligned_files.empty:
        print("\n--- Misaligned Files (Task 1) ---")
        print(
            "The following files had a different best head on the eval set than predicted by the paper's list:"
        )
        print(
            misaligned_files[
                ["Dataset", "Best_TSPulse_Output", "ActualBestHead_Eval"]
            ].to_string()
        )
    else:
        print("\n--- No Misaligned Files (Task 1) ---")


def task_two_reproduce_and_verify_best_heads(tuning_results_df, eval_results_df):
    """
    Task 2: Determines best head by MEAN performance on a dataset from the tuning set,
    then verifies this choice on a PER-FILE basis in the evaluation set.
    """
    print("\n" + "=" * 80)
    print("TASK 2: Verifying Dataset-Level Choice on Per-File Evaluation Data")
    print("=" * 80)

    if tuning_results_df.empty or eval_results_df.empty:
        print("Error: Missing tuning or evaluation data. Skipping Task 2.")
        return
        
    # 1. Determine the best head for each DATASET from the TUNING data (by mean)
    best_head_from_tuning = get_best_head_per_dataset(tuning_results_df)
    best_head_from_tuning.name = "ChosenBestHead_from_Tuning"

    # 2. Determine the actual best head for each FILE from the EVALUATION data
    actual_best_head_from_eval_per_file = get_best_head_per_file(eval_results_df)
    actual_best_head_from_eval_per_file.name = "ActualBestHead_EvalFile"
    
    # 3. Combine the tuning choice with the per-file evaluation results
    # Merge the dataset-level tuning choice back to the per-file eval data
    comparison_df = eval_results_df.join(best_head_from_tuning, on='Dataset')
    
    # Join the actual per-file best heads
    comparison_df = comparison_df.join(actual_best_head_from_eval_per_file, on='file')
    
    # Clean up for comparison
    comparison_df.dropna(subset=['ChosenBestHead_from_Tuning', 'ActualBestHead_EvalFile'], inplace=True)
    comparison_df = comparison_df[['file', 'Dataset', 'ChosenBestHead_from_Tuning', 'ActualBestHead_EvalFile']].drop_duplicates()
    
    if comparison_df.empty:
        print("No common files found between tuning and evaluation sets to compare.")
        return
        
    # 4. Perform the alignment check
    comparison_df['Is_Aligned'] = (comparison_df['ChosenBestHead_from_Tuning'] == comparison_df['ActualBestHead_EvalFile'])
    
    alignment_count = comparison_df['Is_Aligned'].sum()
    total_files = len(comparison_df)
    alignment_percentage = (alignment_count / total_files) * 100 if total_files > 0 else 0

    print(f"Comparison based on {total_files} unique files.")
    print(f"Alignment Count: {alignment_count} / {total_files}")
    print(f"Alignment Percentage: {alignment_percentage:.2f}%")
    print("\nThis means the best head (chosen by mean performance on the tuning set) was also the best head on a per-file basis in the evaluation set for this many files.")
    
    # Display misaligned files for debugging
    misaligned_files = comparison_df[~comparison_df['Is_Aligned']]
    if not misaligned_files.empty:
        print("\n--- Misaligned Files (Task 2) ---")
        print(
            "For these files, the best head chosen from the tuning set (by dataset mean) did not match the actual best head for that file in the eval set:"
        )
        print(
            misaligned_files[
                [
                    "Dataset",
                    "file",
                    "ChosenBestHead_from_Tuning",
                    "ActualBestHead_EvalFile",
                ]
            ].to_string()
        )
    else:
        print("\n--- No Misaligned Files (Task 2) ---")


if __name__ == "__main__":
    # Load all evaluation and tuning results once
    all_eval_results = load_all_results(EVAL_METRICS_DIRS)
    all_tuning_results = load_all_results(TUNING_METRICS_DIRS)

    # Pre-calculate the 'Dataset' column for all dataframes
    if not all_eval_results.empty:
        all_eval_results['Dataset'] = all_eval_results['file'].apply(extract_dataset_name_from_file)
    if not all_tuning_results.empty:
        all_tuning_results['Dataset'] = all_tuning_results['file'].apply(extract_dataset_name_from_file)

    # --- Create a mapping to the provided best heads files ---
    provided_heads_map = {
        "uni": os.path.join(
            os.path.dirname(__file__), "TSPulse_Output_Selection_Univariate.csv"
        ),
        "multi": os.path.join(
            os.path.dirname(__file__), "TSPulse_Output_Selection_Multivariate.csv"
        ),
    }

    for data_type in ["uni", "multi"]:
        print(f"\n\n{'#'*25} {data_type.upper()}VARIATE RESULTS {'#'*25}")
        
        # Filter data for the current type
        eval_df = all_eval_results[all_eval_results['DataType'] == data_type].copy()
        tuning_df = all_tuning_results[all_tuning_results['DataType'] == data_type].copy()
        
        if eval_df.empty and tuning_df.empty:
            print(f"No data found for {data_type}variate analysis.")
            continue
            
        # --- Run Task 1 ---
        # Load the correct provided heads file for the current data_type
        try:
            provided_heads_file = provided_heads_map[data_type]
            provided_heads_df = pd.read_csv(provided_heads_file)
            task_one_verify_with_provided_best_heads(eval_df, provided_heads_df)
        except (FileNotFoundError, KeyError):
            print(
                f"Warning: Provided best heads file for '{data_type}' not found. Skipping Task 1."
            )
        
        # --- Run Task 2 ---
        task_two_reproduce_and_verify_best_heads(tuning_df, eval_df)
