import pandas as pd
import numpy as np
import os
import glob
import re
import sys

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
    elif "TSPulse2_0623" in algo_name:
        return "TSPulse2_0623"
    elif algo_name == "TSPulse2":  # From TSPulse2.csv
        return "Head_scaled_ensemble"
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

        tspulse_files = glob.glob(os.path.join(dir_path, "TSPulse*.csv"))
        for file_path in tspulse_files:
            try:
                algo_name = os.path.basename(file_path).replace(".csv", "")
                df = pd.read_csv(file_path)
                
                if "file" in df.columns and METRIC_TO_COMPARE in df.columns:
                    # Select columns and create a new DataFrame to avoid SettingWithCopyWarning
                    new_df = df[["file", METRIC_TO_COMPARE]].copy()
                    
                    # --- FIX: Normalize file names to remove .csv extension ---
                    new_df["file"] = new_df["file"].str.replace(".csv", "", regex=False) # type: ignore
                    
                    # Standardize to the head name (e.g., Headensemble, Headfft)
                    new_df["Head"] = map_algo_to_head_name(algo_name)
                    new_df["DataType"] = data_type  # Add the column
                    all_results.append(new_df)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                # It's okay to skip empty or missing files
                continue
            except Exception as e:
                print(f"Could not process file {file_path}: {e}")

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def get_best_head_per_file(df):
    """
    Pivots the dataframe and finds the head with the max score for each file,
    EXCLUDING TSPulse2_0623 and Head_scaled_ensemble from the candidates.
    Also returns the full pivot table for debugging.
    """
    # --- FIX: Filter out any 'Unknown' heads before processing ---
    df_filtered = df[df["Head"] != "Unknown"].copy()

    if (
        df_filtered.empty
        or "file" not in df_filtered.columns
        or "Head" not in df_filtered.columns
    ):
        return pd.Series(dtype=str), pd.DataFrame()

    pivot = pd.pivot_table(
        df_filtered,
        values=METRIC_TO_COMPARE,
        index="file",
        columns="Head",
        aggfunc="first",
    )

    # --- FIX: Exclude TSPulse2_0623 from the candidates for 'best head' ---
    candidate_heads = pivot.drop(
        columns=["TSPulse2_0623", "Head_scaled_ensemble"], errors="ignore"
    )

    if candidate_heads.empty:
        return pd.Series(dtype=str), pivot

    # Find the column name (Head) with the maximum value for each row (file)
    best_heads = candidate_heads.idxmax(axis=1)

    return best_heads, pivot


def get_closest_head_to_0623(pivot):
    """
    Pivots the dataframe and finds the head with the VUS-PR score closest
    to TSPulse2_0623's score for each file.
    Returns the closest head names and the difference values.
    """
    if (
        "TSPulse2_0623" not in pivot.columns
        or pivot.empty
    ):
        return pd.Series(dtype=str), pd.Series(dtype=float)

    ref_scores = pivot["TSPulse2_0623"]
    # --- FIX: Include all other heads (including scaled_ensemble) as candidates ---
    candidate_heads = pivot.drop(
        columns=["TSPulse2_0623"], errors="ignore"
    )

    if candidate_heads.empty:
        return pd.Series(dtype=str), pd.Series(dtype=float)

    diffs_df = candidate_heads.subtract(ref_scores, axis=0).abs()

    # --- FIX: Handle FutureWarning by dropping rows where all diffs are NaN ---
    diffs_df.dropna(how="all", inplace=True)
    if diffs_df.empty:
        return pd.Series(dtype=str), pd.Series(dtype=float)

    closest_heads = diffs_df.idxmin(axis=1)
    min_diffs = diffs_df.min(axis=1)

    return closest_heads, min_diffs


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

    # 1. Find the actual best performing head (max score)
    actual_best_heads, eval_pivot = get_best_head_per_file(merged_df)
    actual_best_heads.name = "ActualBestHead_Eval"

    # 2. Find the head closest to TSPulse2_0623
    closest_heads, diffs = get_closest_head_to_0623(eval_pivot)
    closest_heads.name = "Closest_Head_to_0623"
    diffs.name = "Diff_0623_vs_Closest"

    # 3. Create the final summary table
    # Start with the paper's choices on a per-file basis
    summary_df = (
        merged_df[["file", "Dataset", "Best_TSPulse_Output"]]
        .drop_duplicates()
        .set_index("file")
    )

    # Join the other results
    summary_df = summary_df.join(actual_best_heads)
    summary_df = summary_df.join(closest_heads)
    summary_df = summary_df.join(diffs)

    # Add the score comparison column
    if "TSPulse2_0623" in eval_pivot.columns:
        # Helper to get scores safely
        def get_score_from_pivot(row, pivot_table, head_column_name):
            try:
                return pivot_table.loc[row.name, row[head_column_name]]
            except (KeyError, IndexError):
                return np.nan

        paper_choice_scores = summary_df.apply(
            get_score_from_pivot,
            axis=1,
            pivot_table=eval_pivot,
            head_column_name="Best_TSPulse_Output",
        )
        ts_pulse_0623_scores = eval_pivot.loc[summary_df.index, "TSPulse2_0623"]
        summary_df["Paper_Choice_Score < 0623_Score"] = (
            paper_choice_scores < ts_pulse_0623_scores
        )
    else:
        # If the 0623 column doesn't exist, this comparison is not possible.
        summary_df["Paper_Choice_Score < 0623_Score"] = pd.NA

    # Clean up and reorder columns for display
    summary_df.dropna(subset=["ActualBestHead_Eval"], inplace=True)

    # Ensure correct column order as requested
    final_columns = [
        "Dataset",
        "ActualBestHead_Eval",
        "Best_TSPulse_Output",
        "Closest_Head_to_0623",
        "Diff_0623_vs_Closest",
        "Paper_Choice_Score < 0623_Score",
    ]
    # Filter for only columns that exist to prevent KeyErrors
    final_columns_exist = [c for c in final_columns if c in summary_df.columns]
    summary_df = summary_df[final_columns_exist]

    # --- Perform the alignment checks (as before) ---
    summary_df["Is_Aligned_Paper"] = (
        summary_df["Best_TSPulse_Output"]
        == summary_df["ActualBestHead_Eval"]
    )
    # 2. Closest Head vs. Actual Best (per user request to combine these checks)
    summary_df["Is_Aligned_Closest_vs_Actual"] = (
        summary_df["ActualBestHead_Eval"] == summary_df["Closest_Head_to_0623"]
    )

    total_files = len(summary_df)
    print(
        f"\nComparison based on {total_files} unique files present in both evaluation results and provided lists."
    )

    # Report alignment for the paper's choice
    alignment_count_paper = summary_df["Is_Aligned_Paper"].sum()
    alignment_percentage_paper = (
        (alignment_count_paper / total_files) * 100 if total_files > 0 else 0
    )
    print(f"\n1. Alignment of Paper's Choice vs. Actual Best Head:")
    print(f"   - Alignment Count: {alignment_count_paper} / {total_files}")
    print(f"   - Alignment Percentage: {alignment_percentage_paper:.2f}%")

    # Report alignment for Closest Head vs. Actual
    alignment_count_closest = summary_df["Is_Aligned_Closest_vs_Actual"].sum()
    alignment_percentage_closest = (
        (alignment_count_closest / total_files) * 100 if total_files > 0 else 0
    )
    print(f"\n2. Alignment of Closest Head vs. Actual Best Head:")
    print(f"   - Alignment Count: {alignment_count_closest} / {total_files}")
    print(f"   - Alignment Percentage: {alignment_percentage_closest:.2f}%")

    # Report statistics for Paper's Choice vs. 0623 scores
    if "Paper_Choice_Score < 0623_Score" in summary_df.columns:
        score_comparison_col = summary_df["Paper_Choice_Score < 0623_Score"]
        # .count() gives non-NA count. This is important as some comparisons might be NA
        valid_comparisons = score_comparison_col.count()  # type: ignore
        if valid_comparisons > 0:
            is_lower_count = (
                score_comparison_col.sum()
            )  # For booleans, sum() counts True values
            percentage_lower = (is_lower_count / valid_comparisons) * 100
            print(f"\n3. Statistics for 'Paper_Choice_Score < 0623_Score':")
            print(
                f"   - Paper's choice score was lower than 0623's score in {is_lower_count} / {valid_comparisons} cases ({percentage_lower:.2f}%)."
            )

    print(
        "\nColumns: [Dataset], [Actual Best Head], [Paper's Choice], [Closest to 0623], [VUS-PR Diff], [Paper < 0623?]"
    )
    print(summary_df.to_string())


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

    # 2. Find the actual best performing head (max score) from EVAL data
    actual_best_heads_eval, eval_pivot = get_best_head_per_file(eval_results_df)
    actual_best_heads_eval.name = "ActualBestHead_Eval"

    # 3. Find the head closest to TSPulse2_0623 in EVAL data
    closest_heads_eval, diffs_eval = get_closest_head_to_0623(eval_pivot)
    closest_heads_eval.name = "Closest_Head_to_0623"
    diffs_eval.name = "Diff_0623_vs_Closest"

    # 4. Create the final summary table
    # Start with the actual best heads on a per-file basis
    summary_df = pd.DataFrame(actual_best_heads_eval)

    # Add Dataset column to allow merging
    summary_df["Dataset"] = summary_df.index.to_series().apply(
        extract_dataset_name_from_file
    )

    # Join the other results
    summary_df = summary_df.join(best_head_from_tuning, on="Dataset")
    summary_df = summary_df.join(closest_heads_eval)
    summary_df = summary_df.join(diffs_eval)

    # Clean up and reorder columns
    summary_df.dropna(subset=["ActualBestHead_Eval"], inplace=True)
    final_columns = [
        "Dataset",
        "ActualBestHead_Eval",
        "ChosenBestHead_from_Tuning",
        "Closest_Head_to_0623",
        "Diff_0623_vs_Closest",
    ]
    final_columns_exist = [c for c in final_columns if c in summary_df.columns]
    summary_df = summary_df[final_columns_exist]

    print("\n--- Full Comparison Summary (Task 2) ---")
    
    # 1. Tuning choice vs. Actual Best on Eval
    summary_df["Is_Aligned_Tuning"] = (
        summary_df["ChosenBestHead_from_Tuning"]
        == summary_df["ActualBestHead_Eval"]
    )
    # 2. TSPulse2_0623 vs. Actual Best on Eval
    summary_df["Is_Aligned_0623"] = summary_df["ActualBestHead_Eval"] == summary_df["Closest_Head_to_0623"]

    total_files = len(summary_df)
    print(f"\nComparison based on {total_files} unique files.")

    # Report alignment for the tuning-reproduced choice
    alignment_count_tuning = summary_df["Is_Aligned_Tuning"].sum()
    alignment_percentage_tuning = (
        (alignment_count_tuning / total_files) * 100 if total_files > 0 else 0
    )
    print(f"\n1. Alignment of Reproduced Tuning Choice vs. Actual Best Head:")
    print(f"   - Alignment Count: {alignment_count_tuning} / {total_files}")
    print(f"   - Alignment Percentage: {alignment_percentage_tuning:.2f}%")

    # Report alignment for TSPulse2_0623
    alignment_count_0623 = summary_df["Is_Aligned_0623"].sum()
    alignment_percentage_0623 = (
        (alignment_count_0623 / total_files) * 100 if total_files > 0 else 0
    )
    print(f"\n2. Alignment of TSPulse2_0623 vs. Actual Best Head:")
    print(f"   - Alignment Count: {alignment_count_0623} / {total_files}")
    print(f"   - Alignment Percentage: {alignment_percentage_0623:.2f}%")
    
    print(
        "Columns: [Dataset], [Actual Best Head in Eval], [Reproduced Tuning Choice], [Closest Head to TSPulse2_0623], [VUS-PR Difference]"
    )
    print(summary_df.to_string())


if __name__ == "__main__":
    # --- Output Redirection ---
    script_dir = os.path.dirname(__file__)
    output_filename = os.path.join(script_dir, "verify_head_selection_output.txt")
    original_stdout = sys.stdout

    with open(output_filename, "w") as f:
        sys.stdout = f
        try:
            # Load all evaluation and tuning results once
            all_eval_results = load_all_results(EVAL_METRICS_DIRS)
            all_tuning_results = load_all_results(TUNING_METRICS_DIRS)

            # Pre-calculate the 'Dataset' column for all dataframes
            if not all_eval_results.empty:
                all_eval_results["Dataset"] = all_eval_results["file"].apply(
                    extract_dataset_name_from_file
                )
            if not all_tuning_results.empty:
                all_tuning_results["Dataset"] = all_tuning_results["file"].apply(
                    extract_dataset_name_from_file
                )

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
                eval_df = all_eval_results[
                    all_eval_results["DataType"] == data_type
                ].copy()
                tuning_df = all_tuning_results[
                    all_tuning_results["DataType"] == data_type
                ].copy()

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
                except Exception as e:
                    print(f"An error occurred during Task 1 for {data_type}: {e}")

                # --- Run Task 2 ---
                try:
                    task_two_reproduce_and_verify_best_heads(tuning_df, eval_df)
                except Exception as e:
                    print(f"An error occurred during Task 2 for {data_type}: {e}")
        finally:
            sys.stdout = original_stdout

    print(f"Analysis complete. Results written to: {output_filename}")
