import collections
import glob
import os
import re
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def _sanitize_filename(filename: str) -> str:
    """Removes file extensions and special suffixes (e.g., '-dL-rand') from a filename."""
    base, _ = os.path.splitext(filename)
    return re.sub(r"-.+$", "", base)


def get_head_name_from_filename(filepath: str) -> str:
    """Extracts a clean head name from the CSV filename."""
    basename = os.path.basename(filepath)
    # The full filename without extension is the head name.
    return os.path.splitext(basename.replace(".csv.gz", ".csv"))[0]


def get_dataset_name_from_file(filename: str) -> str:
    """Extracts the dataset name from the formatted file string."""
    try:
        # Format: 001_NAB_id_1_...
        return filename.split("_")[1]
    except IndexError:
        return "Unknown"


def load_file_set(filepath: str) -> Set[str]:
    """Loads a list of files and returns a set of sanitized filenames."""
    if not os.path.exists(filepath):
        print(f"Warning: File list not found at {filepath}")
        return set()
    df = pd.read_csv(filepath)
    # The 'file' column in the metrics df is sanitized (no .csv), so we sanitize here too.
    return {_sanitize_filename(f) for f in df["file_name"]}


def load_all_metrics(metrics_path: str) -> pd.DataFrame:
    """Loads and merges VUS-PR scores from all relevant CSV files in a given directory."""
    all_dfs = []

    if not os.path.isdir(metrics_path):
        print(f"Warning: Directory not found, skipping: {metrics_path}")
        return pd.DataFrame()

    glob_pattern = os.path.join(metrics_path, "TSPulse*.csv*")
    metric_files = glob.glob(glob_pattern)

    if not metric_files:
        print(f"No metric files found in {metrics_path} after exclusion.")
        return pd.DataFrame()

    for f in metric_files:
        try:
            # The type checker struggles with usecols, but this is valid pandas code.
            df = pd.read_csv(f, usecols=["file", "VUS-PR"])  # type: ignore
            head_name = get_head_name_from_filename(f)
            # Skip if it's a head we don't want to include
            if head_name == "TSPulse2":  # This was previously the scaled_ensemble
                continue
            df = df.rename(columns={"VUS-PR": head_name}).set_index("file")
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")

    if not all_dfs:
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, axis=1)

    # Sanitize the index to remove file extensions, matching the tuning file lists.
    merged_df.index = merged_df.index.map(
        lambda x: _sanitize_filename(x) if isinstance(x, str) else x
    )

    # If sanitization created duplicate indices, aggregate them.
    if merged_df.index.has_duplicates:
        merged_df = merged_df.groupby(merged_df.index).mean()
        assert isinstance(merged_df, pd.DataFrame)

    merged_df["dataset"] = merged_df.index.to_series().apply(get_dataset_name_from_file)
    return merged_df


def get_best_head_per_file(df: pd.DataFrame, heads_to_consider: List[str]) -> pd.Series:
    """Determines the head with the maximum VUS-PR score for each file (row)."""
    valid_heads = [h for h in heads_to_consider if h in df.columns]
    if not valid_heads or df.empty:
        return pd.Series(dtype=str)

    # For each row (file), find the column name (head) with the max value.
    best_heads = df[valid_heads].idxmax(axis=1)
    assert isinstance(best_heads, pd.Series)
    return best_heads


def calculate_best_heads_per_dataset(
    df: pd.DataFrame, heads_to_consider: List[str]
) -> Tuple[pd.Series, str]:
    """Determines the best performing head for each dataset and the best overall fallback head."""
    valid_heads = [h for h in heads_to_consider if h in df.columns]
    if not valid_heads or df.empty:
        return pd.Series(dtype=str), ""

    dataset_mean_scores = df.groupby("dataset")[valid_heads].mean()
    assert isinstance(dataset_mean_scores, pd.DataFrame)
    best_heads_map = dataset_mean_scores.idxmax(axis=1)
    # The idxmax can return NaN if all values in a row are NaN.
    best_heads_map = best_heads_map.dropna()

    overall_best_head_series = dataset_mean_scores.mean()
    assert isinstance(overall_best_head_series, pd.Series)

    # Handle case where all mean scores are NaN
    if overall_best_head_series.isna().all():
        return best_heads_map, ""

    overall_best_head = overall_best_head_series.idxmax()

    assert isinstance(overall_best_head, str)
    assert isinstance(best_heads_map, pd.Series)
    return best_heads_map, overall_best_head


def apply_strategy_and_evaluate(
    eval_df: pd.DataFrame, best_heads_map: pd.Series, fallback_head: str
) -> float:
    """Calculates the overall VUS-PR by applying the learned best head strategy."""
    if eval_df.empty or best_heads_map.empty or not fallback_head:
        return 0.0

    eval_df_copy = eval_df.copy()
    assert isinstance(eval_df_copy, pd.DataFrame)
    eval_df_copy["chosen_head"] = (
        eval_df_copy["dataset"].map(best_heads_map).fillna(fallback_head)  # type: ignore
    )

    scores = []
    for idx, row in eval_df_copy.iterrows():
        head_to_use = row["chosen_head"]
        score = row.get(head_to_use)
        # This check ensures we only operate on scalar numeric values.
        if isinstance(score, (int, float)) and pd.notna(score):
            scores.append(score)

    if not scores:
        return 0.0

    result = pd.Series(scores).mean()
    assert isinstance(result, float)
    return result


def load_llm_choices(log_files: list[str]) -> Dict[str, str]:
    """
    Parses log files to extract LLM head choices for each series.

    For multivariate series, it determines the choice by finding the most
    frequently selected head across all channels (mode). For univariate,
    it's simply the single choice made.

    Args:
        log_files: A list of paths to the log files.

    Returns:
        A dictionary mapping series filename to the LLM-selected head.
    """
    llm_choices: Dict[str, str] = {}

    # regex patterns
    head_selection_pattern = re.compile(r"LLM: Selected '(\w+)'")
    success_pattern = re.compile(r"Success at (.*?) using")

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: Log file not found: {log_file}")
            continue

        # This list will temporarily hold head choices found between 'Success' lines.
        temp_choices = []
        for line in lines:
            head_match = head_selection_pattern.search(line)
            if head_match:
                temp_choices.append(head_match.group(1))

            success_match = success_pattern.search(line)
            if success_match:
                raw_filename = success_match.group(1).strip()
                filename = _sanitize_filename(raw_filename)

                if temp_choices:
                    # Find the most common head in the list of choices.
                    # This works for both multi-channel (mode) and single-channel (the only item).
                    most_common_head = collections.Counter(temp_choices).most_common(1)[
                        0
                    ][0]
                    llm_choices[filename] = most_common_head

                # Reset the list for the next file's channels.
                temp_choices = []

    return llm_choices


def run_analysis_workflow(
    metrics_path: str,
    tuning_files: Set[str],
    eval_files: Set[str],
    workflow_name: str,
):
    """
    The main workflow for determining and evaluating the head selection strategy.
    """
    print(f"------ Starting Analysis Workflow: {workflow_name} ------")
    oracle_score = 0.0
    llm_score = 0.0

    # --- Step 1: Load Metrics Data ---
    print("\n[STEP 1] Loading all metrics data...")
    all_data = load_all_metrics(metrics_path)
    print(f"Loaded {len(all_data)} total records.")

    # --- Step 2: Split Data into Tuning and Evaluation Sets ---
    print("\n[STEP 2] Splitting data into tuning and evaluation sets...")
    tuning_data = all_data.loc[list(tuning_files)]
    eval_data = all_data.loc[list(eval_files)]
    print(f"Found {len(tuning_files)} files for tuning.")
    print(f"Found {len(eval_files)} files for evaluation.")

    # --- Step 2.5: Load LLM choices from logs ---
    print("\n[STEP 2.5] Loading LLM head choices from logs...")
    log_files = [
        "eval/score/multi/TSPulse2/000_run_TSPulse2.log",
        "eval/score/uni/TSPulse2/000_run_TSPulse2.log",
    ]
    llm_choices = load_llm_choices(log_files)
    if not llm_choices:
        print("Warning: Could not load any LLM choices. Check log file paths.")
    else:
        print(f"Loaded {len(llm_choices)} LLM choices from logs.")

    # --- Step 3: Determine Best Heads from Tuning Data ---
    print("\n[STEP 3] Determining best heads on tuning data...")

    # Create 'dataset' column for grouping
    tuning_data["dataset"] = tuning_data.index.map(get_dataset_name_from_file)
    eval_data["dataset"] = eval_data.index.map(get_dataset_name_from_file)

    # Define two sets of heads to consider for tuning
    original_heads_to_consider = [
        "TSPulse_ZS_ensemble",
        "TSPulse_ZS_fft",
        "TSPulse_ZS_forecast",
        "TSPulse_ZS_time",
    ]
    ablated_heads_to_consider = [
        col for col in tuning_data.columns if "TSPulse2_llm_selection_ablated" in col
    ]

    # Calculate best heads based on ORIGINAL heads
    best_heads_map_original, fallback_head_original = calculate_best_heads_per_dataset(
        tuning_data, original_heads_to_consider
    )

    # Calculate best heads based on ABLATED heads
    best_heads_map_ablated, fallback_head_ablated = calculate_best_heads_per_dataset(
        tuning_data, ablated_heads_to_consider
    )

    print("\n--- Tuning Results (Original Heads) ---")
    print("Best head per dataset:")
    print(best_heads_map_original.to_string(float_format="{:.16f}"))
    print(f"\nDetermined Fallback: '{fallback_head_original}'")

    if not best_heads_map_ablated.empty:
        print("\n--- Tuning Results (Ablated Heads) ---")
        print("Best head per dataset:")
        print(best_heads_map_ablated.to_string(float_format="{:.16f}"))
        print(f"\nDetermined Fallback: '{fallback_head_ablated}'")

    # --- Step 3.5: Verifying head alignment on {workflow_name} evaluation data (per series)...
    print(
        f"\n[STEP 3.5] Verifying head alignment on {workflow_name} evaluation data (per series)..."
    )

    # Define the specific heads to consider for the oracle/ActualBestHead
    oracle_heads_to_consider = [
        "TSPulse_ZS_ensemble",
        "TSPulse_ZS_fft",
        "TSPulse_ZS_forecast",
        "TSPulse_ZS_time",
    ]
    ablated_oracle_heads = [
        col for col in eval_data.columns if "TSPulse2_llm_selection_ablated" in col
    ]

    # Calculate the "actual" best head on the evaluation data
    actual_best_head = get_best_head_per_file(eval_data, oracle_heads_to_consider)
    actual_best_head_ablated = get_best_head_per_file(eval_data, ablated_oracle_heads)

    if actual_best_head.empty:
        print(
            "Could not determine actual best heads per series on eval set. Skipping alignment."
        )
    else:
        # Create a summary dataframe, indexed by series (file).
        comparison_df = pd.DataFrame(
            {
                "ActualBestHead_Eval": actual_best_head,
                "ActualBestHead_Ablated_Eval": actual_best_head_ablated,
            }
        )
        comparison_df["Dataset"] = comparison_df.index.map(eval_data["dataset"])  # type: ignore

        # Map our reproduced choices to the per-series frame. This will have NaNs
        # for datasets in the eval set that were not in the tuning set.
        comparison_df["ReproducedChoice_Original"] = comparison_df["Dataset"].map(best_heads_map_original)  # type: ignore
        comparison_df["ReproducedChoice_Ablated"] = comparison_df["Dataset"].map(best_heads_map_ablated)  # type: ignore

        # --- ADDED: Incorporate LLM choices ---
        comparison_df["LLMChoice"] = comparison_df.index.map(llm_choices)

        # --- Add VUS-PR scores for comparison ---
        # Get score for the 'oracle' best head
        comparison_df["ActualBestHead_VUSPR"] = comparison_df.apply(
            lambda row: eval_data.loc[row.name, row["ActualBestHead_Eval"]], axis=1
        )
        # Get score for the 'ablated oracle' best head
        comparison_df["ActualBestHead_Ablated_VUSPR"] = comparison_df.apply(
            lambda row: (
                eval_data.loc[row.name, row["ActualBestHead_Ablated_Eval"]]
                if pd.notna(row["ActualBestHead_Ablated_Eval"])
                else np.nan
            ),
            axis=1,
        )
        # Get score for our reproduced choice (original)
        comparison_df["ReproducedChoice_Original_VUSPR"] = comparison_df.apply(
            lambda row: (
                eval_data.loc[row.name, row["ReproducedChoice_Original"]]
                if pd.notna(row["ReproducedChoice_Original"])
                else eval_data.loc[row.name, fallback_head_original]
            ),
            axis=1,
        )

        # Get score for our reproduced choice (ablated)
        if not best_heads_map_ablated.empty:
            comparison_df["ReproducedChoice_Ablated_VUSPR"] = comparison_df.apply(
                lambda row: (
                    eval_data.loc[row.name, row["ReproducedChoice_Ablated"]]
                    if pd.notna(row["ReproducedChoice_Ablated"])
                    else eval_data.loc[row.name, fallback_head_ablated]
                ),
                axis=1,
            )

        # --- ADDED: VUS-PR score for LLM choice ---
        def get_llm_vuspr_score(row: pd.Series) -> float:
            """Safely retrieves the VUS-PR score for the LLM's choice."""
            choice = row["LLMChoice"]
            # Handle cases where choice might be a Series due to duplicate indices
            if isinstance(choice, pd.Series):
                choice = choice.iloc[0] if not choice.empty else np.nan

            # After the above, choice should be a scalar. If it's not a string, it's invalid.
            if not isinstance(choice, str):
                return np.nan

            # Construct the full column names to check for
            ablated_col_name = f"TSPulse2_llm_selection_ablated_{choice}"
            original_col_name = f"TSPulse_ZS_{choice}"
            score = np.nan

            if ablated_col_name in eval_data.columns:
                score = eval_data.loc[row.name, ablated_col_name]
            elif original_col_name in eval_data.columns:
                score = eval_data.loc[row.name, original_col_name]
            elif choice in eval_data.columns:  # Fallback for other direct matches
                score = eval_data.loc[row.name, choice]

            # Final check to ensure we are returning a float.
            return (
                float(score)
                if isinstance(score, (int, float)) and pd.notna(score)
                else np.nan
            )

        comparison_df["LLMChoice_VUSPR"] = comparison_df.apply(
            get_llm_vuspr_score, axis=1
        )

        # Get score for the fallback choice
        comparison_df["Fallback_Original_VUSPR"] = eval_data[fallback_head_original]
        if fallback_head_ablated in eval_data.columns:
            comparison_df["Fallback_Ablated_VUSPR"] = eval_data[fallback_head_ablated]

        print("\n--- Alignment & Score Comparison ---")
        # Ensure display columns exist before trying to access them
        display_cols = [
            "ActualBestHead_Eval",
            "ActualBestHead_Ablated_Eval",
            "ReproducedChoice_Original",
            "ReproducedChoice_Ablated",
            "LLMChoice",
            "ActualBestHead_VUSPR",
            "ActualBestHead_Ablated_VUSPR",
            "ReproducedChoice_Original_VUSPR",
            "ReproducedChoice_Ablated_VUSPR",
            "LLMChoice_VUSPR",
            "Fallback_Original_VUSPR",
            "Fallback_Ablated_VUSPR",
        ]
        display_cols_exist = [
            col for col in display_cols if col in comparison_df.columns
        ]
        # Save the comparison dataframe to a CSV file instead of printing
        output_csv_path = f"TSPulse2/comparison_df_{workflow_name}.csv"
        print(f"Saving alignment and score comparison to '{output_csv_path}'...")
        comparison_df[display_cols_exist].sort_index().to_csv(
            output_csv_path, float_format="%.16f"
        )

        # --- Alignment Analysis ---
        # How often did our choice match the actual best head?
        aligned_mask = (
            comparison_df["ReproducedChoice_Original"]
            == comparison_df["ActualBestHead_Eval"]
        )
        alignment_percent = 100 * aligned_mask.sum() / len(comparison_df)
        print(
            f"\nAlignment (Original Tuning Choice vs. Original Best): {alignment_percent:.2f}% ({aligned_mask.sum()}/{len(comparison_df)} series)"
        )

        # --- ADDED: Alignment of Ablated Tuning Choice vs. Ablated Best ---
        if not best_heads_map_ablated.empty:
            ablated_aligned_mask = (
                comparison_df["ReproducedChoice_Ablated"]
                == comparison_df["ActualBestHead_Ablated_Eval"]
            )
            ablated_alignment_percent = (
                100 * ablated_aligned_mask.sum() / len(comparison_df)
            )
            print(
                f"Alignment (Ablated Tuning Choice vs. Ablated Best):  {ablated_alignment_percent:.2f}% ({ablated_aligned_mask.sum()}/{len(comparison_df)} series)"
            )
        else:
            print(
                "Alignment (Ablated Tuning Choice vs. Ablated Best):  N/A (No ablated heads in tuning data)"
            )

        # --- ADDED: LLM Alignment ---
        llm_choices_in_eval = comparison_df["LLMChoice"].dropna()
        if not llm_choices_in_eval.empty:
            # Original Alignment
            # The LLM choice is just the head name, so we need to add the prefix
            # to match the full column name in the metrics data.
            llm_choice_original_names = "TSPulse_ZS_" + llm_choices_in_eval.astype(str)
            actual_best_original_heads = comparison_df.loc[
                llm_choices_in_eval.index, "ActualBestHead_Eval"
            ]
            llm_aligned_mask = (
                llm_choice_original_names.values == actual_best_original_heads.values
            )
            llm_alignment_percent = (
                100 * llm_aligned_mask.sum() / len(llm_choices_in_eval)
            )
            print(
                f"Alignment (LLM Choice vs. Original Best):    {llm_alignment_percent:.2f}% ({llm_aligned_mask.sum()}/{len(llm_choices_in_eval)} series)"
            )

            # Ablated Alignment
            llm_choice_ablated_names = (
                "TSPulse2_llm_selection_ablated_" + llm_choices_in_eval.astype(str)
            )
            actual_best_ablated_heads = comparison_df.loc[
                llm_choices_in_eval.index, "ActualBestHead_Ablated_Eval"
            ]
            ablated_aligned_mask = (
                llm_choice_ablated_names.values == actual_best_ablated_heads.values
            )
            ablated_alignment_percent = (
                100 * ablated_aligned_mask.sum() / len(llm_choices_in_eval)
            )
            print(
                f"Alignment (LLM Choice vs. Ablated Best):     {ablated_alignment_percent:.2f}% ({ablated_aligned_mask.sum()}/{len(llm_choices_in_eval)} series)"
            )
        else:
            print(
                "Alignment (LLM Choice vs. Actual Best):    N/A (no LLM choices found for eval set)"
            )

        # --- Score Analysis ---
        reproduced_score_original = apply_strategy_and_evaluate(
            eval_data, best_heads_map_original, fallback_head_original
        )
        reproduced_score_ablated = np.nan
        if not best_heads_map_ablated.empty:
            reproduced_score_ablated = apply_strategy_and_evaluate(
                eval_data, best_heads_map_ablated, fallback_head_ablated
            )

        # Calculate oracle scores
        oracle_score = comparison_df["ActualBestHead_VUSPR"].mean()
        oracle_score_ablated = comparison_df["ActualBestHead_Ablated_VUSPR"].mean()

        # --- ADDED: LLM Score ---
        llm_score = comparison_df["LLMChoice_VUSPR"].mean()

    print(f"\n--- Final Scores on {workflow_name} evaluation data ---")
    print(f"Score from original tuning strategy:  {reproduced_score_original}")
    if not np.isnan(reproduced_score_ablated):
        print(f"Score from ablated tuning strategy:   {reproduced_score_ablated}")
    else:
        print("Score from ablated tuning strategy:   N/A")
    print(f"Score from LLM-selected heads:        {llm_score}")
    print(f"Oracle score (best possible original):  {oracle_score}")
    print(f"Oracle score (best possible ablated):   {oracle_score_ablated}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    multi_metrics_path = os.path.join(project_root, "eval", "metrics", "multi")
    multi_tuning_list_path = os.path.join(
        project_root, "Datasets", "File_List", "TSB-AD-M-Tuning.csv"
    )
    multi_eval_list_path = os.path.join(
        project_root, "Datasets", "File_List", "TSB-AD-M-Eva.csv"
    )
    multi_tuning_files = load_file_set(multi_tuning_list_path)
    multi_eval_files = load_file_set(multi_eval_list_path)
    run_analysis_workflow(
        multi_metrics_path,
        multi_tuning_files,
        multi_eval_files,
        "multivariate",
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "determine_best_head_output.txt")
    original_stdout = sys.stdout

    print(f"Starting analysis. Output will be saved to: {output_filename}")

    with open(output_filename, "w") as f:
        # Redirect stdout to the file
        sys.stdout = f
        try:
            main()
        except Exception as e:
            # Still print exceptions to the file
            print(f"An error occurred during analysis: {e}", file=f)
            raise  # Optionally re-raise the exception
        finally:
            # Restore stdout
            sys.stdout = original_stdout

    print("Analysis complete.")
