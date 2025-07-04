import os
import glob
import pandas as pd
import sys
from typing import Dict, List, Set, Tuple


def get_head_name_from_filename(filepath: str) -> str:
    """Extracts a clean head name from the CSV filename."""
    basename = os.path.basename(filepath)
    # Extracts 'time' from 'TSPulse_ZS_time.csv' and handles potential .csv.gz
    return basename.replace("TSPulse_ZS_", "").replace(".csv", "").replace(".gz", "")


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
    return {os.path.splitext(f)[0] for f in df["file_name"]}


def load_all_metrics(metrics_path: str) -> pd.DataFrame:
    """Loads and merges VUS-PR scores from all relevant CSV files in a given directory."""
    all_dfs = []

    if not os.path.isdir(metrics_path):
        print(f"Warning: Directory not found, skipping: {metrics_path}")
        return pd.DataFrame()

    glob_pattern = os.path.join(metrics_path, "TSPulse*.csv*")
    all_metric_files = glob.glob(glob_pattern)

    # Filter out any files we want to exclude
    metric_files = [f for f in all_metric_files if "TSPulse2_0623.csv" not in os.path.basename(f)]

    if not metric_files:
        print(f"No metric files found in {metrics_path} after exclusion.")
        return pd.DataFrame()

    for f in metric_files:
        try:
            # The type checker struggles with usecols, but this is valid pandas code.
            df = pd.read_csv(f, usecols=["file", "VUS-PR"]) # type: ignore
            head_name = get_head_name_from_filename(f)
            # Skip if it's a head we don't want to include
            if head_name == "TSPulse2": # This was previously the scaled_ensemble
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
        lambda x: os.path.splitext(x)[0] if isinstance(x, str) else x
    )

    merged_df["dataset"] = merged_df.index.to_series().apply(
        get_dataset_name_from_file
    )
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
    if not valid_heads:
        return pd.Series(dtype=str), ""

    dataset_mean_scores = df.groupby("dataset")[valid_heads].mean()
    assert isinstance(dataset_mean_scores, pd.DataFrame)
    best_heads_map = dataset_mean_scores.idxmax(axis=1)
    overall_best_head_series = dataset_mean_scores.mean()
    assert isinstance(overall_best_head_series, pd.Series)
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
        eval_df_copy["dataset"].map(best_heads_map).fillna(fallback_head) # type: ignore
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


def run_analysis_workflow(
    metrics_path: str,
    tuning_files: Set[str],
    eval_files: Set[str],
    workflow_name: str,
):
    """Executes the full tuning and evaluation workflow for a given variant (uni/multi)."""
    print("\n" + "=" * 80)
    print(f"STARTING {workflow_name.upper()} WORKFLOW")
    print("=" * 80)

    # --- Step 1: Load all metric data from the source directory ---
    print(f"\n[STEP 1] Loading all metric data from {workflow_name} directory...")
    print(f"(Source: {metrics_path})")
    all_data = load_all_metrics(metrics_path)
    if all_data.empty:
        print(f"Error: No data found at {metrics_path}. Cannot proceed.")
        return

    # --- Step 2: Split into Tuning and Evaluation sets based on the file lists ---
    print("\n[STEP 2] Splitting data into Tuning and Evaluation sets...")
    tuning_data = all_data[all_data.index.isin(tuning_files)]
    assert isinstance(tuning_data, pd.DataFrame)
    eval_data = all_data[all_data.index.isin(eval_files)]
    assert isinstance(eval_data, pd.DataFrame)
    print(
        f"Found {len(tuning_data)} tuning records and {len(eval_data)} evaluation records."
    )

    # --- Step 3: Determine Best Heads from Tuning Data ---
    print(f"\n[STEP 3] Determining best heads from {workflow_name} tuning data...")
    if tuning_data.empty:
        print("Error: No tuning data identified after split. Cannot determine strategy.")
        return

    # Now there is only one set of heads to consider.
    all_heads = [col for col in tuning_data.columns if col != "dataset"]

    best_heads_map, fallback_head = calculate_best_heads_per_dataset(
        tuning_data, all_heads
    )

    print("\n--- Tuning Results ---")
    print("Best head per dataset:")
    print(best_heads_map.to_string())
    print(f"\nDetermined Fallback: '{fallback_head}'")

    # --- Step 3.5: Alignment Analysis ---
    print(
        f"\n[STEP 3.5] Verifying head alignment on {workflow_name} evaluation data (per series)..."
    )

    # Calculate the "actual" best head on the evaluation data
    actual_best_head = get_best_head_per_file(eval_data, all_heads)
    oracle_score = 0.0

    if actual_best_head.empty:
        print(
            "Could not determine actual best heads per series on eval set. Skipping alignment."
        )
    else:
        # Create a summary dataframe, indexed by series (file).
        comparison_df = pd.DataFrame(
            {"ActualBestHead_Eval": actual_best_head}
        )
        comparison_df["Dataset"] = comparison_df.index.map(eval_data["dataset"]) # type: ignore

        # Map our reproduced choices to the per-series frame. This will have NaNs
        # for datasets in the eval set that were not in the tuning set.
        comparison_df["ReproducedChoice"] = comparison_df["Dataset"].map(best_heads_map) # type: ignore

        # --- Add VUS-PR scores for comparison ---
        # Get score for the 'oracle' best head
        comparison_df['ActualBestHead_VUSPR'] = comparison_df.apply(
            lambda row: eval_data.loc[row.name, row['ActualBestHead_Eval']], axis=1
        )
        # Get score for our chosen head, handling cases where the choice is NaN
        comparison_df['ReproducedChoice_VUSPR'] = comparison_df.apply(
            lambda row: eval_data.loc[row.name, row['ReproducedChoice']] if pd.notna(row['ReproducedChoice']) else pd.NA,
            axis=1
        )
        # Calculate the performance difference
        comparison_df['VUSPR_Diff'] = comparison_df['ReproducedChoice_VUSPR'] - comparison_df['ActualBestHead_VUSPR']

        # For a fair alignment calculation, we can only consider series where a choice was made
        valid_comparison_df = comparison_df
        oracle_score = valid_comparison_df['ActualBestHead_VUSPR'].mean()

        # --- Calculate Alignment ---
        align_reproduced = (
            valid_comparison_df["ActualBestHead_Eval"]
            == valid_comparison_df["ReproducedChoice"]
        )

        total_series = len(valid_comparison_df)
        avg_vuspr_diff = valid_comparison_df['VUSPR_Diff'].mean()

        print("\n--- Alignment Summary (Per-Series where a choice could be mapped) ---")
        if total_series > 0:
            print(
                f"Alignment of Reproduced Choice vs. ActualBestHead_Eval: {align_reproduced.sum()} / {total_series} ({align_reproduced.mean():.2%})"
            )
            print(f"Average VUS-PR Difference (Reproduced - Actual): {avg_vuspr_diff:.4f}")
        else:
            print("No series available for comparison after mapping choices.")

        print("\n--- Detailed Per-Series Comparison ---")
        display_cols = [
            "Dataset",
            "ActualBestHead_Eval",
            "ReproducedChoice",
            "ActualBestHead_VUSPR",
            "ReproducedChoice_VUSPR",
            "VUSPR_Diff",
        ]
        display_cols_exist = [c for c in display_cols if c in comparison_df.columns]
        print(comparison_df[display_cols_exist].to_string(float_format="{:.4f}".format)) # type: ignore
        # Sort by the difference to highlight the biggest wins/losses
        print("\n--- Sorted by VUS-PR Difference ---")
        print(comparison_df[display_cols_exist].sort_values(by="VUSPR_Diff").to_string(float_format="{:.4f}".format)) # type: ignore

    # --- Step 4: Apply Learned Strategies to Evaluation Set ---
    print(f"\n[STEP 4] Applying strategies to {workflow_name} evaluation set...")
    if eval_data.empty:
        print("Warning: No evaluation data to apply strategy to. Skipping evaluation.")
        return

    print("\n--- Final Evaluation Scores (using empirically best fallback) ---")
    vus_pr_score = apply_strategy_and_evaluate(
        eval_data.copy(), best_heads_map, fallback_head
    )
    print(f"VUS-PR using the derived strategy: {vus_pr_score:.6f}")
    if oracle_score > 0:
        print(f"VUS-PR using 'oracle' best head for every series: {oracle_score:.6f}")

    # --- Step 5: Comprehensive Fallback Analysis on Evaluation Set ---
    print(
        f"\n[STEP 5] Analyzing all possible fallback heads on the {workflow_name} evaluation set..."
    )

    fallback_results = []
    available_heads = [h for h in all_heads if h in eval_data.columns]
    for fallback in available_heads:
        vus_pr = apply_strategy_and_evaluate(
            eval_data.copy(), best_heads_map, fallback
        )
        fallback_results.append(
            {"Fallback Head": fallback, "Resulting VUS-PR": vus_pr}
        )

    if fallback_results:
        print("\n--- Fallback Analysis ---")
        results_df = pd.DataFrame(fallback_results).set_index("Fallback Head")
        print(
            results_df.sort_values(
                by="Resulting VUS-PR", ascending=False
            ).to_string(float_format="%.6f")
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Define paths to metric directories
    uni_metrics_path = os.path.join(project_root, "eval", "metrics", "uni")
    multi_metrics_path = os.path.join(project_root, "eval", "metrics", "multi")

    # Define paths for the TUNING file lists
    uni_tuning_list_path = os.path.join(
        project_root, "Datasets", "File_List", "TSB-AD-U-Tuning.csv"
    )
    multi_tuning_list_path = os.path.join(
        project_root, "Datasets", "File_List", "TSB-AD-M-Tuning.csv"
    )

    # Define paths for the EVALUATION file lists
    uni_eval_list_path = os.path.join(
        project_root, "Datasets", "File_List", "TSB-AD-U-Eva.csv"
    )
    multi_eval_list_path = os.path.join(
        project_root, "Datasets", "File_List", "TSB-AD-M-Eva.csv"
    )

    # Load the sets of filenames for both sets
    uni_tuning_files = load_file_set(uni_tuning_list_path)
    multi_tuning_files = load_file_set(multi_tuning_list_path)
    uni_eval_files = load_file_set(uni_eval_list_path)
    multi_eval_files = load_file_set(multi_eval_list_path)

    run_analysis_workflow(
        uni_metrics_path,
        uni_tuning_files,
        uni_eval_files,
        "univariate",
    )
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
            raise # Optionally re-raise the exception
        finally:
            # Restore stdout
            sys.stdout = original_stdout

    print("Analysis complete.")
