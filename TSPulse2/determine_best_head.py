import os
import glob
import pandas as pd
import sys


def get_head_name_from_filename(filepath):
    """Extracts a clean head name from the CSV filename."""
    basename = os.path.basename(filepath)
    if "TSPulse2.csv" in basename:
        return "scaled_ensemble"
    # Extracts 'time' from 'TSPulse_ZS_time.csv' and handles potential .csv.gz
    return basename.replace("TSPulse_ZS_", "").replace(".csv", "").replace(".gz", "")


def get_dataset_name_from_file(filename):
    """Extracts the dataset name from the formatted file string."""
    try:
        # Format: 001_NAB_id_1_...
        return filename.split("_")[1]
    except IndexError:
        return "Unknown"


def load_file_set(filepath: str) -> set:
    """Loads a list of files and returns a set of sanitized filenames."""
    if not os.path.exists(filepath):
        print(f"Warning: File list not found at {filepath}")
        return set()
    df = pd.read_csv(filepath)
    # The 'file' column in the metrics df is sanitized (no .csv), so we sanitize here too.
    return {os.path.splitext(f)[0] for f in df["file_name"]}


def load_all_metrics(metrics_path):
    """Loads and merges VUS-PR scores from all relevant CSV files in a given directory."""
    all_dfs = []

    if not os.path.isdir(metrics_path):
        print(f"Warning: Directory not found, skipping: {metrics_path}")
        return pd.DataFrame()

    glob_pattern = os.path.join(metrics_path, "TSPulse*.csv*")
    all_metric_files = glob.glob(glob_pattern)

    metric_files = [
        f for f in all_metric_files if "TSPulse2_0623.csv" not in os.path.basename(f)
    ]

    if not metric_files:
        print(f"No metric files found in {metrics_path} after exclusion.")
        return pd.DataFrame()

    for f in metric_files:
        try:
            df = pd.read_csv(f, usecols=["file", "VUS-PR"])
            head_name = get_head_name_from_filename(f)
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


def load_paper_best_heads(filepath: str, head_name_map: dict) -> pd.Series:
    """
    Loads the best head choices from the paper's supplementary CSV file,
    maps the head names to match our internal representation, and returns a
    Series mapping Dataset -> Best Head.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Paper's best head file not found at {filepath}")
        return pd.Series(dtype=str)
    df = pd.read_csv(filepath)
    df["Best_TSPulse_Output"] = df["Best_TSPulse_Output"].map(head_name_map)
    return df.set_index("Dataset")["Best_TSPulse_Output"]


def get_best_head_per_file(df, heads_to_consider):
    """Determines the head with the maximum VUS-PR score for each file (row)."""
    valid_heads = [h for h in heads_to_consider if h in df.columns]
    if not valid_heads or df.empty:
        return pd.Series(dtype=str)

    # For each row (file), find the column name (head) with the max value.
    best_heads = df[valid_heads].idxmax(axis=1)
    return best_heads


def calculate_best_heads_per_dataset(df, heads_to_consider):
    """Determines the best performing head for each dataset and the best overall fallback head."""
    valid_heads = [h for h in heads_to_consider if h in df.columns]
    if not valid_heads:
        return pd.Series(dtype=str), None

    dataset_mean_scores = df.groupby("dataset")[valid_heads].mean()
    best_heads_map = dataset_mean_scores.idxmax(axis=1)
    overall_best_head = dataset_mean_scores.mean().idxmax()

    return best_heads_map, overall_best_head


def apply_strategy_and_evaluate(eval_df, best_heads_map, fallback_head):
    """Calculates the overall VUS-PR by applying the learned best head strategy."""
    if eval_df.empty or best_heads_map.empty or fallback_head is None:
        return 0.0

    eval_df_copy = eval_df.copy()
    eval_df_copy["chosen_head"] = (
        eval_df_copy["dataset"].map(best_heads_map).fillna(fallback_head)
    )

    scores = []
    for idx, row in eval_df_copy.iterrows():
        head_to_use = row["chosen_head"]
        score = row.get(head_to_use)
        if pd.notna(score):
            scores.append(score)

    if not scores:
        return 0.0

    return pd.Series(scores).mean()


def run_analysis_workflow(
    metrics_path, tuning_files, eval_files, paper_best_heads, workflow_name
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
    eval_data = all_data[all_data.index.isin(eval_files)]
    print(
        f"Found {len(tuning_data)} tuning records and {len(eval_data)} evaluation records."
    )

    # --- Step 3: Determine Best Heads from Tuning Data ---
    print(f"\n[STEP 3] Determining best heads from {workflow_name} tuning data...")
    if tuning_data.empty:
        print("Error: No tuning data identified after split. Cannot determine strategy.")
        return

    base_heads = [
        col for col in tuning_data.columns if col not in ["dataset", "scaled_ensemble"]
    ]
    all_heads = [col for col in tuning_data.columns if col != "dataset"]

    best_heads_map_base, fallback_head_base = calculate_best_heads_per_dataset(
        tuning_data, base_heads
    )
    best_heads_map_all, fallback_head_all = calculate_best_heads_per_dataset(
        tuning_data, all_heads
    )

    print("\n--- Tuning Results ---")
    print("Best head per dataset (ZS only):")
    print(best_heads_map_base.to_string())
    print(f"\nDetermined Fallback (ZS only): '{fallback_head_base}'")

    print("\nBest head per dataset (including scaled_ensemble):")
    print(best_heads_map_all.to_string())
    print(f"\nDetermined Fallback (with ensemble): '{fallback_head_all}'")

    # --- Step 3.5: Alignment Analysis ---
    print(
        f"\n[STEP 3.5] Verifying head alignment on {workflow_name} evaluation data (per series)..."
    )

    # Calculate two versions of the "actual" best head on the evaluation data
    actual_best_ts_only = get_best_head_per_file(eval_data, base_heads)
    actual_best_with_ensemble = get_best_head_per_file(eval_data, all_heads)

    if actual_best_ts_only.empty or actual_best_with_ensemble.empty:
        print(
            "Could not determine actual best heads per series on eval set. Skipping alignment."
        )
    else:
        # Create a summary dataframe, indexed by series (file).
        comparison_df = pd.DataFrame(
            {
                "ActualBestHead_Eval_TSPulse_Only": actual_best_ts_only,
                "ActualBestHead_Eval_With_Scaled_Ensemble": actual_best_with_ensemble,
            }
        )
        comparison_df["Dataset"] = comparison_df.index.map(eval_data["dataset"])

        # Map all choices to the per-series frame
        comparison_df["PaperChoice"] = comparison_df["Dataset"].map(paper_best_heads)
        comparison_df["ReproducedChoice_TSPulse_Only"] = comparison_df[
            "Dataset"
        ].map(best_heads_map_base)
        comparison_df["ReproducedChoice_With_Scaled_Ensemble"] = comparison_df[
            "Dataset"
        ].map(best_heads_map_all)

        # Drop series where a mapping couldn't be made for fair comparison
        comparison_df.dropna(
            subset=[
                "PaperChoice",
                "ReproducedChoice_TSPulse_Only",
                "ReproducedChoice_With_Scaled_Ensemble",
            ],
            inplace=True,
        )

        # --- Calculate Alignments ---
        align_paper = (
            comparison_df["ActualBestHead_Eval_TSPulse_Only"]
            == comparison_df["PaperChoice"]
        )
        align_reproduced_ts_only = (
            comparison_df["ActualBestHead_Eval_TSPulse_Only"]
            == comparison_df["ReproducedChoice_TSPulse_Only"]
        )
        align_reproduced_with_ensemble = (
            comparison_df["ActualBestHead_Eval_With_Scaled_Ensemble"]
            == comparison_df["ReproducedChoice_With_Scaled_Ensemble"]
        )

        total_series = len(comparison_df)

        print("\n--- Alignment Summary (Per-Series) ---")
        if total_series > 0:
            print(
                f"Alignment of Paper's Choice vs. ActualBestHead_Eval_TSPulse_Only: {align_paper.sum()} / {total_series} ({align_paper.mean():.2%})"
            )
            print(
                f"Alignment of Reproduced TSPulse-Only Choice vs. ActualBestHead_Eval_TSPulse_Only: {align_reproduced_ts_only.sum()} / {total_series} ({align_reproduced_ts_only.mean():.2%})"
            )
            print(
                f"Alignment of Reproduced With-Scaled-Ensemble Choice vs. ActualBestHead_Eval_With_Scaled_Ensemble: {align_reproduced_with_ensemble.sum()} / {total_series} ({align_reproduced_with_ensemble.mean():.2%})"
            )
        else:
            print("No series available for comparison after mapping choices.")

        print("\n--- Detailed Per-Series Comparison ---")
        display_cols = [
            "Dataset",
            "ActualBestHead_Eval_TSPulse_Only",
            "ActualBestHead_Eval_With_Scaled_Ensemble",
            "PaperChoice",
            "ReproducedChoice_TSPulse_Only",
            "ReproducedChoice_With_Scaled_Ensemble",
        ]
        display_cols_exist = [c for c in display_cols if c in comparison_df.columns]
        print(comparison_df[display_cols_exist].to_string())

    # --- Step 4: Apply Learned Strategies to Evaluation Set ---
    print(f"\n[STEP 4] Applying strategies to {workflow_name} evaluation set...")
    if eval_data.empty:
        print("Warning: No evaluation data to apply strategy to. Skipping evaluation.")
        return

    print("\n--- Final Evaluation Scores (using empirically best fallback) ---")
    vus_pr_base = apply_strategy_and_evaluate(
        eval_data.copy(), best_heads_map_base, fallback_head_base
    )
    print(f"VUS-PR using ZS-only strategy:      {vus_pr_base:.6f}")

    vus_pr_all = apply_strategy_and_evaluate(
        eval_data.copy(), best_heads_map_all, fallback_head_all
    )
    print(f"VUS-PR using with-ensemble strategy: {vus_pr_all:.6f}")

    # --- Step 5: Comprehensive Fallback Analysis on Evaluation Set ---
    print(
        f"\n[STEP 5] Analyzing all possible fallback heads on the {workflow_name} evaluation set..."
    )

    # Fallback analysis for Strategy 1
    fallback_results_base = []
    available_base_heads = [h for h in base_heads if h in eval_data.columns]
    for fallback in available_base_heads:
        vus_pr = apply_strategy_and_evaluate(
            eval_data.copy(), best_heads_map_base, fallback
        )
        fallback_results_base.append(
            {"Fallback Head": fallback, "Resulting VUS-PR": vus_pr}
        )

    if fallback_results_base:
        print("\n--- Fallback Analysis for ZS-only Strategy ---")
        results_df_base = pd.DataFrame(fallback_results_base).set_index("Fallback Head")
        print(
            results_df_base.sort_values(
                by="Resulting VUS-PR", ascending=False
            ).to_string(float_format="%.6f")
        )

    # Fallback analysis for Strategy 2
    fallback_results_all = []
    available_all_heads = [h for h in all_heads if h in eval_data.columns]
    for fallback in available_all_heads:
        vus_pr = apply_strategy_and_evaluate(
            eval_data.copy(), best_heads_map_all, fallback
        )
        fallback_results_all.append(
            {"Fallback Head": fallback, "Resulting VUS-PR": vus_pr}
        )

    if fallback_results_all:
        print("\n--- Fallback Analysis for with-ensemble Strategy ---")
        results_df_all = pd.DataFrame(fallback_results_all).set_index("Fallback Head")
        print(
            results_df_all.sort_values(
                by="Resulting VUS-PR", ascending=False
            ).to_string(float_format="%.6f")
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # --- Head Name Mapping ---
    # Maps head names from the paper's CSVs to the names used in this script
    head_name_map = {
        "Headtime": "time",
        "Headfft": "fft",
        "Headfuture": "future",
        "Headensemble": "ensemble",
        "Head_scaled_ensemble": "scaled_ensemble",
    }

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

    # Define paths for the paper's provided BEST HEADS file lists
    uni_paper_heads_path = os.path.join(
        script_dir, "TSPulse_Output_Selection_Univariate.csv"
    )
    multi_paper_heads_path = os.path.join(
        script_dir, "TSPulse_Output_Selection_Multivariate.csv"
    )

    # Load the sets of filenames for both sets
    uni_tuning_files = load_file_set(uni_tuning_list_path)
    multi_tuning_files = load_file_set(multi_tuning_list_path)
    uni_eval_files = load_file_set(uni_eval_list_path)
    multi_eval_files = load_file_set(multi_eval_list_path)
    uni_paper_heads = load_paper_best_heads(uni_paper_heads_path, head_name_map)
    multi_paper_heads = load_paper_best_heads(multi_paper_heads_path, head_name_map)

    run_analysis_workflow(
        uni_metrics_path,
        uni_tuning_files,
        uni_eval_files,
        uni_paper_heads,
        "univariate",
    )
    run_analysis_workflow(
        multi_metrics_path,
        multi_tuning_files,
        multi_eval_files,
        multi_paper_heads,
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
