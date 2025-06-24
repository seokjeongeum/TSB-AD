import os
import glob
import pandas as pd


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
    merged_df["dataset"] = merged_df.index.to_series().apply(get_dataset_name_from_file)
    return merged_df


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


def run_analysis_workflow(tuning_path, eval_path, workflow_name):
    """Executes the full tuning and evaluation workflow for a given variant (uni/multi)."""
    print("\n" + "=" * 80)
    print(f"STARTING {workflow_name.upper()} WORKFLOW")
    print("=" * 80)

    # --- Step 1: Determine Best Heads from Tuning Data ---
    print(f"\n[STEP 1] Determining best heads from {workflow_name} tuning data...")
    print(f"(Source: {tuning_path})")

    tuning_data = load_all_metrics(tuning_path)
    if tuning_data.empty:
        print(f"Error: No tuning data found at {tuning_path}. Cannot proceed.")
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

    # --- Step 2: Apply Learned Strategies to Evaluation Set ---
    print(f"\n[STEP 2] Applying strategies to {workflow_name} evaluation set...")
    print(f"(Source: {eval_path})")
    eval_data = load_all_metrics(eval_path)
    if eval_data.empty:
        print(f"Error: No evaluation data found at {eval_path}. Cannot proceed.")
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

    # --- Step 3: Comprehensive Fallback Analysis on Evaluation Set ---
    print(
        f"\n[STEP 3] Analyzing all possible fallback heads on the {workflow_name} evaluation set..."
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

    uni_tuning_path = os.path.join(project_root, "eval", "metrics", "uni-tuning")
    uni_eval_path = os.path.join(project_root, "eval", "metrics", "uni")

    multi_tuning_path = os.path.join(project_root, "eval", "metrics", "multi-tuning")
    multi_eval_path = os.path.join(project_root, "eval", "metrics", "multi")

    run_analysis_workflow(uni_tuning_path, uni_eval_path, "univariate")
    run_analysis_workflow(multi_tuning_path, multi_eval_path, "multivariate")


if __name__ == "__main__":
    main()
