import glob
import os
import pandas as pd

# --- FIX 1: Dynamically determine the project's root directory ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Configuration (now using PROJECT_ROOT to build paths) ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "comparison_results")
METRIC_TO_COMPARE = "VUS-PR"
BENCHMARK_CONFIGS = [
    {
        "name": "uni",
        "metrics_dir": os.path.join(PROJECT_ROOT, "eval", "metrics", "uni"),
        "benchmark_file": os.path.join(
            PROJECT_ROOT,
            "benchmark_exp",
            "benchmark_eval_results",
            "uni_mergedTable_VUS-PR.csv",
        ),
    },
    {
        "name": "multi",
        "metrics_dir": os.path.join(PROJECT_ROOT, "eval", "metrics", "multi"),
        "benchmark_file": os.path.join(
            PROJECT_ROOT,
            "benchmark_exp",
            "benchmark_eval_results",
            "multi_mergedTable_VUS-PR.csv",
        ),
    },
]


# --- Main Logic ---


def save_final_comparison(name, metrics_dir, benchmark_file, output_dir, metric):
    """
    Merges TSPulse results with a specific benchmark, creates a consolidated comparison,
    and saves the results to separate CSV files for that benchmark.
    """
    print(f"\n--- Processing Benchmark Set: {name.upper()} ---")

    # 1. Load the main benchmark table
    try:
        benchmark_df = pd.read_csv(benchmark_file)
        if "file" not in benchmark_df.columns:
            benchmark_df.rename(columns={"Unnamed: 0": "file"}, inplace=True)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{benchmark_file}'")
        return

    # --- THE FIX (Part 1): Normalize the 'file' column by removing the .csv extension ---
    # This ensures it will match the index of your TSPulse results.
    benchmark_df["file"] = benchmark_df["file"].str.replace(r"\.csv$", "", regex=True)

    # Set the index AFTER normalization
    benchmark_df.set_index("file", inplace=True)

    # Store original baseline algorithm columns
    baseline_algos = [
        col
        for col in benchmark_df.columns
        if not col.endswith(("_len", "_ratio", "_anomaly"))
    ]

    # --- THE FIX (Part 2): Create the merged_df *after* the benchmark_df is fixed ---
    # Now, merged_df will have the correct index from the start.
    merged_df = benchmark_df.copy()

    # 2. Find and merge all TSPulse variant result files from the specific directory
    tspulse_files = sorted(glob.glob(os.path.join(metrics_dir, "TSPulse*.csv")))

    if not tspulse_files:
        print(f"Warning: No TSPulse result files found in '{metrics_dir}', skipping.")
        return

    tspulse_algo_names = []
    # --- FIX: Define a list of TSPulse2 variants to handle multiple versions ---
    tspulse2_variants = ["TSPulse2", "TSPulse2_0623"]
    tspulse_ensemble_algo_name = "TSPulse_ZS_ensemble"

    for file_path in tspulse_files:
        algo_name = os.path.basename(file_path).replace(".csv", "")
        tspulse_algo_names.append(algo_name)
        try:
            tspulse_df = pd.read_csv(file_path)
            
            if "file" not in tspulse_df.columns or metric not in tspulse_df.columns:
                continue
            
            # --- FIX: Normalize the 'file' column in both sources at runtime ---
            # This makes the script robust to whether the .csv extension is present or not.
            tspulse_df["file"] = tspulse_df["file"].str.replace(r"\.csv$", "", regex=True)

            tspulse_scores = tspulse_df[["file", metric]].set_index("file")

            # This assignment will now work because the indices of merged_df and tspulse_scores match.
            merged_df[algo_name] = tspulse_scores[metric]

        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    # --- Create a Consolidated Per-File Comparison DataFrame ---
    baseline_scores_df = merged_df[baseline_algos]
    best_baseline_scores = baseline_scores_df.max(axis=1)
    best_baseline_algos = baseline_scores_df.idxmax(axis=1)

    # Separate TSPulse2 from other TSPulse variants
    zs_algo_names = [
        name for name in tspulse_algo_names if name not in tspulse2_variants
    ]

    # Safely select only the ZS algorithm columns that actually exist in the merged dataframe
    existing_zs_algos = [name for name in zs_algo_names if name in merged_df.columns]
    tspulse_zs_scores_df = merged_df[existing_zs_algos]

    Best_TSPulse_Scores = tspulse_zs_scores_df.max(axis=1)
    Best_TSPulse_Algos = tspulse_zs_scores_df.idxmax(axis=1)

    summary_data = {
        "Best_Baseline_Algo": best_baseline_algos,
        "Best_Baseline_Score": best_baseline_scores,
        "Best_TSPulse_Algo": Best_TSPulse_Algos,
        "Best_TSPulse_Score": Best_TSPulse_Scores,
    }

    # Add TSPulse_ZS_ensemble to summary_data if it exists, to make it available in the final CSV.
    if tspulse_ensemble_algo_name in merged_df.columns:
        summary_data[tspulse_ensemble_algo_name] = merged_df[tspulse_ensemble_algo_name]

    # --- UPDATED: Loop through all specified TSPulse2 variants ---
    for variant_name in tspulse2_variants:
        if variant_name in merged_df.columns:
            summary_data[variant_name] = merged_df[variant_name]
            summary_data[f"Diff_{variant_name}_vs_Baseline"] = (
                summary_data[variant_name] - summary_data["Best_Baseline_Score"]
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df["Diff_ZS_vs_Baseline"] = (
        summary_df["Best_TSPulse_Score"] - summary_df["Best_Baseline_Score"]
    )

    # --- UPDATED: Loop through TSPulse2 variants for specific comparisons ---
    for variant_name in tspulse2_variants:
        if variant_name in summary_df.columns:
            # Compare against the best of the other TSPulse variants
            if "Best_TSPulse_Score" in summary_df.columns:
                summary_df[f"Diff_{variant_name}_vs_Best_TSPulse"] = (
                    summary_df[variant_name] - summary_df["Best_TSPulse_Score"]
                )

            # Compare against the specific 'ensemble' variant
            if tspulse_ensemble_algo_name in summary_df.columns:
                summary_df[f"Diff_{variant_name}_vs_Ensemble"] = (
                    summary_df[variant_name] - summary_df[tspulse_ensemble_algo_name]
                )

    summary_df.fillna(0.0, inplace=True)

    # Sort by the difference to easily see where TSPulse performs best/worst
    sort_key = None
    for variant_name in tspulse2_variants:
        key = f"Diff_{variant_name}_vs_Baseline"
        if key in summary_df.columns:
            sort_key = key
            break

    if sort_key:
        sort_columns = [sort_key, "Best_Baseline_Score"]
    else:
        sort_columns = ["Diff_ZS_vs_Baseline", "Best_Baseline_Score"]
    sorted_summary = summary_df.sort_values(by=sort_columns, ascending=[False, True])

    # --- Save Detailed Per-File Comparison to CSV ---
    os.makedirs(output_dir, exist_ok=True)
    detailed_output_path = os.path.join(
        output_dir, f"{name}_detailed_comparison_{metric}.csv"
    )
    sorted_summary.to_csv(detailed_output_path)
    print(f"Detailed per-file comparison saved to: {detailed_output_path}")

    # --- Save Overall Mean Score Summary to CSV ---
    mean_scores_series = merged_df[baseline_algos].mean()
    if isinstance(mean_scores_series, pd.Series):
        mean_scores_dict = mean_scores_series.to_dict()
    else:
        mean_scores_dict = {}

    if not summary_df.empty:
        mean_scores_dict["TSPulse_Best"] = summary_df["Best_TSPulse_Score"].mean()
        for variant_name in tspulse2_variants:
            if variant_name in summary_df.columns:
                mean_scores_dict[variant_name] = summary_df[variant_name].mean()
        if tspulse_ensemble_algo_name in summary_df.columns:
            mean_scores_dict[tspulse_ensemble_algo_name] = summary_df[
                tspulse_ensemble_algo_name
            ].mean()

    mean_scores = pd.Series(mean_scores_dict).sort_values(ascending=False)

    mean_scores_df = mean_scores.reset_index()
    mean_scores_df.columns = ["Algorithm", f"Mean_{metric}"]

    mean_output_path = os.path.join(
        output_dir, f"{name}_mean_score_summary_{metric}.csv"
    )
    mean_scores_df.to_csv(mean_output_path, index=False)
    print(f"Mean score summary saved to: {mean_output_path}")

    # --- Print Final Summary to Console ---
    print(f"\n--- Average Performance Difference ({name.upper()}) ---")
    if not summary_df.empty:
        avg_diff_zs = summary_df["Diff_ZS_vs_Baseline"].mean()
        print(
            f"On average, TSPulse_Best performs {avg_diff_zs:+.4f} points different than the best baseline per file."
        )
        for variant_name in tspulse2_variants:
            if f"Diff_{variant_name}_vs_Baseline" in summary_df.columns:
                avg_diff_ts2 = summary_df[f"Diff_{variant_name}_vs_Baseline"].mean()
                print(
                    f"On average, {variant_name} performs {avg_diff_ts2:+.4f} points different than the best baseline per file."
                )

            if f"Diff_{variant_name}_vs_Best_TSPulse" in summary_df.columns:
                avg_diff = summary_df[f"Diff_{variant_name}_vs_Best_TSPulse"].mean()
                print(
                    f"On average, {variant_name} performs {avg_diff:+.4f} points different than TSPulse_Best per file."
                )

            if f"Diff_{variant_name}_vs_Ensemble" in summary_df.columns:
                avg_diff = summary_df[f"Diff_{variant_name}_vs_Ensemble"].mean()
                print(
                    f"On average, {variant_name} performs {avg_diff:+.4f} points different than TSPulse_ZS_ensemble per file."
                )


if __name__ == "__main__":
    # Now, run the main comparison logic
    for config in BENCHMARK_CONFIGS:
        save_final_comparison(
            name=config["name"],
            metrics_dir=config["metrics_dir"],
            benchmark_file=config["benchmark_file"],
            output_dir=OUTPUT_DIR,
            metric=METRIC_TO_COMPARE,
        )
