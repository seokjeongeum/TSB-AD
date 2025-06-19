import glob
import os

import pandas as pd

# --- Configuration ---

# Directory to save the final comparison files
OUTPUT_DIR = "/workspaces/TSB-AD/comparison_results/"

# The performance metric we want to compare
METRIC_TO_COMPARE = "VUS-PR"

# Group configurations for different benchmark sets
BENCHMARK_CONFIGS = [
    {
        "name": "uni",
        "metrics_dir": "/workspaces/TSB-AD/eval/metrics/uni/",
        "benchmark_file": "/workspaces/TSB-AD/benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv",
    },
    {
        "name": "multi",
        "metrics_dir": "/workspaces/TSB-AD/eval/metrics/multi/",
        "benchmark_file": "/workspaces/TSB-AD/benchmark_exp/benchmark_eval_results/multi_mergedTable_VUS-PR.csv",
    },
]


# --- Main Logic ---


def save_final_comparison(name, metrics_dir, benchmark_file, output_dir, metric):
    """
    Merges TSPulse results with a specific benchmark, creates a consolidated comparison,
    and saves the results to separate CSV files for that benchmark.
    """
    print(f"\n--- Processing Benchmark Set: {name.upper()} ---")

    # 1. Load and prepare the main benchmark table
    try:
        benchmark_df = pd.read_csv(benchmark_file)
        if "file" not in benchmark_df.columns:
            benchmark_df.rename(columns={"Unnamed: 0": "file"}, inplace=True)
        benchmark_df.set_index("file", inplace=True)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{benchmark_file}'")
        return

    # Store original baseline algorithm columns
    baseline_algos = [
        col
        for col in benchmark_df.columns
        if not col.endswith(("_len", "_ratio", "_anomaly"))
    ]

    # 2. Find and merge all TSPulse variant result files from the specific directory
    tspulse_files = sorted(glob.glob(os.path.join(metrics_dir, "TSPulse*.csv")))

    if not tspulse_files:
        print(f"Warning: No TSPulse result files found in '{metrics_dir}', skipping.")
        return

    merged_df = benchmark_df.copy()
    tspulse_algo_names = []
    tspulse2_algo_name = "TSPulse2"

    for file_path in tspulse_files:
        algo_name = os.path.basename(file_path).replace(".csv", "")
        tspulse_algo_names.append(algo_name)
        try:
            tspulse_df = pd.read_csv(file_path)
            if "file" not in tspulse_df.columns or metric not in tspulse_df.columns:
                continue
            tspulse_scores = tspulse_df[["file", metric]].set_index("file")
            merged_df[algo_name] = tspulse_scores[metric]
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    # --- Create a Consolidated Per-File Comparison DataFrame ---
    baseline_scores_df = merged_df[baseline_algos]
    best_baseline_scores = baseline_scores_df.max(axis=1)
    best_baseline_algos = baseline_scores_df.idxmax(axis=1)

    # Separate TSPulse2 from other TSPulse variants
    zs_algo_names = [name for name in tspulse_algo_names if name != tspulse2_algo_name]
    tspulse_zs_scores_df = merged_df.get(zs_algo_names, pd.DataFrame())
    best_tspulse_zs_scores = tspulse_zs_scores_df.max(axis=1)

    summary_data = {
        "Best_Baseline_Algo": best_baseline_algos,
        "Best_Baseline_Score": best_baseline_scores,
        "TSPulse_ZS_Best": best_tspulse_zs_scores,
    }

    # Add TSPulse2 if it was found
    if tspulse2_algo_name in merged_df.columns:
        summary_data["TSPulse2"] = merged_df[tspulse2_algo_name]
        summary_data["Diff_TSPulse2_vs_Baseline"] = (
            summary_data["TSPulse2"] - summary_data["Best_Baseline_Score"]
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df["Diff_ZS_vs_Baseline"] = (
        summary_df["TSPulse_ZS_Best"] - summary_df["Best_Baseline_Score"]
    )
    summary_df.fillna(0.0, inplace=True)

    # Sort by the difference to easily see where TSPulse performs best/worst
    sort_columns = (
        ["Diff_TSPulse2_vs_Baseline", "Best_Baseline_Score"]
        if "Diff_TSPulse2_vs_Baseline" in summary_df.columns
        else ["Diff_ZS_vs_Baseline", "Best_Baseline_Score"]
    )
    sorted_summary = summary_df.sort_values(by=sort_columns, ascending=[False, True])

    # --- Print Detailed Per-File Comparison to Console ---
    print(f"\n--- Detailed Per-File Comparison ({name.upper()}) ---")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(sorted_summary)

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
        mean_scores_dict["TSPulse_ZS_Best"] = summary_df["TSPulse_ZS_Best"].mean()
        if tspulse2_algo_name in summary_df.columns:
            mean_scores_dict[tspulse2_algo_name] = summary_df[tspulse2_algo_name].mean()

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
            f"On average, TSPulse_ZS_Best performs {avg_diff_zs:+.4f} points different than the best baseline per file."
        )
        if "Diff_TSPulse2_vs_Baseline" in summary_df.columns:
            avg_diff_ts2 = summary_df["Diff_TSPulse2_vs_Baseline"].mean()
            print(
                f"On average, TSPulse2 performs {avg_diff_ts2:+.4f} points different than the best baseline per file."
            )


if __name__ == "__main__":
    for config in BENCHMARK_CONFIGS:
        save_final_comparison(
            name=config["name"],
            metrics_dir=config["metrics_dir"],
            benchmark_file=config["benchmark_file"],
            output_dir=OUTPUT_DIR,
            metric=METRIC_TO_COMPARE,
        )
