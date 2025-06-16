import pandas as pd
import os
import glob

# --- Configuration ---
# Directory where your TSPulse metric CSVs are stored
METRICS_DIR = "/workspaces/TSB-AD/eval/metrics/uni/"

# Path to the official benchmark results from the TSB-AD repository
BENCHMARK_FILE = (
    "/workspaces/TSB-AD/benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv"
)

# Directory to save the final comparison files
OUTPUT_DIR = "/workspaces/TSB-AD/comparison_results/"

# The performance metric we want to compare
METRIC_TO_COMPARE = "VUS-PR"

# --- Main Logic ---


def save_final_comparison(metrics_dir, benchmark_file, output_dir, metric):
    """
    Merges TSPulse results with the benchmark, creates a consolidated comparison,
    and saves the results to CSV files.
    """
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

    # 2. Find and merge all TSPulse variant result files
    tspulse_files = sorted(glob.glob(os.path.join(metrics_dir, "TSPulse_*.csv")))
    if not tspulse_files:
        print(f"Error: No TSPulse result files found in '{metrics_dir}'")
        return

    merged_df = benchmark_df.copy()
    tspulse_algos = []

    for file_path in tspulse_files:
        algo_name = os.path.basename(file_path).replace(".csv", "")
        tspulse_algos.append(algo_name)
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

    tspulse_scores_df = merged_df[tspulse_algos]
    best_tspulse_scores = tspulse_scores_df.max(axis=1)

    summary_df = pd.DataFrame(
        {
            "Best_Baseline_Algo": best_baseline_algos,
            "Best_Baseline_Score": best_baseline_scores,
            "TSPulse_ZS": best_tspulse_scores,
        }
    )
    summary_df["Diff_vs_Baseline"] = (
        summary_df["TSPulse_ZS"] - summary_df["Best_Baseline_Score"]
    )
    summary_df.fillna(0.0, inplace=True)

    # Sort by the difference to easily see where TSPulse performs best/worst
    sorted_summary = summary_df.sort_values(
        by=["Diff_vs_Baseline", "Best_Baseline_Score"], ascending=[False, True]
    )

    # --- Save Detailed Per-File Comparison to CSV ---
    os.makedirs(output_dir, exist_ok=True)
    detailed_output_path = os.path.join(output_dir, f"detailed_comparison_{metric}.csv")
    sorted_summary.to_csv(detailed_output_path)
    print(f"Detailed per-file comparison saved to: {detailed_output_path}")

    # --- Save Overall Mean Score Summary to CSV ---
    all_algos_for_mean = baseline_algos + ["TSPulse_ZS"]
    mean_scores = pd.concat(
        [
            merged_df[baseline_algos].mean(),
            pd.Series({"TSPulse_ZS": summary_df["TSPulse_ZS"].mean()}),
        ]
    ).sort_values(ascending=False)

    mean_scores_df = mean_scores.reset_index()
    mean_scores_df.columns = ["Algorithm", f"Mean_{metric}"]

    mean_output_path = os.path.join(output_dir, f"mean_score_summary_{metric}.csv")
    mean_scores_df.to_csv(mean_output_path, index=False)
    print(f"Mean score summary saved to: {mean_output_path}")

    # --- Print Final Summary to Console ---
    print("\n--- Average Performance Difference ---")
    avg_diff = summary_df["Diff_vs_Baseline"].mean()
    print(
        f"On average, TSPulse_ZS performs {avg_diff:+.4f} points better than the best baseline per file."
    )


if __name__ == "__main__":
    save_final_comparison(METRICS_DIR, BENCHMARK_FILE, OUTPUT_DIR, METRIC_TO_COMPARE)
