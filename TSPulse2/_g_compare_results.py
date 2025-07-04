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
    Merges all algorithm results from a directory with a benchmark, creates a consolidated comparison,
    and saves the results.
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

    # --- THE FIX (Part 2): Create the merged_df *after* the benchmark_df is fixed ---
    # Now, merged_df will have the correct index from the start.
    merged_df = benchmark_df.copy()

    # 2. Find and merge all TSPulse variant result files from the specific directory
    result_files = sorted(glob.glob(os.path.join(metrics_dir, "*.csv")))

    if not result_files:
        print(f"Warning: No result files found in '{metrics_dir}', skipping.")
        return

    for file_path in result_files:
        algo_name = os.path.basename(file_path).replace(".csv", "")
        if algo_name in merged_df.columns:
            print(f"Skipping {algo_name} as it is already present.")
            continue
        try:
            tspulse_df = pd.read_csv(file_path)

            if "file" not in tspulse_df.columns or metric not in tspulse_df.columns:
                continue

            # --- FIX: Normalize the 'file' column in both sources at runtime ---
            # This makes the script robust to whether the .csv extension is present or not.
            tspulse_df["file"] = tspulse_df["file"].str.replace(
                r"\.csv$", "", regex=True
            )

            tspulse_scores = tspulse_df[["file", metric]].set_index("file")

            # This assignment will now work because the indices of merged_df and tspulse_scores match.
            merged_df[algo_name] = tspulse_scores[metric]

        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    # --- Create a Consolidated Per-File Comparison DataFrame ---
    algo_columns = [
        col
        for col in merged_df.columns
        if not col.endswith(("_len", "_ratio", "_anomaly"))
    ]
    scores_df = merged_df[algo_columns].copy()
    scores_df.fillna(0.0, inplace=True)

    best_scores = scores_df.max(axis=1)
    best_algos = scores_df.idxmax(axis=1)

    summary_df = merged_df.copy()
    summary_df["Best_Algo"] = best_algos
    summary_df["Best_Score"] = best_scores
    summary_df.fillna(0.0, inplace=True)

    # Reorder columns for clarity
    all_algo_names = sorted(algo_columns)
    detailed_cols = (
        ["Best_Algo", "Best_Score"]
        + all_algo_names
        + [
            col
            for col in summary_df.columns
            if col not in all_algo_names and col not in ["Best_Algo", "Best_Score"]
        ]
    )
    # Get unique columns in order
    detailed_cols = list(dict.fromkeys(detailed_cols))

    sorted_summary = summary_df[detailed_cols].sort_values(
        by="Best_Score", ascending=False
    )

    # --- Save Detailed Per-File Comparison to CSV ---
    os.makedirs(output_dir, exist_ok=True)
    detailed_output_path = os.path.join(
        output_dir, f"{name}_detailed_comparison_{metric}.csv"
    )
    sorted_summary.to_csv(detailed_output_path)
    print(f"Detailed per-file comparison saved to: {detailed_output_path}")

    # --- Save Overall Mean Score Summary to CSV ---
    mean_scores_data = scores_df.mean()
    if not isinstance(mean_scores_data, pd.Series):
        mean_scores_data = pd.Series(mean_scores_data)

    mean_scores = mean_scores_data.sort_values(ascending=False)

    mean_scores_df = mean_scores.reset_index()
    mean_scores_df.columns = ["Algorithm", f"Mean_{metric}"]

    mean_output_path = os.path.join(
        output_dir, f"{name}_mean_score_summary_{metric}.csv"
    )
    mean_scores_df.to_csv(mean_output_path, index=False)
    print(f"Mean score summary saved to: {mean_output_path}")

    # --- Print Final Summary to Console ---
    print(f"\n--- Mean Scores ({name.upper()}) ---")
    print(mean_scores_df.to_string(index=False))


def merge_csv_files(glob_pattern, output_filepath):
    """
    Merges multiple CSV files found via a glob pattern into a single CSV file.
    It concatenates the files row-wise and sorts the result by the 'file' column.
    """
    print(f"\n--- Merging files for pattern: {glob_pattern} ---")

    result_files = sorted(glob.glob(glob_pattern))

    if not result_files:
        print(f"Warning: No files found for pattern '{glob_pattern}'. Nothing to merge.")
        return

    all_dfs = []
    try:
        all_dfs = [pd.read_csv(f) for f in result_files]
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    if not all_dfs:
        print("No CSVs could be read. Aborting merge.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by the 'file' column before saving
    if "file" in merged_df.columns:
        print("Sorting merged file by 'file' column.")
        merged_df.sort_values(by="file", inplace=True)

        # --- FIX: Remove duplicates based on the 'file' column ---
        print(f"Original row count: {len(merged_df)}")
        merged_df.drop_duplicates(subset=["file"], keep="first", inplace=True)
        print(f"Row count after removing duplicates: {len(merged_df)}")
    else:
        print("Warning: 'file' column not found in merged data. Cannot sort or deduplicate.")

    # Ensure parent directory exists
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    merged_df.to_csv(output_filepath, index=False)
    print(f"Successfully merged {len(result_files)} files into: {output_filepath}")


if __name__ == "__main__":
    # --- Task 1: Merge user-requested ablation results ---
    multi_metrics_dir = os.path.join(PROJECT_ROOT, "eval", "metrics", "multi")
    merge_glob_pattern = os.path.join(
        multi_metrics_dir, "TSPulse2_dimensionality_reduction_ablated_*.csv"
    )
    merge_output_file = os.path.join(
        multi_metrics_dir, "TSPulse2_dimensionality_reduction_ablated.csv"
    )
    merge_csv_files(glob_pattern=merge_glob_pattern, output_filepath=merge_output_file)

    print("\n" + "=" * 50 + "\n")
    print("--- Task 2: Running original comparison logic ---")
    # Now, run the main comparison logic
    for config in BENCHMARK_CONFIGS:
        save_final_comparison(
            name=config["name"],
            metrics_dir=config["metrics_dir"],
            benchmark_file=config["benchmark_file"],
            output_dir=OUTPUT_DIR,
            metric=METRIC_TO_COMPARE,
        )
