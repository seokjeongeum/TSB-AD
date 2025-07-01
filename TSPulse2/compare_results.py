import glob
import os

import pandas as pd

# --- Configuration ---
# Base directories where the result files are stored.
# Adjust these paths if your script is not in the project's root directory.
MY_RUNS_BASE_DIR = "eval/metrics"
BENCHMARK_BASE_DIR = (
    "granite-tsfm/notebooks/hfdemo/tspulse/anomaly_detection/benchmarks"
)

# The core algorithm types we want to compare.
ALGO_TYPES = ["ensemble", "fft", "forecast", "time"]

# The dataset types (uni for univariate, multi for multivariate).
DATASET_TYPES = ["multi", "uni"]

# --- Main Logic ---


def find_benchmark_files(data_type, algo):
    """
    Finds all relevant benchmark files (Eva and Tuning).
    Example pattern: TSB-AD-M-Eva-ensemble.csv
    """
    # Map algorithm names if they differ between your runs and benchmark runs
    algo_for_benchmark = "future" if algo == "forecast" else algo

    # Construct a search pattern. The '*' allows for variants like 'Eva' or 'Tuning'.
    pattern = os.path.join(
        BENCHMARK_BASE_DIR,
        f"TSB-AD-{data_type[0].upper()}-*-{algo_for_benchmark}.csv",
    )
    found_files = glob.glob(pattern)

    return found_files


def compare_and_summarize(my_file_path, bench_file_paths):
    """
    Loads, merges, and calculates differences between your result CSV
    and a union of benchmark CSVs.
    """
    if not os.path.exists(my_file_path):
        print(f"SKIPPING: Your run file not found at '{my_file_path}'")
        return None, None

    if not bench_file_paths:
        print(f"SKIPPING: No benchmark files found for the pattern near '{my_file_path}'")
        return None, None

    print(f"\nComparing:")
    print(f"  - Your Run:    {os.path.basename(my_file_path)}")
    print(f"  - Benchmarks (Union):")
    for bench_path in bench_file_paths:
        print(f"    - {os.path.basename(bench_path)}")


    # Load the CSV files into pandas DataFrames
    df_my = pd.read_csv(my_file_path)

    # Load and concatenate all found benchmark files
    list_of_bench_dfs = []
    for bench_path in bench_file_paths:
        df_bench_single = pd.read_csv(bench_path)
        list_of_bench_dfs.append(df_bench_single)
    df_bench = pd.concat(list_of_bench_dfs, ignore_index=True)


    # Standardize column names in the benchmark DataFrame before merging
    if "file_name" in df_bench.columns:
        df_bench.rename(columns={"file_name": "file"}, inplace=True)
    elif "file_list" in df_bench.columns:
        df_bench.rename(columns={"file_list": "file"}, inplace=True)

    # Perform an inner merge to only compare results for common dataset files
    df_merged = pd.merge(df_my, df_bench, on="file", suffixes=("_my", "_bench"))

    if df_merged.empty:
        print("  -> No common 'file' entries found between the two CSVs.")
        return None, None

    # --- Calculate the difference in performance (Your Run - Benchmark Run) ---
    # A positive value means your run performed better.
    df_merged["VUS-PR_Improvement"] = df_merged["VUS-PR_my"] - df_merged["VUS-PR_bench"]
    df_merged["VUS-ROC_Improvement"] = (
        df_merged["VUS-ROC_my"] - df_merged["VUS-ROC_bench"]
    )

    # --- Create a summary of the comparison ---
    summary = {
        "My_Avg_VUS-PR": df_merged["VUS-PR_my"].mean(),
        "Bench_Avg_VUS-PR": df_merged["VUS-PR_bench"].mean(),
        "Avg_VUS-PR_Improvement": df_merged["VUS-PR_Improvement"].mean(),
        "My_Avg_VUS-ROC": df_merged["VUS-ROC_my"].mean(),
        "Bench_Avg_VUS-ROC": df_merged["VUS-ROC_bench"].mean(),
        "Avg_VUS-ROC_Improvement": df_merged["VUS-ROC_Improvement"].mean(),
        "Common_Files": len(df_merged),
    }

    # --- Filter and display rows with VUS-PR differences ---
    # Treat differences smaller than 1e-6 as negligible
    df_diff = df_merged[df_merged["VUS-PR_Improvement"].abs() >= 9e-4]

    if not df_diff.empty:
        print("\nDetailed comparison (only showing VUS-PR differences >= 9e-4):")
        display_cols = [
            "file",
            "VUS-PR_my",
            "VUS-PR_bench",
            "VUS-PR_Improvement",
            "VUS-ROC_my",
            "VUS-ROC_bench",
            "VUS-ROC_Improvement",
        ]
        print(df_diff[display_cols].to_string())
    else:
        print("\n-> No significant VUS-PR differences found for this comparison.")

    return summary, df_merged


if __name__ == "__main__":
    all_summaries = []

    for data_type in DATASET_TYPES:
        for algo in ALGO_TYPES:
            # Construct the path to your result file
            # e.g., eval/metrics/multi/TSPulse_ZS_ensemble.csv
            my_file = os.path.join(
                MY_RUNS_BASE_DIR, data_type, f"TSPulse_ZS_{algo}.csv"
            )

            # Find the corresponding benchmark files
            bench_files = find_benchmark_files(data_type, algo)

            # Perform the comparison
            summary_stats, detailed_df = compare_and_summarize(my_file, bench_files)

            if summary_stats and detailed_df is not None:
                summary_stats["Algorithm"] = algo
                summary_stats["Dataset"] = data_type.upper()
                all_summaries.append(summary_stats)

                # Optional: Save the detailed merged DataFrame to a new CSV for further analysis
                output_dir = "comparison_results"
                os.makedirs(output_dir, exist_ok=True)
                output_filename = os.path.join(
                    output_dir, f"comparison_{data_type}_{algo}.csv"
                )
                detailed_df.to_csv(output_filename, index=False)
                print(f"\n-> Detailed comparison saved to '{output_filename}'")

            print("-" * 70)

    # --- Display Final Summary Table ---
    if all_summaries:
        df_final_summary = pd.DataFrame(all_summaries)

        # Reorder columns for better presentation
        ordered_cols = [
            "Dataset",
            "Algorithm",
            "Common_Files",
            "Avg_VUS-PR_Improvement",
            "Avg_VUS-ROC_Improvement",
        ]
        df_final_summary = df_final_summary[ordered_cols]

        # Treat small improvement values as zero for clarity in the final summary
        for col in ["Avg_VUS-PR_Improvement", "Avg_VUS-ROC_Improvement"]:
            df_final_summary.loc[df_final_summary[col].abs() < 9e-4, col] = 0

        print("\n" * 2)
        print("=" * 80)
        print(" " * 25 + "OVERALL PERFORMANCE SUMMARY")
        print("=" * 80)
        # Use .to_string() to ensure the full table is printed without truncation
        print(df_final_summary.to_string(index=False))
        print("=" * 80)
        print("(Positive 'Improvement' value means your run performed better)")
