import glob
import os
import re

import pandas as pd

# --- Configuration ---
METRICS_DIR = "/workspaces/TSB-AD/eval/metrics/uni/"
METRIC_TO_COMPARE = "VUS-PR"


def create_performance_pivot_table(metrics_dir, metric):
    """
    Creates and displays a pivot table comparing the performance of all
    TSPulse variants across all datasets.
    """
    # 1. Find and load all TSPulse result CSVs
    tspulse_files = glob.glob(os.path.join(metrics_dir, "TSPulse*.csv"))
    if not tspulse_files:
        print(f"Error: No TSPulse result files found in '{metrics_dir}'")
        return

    all_results = []

    for file_path in tspulse_files:
        try:
            algo_name = os.path.basename(file_path).replace(".csv", "")
            df = pd.read_csv(file_path)
            if "file" in df.columns and metric in df.columns:
                df = df[["file", metric]]
                df["Algorithm"] = algo_name
                all_results.append(df)
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    if not all_results:
        print("No valid result files could be processed.")
        return

    combined_df = pd.concat(all_results, ignore_index=True)

    # 2. Extract the clean Dataset Name from the filename
    def extract_dataset_name(filename):
        match = re.search(r"^\d+_(.*?)_id_", filename)
        return match.group(1) if match else "Unknown"

    combined_df["Dataset"] = combined_df["file"].apply(extract_dataset_name)

    # 3. Create the pivot table
    # This reshapes the data to have datasets as rows, algorithms as columns,
    # and the mean metric score as the values.
    pivot_table = pd.pivot_table(
        combined_df,
        values=metric,
        index="Dataset",
        columns="Algorithm",
        aggfunc="mean",  # Aggregate by mean for datasets with multiple files
    )

    # 4. Add a 'Best_Variant' column for easy identification
    # .idxmax(axis=1) finds the column name (algorithm) with the max value in each row
    variant_columns = [col for col in pivot_table.columns if col != "TSPulse2"]
    pivot_table["Best_Variant"] = pivot_table[variant_columns].idxmax(axis=1)

    # Fill any potential NaN values with 0 for cleaner output
    pivot_table.fillna(0.0, inplace=True)

    # Sort by dataset name
    pivot_table.sort_index(inplace=True)

    # --- Display Results ---
    print(f"\n--- TSPulse Variant Performance Comparison ({metric}) ---")
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    # Reorder columns to have 'Best_Variant' first for clarity
    cols = ["Best_Variant"] + [
        col for col in pivot_table.columns if col != "Best_Variant"
    ]
    print(pivot_table[cols])


if __name__ == "__main__":
    create_performance_pivot_table(METRICS_DIR, METRIC_TO_COMPARE)
