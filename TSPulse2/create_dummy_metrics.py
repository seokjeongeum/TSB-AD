# /home/seokjeongeum/TSB-AD/TSPulse2/create_dummy_metrics.py

import os
import random

import pandas as pd


def create_dummy_metrics_file(output_path, file_list_path, header):
    """
    Generates a dummy metrics CSV file.

    Args:
        output_path (str): The full path for the output CSV file.
        file_list_path (str): The path to the CSV containing the list of filenames.
        header (list): The header row for the output CSV.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read the list of files to generate metrics for
    try:
        file_df = pd.read_csv(file_list_path)
        filenames = file_df["file_name"]
    except FileNotFoundError:
        print(f"Error: Input file list not found at {file_list_path}")
        return
    except KeyError:
        print(f"Error: 'file_name' column not found in {file_list_path}")
        return

    # Generate dummy data and write to the output file
    with open(output_path, "w", newline="") as f:
        # Using pandas to_csv for simplicity and correct quoting
        rows = []
        rows.append(header)
        for filename in filenames:
            # Create a row of dummy data
            # [filename, Time, AUC-PR, AUC-ROC, ..., Affiliation-F]
            time = random.uniform(10.0, 120.0)
            # 9 other metric scores, mostly between 0 and 1
            metrics = [random.random() for _ in range(9)]

            row = [filename, f"{time:.6f}"] + [f"{m:.6f}" for m in metrics]
            rows.append(row)

        # Create a DataFrame and save to CSV
        df_out = pd.DataFrame(rows[1:], columns=rows[0])
        df_out.to_csv(f, index=False)

    print(f"Successfully created: {output_path}")


if __name__ == "__main__":
    # --- Configuration ---

    # The header for the output CSV files
    HEADER = [
        "file",
        "Time",
        "AUC-PR",
        "AUC-ROC",
        "VUS-PR",
        "VUS-ROC",
        "Standard-F1",
        "PA-F1",
        "Event-based-F1",
        "R-based-F1",
        "Affiliation-F",
    ]

    # This is the key part for relative paths:
    # 1. Get the directory of the current script.
    #    e.g., /home/seokjeongeum/TSB-AD/TSPulse2
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up one level to get the project's root directory.
    #    e.g., /home/seokjeongeum/TSB-AD
    project_root = os.path.dirname(script_dir)

    print(f"Detected project root: {project_root}")

    # Define relative paths for inputs and outputs based on the project root
    # Input File Lists
    multi_file_list = os.path.join(project_root, "Datasets/File_List/TSB-AD-M.csv")
    uni_file_list = os.path.join(project_root, "Datasets/File_List/TSB-AD-U.csv")
    multi_as_uni_file_list = os.path.join(
        project_root, "Datasets/File_List/TSB-AD-M-univariate.csv"
    )

    # Target Output Files
    target_paths = [
        # Multivariate targets
        "eval/metrics/multi/TSPulse_ZS_ensemble.csv",
        "eval/metrics/multi/TSPulse_ZS_fft.csv",
        "eval/metrics/multi/TSPulse_ZS_future.csv",
        "eval/metrics/multi/TSPulse_ZS_time.csv",
        "eval/metrics/multi/TSPulse2.csv",
        # Univariate targets
        "eval/metrics/uni/TSPulse_ZS_ensemble.csv",
        "eval/metrics/uni/TSPulse_ZS_fft.csv",
        "eval/metrics/uni/TSPulse_ZS_future.csv",
        "eval/metrics/uni/TSPulse_ZS_time.csv",
        "eval/metrics/uni/TSPulse2.csv",
        # Multivariate-as-Univariate targets
        "eval/metrics/multi_as_uni/TSPulse_ZS_ensemble.csv",
        "eval/metrics/multi_as_uni/TSPulse_ZS_fft.csv",
        "eval/metrics/multi_as_uni/TSPulse_ZS_future.csv",
        "eval/metrics/multi_as_uni/TSPulse_ZS_time.csv",
        "eval/metrics/multi_as_uni/TSPulse2.csv",
    ]

    # --- Main Execution Logic ---
    print("\nStarting to generate dummy metric files...")

    for rel_path in target_paths:
        # Construct the full, absolute path for the output file
        full_output_path = os.path.join(project_root, rel_path)

        # Determine whether to use the 'multi' or 'uni' file list
        if "multi_as_uni" in rel_path:
            create_dummy_metrics_file(
                full_output_path, multi_as_uni_file_list, HEADER
            )
        elif "multi" in rel_path:
            create_dummy_metrics_file(full_output_path, multi_file_list, HEADER)
        elif "uni" in rel_path:
            create_dummy_metrics_file(full_output_path, uni_file_list, HEADER)
        else:
            print(
                f"Warning: Could not determine type (multi/uni) for path: {rel_path}. Skipping."
            )

    print("\nAll tasks completed.")
