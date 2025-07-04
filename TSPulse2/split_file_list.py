import os

import numpy as np
import pandas as pd

# --- Configuration ---

# This script assumes it is located in a subdirectory of the main project.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_LIST_DIR = os.path.join(PROJECT_ROOT, "Datasets", "File_List")
INPUT_FILENAME = "TSB-AD-M-Eva.csv"
INPUT_FILE_PATH = os.path.join(FILE_LIST_DIR, INPUT_FILENAME)
BASE_OUTPUT_FILENAME = "TSB-AD-M-Eva"


def split_csv_file(input_path, output_dir, base_output_name, num_splits=6):
    """
    Reads a CSV file and splits it into a specified number of smaller CSV files
    based on the row index.
    """
    # 1. Load the source CSV data
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {len(df)} records from '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: The source file was not found at '{input_path}'.")
        return
    except Exception as e:
        print(f"Error reading CSV data from '{input_path}': {e}")
        return

    if df.empty:
        print("The source file is empty. No files will be created.")
        return

    # 2. Assign each row to a split file group (0 to num_splits-1)
    # np.arange(len(df)) creates an array [0, 1, 2, ..., n-1]
    # The modulo operator (%) then assigns the group for each row.
    df["split_group"] = np.arange(len(df)) % num_splits

    # 3. Loop through each group to create the corresponding file
    for i in range(num_splits):
        # The output filename uses 1-based indexing (e.g., _1, _2)
        output_filename = f"{base_output_name}_{i+1}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Select the subset of the DataFrame for the current group
        subset_df = df[df["split_group"] == i]

        # Ensure we only save the original columns, not the temporary 'split_group'
        original_columns = [col for col in df.columns if col != "split_group"]
        df_to_save = subset_df[original_columns]

        # Save the subset to a new CSV file, without the pandas index column
        df_to_save.to_csv(output_path, index=False)

        print(f"Created '{output_filename}' with {len(df_to_save)} records.")

    print("\nSplitting complete.")


if __name__ == "__main__":
    split_csv_file(
        input_path=INPUT_FILE_PATH,
        output_dir=FILE_LIST_DIR,
        base_output_name=BASE_OUTPUT_FILENAME,
        num_splits=6,
    )
