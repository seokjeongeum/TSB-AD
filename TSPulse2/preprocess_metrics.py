import glob
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --- NEW: Pre-processing Function to Sort Result Files ---
def align_and_clean_all_result_files():
    """
    Finds all result .csv files, aligns them with the official master file lists,
    cleans inconsistent .csv extensions, fills missing values, sorts them,
    and saves them back to disk.
    """
    print("=" * 60)
    print("Starting pre-processing: Aligning, cleaning, and sorting all result CSV files...")
    print("=" * 60)

    # Path to the parent metrics directory and file lists
    all_metrics_dir = os.path.join(PROJECT_ROOT, "eval", "metrics")
    file_list_dir = os.path.join(PROJECT_ROOT, "Datasets", "File_List")

    # Find all .csv files in any subdirectory of eval/metrics/
    all_csv_files = glob.glob(
        os.path.join(all_metrics_dir, "**", "*.csv"), recursive=True
    )

    if not all_csv_files:
        print("No result files found to process.")
        return

    # Create a mapping from experiment type to the official file list
    file_list_map = {
        "uni": os.path.join(file_list_dir, "TSB-AD-U-Eva.csv"),
        "multi": os.path.join(file_list_dir, "TSB-AD-M-Eva.csv"),
        "uni-tuning": os.path.join(file_list_dir, "TSB-AD-U-Tuning.csv"),
        "multi-tuning": os.path.join(file_list_dir, "TSB-AD-M-Tuning.csv"),
    }

    for file_path in all_csv_files:
        try:
            # 1. Determine experiment type and get the official file list path
            exp_type = os.path.basename(os.path.dirname(file_path))
            official_list_path = file_list_map.get(exp_type)

            if not official_list_path:
                print(
                    f"  - Warning: Skipping file with unknown experiment type '{exp_type}': {file_path}"
                )
                continue

            if not os.path.exists(official_list_path):
                print(
                    f"  - Warning: Official file list not found for '{exp_type}', skipping: {official_list_path}"
                )
                continue

            # 2. Load the official list of filenames (without .csv)
            official_df = pd.read_csv(official_list_path)
            official_filenames = set(
                official_df["file_name"].str.replace(r"\.csv$", "", regex=True)
            )

            # 3. Load the result file and clean its 'file' column
            result_df = pd.read_csv(file_path)
            if "file" not in result_df.columns:
                print(
                    f"  - Warning: 'file' column not found in {file_path}. Skipping alignment."
                )
                continue

            result_df["file"] = result_df["file"].str.replace(
                r"\.csv$", "", regex=True
            )

            # 4. Align the result data with the official list
            # Create a clean DataFrame based on the official list
            aligned_df = pd.DataFrame({'file': sorted(list(official_filenames))})

            # Merge the actual results onto this clean frame
            # This keeps only official files, drops unofficial ones, and adds rows for missing ones
            final_df = pd.merge(aligned_df, result_df, on="file", how="left")

            # Fill missing results with 0.0, but keep 'file' column as is.
            # Identify columns to fill (all except 'file')
            fill_cols = [col for col in final_df.columns if col != "file"]
            final_df[fill_cols] = final_df[fill_cols].fillna(0.0)

            # 5. Save the aligned, cleaned, and sorted DataFrame back to disk
            final_df.to_csv(file_path, index=False)
            print(
                f"  - Processed and aligned: {os.path.relpath(file_path, PROJECT_ROOT)}"
            )

        except pd.errors.EmptyDataError:
            print(f"  - Warning: File is empty, skipping processing for {file_path}")
        except Exception as e:
            print(f"  - Error processing {file_path}: {e}")

    print("\nPre-processing finished.")


if __name__ == "__main__":
    # --- ADDED: Run the pre-processing step once before everything else ---
    align_and_clean_all_result_files()
