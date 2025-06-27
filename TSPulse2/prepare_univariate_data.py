# prepare_univariate_data.py
import argparse
import os

import pandas as pd
from tqdm import tqdm


def rename_incorrectly_named_files(dest_dir, file_list_path):
    """
    Renames files in the destination directory. This does two things:
    1. Ensures the separator between base name and feature name is a hyphen.
    2. Replaces all underscores in the feature part of the filename with hyphens.
    """
    if not os.path.exists(dest_dir):
        return

    try:
        source_df = pd.read_csv(file_list_path)
        base_names = [os.path.splitext(f)[0] for f in source_df["file_name"].tolist()]
        base_names.sort(key=len, reverse=True)  # Handle overlapping names
    except FileNotFoundError:
        print(
            f"Error: Source file list not found at {file_list_path}, skipping rename."
        )
        return

    print(f"Sanitizing filenames in '{dest_dir}'...")
    renamed_count = 0
    files_in_dest = os.listdir(dest_dir)

    for filename in tqdm(files_in_dest, desc="Sanitizing"):
        matched_base = None
        for base in base_names:
            if filename.startswith(base + "_") or filename.startswith(base + "-"):
                matched_base = base
                break

        if matched_base:
            # Extract feature part (everything after base name and separator)
            feature_part_with_ext = filename[len(matched_base) + 1 :]
            name_part, ext_part = os.path.splitext(feature_part_with_ext)

            # Sanitize by replacing all underscores with hyphens
            sanitized_name = name_part.replace("_", "-")
            sanitized_feature = sanitized_name + ext_part

            # Construct the new filename with a hyphen separator
            new_filename = f"{matched_base}-{sanitized_feature}"

            if new_filename != filename:
                source_path = os.path.join(dest_dir, filename)
                dest_path = os.path.join(dest_dir, new_filename)
                os.rename(source_path, dest_path)
                renamed_count += 1

    if renamed_count > 0:
        print(f"Renamed {renamed_count} files.")


def convert_multivariate_to_univariate(source_dir, dest_dir, file_list_path):
    """
    Converts all multivariate CSV files from a source directory into multiple
    univariate CSV files in a destination directory.

    For each input file, it creates multiple univariate files where the original
    base name is separated from the column name by a hyphen.

    Args:
        source_dir (str): Directory containing the original multivariate CSV files.
        dest_dir (str): Directory to save the new univariate CSV files.
        file_list_path (str): Path to the original file list (e.g., TSB-AD-M.csv).
    """
    os.makedirs(dest_dir, exist_ok=True)

    try:
        source_df = pd.read_csv(file_list_path)
        source_files = source_df["file_name"].tolist()
    except FileNotFoundError:
        print(f"Error: Source file list not found at {file_list_path}")
        return

    new_univariate_files = []
    created_count = 0
    skipped_count = 0

    print(f"Converting files from '{source_dir}' to '{dest_dir}'...")
    for filename in tqdm(source_files, desc="Converting files"):
        source_path = os.path.join(source_dir, filename)

        try:
            df = pd.read_csv(source_path)
        except FileNotFoundError:
            print(f"Warning: File {filename} not found in {source_dir}. Skipping.")
            continue

        if "Label" not in df.columns:
            print(f"Warning: 'Label' column not found in {filename}. Skipping.")
            continue

        label_series = df["Label"]
        feature_columns = [col for col in df.columns if col != "Label"]

        base_name = os.path.splitext(filename)[0]

        for col in feature_columns:
            # Sanitize column name to be filesystem-friendly
            sanitized_col_name = (
                "".join(c for c in col if c.isalnum() or c in ("_", "-"))
                .replace("_", "-")  # Also replace underscores with hyphens
                .rstrip()
            )

            # Define the new filename using a hyphen as a separator
            new_filename = f"{base_name}-{sanitized_col_name}.csv"
            dest_path = os.path.join(dest_dir, new_filename)

            # The file list should contain all generated files, even if they already exist.
            new_univariate_files.append(new_filename)

            # Skip creating the file if it already exists
            if os.path.exists(dest_path):
                skipped_count += 1
                continue

            # Create a new DataFrame for the univariate series with 'Data' and 'Label' columns
            univariate_df = pd.DataFrame({"Data": df[col], "Label": label_series})
            univariate_df.to_csv(dest_path, index=False)
            created_count += 1

    # Create a new file list for the generated univariate files
    new_file_list_df = pd.DataFrame({"file_name": sorted(new_univariate_files)})

    # Save the new file list in the same directory as the original
    file_list_dir = os.path.dirname(file_list_path)
    new_list_filename = os.path.basename(file_list_path).replace(
        ".csv", "-univariate.csv"
    )
    new_file_list_path = os.path.join(file_list_dir, new_list_filename)

    new_file_list_df.to_csv(new_file_list_path, index=False)

    print("\nConversion complete.")
    print(f"{created_count} new univariate files created.")
    print(f"{skipped_count} files already existed and were skipped.")
    print(f"New file list saved to: {new_file_list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert multivariate time series data to univariate."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="Datasets/TSB-AD-M/",
        help="Directory of the original multivariate CSV files.",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="Datasets/TSB-AD-M-univariate/",
        help="Directory to save the new univariate CSV files.",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default="Datasets/File_List/TSB-AD-M.csv",
        help="Path to the file list of the multivariate data.",
    )
    args = parser.parse_args()

    # First, rename any incorrectly named files from a previous run.
    rename_incorrectly_named_files(args.dest_dir, args.file_list)

    # Now, proceed with the conversion.
    convert_multivariate_to_univariate(args.source_dir, args.dest_dir, args.file_list)
