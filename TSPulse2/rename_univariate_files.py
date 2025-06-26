# /home/seokjeongeum/TSB-AD/TSPulse2/rename_univariate_files.py

import os
import pandas as pd
from tqdm import tqdm
import re

def find_nth_occurrence(string, sub, n):
    """Finds the starting index of the nth occurrence of a substring."""
    start = string.find(sub)
    while start >= 0 and n > 1:
        start = string.find(sub, start + len(sub))
        n -= 1
    return start

def rename_files_and_update_list(target_dir, file_list_path, n=9):
    """
    Renames files in a directory by replacing the nth and subsequent underscores
    with hyphens, and updates the corresponding file list CSV.
    
    Args:
        target_dir (str): The directory containing files to rename.
        file_list_path (str): The path to the CSV file list to update.
        n (int): The occurrence of the underscore to start replacing from.
    """
    print(f"Processing directory: {target_dir}")
    
    # --- Step 1: Rename the files on the filesystem ---
    renamed_count = 0
    files_in_dir = os.listdir(target_dir)
    for old_filename in tqdm(files_in_dir, desc="Renaming files"):
        if not old_filename.endswith(".csv"):
            continue

        # Find the position of the 9th underscore
        split_pos = find_nth_occurrence(old_filename, '_', n)
        
        if split_pos != -1:
            base_part = old_filename[:split_pos]
            suffix_part = old_filename[split_pos:]
            
            # Replace all remaining underscores in the suffix part with hyphens
            new_suffix = suffix_part.replace('_', '-')
            new_filename = base_part + new_suffix
            
            if new_filename != old_filename:
                old_path = os.path.join(target_dir, old_filename)
                new_path = os.path.join(target_dir, new_filename)
                os.rename(old_path, new_path)
                renamed_count += 1
                
    print(f"Renamed {renamed_count} files on the filesystem.")

    # --- Step 2: Update the file list CSV ---
    print(f"\nUpdating file list: {file_list_path}")
    try:
        df = pd.read_csv(file_list_path)
    except FileNotFoundError:
        print(f"Error: File list not found at {file_list_path}. Skipping update.")
        return

    updated_filenames = []
    for old_filename in tqdm(df['file_name'], desc="Updating file list entries"):
        split_pos = find_nth_occurrence(old_filename, '_', n)
        
        if split_pos != -1:
            base_part = old_filename[:split_pos]
            suffix_part = old_filename[split_pos:]
            new_suffix = suffix_part.replace('_', '-')
            new_filename = base_part + new_suffix
            updated_filenames.append(new_filename)
        else:
            updated_filenames.append(old_filename) # Append unchanged if no 9th underscore
            
    df['file_name'] = updated_filenames
    df.to_csv(file_list_path, index=False)
    print(f"File list updated successfully.")


if __name__ == "__main__":
    # Get the directory of the current script to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes script is in TSPulse2/

    # Define paths relative to the project root
    target_data_dir = os.path.join(project_root, "Datasets/TSB-AD-M-univariate")
    target_list_path = os.path.join(project_root, "Datasets/File_List/TSB-AD-M-univariate.csv")
    
    rename_files_and_update_list(target_data_dir, target_list_path, n=9) 