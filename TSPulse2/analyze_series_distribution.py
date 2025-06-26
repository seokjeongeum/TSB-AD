import os
import pandas as pd
from collections import defaultdict

# Define the base paths for datasets and file lists
BASE_DIR = "Datasets"
MULTI_DIR = os.path.join(BASE_DIR, "TSB-AD-M")
UNI_DIR = os.path.join(BASE_DIR, "TSB-AD-U")
FILE_LIST_DIR = os.path.join(BASE_DIR, "File_List")

# Paths to the CSV files that define the tuning sets
MULTI_TUNING_LIST = os.path.join(FILE_LIST_DIR, "TSB-AD-M-Tuning.csv")
UNI_TUNING_LIST = os.path.join(FILE_LIST_DIR, "TSB-AD-U-Tuning.csv")

def load_tuning_files(filepath: str) -> set:
    """
    Loads the list of tuning files from a given CSV path and returns a set
    of filenames for efficient lookup.
    """
    try:
        df = pd.read_csv(filepath)
        return set(df['file_name'])
    except FileNotFoundError:
        print(f"Warning: Tuning file list not found at '{filepath}'.")
        return set()

def analyze_directory(directory_path: str, domain: str, all_tuning_files: set, counts: defaultdict):
    """
    Iterates through a directory, parses filenames, determines if each file
    belongs to the tuning or evaluation set, and updates the counts for the
    given domain.
    """
    if not os.path.isdir(directory_path):
        print(f"Warning: Data directory not found, skipping: '{directory_path}'")
        return

    print(f"Analyzing directory: {directory_path} (Domain: {domain})")
    for filename in os.listdir(directory_path):
        if not filename.endswith(".csv"):
            continue

        # Assumes format: [index]_[Dataset Name]_...
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"Warning: Skipping malformed filename: {filename}")
            continue

        dataset_name = parts[1]

        # Categorize the file as "Tuning" or "Evaluation"
        set_type = "Tuning" if filename in all_tuning_files else "Evaluation"
        
        # Store counts under the dataset and its domain
        counts[dataset_name][domain][set_type] += 1

def main():
    """
    Main function to drive the analysis and print a formatted summary of the
    time series distribution across different datasets and domains.
    """
    print("Starting analysis of series distribution...\n")

    # 1. Load the complete list of tuning files
    multi_tuning_files = load_tuning_files(MULTI_TUNING_LIST)
    uni_tuning_files = load_tuning_files(UNI_TUNING_LIST)
    all_tuning_files = multi_tuning_files.union(uni_tuning_files)

    if not all_tuning_files:
        print("Error: No tuning files could be loaded. Cannot determine sets.")
        return

    # 2. Initialize a 3-level nested dictionary for counts:
    #    e.g., {'smd': {'M': {'Tuning': 10, 'Evaluation': 2}}}
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # 3. Analyze both data directories, passing the domain for each
    analyze_directory(MULTI_DIR, "Multivariate", all_tuning_files, counts)
    analyze_directory(UNI_DIR, "Univariate", all_tuning_files, counts)
    print("\nAnalysis complete.")

    # 4. Format and print the results in a table
    if not counts:
        print("No .csv files were found in the data directories.")
        return

    # Convert the nested dictionary into a list of records for DataFrame creation
    records = []
    for dataset, domains in sorted(counts.items()):
        for domain, set_counts in sorted(domains.items()):
            records.append({
                "Dataset": dataset,
                "Domain": domain,
                "Tuning Set": set_counts.get("Tuning", 0),
                "Evaluation Set": set_counts.get("Evaluation", 0)
            })

    df = pd.DataFrame(records)
    df['Total'] = df['Tuning Set'] + df['Evaluation Set']

    # Create a summary row for the table footer
    summary_row = pd.DataFrame({
        "Dataset": ["--- TOTAL ---"],
        "Domain": [""],
        "Tuning Set": [df['Tuning Set'].sum()],
        "Evaluation Set": [df['Evaluation Set'].sum()],
        "Total": [df['Total'].sum()],
    })

    # Combine the data and the summary row
    df_final = pd.concat([df, summary_row], ignore_index=True)

    print("\n" + "="*75)
    print("                Distribution of Time Series per Dataset and Domain")
    print("="*75)
    # Use to_string() for clean, aligned console output
    print(df_final.to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    main()