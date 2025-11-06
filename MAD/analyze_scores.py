import os
import re
from typing import Dict

import numpy as np
import pandas as pd


def get_parent_map(univariate_files, parents) -> Dict[str, str]:
    """Maps a univariate file to its multivariate parent."""
    base_name_pattern = "|".join(re.escape(b) for b in parents)
    base_name_regex = re.compile(f"^({base_name_pattern})-")
    mapping = {}
    for f in univariate_files:
        f_base = os.path.splitext(f)[0]
        match = base_name_regex.match(f_base)
        if match:
            mapping[f] = match.group(1) + ".csv"
    return mapping


def analyze_vus_pr_scores():
    """
    Reads VUS-PR scores, determines the best channel per dataset based on tuning data,
    filters for those channels, and then finds the top-performing file per head.
    """
    # Define the mapping from method name to its corresponding file path
    base_path = os.path.join("eval", "metrics", "multi_as_uni")
    files_to_read = {
        "ensemble": os.path.join(base_path, "TSPulse_ZS_ensemble.csv"),
        "fft": os.path.join(base_path, "TSPulse_ZS_fft.csv"),
        "forecast": os.path.join(base_path, "TSPulse_ZS_forecast.csv"),
        "time": os.path.join(base_path, "TSPulse_ZS_time.csv"),
    }

    all_scores = []

    # Read each file and extract the relevant columns
    for method, file_path in files_to_read.items():
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Keep only the file name and VUS-PR score
            df_subset = df[["file", "VUS-PR"]].copy()

            # Rename the 'VUS-PR' column to the method name for clarity
            df_subset.rename(columns={"VUS-PR": method}, inplace=True)

            # Set the 'file' column as the index for easy merging
            df_subset.set_index("file", inplace=True)

            all_scores.append(df_subset)
            print(f"Successfully loaded and processed {file_path}")

        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
        except KeyError:
            print(
                f"Warning: 'file' or 'VUS-PR' column not found in {file_path}. Skipping."
            )
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # Check if any data was loaded
    if not all_scores:
        print("\nNo data was loaded. Please check the file paths and content.")
        return

    # Merge all dataframes into a single dataframe, aligning by file name
    # The concat function aligns data based on the index (which we set to 'file')
    consolidated_df = pd.concat(all_scores, axis=1)

    # Calculate the difference between the max and min of specified columns
    cols_for_range = ["ensemble", "fft", "forecast", "time"]
    consolidated_df["range"] = consolidated_df[cols_for_range].max(
        axis=1
    ) - consolidated_df[cols_for_range].min(axis=1)

    # Calculate the difference between the max and second max
    sorted_scores = np.sort(consolidated_df[cols_for_range].values, axis=1)
    consolidated_df["max_minus_second_max"] = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
    )

    # Find which method produced the max score
    consolidated_df["max_method"] = consolidated_df[cols_for_range].idxmax(axis=1)

    # Add columns for avg score, parent, group, and channel to assist with filtering
    consolidated_df["avg_score"] = consolidated_df[cols_for_range].mean(axis=1)

    try:
        # Load multivariate file list to identify parent files
        multi_file_list_path = os.path.join("Datasets", "File_List", "TSB-AD-M.csv")
        multi_full_df = pd.read_csv(multi_file_list_path)
        multi_base_names = sorted(
            [os.path.splitext(f)[0] for f in multi_full_df["file_name"]],
            key=len,
            reverse=True,
        )

        # Map univariate files to their multivariate parents
        uni_to_multi_map = get_parent_map(
            consolidated_df.index.tolist(), multi_base_names
        )
        consolidated_df["parent"] = consolidated_df.index.map(uni_to_multi_map.get)
        consolidated_df["parent_base"] = consolidated_df["parent"].apply(
            lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
        )
        consolidated_df["channel_name"] = consolidated_df.apply(
            lambda row: (
                os.path.splitext(row.name)[0][len(row["parent_base"]) + 1 :]
                if pd.notna(row["parent_base"])
                else None
            ),
            axis=1,
        )
        consolidated_df["group"] = consolidated_df["parent"].apply(
            lambda x: os.path.splitext(x)[0].split("_")[1] if pd.notna(x) else None
        )
        consolidated_df.dropna(subset=["parent", "group", "channel_name"], inplace=True)

        # Determine the best channel for each group using the tuning set
        tuning_list_path = os.path.join("Datasets", "File_List", "TSB-AD-M-Tuning.csv")
        tuning_files_df = pd.read_csv(tuning_list_path)
        tuning_basenames = (
            tuning_files_df["file_name"].str.replace(".csv", "", regex=False).tolist()
        )
        tuning_files = set(tuning_files_df["file_name"])

        tuning_df = consolidated_df[consolidated_df["parent"].isin(tuning_files)]

        if tuning_df.empty:
            raise ValueError("Tuning set is empty, cannot determine best channels.")

        # For each group, find the channel with the highest average score across heads
        group_channel_scores = (
            tuning_df.groupby(["group", "channel_name"])["avg_score"]
            .mean()
            .reset_index()
        )
        best_channels_df = group_channel_scores.loc[
            group_channel_scores.groupby("group")["avg_score"].idxmax()
        ]
        best_channel_map = best_channels_df.set_index("group")["channel_name"].to_dict()

        print("\n--- Learned Best Channel per Group (from Tuning Set) ---")
        print(pd.Series(best_channel_map))
        print("--------------------------------------------------------")

        # Create a boolean mask to identify rows that match the tuning files
        mask = consolidated_df.index.to_series().apply(
            lambda x: any(x.startswith(basename) for basename in tuning_basenames)
        )
        consolidated_df = consolidated_df[mask]
        # Filter the main dataframe to only include the best channels
        consolidated_df = consolidated_df[
            consolidated_df.apply(
                lambda row: row["channel_name"] == best_channel_map.get(row["group"]),
                axis=1,
            )
        ]
        print(
            f"\nFiltered to {len(consolidated_df)} files corresponding to best channels."
        )

    except (FileNotFoundError, ValueError) as e:
        print(
            f"\nWarning: Could not determine or apply best channel strategy. Error: {e}"
        )
        print("Proceeding with analysis on all available files.")
    except Exception as e:
        print(f"An unexpected error occurred during best channel filtering: {e}")

    # Define the output path and save the consolidated dataframe to a CSV file
    output_path = os.path.join("MAD", "consolidated_vus_pr_scores.csv")
    # Save the filtered dataframe for inspection
    consolidated_df.to_csv(output_path)

    print(f"\nConsolidated scores for best channels have been saved to {output_path}")

    # --- Final Analysis ---
    # The 'group' column is now the reliable way to identify the dataset
    for method in cols_for_range:
        # Filter for rows where the max method is the current method
        method_max_df = consolidated_df[consolidated_df["max_method"] == method].copy()

        if not method_max_df.empty:
            # For each group, find the file with the highest 'max_minus_second_max'
            idx = method_max_df.groupby("group")["max_minus_second_max"].idxmax()
            top_per_dataset = method_max_df.loc[idx]

            # Print the results, sorted by the score
            print(
                f"\n--- Top file per dataset where {method} is max and (max - 2nd max) is highest (Best Channels Only) ---"
            )
            print(
                top_per_dataset.sort_values(
                    by="max_minus_second_max", ascending=False
                ).to_string()
            )
        else:
            print(f"\nNo files found where '{method}' was the max method.")


if __name__ == "__main__":
    analyze_vus_pr_scores()
