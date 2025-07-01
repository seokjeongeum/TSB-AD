# Copyright contributors to the TSFM project
#

import argparse
import os
import re
from typing import Dict, Set

import numpy as np
import pandas as pd


def load_split_files(base_data_path: str, dataset_type: str) -> Dict[str, Set[str]]:
    """Loads the split files for a given dataset type into sets for efficient lookup."""
    try:
        file_list_dir = os.path.join(base_data_path, "File_List")
        if dataset_type == "uni":
            eva_df = pd.read_csv(os.path.join(file_list_dir, "TSB-AD-U-Eva.csv"))
            tuning_df = pd.read_csv(os.path.join(file_list_dir, "TSB-AD-U-Tuning.csv"))
        elif dataset_type == "multi":
            eva_df = pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M-Eva.csv"))
            tuning_df = pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M-Tuning.csv"))
        else:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Choose 'uni' or 'multi'."
            )

        eval_files = set(eva_df["file_name"])
        tuning_files = set(tuning_df["file_name"])

        return {"eval": eval_files, "tuning": tuning_files}
    except FileNotFoundError as e:
        print(
            f"Error: Could not load split files for '{dataset_type}' from '{file_list_dir}'. {e}"
        )
        raise


def triangulation_performance(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    dataset_type: str,
    heads_to_load: Dict[str, str],
):
    """
    Performs triangulation scoring based on a new file structure and split logic.

    1. Loads all ZS metric files from multi, multi_as_uni, and uni directories.
    2. Combines them into a single DataFrame.
    3. Splits the data into tuning and evaluation sets based on provided file lists.
    4. Determines the best head for each data group from the tuning set.
    5. Calculates the final triangulated score on the evaluation set.
    """
    if dataset_type == "uni":
        metric_dirs = [os.path.join(root_directory, "uni")]
    elif dataset_type == "multi":
        metric_dirs = [
            os.path.join(root_directory, "multi"),
            os.path.join(root_directory, "multi_as_uni"),
        ]
    else:
        # This case should ideally not be hit if arg parser has choices
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. Choose 'uni' or 'multi'."
        )

    # 1. Load all metric files and combine them
    all_dfs = {}
    for head, fname in heads_to_load.items():
        head_dfs = []
        for d in metric_dirs:
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    # The file column in metrics might not have .csv, ensure consistency
                    df["file"] = df["file"].apply(
                        lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
                    )
                    head_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load or process {fpath}. Error: {e}")
        if head_dfs:
            all_dfs[head] = pd.concat(head_dfs, ignore_index=True)

    if not all_dfs:
        raise ValueError("No metric files were found. Aborting.")

    # 2. Create a single unified DataFrame with files as index and heads as columns
    unified_df = None
    for head, df in all_dfs.items():
        if metric in df.columns:
            df = df.rename(columns={metric: head})
            df = df[["file", head]]
            df = df.set_index("file")
            if unified_df is None:
                unified_df = df
            else:
                unified_df = unified_df.join(df, how="outer")
        else:
            print(
                f"Warning: Metric '{metric}' not found in data for head '{head}'. Skipping."
            )

    if unified_df is None:
        raise ValueError("No metric files were found and unified_df is None. Aborting.")

    # 3. Split into tuning and evaluation sets
    tuning_df = unified_df[unified_df.index.isin(list(split_files["tuning"]))]
    eval_df = unified_df[unified_df.index.isin(list(split_files["eval"]))]

    print(
        f"Found {len(tuning_df)} records for tuning and {len(eval_df)} for evaluation."
    )

    # 4. Helper function to group and aggregate performance
    def _group_and_agg(df: pd.DataFrame):
        index_list = df.index.tolist()
        index_group = {}
        for index in index_list:
            # Assumes file format like '001_UCR_....csv'
            try:
                # remove .csv before splitting
                dataset_name = os.path.splitext(index)[0].split("_")[1]
                if dataset_name not in index_group:
                    index_group[dataset_name] = []
                index_group[dataset_name].append(index)
            except IndexError:
                print(
                    f"Warning: Could not parse dataset group from filename '{index}'. Skipping."
                )

        scores = {}
        for grp, files in index_group.items():
            scores[grp] = df.loc[files].mean().to_dict()

        perf_df = pd.DataFrame(scores).T  # Transpose to have groups as index
        grp_size = {k: len(v) for k, v in index_group.items()}
        return perf_df, grp_size

    # 5. Get performance DataFrames for tuning and evaluation sets
    tuning_performance, _ = _group_and_agg(tuning_df)

    if tuning_performance.empty:
        raise ValueError("Tuning set is empty after processing. Cannot triangulate.")

    # 6. Perform triangulation
    cols = tuning_performance.columns.tolist()
    tuning_performance["best"] = [
        cols[c] for c in np.argmax(tuning_performance.values, axis=1)
    ]

    detailed_results = []

    # Helper to map a file to its group
    def get_group(filename):
        try:
            return os.path.splitext(filename)[0].split("_")[1]
        except IndexError:
            return None

    for file_name, row in eval_df.iterrows():
        group = get_group(file_name)
        if not group:
            print(f"Warning: Could not parse dataset group for {file_name}. Skipping.")
            continue

        sel_mode = (
            tuning_performance.loc[group, "best"]
            if group in tuning_performance.index
            else "time"
        )

        # Get the actual score for this series using the selected head
        series_score = row.get(sel_mode)

        if pd.isna(series_score):
            # Fallback: if the selected head has no score for this file,
            # use the best score available from any other head for THIS file.
            best_available_score = row.max()
            best_available_head = row.idxmax() if not row.empty else "N/A"
            print(
                f"Warning: No score for selected head '{sel_mode}' in file '{file_name}'. "
                f"Falling back to best available head for this file: '{best_available_head}'."
            )
            series_score = best_available_score if not pd.isna(best_available_score) else 0

        detailed_results.append(
            {
                "file": file_name,
                "group": group,
                "selected_head": sel_mode,
                metric: series_score,
            }
        )

    if not detailed_results:
        final_metric = 0
        detailed_results_df = pd.DataFrame()
    else:
        detailed_results_df = pd.DataFrame(detailed_results).set_index("file")
        final_metric = detailed_results_df[metric].mean()

    # The group-aggregated eval performance is still useful for a summary view
    eval_performance, _ = _group_and_agg(eval_df)

    return {
        "tuning": tuning_performance,
        "evaluation": eval_performance,  # Group-aggregated
        "detailed_evaluation": detailed_results_df,  # Per-series
        "metric": final_metric,
    }


def compute_best_channel_by_avg_head_performance(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    base_data_path: str,
    heads_to_load: Dict[str, str],
):
    """
    Computes performance by:
    1. Learning a 'best channel by group' strategy from the tuning set, where 'best'
       is determined by the highest average score across all heads.
    2. Applying this strategy to the evaluation set.
    3. For groups not seen in tuning, it falls back to using the average score of the
       multivariate heads.
    """
    # 1. Load all `multi_as_uni` metrics for all heads and pivot
    all_uni_metrics = []
    for head, filename in heads_to_load.items():
        metric_file_path = os.path.join(root_directory, "multi_as_uni", filename)
        if os.path.exists(metric_file_path):
            try:
                df = pd.read_csv(metric_file_path)
                if metric in df.columns:
                    df["file"] = df["file"].apply(
                        lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
                    )
                    df = df[["file", metric]].rename(columns={metric: head})
                    all_uni_metrics.append(df.set_index("file"))
                else:
                    print(
                        f"Warning: Metric '{metric}' not found in {metric_file_path}. Skipping."
                    )
            except Exception as e:
                print(
                    f"Warning: Could not load or process {metric_file_path}. Error: {e}"
                )

    if not all_uni_metrics:
        print("\nWarning: No 'multi_as_uni' metric files found. Skipping.")
        return None, None, None

    uni_pivoted_df = pd.concat(all_uni_metrics, axis=1, join="outer")
    uni_pivoted_df["avg_score"] = uni_pivoted_df.mean(axis=1)
    combined_df = uni_pivoted_df.reset_index()

    # 2. Add parent, group, and channel info
    multi_file_list_path = os.path.join(base_data_path, "File_List", "TSB-AD-M.csv")
    multi_full_df = pd.read_csv(multi_file_list_path)
    multi_base_names = sorted(
        [os.path.splitext(f)[0] for f in multi_full_df["file_name"]],
        key=len,
        reverse=True,
    )

    def get_parent_map(univariate_files, parents):
        base_name_pattern = "|".join(re.escape(b) for b in parents)
        base_name_regex = re.compile(f"^({base_name_pattern})-")
        mapping = {}
        for f in univariate_files:
            f_base = os.path.splitext(f)[0]
            match = base_name_regex.match(f_base)
            if match:
                mapping[f] = match.group(1) + ".csv"
        return mapping

    uni_to_multi_map = get_parent_map(combined_df["file"], multi_base_names)
    combined_df["parent"] = combined_df["file"].map(uni_to_multi_map)
    combined_df["parent_base"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
    )
    combined_df["channel_name"] = combined_df.apply(
        lambda row: os.path.splitext(row["file"])[0][len(row["parent_base"]) + 1 :]
        if pd.notna(row["parent_base"])
        else None,
        axis=1,
    )
    combined_df["group"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0].split("_")[1] if pd.notna(x) else None
    )
    combined_df = combined_df.dropna(subset=["parent", "group", "channel_name"])

    # 3. Load multivariate data for fallback
    all_multi_metrics = {}
    for head, filename in heads_to_load.items():
        fpath = os.path.join(root_directory, "multi", filename)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                if metric in df.columns:
                    df["file"] = df["file"].apply(
                        lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
                    )
                    all_multi_metrics[head] = df.set_index("file")[metric]
                else:
                    print(
                        f"Warning: Metric '{metric}' not found in {fpath}. Skipping for head '{head}'."
                    )
            except Exception as e:
                print(
                    f"Warning: Could not process {fpath} for head '{head}'. Error: {e}"
                )

    if not all_multi_metrics:
        multi_pivoted_df = pd.DataFrame(columns=list(heads_to_load.keys()))
    else:
        multi_pivoted_df = pd.concat(all_multi_metrics, axis=1, join="outer")

    # 4. Split into tuning and evaluation sets
    tuning_df = combined_df[
        combined_df["parent"].isin(list(split_files["tuning"]))
    ].copy()
    eval_df = combined_df[combined_df["parent"].isin(list(split_files["eval"]))].copy()

    if tuning_df.empty:
        print("Warning: Tuning set for this strategy is empty. Skipping.")
        return None, None, None

    # 5. Determine best channel NAME for each DATASET GROUP on the tuning set
    group_channel_scores = (
        tuning_df.groupby(["group", "channel_name"])["avg_score"].mean().reset_index()
    )
    best_channels_df = group_channel_scores.loc[
        group_channel_scores.groupby("group")["avg_score"].idxmax()
    ]
    group_to_best_channel_map = (
        best_channels_df.set_index("group")["channel_name"].to_dict()
    )

    # 6. Apply strategy to evaluation set
    detailed_results = []
    processed_parents = set()

    for parent_file in eval_df["parent"].unique():
        if parent_file in processed_parents:
            continue

        group = (
            os.path.splitext(parent_file)[0].split("_")[1]
            if "_" in parent_file
            else None
        )
        if not group:
            continue

        strat = "N/A"
        scores_row = None

        if group in group_to_best_channel_map:
            best_channel = group_to_best_channel_map[group]
            # Find the original univariate filename for the best channel
            target_series_df = eval_df[
                (eval_df["parent"] == parent_file)
                & (eval_df["channel_name"] == best_channel)
            ]
            if not target_series_df.empty:
                uni_filename = target_series_df.iloc[0]["file"]
                if uni_filename in uni_pivoted_df.index:
                    scores_row = uni_pivoted_df.loc[uni_filename]
                    strat = f"Best Channel ({best_channel})"

            # Fallback for best channel if it wasn't found in eval set
            if scores_row is None and parent_file in multi_pivoted_df.index:
                scores_row = multi_pivoted_df.loc[parent_file]
                strat = "Fallback-Multivariate (avg)"

        # Fallback for unknown group
        elif parent_file in multi_pivoted_df.index:
            scores_row = multi_pivoted_df.loc[parent_file]
            strat = "Fallback-Unknown Group (avg)"

        if scores_row is not None and not scores_row.empty:
            record = {
                "parent": parent_file,
                "group": group,
                "strategy_or_channel": strat,
                metric: scores_row.mean(),  # The final metric is the average of the selected series.
            }
            # Add all individual head scores for detailed breakdown
            for head in heads_to_load.keys():
                record[head] = scores_row.get(head)

            detailed_results.append(record)

        processed_parents.add(parent_file)

    if not detailed_results:
        return pd.Series(dtype=float), pd.DataFrame(), group_to_best_channel_map

    results_df = pd.DataFrame(detailed_results)

    # Calculate final score for each head by taking the mean of their respective columns
    head_columns = list(heads_to_load.keys())
    final_scores_by_head = results_df[head_columns].mean()
    # Also add the main metric (average of best) to the series for reference
    final_scores_by_head[metric] = results_df[metric].mean()

    return final_scores_by_head, results_df, group_to_best_channel_map


def compute_best_head_and_channel_strategy(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    base_data_path: str,
    heads_to_load: Dict[str, str],
    unknown_group_fallback_head: str,
    triangulation_best_head_map: Dict[str, str],
):
    """
    Learns the best (head, channel) combination from the tuning set and applies it.
    """
    # 1. Load all `multi_as_uni` metrics for all heads and combine them
    all_uni_metrics = []
    for head, filename in heads_to_load.items():
        metric_file_path = os.path.join(root_directory, "multi_as_uni", filename)
        if os.path.exists(metric_file_path):
            try:
                df = pd.read_csv(metric_file_path)
                if metric in df.columns:
                    df["file"] = df["file"].apply(
                        lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
                    )
                    df["head"] = head
                    all_uni_metrics.append(df)
                else:
                    print(
                        f"Warning: Metric '{metric}' not found in {metric_file_path}. Skipping."
                    )
            except Exception as e:
                print(
                    f"Warning: Could not load or process {metric_file_path}. Error: {e}"
                )

    if not all_uni_metrics:
        print("\nWarning: No 'multi_as_uni' metric files found. Skipping Scenario 4.")
        return None, None, None

    combined_df = pd.concat(all_uni_metrics, ignore_index=True)

    # 2. Add parent, group, and channel info
    multi_file_list_path = os.path.join(base_data_path, "File_List", "TSB-AD-M.csv")
    multi_full_df = pd.read_csv(multi_file_list_path)
    multi_base_names = sorted(
        [os.path.splitext(f)[0] for f in multi_full_df["file_name"]],
        key=len,
        reverse=True,
    )

    def get_parent_map(univariate_files, parents):
        base_name_pattern = "|".join(re.escape(b) for b in parents)
        base_name_regex = re.compile(f"^({base_name_pattern})-")
        mapping = {}
        for f in univariate_files:
            f_base = os.path.splitext(f)[0]
            match = base_name_regex.match(f_base)
            if match:
                mapping[f] = match.group(1) + ".csv"
        return mapping

    uni_to_multi_map = get_parent_map(combined_df["file"], multi_base_names)
    combined_df["parent"] = combined_df["file"].map(uni_to_multi_map)
    combined_df["parent_base"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
    )
    combined_df["channel_name"] = combined_df.apply(
        lambda row: os.path.splitext(row["file"])[0][len(row["parent_base"]) + 1 :]
        if pd.notna(row["parent_base"])
        else None,
        axis=1,
    )
    combined_df["group"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0].split("_")[1] if pd.notna(x) else None
    )
    combined_df = combined_df.dropna(subset=["parent", "group", "channel_name"])

    # 3. Split into tuning and evaluation sets
    tuning_df = combined_df[
        combined_df["parent"].isin(list(split_files["tuning"]))
    ].copy()
    eval_df = combined_df[combined_df["parent"].isin(list(split_files["eval"]))].copy()

    if tuning_df.empty:
        print("Warning: Tuning set is empty for Scenario 4. Skipping.")
        return None, None, None

    # 4. Find best (head, channel) for each group on the tuning set
    group_scores = (
        tuning_df.groupby(["group", "head", "channel_name"])[metric]
        .mean()
        .reset_index()
    )
    best_strategies_df = group_scores.loc[
        group_scores.groupby("group")[metric].idxmax()
    ]
    best_strategy_map = best_strategies_df.set_index("group")[
        ["head", "channel_name"]
    ].to_dict("index")

    # 5. Load multivariate data for fallback
    all_multi_metrics = {}
    for head, filename in heads_to_load.items():
        fpath = os.path.join(root_directory, "multi", filename)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                if metric in df.columns:
                    df["file"] = df["file"].apply(
                        lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
                    )
                    all_multi_metrics[head] = df.set_index("file")[metric]
                else:
                    print(
                        f"Warning: Metric '{metric}' not found in {fpath}. Skipping for head '{head}'."
                    )
            except Exception as e:
                print(
                    f"Warning: Could not process {fpath} for head '{head}'. Error: {e}"
                )

    # 6. Apply strategy to evaluation set
    detailed_results = []
    processed_parents = set()

    for parent_file in eval_df["parent"].unique():
        if parent_file in processed_parents:
            continue

        group = (
            os.path.splitext(parent_file)[0].split("_")[1]
            if "_" in parent_file
            else None
        )

        if not group:
            continue

        parent_file_channels = eval_df[eval_df["parent"] == parent_file]

        if group in best_strategy_map:
            # Strategy: Apply learned best (head, channel)
            strategy = best_strategy_map[group]
            best_head = strategy["head"]
            best_channel = strategy["channel_name"]

            target_series = parent_file_channels[
                (parent_file_channels["head"] == best_head)
                & (parent_file_channels["channel_name"] == best_channel)
            ]

            if not target_series.empty:
                score = target_series.iloc[0][metric]
                strat = f"Best Channel ({best_head}/{best_channel})"
            else:
                # Fallback 1: Chan not present. Use best head from simple triangulation.
                fallback_head = triangulation_best_head_map.get(group, "time")
                score = all_multi_metrics.get(fallback_head, {}).get(parent_file, 0)
                strat = f"Fallback-Triangulation ({fallback_head})"

        else:
            # Fallback 2: Group is unknown. Use the specified fallback head.
            score = all_multi_metrics.get(unknown_group_fallback_head, {}).get(
                parent_file, 0
            )
            strat = f"Fallback-Unknown Group ({unknown_group_fallback_head})"

        detailed_results.append(
            {
                "parent": parent_file,
                "group": group,
                "strategy": strat,
                metric: score,
            }
        )
        processed_parents.add(parent_file)

    if not detailed_results:
        return 0.0, pd.DataFrame(), best_strategy_map

    results_df = pd.DataFrame(detailed_results)
    final_score = results_df[metric].mean()

    return final_score, results_df, best_strategy_map


def generate_best_channel_eval_file(
    root_directory: str,
    metric: str,
    base_data_path: str,
    heads_to_load: Dict[str, str],
    output_filename: str,
):
    """
    Generates a new evaluation file by replacing multivariate files with their
    best performing univariate channel, based on tuning data.
    """
    print("--- Generating Best Channel Evaluation File ---")
    file_list_dir = os.path.join(base_data_path, "File_List")

    # 1. Load existing file lists
    try:
        tuning_files = set(
            pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M-Tuning.csv"))["file_name"]
        )
        eva_df = pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M-Eva.csv"))
        all_univariate_files = set(
            pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M-univariate.csv"))[
                "file_name"
            ]
        )
        print(f"Loaded {len(tuning_files)} tuning files, {len(eva_df)} eval files, and {len(all_univariate_files)} total univariate files.")
    except FileNotFoundError as e:
        print(f"Error: A required file list was not found. {e}")
        return

    # 2. Get the best channel map (reusing logic from Scenario 3)
    _, _, best_channel_map = compute_best_channel_by_avg_head_performance(
        root_directory=root_directory,
        metric=metric,
        split_files={"tuning": tuning_files, "eval": set()}, # only need tuning
        base_data_path=base_data_path,
        heads_to_load=heads_to_load,
    )

    if not best_channel_map:
        print("Error: Could not determine best channel map. Aborting file generation.")
        return

    print("\nLearned Best Channel per Group:")
    print(pd.Series(best_channel_map, name="best_channel"))

    # 3. Generate the new file list
    new_eval_files = []
    for _, row in eva_df.iterrows():
        original_file = row["file_name"]
        base_name = os.path.splitext(original_file)[0]
        try:
            group = base_name.split("_")[1]
        except IndexError:
            new_eval_files.append(original_file)
            continue

        best_channel = best_channel_map.get(group)

        if best_channel:
            # e.g., 174_Exathlon_id_1_...-node8-NET-ib0-read-KBs.csv
            potential_file = f"{base_name}-{best_channel}.csv"
            if potential_file in all_univariate_files:
                new_eval_files.append(potential_file)
                print(f"  - Group '{group}': Replaced '{original_file}' with '{potential_file}'")
            else:
                new_eval_files.append(original_file) # Fallback to original
                print(f"  - Group '{group}': Best channel file '{potential_file}' not found. Keeping original.")
        else:
            new_eval_files.append(original_file) # Fallback to original
            print(f"  - Group '{group}': No best channel learned. Keeping original.")


    # 4. Save the new file
    output_path = os.path.join(file_list_dir, output_filename)
    new_eval_df = pd.DataFrame({"file_name": new_eval_files})
    new_eval_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully generated new evaluation file at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run triangulation scoring.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="multi",
        choices=["uni", "multi"],
        help="Specify the dataset type to process: 'uni' or 'multi'.",
    )
    parser.add_argument(
        "--root_directory",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../eval/metrics"
        ),
        help="Specify the root directory where 'multi', 'uni', etc. metric folders are stored.",
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Datasets"),
        help="Specify the root directory of the Datasets, where 'File_List' is located.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="VUS-PR",
        help="AD metric to report, default VUS-PR.",
    )
    parser.add_argument(
        "--generate_file",
        action="store_true",
        help="If set, generates a new evaluation file based on the best channel strategy instead of running scoring."
    )
    args = parser.parse_args()

    # Define the two sets of heads for comparison
    zs_heads_only = {
        "ensemble": "TSPulse_ZS_ensemble.csv",
        "fft": "TSPulse_ZS_fft.csv",
        "forecast": "TSPulse_ZS_forecast.csv",
        "time": "TSPulse_ZS_time.csv",
    }

    zs_and_scaled_heads = {
        "ensemble": "TSPulse_ZS_ensemble.csv",
        "fft": "TSPulse_ZS_fft.csv",
        "forecast": "TSPulse_ZS_forecast.csv",
        "time": "TSPulse_ZS_time.csv",
        "scaled_ensemble": "TSPulse_ZS_scaled_ensemble.csv",  # Add the scaled ensemble
    }

    if args.generate_file:
        generate_best_channel_eval_file(
            root_directory=args.root_directory,
            metric=args.metric,
            base_data_path=args.data_directory,
            heads_to_load=zs_and_scaled_heads,
            output_filename="TSPulse2-M-Eva.csv",
        )
        exit()

    # Load file lists to determine splits
    split_files = load_split_files(args.data_directory, args.dataset_type)

    print("\n" + "=" * 80)
    print("SCENARIO 1: Scoring with Zero-Shot (ZS) Heads Only")
    print("=" * 80)

    # Calculate triangulated performance for the first scenario
    result_zs = triangulation_performance(
        root_directory=args.root_directory,
        metric=args.metric,
        split_files=split_files,
        dataset_type=args.dataset_type,
        heads_to_load=zs_heads_only,
    )

    print(
        f"\nTriangulation Results On Tuning Data ({args.dataset_type.upper()}) (Best Head per Group)"
    )
    print("-" * 60)
    print(result_zs["tuning"].sort_index())
    print("-" * 60)
    print(
        f"Triangulated {args.metric} ({args.dataset_type.upper()}) [ZS ONLY]: {result_zs['metric']:0.3f}\n\n"
    )

    print("=" * 80)
    print("SCENARIO 2: Scoring with ZS Heads + Scaled Ensemble")
    print("=" * 80)

    # Calculate triangulated performance for the second scenario
    result_scaled = triangulation_performance(
        root_directory=args.root_directory,
        metric=args.metric,
        split_files=split_files,
        dataset_type=args.dataset_type,
        heads_to_load=zs_and_scaled_heads,
    )

    print(
        f"\nTriangulation Results On Tuning Data ({args.dataset_type.upper()}) (Best Head per Group)"
    )
    print("-" * 60)
    print(result_scaled["tuning"].sort_index())
    print("-" * 60)
    print(
        f"Triangulated {args.metric} ({args.dataset_type.upper()}) [ZS + SCALED]: {result_scaled['metric']:0.3f}\n\n"
    )

    # Scenario 3: Best Channel Selection (only for multivariate)
    if args.dataset_type == "multi":
        print("=" * 80)
        print("SCENARIO 3: Best Channel by Group (Learned on Avg Head Performance)")
        print("=" * 80)

        (
            final_scores,
            best_channel_details,
            best_channel_map,
        ) = compute_best_channel_by_avg_head_performance(
            root_directory=args.root_directory,
            metric=args.metric,
            split_files=split_files,
            base_data_path=args.data_directory,
            heads_to_load=zs_and_scaled_heads,  # Use same heads as S2 for consistency
        )

        if final_scores is not None and not final_scores.empty:
            print("Learned Best Channel per Group (from avg head performance):")
            print(pd.Series(best_channel_map, name="selected_channel"))

            if best_channel_details is not None and not best_channel_details.empty:
                print("\n--- Detailed Results for Scenario 3 ---")
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    1000,
                ):
                    print(best_channel_details.set_index("parent").sort_index())

            print("\n--- Final Scores by Head (Best Channel Strategy) ---")
            print(final_scores.to_string(float_format="%.3f"))

        # SCENARIO 4: Best Head and Channel Strategy with Fallback Experiments
        print("\n" + "=" * 80)
        print(
            "SCENARIO 4: Best Head/Channel with Fallback Experiments for Unknown Groups"
        )
        print("=" * 80)

        fallback_heads_to_test = list(zs_and_scaled_heads.keys())
        s4_results_list = []
        triangulation_best_heads = result_scaled["tuning"]["best"].to_dict()

        # We only need to print the learned strategy map once, as it's the same
        # for all fallback experiments. Run once just to get the map.
        _, _, s4_map = compute_best_head_and_channel_strategy(
            root_directory=args.root_directory,
            metric=args.metric,
            split_files=split_files,
            base_data_path=args.data_directory,
            heads_to_load=zs_and_scaled_heads,
            unknown_group_fallback_head=fallback_heads_to_test[
                0
            ],  # Dummy for first run
            triangulation_best_head_map=triangulation_best_heads,
        )

        if s4_map:
            print(
                "Learned Best (Head, Channel) Strategy per Group (used across all S4 experiments):"
            )
            pretty_map = {
                k: f"{v['head']} / {v['channel_name']}" for k, v in s4_map.items()
            }
            print(pd.Series(pretty_map, name="best_strategy"))
            print("-" * 80)

        # Now, run for each fallback head and gather results
        for i, fallback_head in enumerate(fallback_heads_to_test):
            score, details, _ = compute_best_head_and_channel_strategy(
                root_directory=args.root_directory,
                metric=args.metric,
                split_files=split_files,
                base_data_path=args.data_directory,
                heads_to_load=zs_and_scaled_heads,
                unknown_group_fallback_head=fallback_head,
                triangulation_best_head_map=triangulation_best_heads,
            )

            if score is not None:
                s4_results_list.append(
                    {"fallback_head": fallback_head, "score": score}
                )

        print("\nSummary of Scenario 4: Final Scores by Fallback Head")
        summary_df = pd.DataFrame(s4_results_list).set_index("fallback_head")
        print(summary_df.to_string(float_format="%.3f"))
        print("=" * 80)
