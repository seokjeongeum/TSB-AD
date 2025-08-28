# Copyright contributors to the TSFM project
#

import argparse
import os
import re
import sys
from typing import Dict, Set, cast, Optional

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


def evaluate_best_head_strategy(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    dataset_type: str,
    heads_to_load: Dict[str, str],
    fallback_head: Optional[str] = None,
):
    """
    Evaluates the 'best head' strategy (formerly triangulation).

    1. Loads all ZS metric files from multi, multi_as_uni, and uni directories.
    2. Combines them into a single DataFrame.
    3. Splits the data into tuning and evaluation sets based on provided file lists.
    4. Determines the best head for each data group from the tuning set.
    5. Calculates the final score on the evaluation set by applying this strategy.
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
        raise ValueError(
            "Tuning set is empty after processing. Cannot determine best head strategy."
        )

    # 6. Perform best-head selection
    cols = tuning_performance.columns.tolist()
    tuning_performance["best"] = [
        cols[c] for c in np.argmax(tuning_performance.values, axis=1)
    ]

    # Determine a global best head from the tuning set to use as a fallback
    # if a specific one is not provided.
    final_fallback_head = fallback_head
    if final_fallback_head is None and not tuning_performance.empty:
        final_fallback_head = tuning_performance[cols].mean().idxmax()

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
            else final_fallback_head
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
            series_score = (
                best_available_score if not pd.isna(best_available_score) else 0
            )

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


def _generate_s2_detailed_results(
    df: pd.DataFrame,
    group_to_best_channel_map: Dict[str, str],
    uni_pivoted_df: pd.DataFrame,
    multi_pivoted_df: pd.DataFrame,
    metric: str,
    heads_to_load: Dict[str, str],
) -> pd.DataFrame:
    """Generates a detailed per-series result DataFrame for Scenario 2."""
    detailed_results = []
    processed_parents = set()

    # Use a copy of the incoming df to avoid SettingWithCopyWarning
    df = df.copy()

    for parent_file in pd.Series(df["parent"].unique()):
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
            target_series_df = df[
                (df["parent"] == parent_file)
                & (df["channel_name"] == best_channel)
            ]
            if not target_series_df.empty:
                uni_filename = target_series_df.iloc[0]["file"]
                if uni_filename in uni_pivoted_df.index:
                    scores_row = uni_pivoted_df.loc[uni_filename]
                    strat = f"Best Channel ({best_channel})"

            # Fallback for best channel if it wasn't found in the current dataset split
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
                # The final metric is the average of the selected series' head scores.
                metric: scores_row.mean(),
            }
            # Add all individual head scores for detailed breakdown
            for head in heads_to_load.keys():
                record[head] = scores_row.get(head)

            detailed_results.append(record)

        processed_parents.add(parent_file)

    if not detailed_results:
        return pd.DataFrame()

    return pd.DataFrame(detailed_results)


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
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), {}

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

    def get_parent_map(univariate_files, parents) -> Dict[str, str]:
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
    combined_df["parent"] = combined_df["file"].map(uni_to_multi_map.get)
    combined_df["parent_base"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
    )
    combined_df["channel_name"] = combined_df.apply(
        lambda row: (
            os.path.splitext(row["file"])[0][len(row["parent_base"]) + 1 :]
            if pd.notna(row["parent_base"])
            else None
        ),
        axis=1,
    )
    combined_df["group"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0].split("_")[1] if pd.notna(x) else None
    )
    print(
        f"DEBUG: Found parents for {len(combined_df) - combined_df['parent'].isna().sum()} of {len(combined_df)} files."
    )
    print(
        f"DEBUG: Found groups for {len(combined_df) - combined_df['group'].isna().sum()} of {len(combined_df)} files."
    )
    rows_before_drop = len(combined_df)
    combined_df = combined_df.dropna(subset=["parent", "group", "channel_name"])
    print(
        f"DEBUG: Dropped {rows_before_drop - len(combined_df)} rows due to missing parent/group/channel info."
    )

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
        multi_pivoted_df = pd.DataFrame(columns=pd.Index(heads_to_load.keys())).astype(
            {k: float for k in heads_to_load}
        )
    else:
        multi_pivoted_df = pd.concat(all_multi_metrics, axis=1, join="outer")

    # 4. Split into tuning and evaluation sets
    tuning_df = combined_df[
        combined_df["parent"].isin(list(split_files["tuning"]))
    ].copy()
    eval_df = combined_df[combined_df["parent"].isin(list(split_files["eval"]))].copy()
    assert isinstance(eval_df, pd.DataFrame)

    if tuning_df.empty:
        print("Warning: Tuning set for this strategy is empty. Skipping.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), {}

    # 5. Determine best channel NAME for each DATASET GROUP on the tuning set (manual implementation)
    # This is a workaround for a suspected bug in pandas.groupby that causes a hard crash.
    print("--- Manually calculating group scores (workaround) ---")

    group_scores = {}  # {(group, channel_name): [scores]}
    for _, row in tuning_df.iterrows():
        # Ensure avg_score is a valid float, skip if not
        try:
            score = float(row["avg_score"])
            if pd.isna(score):
                continue
        except (ValueError, TypeError):
            continue

        key = (row["group"], row["channel_name"])
        if key not in group_scores:
            group_scores[key] = []
        group_scores[key].append(score)

    # Calculate the mean score for each group/channel
    mean_scores = {
        key: sum(scores) / len(scores) for key, scores in group_scores.items()
    }

    # Format and print the tuning data as requested
    print("\n--- Tuning Data: Avg Head Performance per Channel (S2) ---")
    printable_scores = pd.DataFrame(
        [(k[0], k[1], v) for k, v in mean_scores.items()],
        columns=["group", "channel", "avg_score_across_heads"],
    )
    # Sort by group, then by score descending to see best channels per group
    printable_scores = printable_scores.sort_values(
        by=["group", "avg_score_across_heads"], ascending=[True, False]
    ).set_index(["group", "channel"])

    with pd.option_context("display.max_rows", None):
        print(printable_scores.to_string(float_format="{:.16f}".format))
    print("----------------------------------------------------\n")

    # Find the best channel for each group
    best_channels = {}  # {group: (channel_name, best_score)}
    for (group, channel), avg_score in mean_scores.items():
        if group not in best_channels or avg_score > best_channels[group][1]:
            best_channels[group] = (channel, avg_score)

    # Final map from group to best channel name
    group_to_best_channel_map = {
        group: channel_info[0] for group, channel_info in best_channels.items()
    }

    print(
        f"Manual calculation complete. Found best channels for {len(group_to_best_channel_map)} groups."
    )

    # 6. Apply strategy to get detailed results for both sets
    eval_details_df = _generate_s2_detailed_results(
        df=eval_df,
        group_to_best_channel_map=group_to_best_channel_map,
        uni_pivoted_df=uni_pivoted_df,
        multi_pivoted_df=multi_pivoted_df,
        metric=metric,
        heads_to_load=heads_to_load,
    )

    tuning_details_df = _generate_s2_detailed_results(
        df=tuning_df,
        group_to_best_channel_map=group_to_best_channel_map,
        uni_pivoted_df=uni_pivoted_df,
        multi_pivoted_df=multi_pivoted_df,
        metric=metric,
        heads_to_load=heads_to_load,
    )

    if eval_details_df.empty:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            tuning_details_df,
            group_to_best_channel_map,
        )

    # Calculate final score for each head by taking the mean of their respective columns
    head_columns = list(heads_to_load.keys())
    final_scores_by_head = eval_details_df[head_columns].mean()
    assert isinstance(final_scores_by_head, pd.Series)
    # Also add the main metric (average of best) to the series for reference
    final_scores_by_head[metric] = eval_details_df[metric].mean()

    return (
        final_scores_by_head,
        eval_details_df,
        tuning_details_df,
        group_to_best_channel_map,
    )


def compute_best_head_and_channel_strategy(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    base_data_path: str,
    heads_to_load: Dict[str, str],
    unknown_group_fallback_head: str,
    s1_best_head_map: Dict[str, str],
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
        print("\nWarning: No 'multi_as_uni' metric files found. Skipping Scenario 3.")
        return 0.0, pd.DataFrame(), {}

    combined_df = pd.concat(all_uni_metrics, ignore_index=True)

    # 2. Add parent, group, and channel info
    multi_file_list_path = os.path.join(base_data_path, "File_List", "TSB-AD-M.csv")
    multi_full_df = pd.read_csv(multi_file_list_path)
    multi_base_names = sorted(
        [os.path.splitext(f)[0] for f in multi_full_df["file_name"]],
        key=len,
        reverse=True,
    )

    def get_parent_map(univariate_files, parents) -> Dict[str, str]:
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
    combined_df["parent"] = combined_df["file"].map(uni_to_multi_map.get)
    combined_df["parent_base"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
    )
    combined_df["channel_name"] = combined_df.apply(
        lambda row: (
            os.path.splitext(row["file"])[0][len(row["parent_base"]) + 1 :]
            if pd.notna(row["parent_base"])
            else None
        ),
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
    assert isinstance(eval_df, pd.DataFrame)

    if tuning_df.empty:
        print("Warning: Tuning set is empty for Scenario 3. Skipping.")
        return 0.0, pd.DataFrame(), {}

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

    for parent_file in pd.Series(eval_df["parent"].unique()):
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
            assert isinstance(target_series, pd.DataFrame)
            if not target_series.empty:
                score = target_series.iloc[0][metric]
                strat = f"Best Channel ({best_head}/{best_channel})"
            else:
                # Fallback 1: Chan not present. Use best head from Scenario 1.
                fallback_head = s1_best_head_map.get(group, "time")
                score = all_multi_metrics.get(fallback_head, {}).get(parent_file, 0)
                strat = f"Fallback-S1 Best Head ({fallback_head})"

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
        return 0.0, pd.DataFrame(), {}

    results_df = pd.DataFrame(detailed_results)
    final_score = results_df[metric].mean()

    return final_score, results_df, best_strategy_map


def compute_best_channel_by_max_head_performance(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    base_data_path: str,
    heads_to_load: Dict[str, str],
):
    """
    Computes performance by:
    1. Learning a 'best channel by group' strategy from the tuning set, where 'best'
       is determined by the highest maximum score across all heads.
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
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), {}

    uni_pivoted_df = pd.concat(all_uni_metrics, axis=1, join="outer")
    uni_pivoted_df["max_score"] = uni_pivoted_df[list(heads_to_load.keys())].max(axis=1)
    combined_df = uni_pivoted_df.reset_index()

    # 2. Add parent, group, and channel info
    multi_file_list_path = os.path.join(base_data_path, "File_List", "TSB-AD-M.csv")
    multi_full_df = pd.read_csv(multi_file_list_path)
    multi_base_names = sorted(
        [os.path.splitext(f)[0] for f in multi_full_df["file_name"]],
        key=len,
        reverse=True,
    )

    def get_parent_map(univariate_files, parents) -> Dict[str, str]:
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
    combined_df["parent"] = combined_df["file"].map(uni_to_multi_map.get)
    combined_df["parent_base"] = combined_df["parent"].apply(
        lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
    )
    combined_df["channel_name"] = combined_df.apply(
        lambda row: (
            os.path.splitext(row["file"])[0][len(row["parent_base"]) + 1 :]
            if pd.notna(row["parent_base"])
            else None
        ),
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
        multi_pivoted_df = pd.DataFrame(columns=pd.Index(heads_to_load.keys())).astype(
            {k: float for k in heads_to_load}
        )
    else:
        multi_pivoted_df = pd.concat(all_multi_metrics, axis=1, join="outer")

    # 4. Split into tuning and evaluation sets
    tuning_df = combined_df[
        combined_df["parent"].isin(list(split_files["tuning"]))
    ].copy()
    eval_df = combined_df[combined_df["parent"].isin(list(split_files["eval"]))].copy()
    assert isinstance(eval_df, pd.DataFrame)

    if tuning_df.empty:
        print("Warning: Tuning set for this strategy is empty. Skipping.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), {}

    # 5. Determine best channel NAME for each DATASET GROUP on the tuning set
    group_channel_scores = (
        tuning_df.groupby(["group", "channel_name"])["max_score"].mean().reset_index()
    )
    best_channels_df = group_channel_scores.loc[
        group_channel_scores.groupby("group")["max_score"].idxmax()
    ]
    # Format and print the tuning data as requested
    print("\n--- Tuning Data: Avg of Max Head Performance per Channel (S2B) ---")
    printable_scores = group_channel_scores.rename(
        columns={"max_score": "avg_of_max_scores"}
    )
    # Sort by group, then by score descending to see best channels per group
    printable_scores = printable_scores.sort_values(
        by=["group", "avg_of_max_scores"], ascending=[True, False]
    ).set_index(["group", "channel_name"])
    with pd.option_context("display.max_rows", None):
        print(printable_scores.to_string(float_format="{:.16f}".format))
    print("-----------------------------------------------------------\n")
    group_to_best_channel_map = best_channels_df.set_index("group")[
        "channel_name"
    ].to_dict()

    # 6. Apply strategy to get detailed results for both sets
    eval_details_df = _generate_s2_detailed_results(
        df=eval_df,
        group_to_best_channel_map=group_to_best_channel_map,
        uni_pivoted_df=uni_pivoted_df,
        multi_pivoted_df=multi_pivoted_df,
        metric=metric,
        heads_to_load=heads_to_load,
    )

    tuning_details_df = _generate_s2_detailed_results(
        df=tuning_df,
        group_to_best_channel_map=group_to_best_channel_map,
        uni_pivoted_df=uni_pivoted_df,
        multi_pivoted_df=multi_pivoted_df,
        metric=metric,
        heads_to_load=heads_to_load,
    )

    if eval_details_df.empty:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            tuning_details_df,
            group_to_best_channel_map,
        )

    # Calculate final score for each head by taking the mean of their respective columns
    head_columns = list(heads_to_load.keys())
    final_scores_by_head = eval_details_df[head_columns].mean()
    assert isinstance(final_scores_by_head, pd.Series)
    # Also add the main metric (average of best) to the series for reference
    final_scores_by_head[metric] = eval_details_df[metric].mean()

    return (
        final_scores_by_head,
        eval_details_df,
        tuning_details_df,
        group_to_best_channel_map,
    )


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
        # We need the main file list to iterate through all original multi files.
        multi_df = pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M.csv"))
        all_univariate_files = set(
            pd.read_csv(os.path.join(file_list_dir, "TSB-AD-M-univariate.csv"))[
                "file_name"
            ]
        )
        print(
            f"Loaded {len(tuning_files)} tuning files, {len(multi_df)} eval files, and {len(all_univariate_files)} total univariate files."
        )
    except FileNotFoundError as e:
        print(f"Error: A required file list was not found. {e}")
        return

    # 2. Get the best channel map using the 'avg' performance strategy.
    # We only care about the map, so we ignore the other return values.
    (
        _,
        _,
        _,
        best_channel_map_avg,
    ) = compute_best_channel_by_avg_head_performance(
        root_directory=root_directory,
        metric=metric,
        split_files={"tuning": tuning_files, "eval": set()},  # Only need tuning data
        base_data_path=base_data_path,
        heads_to_load=heads_to_load,
    )

    if not best_channel_map_avg:
        print(
            "Error: Could not determine best channel map using AVG strategy. Aborting file generation."
        )
        return

    print("\nLearned Best Channel per Group (using AVG head performance):")
    print(pd.Series(best_channel_map_avg, name="best_channel"))

    # 3. Generate the new file list
    new_eval_files = []
    for _, row in multi_df.iterrows():
        original_file = cast(str, row["file_name"])
        base_name = os.path.splitext(original_file)[0]
        try:
            group = base_name.split("_")[1]
        except IndexError:
            # If we can't get a group, keep the original file
            new_eval_files.append(original_file)
            continue

        # Look up the best channel for this group from our learned map
        best_channel = best_channel_map_avg.get(group)

        if best_channel:
            # Construct the potential filename for the best univariate channel
            potential_file = f"{base_name}-{best_channel}.csv"
            if potential_file in all_univariate_files:
                new_eval_files.append(potential_file)
                print(
                    f"  - Group '{group}': Replaced '{original_file}' with '{potential_file}'"
                )
            else:
                # If the specific univariate file doesn't exist, fall back to the original
                new_eval_files.append(original_file)
                print(
                    f"  - Group '{group}': Best channel file '{potential_file}' not found. Keeping original."
                )
        else:
            # If the group wasn't in our tuning map, keep the original file
            new_eval_files.append(original_file)
            print(f"  - Group '{group}': No best channel learned. Keeping original.")

    # 4. Save the new file
    output_path = os.path.join(file_list_dir, output_filename)
    new_eval_df = pd.DataFrame({"file_name": new_eval_files})
    new_eval_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully generated new evaluation file at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model strategy analysis.")
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
        help="If set, generates a new evaluation file based on the best channel strategy instead of running scoring.",
    )
    args = parser.parse_args()

    # Define the heads for comparison
    zs_heads_only = {
        "ensemble": "TSPulse_ZS_ensemble.csv",
        "fft": "TSPulse_ZS_fft.csv",
        "forecast": "TSPulse_ZS_forecast.csv",
        "time": "TSPulse_ZS_time.csv",
    }

    # Redirect output to a fixed file in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create filenames based on dataset type to avoid overwrites
    base_output_name = f"analysis_{args.dataset_type}"
    output_filename = os.path.join(script_dir, f"{base_output_name}_summary.txt")
    s1_csv_path = os.path.join(script_dir, f"{base_output_name}_s1_details.csv")
    s2_csv_path = os.path.join(script_dir, f"{base_output_name}_s2_details.csv")
    s2_tuning_csv_path = os.path.join(
        script_dir, f"{base_output_name}_s2_tuning_details.csv"
    )
    s3_csv_path = os.path.join(script_dir, f"{base_output_name}_s3_details.csv")
    output_file_handle = None
    original_stdout = sys.stdout
    # Print to console before redirection
    print(f"All subsequent output will be written to {output_filename}")
    output_file_handle = open(output_filename, "w")
    sys.stdout = output_file_handle

    try:
        if args.generate_file:
            generate_best_channel_eval_file(
                root_directory=args.root_directory,
                metric=args.metric,
                base_data_path=args.data_directory,
                heads_to_load=zs_heads_only,
                output_filename="TSPulse2-M.csv",
            )
            exit()

        # Load file lists to determine splits
        split_files = load_split_files(args.data_directory, args.dataset_type)

        print("\n" + "=" * 80)
        print("SCENARIO 1: Best Head Strategy")
        print("=" * 80)

        # Calculate performance for the 'best head' strategy
        result_s1 = evaluate_best_head_strategy(
            root_directory=args.root_directory,
            metric=args.metric,
            split_files=split_files,
            dataset_type=args.dataset_type,
            heads_to_load=zs_heads_only,
            fallback_head="time",
        )

        print(
            f"\nBest Head Strategy Results On Tuning Data ({args.dataset_type.upper()}) (Best Head per Group)"
        )
        print("-" * 60)
        print(result_s1["tuning"].sort_index().to_string(float_format="{:.16f}".format))
        print("-" * 60)
        print(
            f"Final Score ({args.dataset_type.upper()}) [S1 - Best Head]: {result_s1['metric']}\n\n"
        )
        if (
            "detailed_evaluation" in result_s1
            and not result_s1["detailed_evaluation"].empty
        ):
            result_s1["detailed_evaluation"].to_csv(s1_csv_path)
            print(f"--> Scenario 1 detailed results saved to {s1_csv_path}")

        # Scenario 2 & 2B: Best Channel Selection (only for multivariate)
        if args.dataset_type == "multi":            
            # --- Run both scenarios first ---
            (
                final_scores_avg,
                details_avg,
                tuning_details_avg,
                map_avg,
            ) = compute_best_channel_by_avg_head_performance(
                root_directory=args.root_directory,
                metric=args.metric,
                split_files=split_files,
                base_data_path=args.data_directory,
                heads_to_load=zs_heads_only,
            )
            (
                final_scores_max,
                details_max,
                tuning_details_max,
                map_max,
            ) = compute_best_channel_by_max_head_performance(
                root_directory=args.root_directory,
                metric=args.metric,
                split_files=split_files,
                base_data_path=args.data_directory,
                heads_to_load=zs_heads_only,
            )

            print("=" * 80)
            print(
                "SCENARIO 2 & 2B: Best Channel by Group (Avg vs Max Head Performance)"
            )
            print("=" * 80)

            # --- 1. Print learned strategies ---
            if map_avg or map_max:
                s2_map_df = pd.Series(map_avg, name="S2_Avg_Perf_Channel").to_frame()
                s2b_map_df = pd.Series(map_max, name="S2B_Max_Perf_Channel").to_frame()
                combined_maps = s2_map_df.join(s2b_map_df, how="outer").sort_index()
                print("Learned Best Channel per Group (Side-by-Side):")
                print(combined_maps.to_string(float_format="{:.16f}".format))

            # --- 2. Print detailed results ---
            if (
                details_avg is not None
                and not details_avg.empty
                and details_max is not None
                and not details_max.empty
            ):
                s2_details = details_avg.rename(
                    columns=lambda c: f"S2_{c}" if c not in ["parent", "group"] else c
                )
                s2b_details = details_max.rename(
                    columns=lambda c: f"S2B_{c}" if c not in ["parent", "group"] else c
                )

                combined_details = pd.merge(
                    s2_details, s2b_details, on=["parent", "group"], how="outer"
                )
                combined_details.to_csv(s2_csv_path, index=False)
                print(f"--> Scenario 2/2B detailed results saved to {s2_csv_path}")

                print("\n--- Detailed Per-Series Results (Scenario 2/2B) ---")
                with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
                    print(combined_details.to_string())

            if (
                tuning_details_avg is not None
                and not tuning_details_avg.empty
                and tuning_details_max is not None
                and not tuning_details_max.empty
            ):
                s2_tuning_details = tuning_details_avg.rename(
                    columns=lambda c: f"S2_{c}" if c not in ["parent", "group"] else c
                )
                s2b_tuning_details = tuning_details_max.rename(
                    columns=lambda c: f"S2B_{c}" if c not in ["parent", "group"] else c
                )
                combined_tuning_details = pd.merge(
                    s2_tuning_details,
                    s2b_tuning_details,
                    on=["parent", "group"],
                    how="outer",
                )
                combined_tuning_details.to_csv(s2_tuning_csv_path, index=False)
                print(
                    f"--> Scenario 2/2B detailed TUNING results saved to {s2_tuning_csv_path}"
                )

            # --- 3. Print final scores ---
            if (
                final_scores_avg is not None
                and not final_scores_avg.empty
                and final_scores_max is not None
                and not final_scores_max.empty
            ):
                summary_scores = pd.DataFrame(
                    {"S2_Avg_Perf": final_scores_avg, "S2B_Max_Perf": final_scores_max}
                )
                print("\n--- Final Scores by Head (Side-by-Side) ---")
                print(summary_scores.to_string(float_format="{:.16f}".format))

                s1_score = result_s1["metric"]
                s2_score = final_scores_avg.get(args.metric, 0)
                s2b_score = final_scores_max.get(args.metric, 0)
                print(f"\nS2 vs S1:  {s2_score} - {s1_score} = {s2_score - s1_score}")
                print(f"S2B vs S1: {s2b_score} - {s1_score} = {s2b_score - s1_score}")

        # SCENARIO 3: Best Head/Channel with Fallback Experiments for Unknown Groups
        print("\n" + "=" * 80)
        print(
            "SCENARIO 3: Best Head/Channel with Fallback Experiments for Unknown Groups"
        )
        print("=" * 80)

        fallback_heads_to_test = list(zs_heads_only.keys())
        s3_results_list = []
        s3_all_details = []
        s1_best_head_map = result_s1["tuning"]["best"].to_dict()

        # We only need to print the learned strategy map once, as it's the same
        # for all fallback experiments. Run once just to get the map.
        _, _, s3_map = compute_best_head_and_channel_strategy(
            root_directory=args.root_directory,
            metric=args.metric,
            split_files=split_files,
            base_data_path=args.data_directory,
            heads_to_load=zs_heads_only,
            unknown_group_fallback_head=fallback_heads_to_test[
                0
            ],  # Dummy for first run
            s1_best_head_map=s1_best_head_map,
        )

        if s3_map:
            print(
                "Learned Best (Head, Channel) Strategy per Group (used across all S3 experiments):"
            )
            pretty_map = {
                k: f"{v['head']} / {v['channel_name']}" for k, v in s3_map.items()
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
                heads_to_load=zs_heads_only,
                unknown_group_fallback_head=fallback_head,
                s1_best_head_map=s1_best_head_map,
            )

            if score is not None:
                s3_results_list.append({"fallback_head": fallback_head, "score": score})
                if details is not None and not details.empty:
                    details["fallback_head_tested"] = fallback_head
                    s3_all_details.append(details)

        if s3_all_details:
            s3_combined_details_df = pd.concat(s3_all_details, ignore_index=True)
            s3_combined_details_df.to_csv(s3_csv_path, index=False)
            print(f"--> Scenario 3 detailed results saved to {s3_csv_path}")

        print("\nSummary of Scenario 3: Final Scores by Fallback Head")
        summary_df = pd.DataFrame(s3_results_list).set_index("fallback_head")
        summary_df["diff_vs_s1"] = summary_df["score"] - result_s1["metric"]
        print(summary_df.to_string(float_format="{:.16f}".format))
        print("=" * 80)
    finally:
        # Restore stdout and close the file handle
        if output_file_handle:
            sys.stdout = original_stdout
            output_file_handle.close()
            # Print to console after restoration
            print(f"Output successfully written to {output_filename}")
