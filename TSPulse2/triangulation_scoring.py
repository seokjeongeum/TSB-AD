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
        df = df.rename(columns={metric: head})
        df = df[["file", head]]
        df = df.set_index("file")
        if unified_df is None:
            unified_df = df
        else:
            unified_df = unified_df.join(df, how="outer")

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
            print(
                f"Warning: No score for selected head '{sel_mode}' in file '{file_name}'. Defaulting to 'time' head."
            )
            series_score = row.get("time", 0)

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


def compute_best_channel_performance(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    base_data_path: str,
    head_to_evaluate: str,
    head_filename: str,
):
    """
    Computes performance for a single head by:
    1. Learning a 'best channel by group' strategy from the tuning set.
    2. Applying this strategy to evaluation series from groups seen in tuning.
    3. Applying a fallback (using the multivariate head score) for evaluation
       series from groups NOT seen in tuning (e.g., Genesis).
    """
    # 1. Load multi_as_uni data for the specific head
    metric_dir = os.path.join(root_directory, "multi_as_uni")
    metric_file_path = os.path.join(metric_dir, head_filename)

    if not os.path.exists(metric_file_path):
        print(
            f"\nWarning: Metric file not found for head '{head_to_evaluate}'. Skipping."
        )
        return None, None, None

    try:
        df = pd.read_csv(metric_file_path)
        df["file"] = df["file"].apply(
            lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
        )
    except Exception as e:
        print(f"Warning: Could not load or process {metric_file_path}. Error: {e}")
        return None, None, None

    # 2. Load corresponding multivariate data for the fallback
    multi_metric_path = os.path.join(root_directory, "multi", head_filename)
    if not os.path.exists(multi_metric_path):
        print(
            f"Warning: Multivariate metric file for fallback not found for head '{head_to_evaluate}'. Fallback will fail."
        )
        multi_metrics_df = pd.DataFrame(columns=["file", metric]).set_index("file")
    else:
        multi_metrics_df = pd.read_csv(multi_metric_path)
        multi_metrics_df["file"] = multi_metrics_df["file"].apply(
            lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
        )
        multi_metrics_df = multi_metrics_df.set_index("file")

    # 3. Map to parents and extract group/channel names
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

    uni_to_multi_map = get_parent_map(df["file"], multi_base_names)
    df["parent"] = df["file"].map(uni_to_multi_map)
    df["parent_base"] = df["parent"].apply(
        lambda x: os.path.splitext(x)[0] if pd.notna(x) else None
    )
    df["channel_name"] = df.apply(
        lambda row: os.path.splitext(row["file"])[0][len(row["parent_base"]) + 1 :]
        if pd.notna(row["parent_base"])
        else None,
        axis=1,
    )
    df["group"] = df["parent"].apply(
        lambda x: os.path.splitext(x)[0].split("_")[1] if pd.notna(x) else None
    )
    df = df.dropna(subset=["parent", "group", "channel_name"])

    # 4. Split into tuning and evaluation sets
    tuning_df = df[df["parent"].isin(list(split_files["tuning"]))].copy()
    eval_df = df[df["parent"].isin(list(split_files["eval"]))].copy()

    if tuning_df.empty:
        print(
            f"Warning: Tuning set for head '{head_to_evaluate}' is empty. Skipping."
        )
        return None, None, None

    # 5. Determine best channel NAME for each DATASET GROUP on the tuning set
    group_channel_scores = (
        tuning_df.groupby(["group", "channel_name"])[metric].mean().reset_index()
    )
    best_channels_df = group_channel_scores.loc[
        group_channel_scores.groupby("group")[metric].idxmax()
    ]
    group_to_best_channel_map = (
        best_channels_df.set_index("group")["channel_name"].to_dict()
    )

    # 6. Apply strategy and handle fallbacks on the evaluation set
    eval_df["best_channel_for_group"] = eval_df["group"].map(group_to_best_channel_map)

    # Part 1: Apply the learned rule
    eval_with_rule_df = eval_df.dropna(subset=["best_channel_for_group"])
    selected_channels_df = eval_with_rule_df[
        eval_with_rule_df["channel_name"] == eval_with_rule_df["best_channel_for_group"]
    ]

    # Part 2: Handle groups with no rule (fallback)
    eval_without_rule_df = eval_df[eval_df["best_channel_for_group"].isna()]
    parents_to_fallback = eval_without_rule_df["parent"].unique()
    fallback_scores_df = multi_metrics_df.loc[
        multi_metrics_df.index.isin(parents_to_fallback)
    ]

    # 7. Combine results and calculate final score
    scores_from_rule = selected_channels_df[metric]
    scores_from_fallback = fallback_scores_df[metric]

    all_scores = pd.concat([scores_from_rule, scores_from_fallback])
    final_score = all_scores.mean() if not all_scores.empty else 0.0

    # 8. Prepare detailed results for printing
    details_from_rule = selected_channels_df[
        ["parent", "channel_name", metric]
    ].rename(columns={"channel_name": "strategy_or_channel"})
    details_from_fallback = (
        fallback_scores_df[[metric]]
        .reset_index()
        .rename(columns={"file": "parent"})
    )
    details_from_fallback["strategy_or_channel"] = "multivariate_fallback"

    final_details_df = pd.concat(
        [details_from_rule, details_from_fallback], ignore_index=True
    )

    return final_score, final_details_df, group_to_best_channel_map


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
                df["file"] = df["file"].apply(
                    lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
                )
                df["head"] = head
                all_uni_metrics.append(df)
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
            df = pd.read_csv(fpath)
            df["file"] = df["file"].apply(
                lambda x: x if str(x).endswith(".csv") else f"{x}.csv"
            )
            all_multi_metrics[head] = df.set_index("file")[metric]

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
    args = parser.parse_args()

    # Define the two sets of heads for comparison
    zs_heads_only = {
        "ensemble": "TSPulse_ZS_ensemble.csv",
        "fft": "TSPulse_ZS_fft.csv",
        "future": "TSPulse_ZS_future.csv",
        "time": "TSPulse_ZS_time.csv",
    }

    zs_and_scaled_heads = {
        "ensemble": "TSPulse_ZS_ensemble.csv",
        "fft": "TSPulse_ZS_fft.csv",
        "future": "TSPulse_ZS_future.csv",
        "time": "TSPulse_ZS_time.csv",
        "scaled_ensemble": "TSPulse2.csv",  # Add the scaled ensemble
    }

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
        print("SCENARIO 3: Best Channel by Group Strategy (Learned on Tuning Set)")
        print("=" * 80)

        heads_for_scenario_3 = {
            "ensemble": "TSPulse_ZS_ensemble.csv",
            "fft": "TSPulse_ZS_fft.csv",
            "future": "TSPulse_ZS_future.csv",
            "time": "TSPulse_ZS_time.csv",
            "scaled_ensemble": "TSPulse_ZS_scaled_ensemble.csv",
        }

        scenario_3_results = []

        for head_name, head_file in heads_for_scenario_3.items():
            print(f"\n--- Evaluating for Head: {head_name.upper()} ---")

            (
                best_channel_score,
                best_channel_details,
                best_channel_map,
            ) = compute_best_channel_performance(
                root_directory=args.root_directory,
                metric=args.metric,
                split_files=split_files,
                base_data_path=args.data_directory,
                head_to_evaluate=head_name,
                head_filename=head_file,
            )

            if best_channel_score is not None:
                print("Learned Best Channel per Group:")
                print(pd.Series(best_channel_map, name="selected_channel"))

                # if best_channel_details is not None and not best_channel_details.empty:
                #     print("\n--- Details of Evaluation Set Files Used ---")
                #     with pd.option_context(
                #         "display.max_rows",
                #         None,
                #         "display.max_columns",
                #         None,
                #         "display.width",
                #         1000,
                #     ):
                #         print(best_channel_details)

                print(
                    f"\nFinal Score (Best Channel Strategy): {best_channel_score:0.3f}"
                )
                scenario_3_results.append(
                    {"head": head_name, "score": best_channel_score}
                )
            else:
                scenario_3_results.append({"head": head_name, "score": float("nan")})

        print("\n" + "=" * 80)
        print("Summary of Best Channel by Group Strategy")
        print("=" * 80)
        summary_df = pd.DataFrame(scenario_3_results).set_index("head")
        print(summary_df.to_string(float_format="%.3f"))
        print("=" * 80)

        # SCENARIO 4: Best Head and Channel Strategy with Fallback Experiments
        print("\n" + "=" * 80)
        print("SCENARIO 4: Best Head/Channel with Fallback Experiments for Unknown Groups")
        print("=" * 80)

        fallback_heads_to_test = list(heads_for_scenario_3.keys())
        s4_results_list = []
        triangulation_best_heads = result_scaled["tuning"]["best"].to_dict()

        # We only need to print the learned strategy map once, as it's the same
        # for all fallback experiments. Run once just to get the map.
        _, _, s4_map = compute_best_head_and_channel_strategy(
            root_directory=args.root_directory,
            metric=args.metric,
            split_files=split_files,
            base_data_path=args.data_directory,
            heads_to_load=heads_for_scenario_3,
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
                heads_to_load=heads_for_scenario_3,
                unknown_group_fallback_head=fallback_head,
                triangulation_best_head_map=triangulation_best_heads,
            )

            if score is not None:
                s4_results_list.append(
                    {"fallback_head": fallback_head, "score": score}
                )
                # Print detailed results only for the first experiment run
                if i == 0 and details is not None and not details.empty:
                    print(f"\n--- Details for First Fallback Head: {fallback_head.upper()} ---")
                    with pd.option_context(
                        "display.max_rows", None, "display.width", 1000
                    ):
                        print(details.sort_values(by="group"))
                    print("-" * 80)

        print("\nSummary of Scenario 4: Final Scores by Fallback Head")
        summary_df = pd.DataFrame(s4_results_list).set_index("fallback_head")
        print(summary_df.to_string(float_format="%.3f"))
        print("=" * 80)
