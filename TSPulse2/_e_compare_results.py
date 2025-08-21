import glob
import os
from typing import Dict, Optional, Set, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))


def clean_filenames(s: pd.Series) -> pd.Series:
    """Applies a standard cleaning regex to filenames."""
    # Removes any suffix starting with '-' and the '.csv' extension.
    return s.str.replace(r"(-.*)?\.csv$", "", regex=True)


def plot_bar_on_ax(
    ax,
    df: pd.DataFrame,
    title: str,
    x_col: str,
    y_col: str,
    y_label: Optional[str] = None,
    colors: Optional[list] = None,
):
    """Generates a bar chart on a given matplotlib axis."""
    if df.empty:
        ax.text(
            0.5,
            0.5,
            f"No data for '{title}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return

    bar_colors = (
        colors
        if colors is not None
        else plt.cm.viridis(np.linspace(0.4, 0.8, len(df[x_col])))
    )

    bars = ax.bar(df[x_col], df[y_col], color=bar_colors)

    if y_label:
        ax.set_ylabel(y_label, fontsize=40)
    ax.set_title(title, fontsize=48)
    ax.tick_params(axis="x", labelsize=36)
    ax.tick_params(axis="y", labelsize=36)

    # Rotate labels to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Add data labels outside of the bars
    ax.bar_label(bars, fmt="%.3f", fontsize=32, padding=3)

    # Add some margin to the top to make space for labels
    ax.margins(y=0.15)


def evaluate_best_head_strategy(
    root_directory: str,
    metric: str,
    split_files: Dict[str, Set[str]],
    dataset_type: str,
    heads_to_load: Dict[str, str],
    clean_filenames_flag: bool = False,
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
                    file_col_s = cast(pd.Series, df["file"]).astype(str)
                    if clean_filenames_flag:
                        # Normalize filenames by removing suffixes and ensuring .csv extension
                        df["file"] = clean_filenames(file_col_s) + ".csv"
                    else:
                        df["file"] = file_col_s.apply(
                            lambda x: x if x.endswith(".csv") else f"{x}.csv"
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
        # The result of .mean() can be a float if there's only one row.
        mean_scores = tuning_performance[cols].mean()
        if isinstance(mean_scores, pd.Series):
            final_fallback_head = mean_scores.idxmax()
        else:
            # If it's a float (or anything else), there's no 'best' head to determine.
            # We can either pick the first col or handle it as an error/default.
            # Picking the first column is a reasonable default.
            final_fallback_head = cols[0] if cols else None

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

        # Head used for scoring (fallback allowed)
        sel_mode = (
            tuning_performance.loc[group, "best"]
            if group in tuning_performance.index
            else final_fallback_head
        )

        # Head shown in output: N/A if not determined by tuning set
        selected_head_output = (
            tuning_performance.loc[group, "best"]
            if group in tuning_performance.index
            else "N/A"
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
                "selected_head": selected_head_output,
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
        "raw_eval_scores": eval_df,
    }


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


# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "comparison_results")
METRIC_TO_COMPARE = "VUS-PR"
# Directories needed by analyze_model_strategies functions
METRICS_ROOT_DIR = os.path.join(PROJECT_ROOT, "eval", "metrics")
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "Datasets")


def load_tspulse_ft() -> pd.DataFrame:
    """Loads results for TSPulse (FT) from its specific CSV file."""
    print("\n--- Loading TSPulseFT Results ---")
    tspulse_ft_path = os.path.join(script_dir, "TSPulseFT.csv")
    if not os.path.exists(tspulse_ft_path):
        print("Warning: TSPulseFT.csv not found. Skipping.")
        return pd.DataFrame()

    try:
        ft_df = pd.read_csv(tspulse_ft_path)
        ft_df = ft_df.set_index("Method").T
        ft_df.index.name = "file"
        if "TSPulse (FT)" in ft_df.columns:
            ft_df.rename(columns={"TSPulse (FT)": "TSPulseFT"}, inplace=True)
            print("Successfully loaded TSPulseFT.csv")
            # Ensure it returns a DataFrame
            return ft_df.loc[:, ["TSPulseFT"]]
        else:
            print("Warning: 'TSPulse (FT)' column not found in TSPulseFT.csv")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error processing TSPulseFT.csv: {e}")
        return pd.DataFrame()


def load_benchmark_results() -> pd.DataFrame:
    """Loads results for other benchmark algorithms from a pre-compiled CSV."""
    print("\n--- Loading Benchmark Results ---")
    benchmark_path = os.path.join(
        PROJECT_ROOT,
        "benchmark_exp",
        "benchmark_eval_results",
        "multi_mergedTable_VUS-PR.csv",
    )
    if not os.path.exists(benchmark_path):
        print(f"Warning: Benchmark file not found at {benchmark_path}. Skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(benchmark_path)
        # Assuming the first column is the file/dataset name, which becomes the index
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "file"}, inplace=True)

        if "file" not in df.columns:
            print(
                "Warning: 'file' column not found in benchmark file. Using first column as index."
            )
            df.set_index(df.columns[0], inplace=True)
        else:
            df.set_index("file", inplace=True)

        # Clean file names by removing any extension
        df.index = df.index.str.replace(r"\.csv$", "", regex=True)
        df.index.name = "file"

        if "CNN" in df.columns:
            print(
                "Successfully loaded CNN benchmark algorithm from "
                f"{os.path.basename(benchmark_path)}."
            )
            return df[["CNN"]].copy()

        print(
            f"Warning: 'CNN' column not found in {os.path.basename(benchmark_path)}. "
            "Skipping benchmark results."
        )
        return pd.DataFrame()

    except Exception as e:
        print(f"Error processing benchmark file {benchmark_path}: {e}")
        return pd.DataFrame()


def load_stl_ad_results(metrics_dir: str, metric: str, split_files: dict) -> pd.DataFrame:
    """Loads STL_AD results from eval/metrics/multi/STL_AD.csv filtered to eval files."""
    print("\n--- Loading STL_AD Results ---")
    stl_path = os.path.join(metrics_dir, "multi", "STL_AD.csv")
    if not os.path.exists(stl_path):
        print(f"Warning: STL_AD file not found at {stl_path}. Skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(stl_path)
        if "file" not in df.columns:
            if "Unnamed: 0" in df.columns:
                df.rename(columns={"Unnamed: 0": "file"}, inplace=True)
            else:
                print("Warning: 'file' column not found in STL_AD.csv. Skipping.")
                return pd.DataFrame()

        if metric not in df.columns:
            print(
                f"Warning: Metric '{metric}' not found in STL_AD.csv. Available: {list(df.columns)}"
            )
            return pd.DataFrame()

        eval_files = split_files.get("eval", set())
        file_col = cast(pd.Series, df["file"]).astype(str)
        df_files_with_ext = clean_filenames(file_col) + ".csv"
        df = df[df_files_with_ext.isin(eval_files)].copy()
        if df.empty:
            print("Warning: No STL_AD eval rows after filtering. Skipping.")
            return pd.DataFrame()

        scores = df[["file", metric]].copy()
        scores["file"] = clean_filenames(cast(pd.Series, scores["file"]))
        final_scores = scores.set_index("file")  # type: ignore
        final_scores.rename(columns={metric: "STL_AD"}, inplace=True)
        print(f"Successfully loaded STL_AD for {len(final_scores)} eval files.")
        return final_scores
    except Exception as e:
        print(f"Error processing STL_AD.csv: {e}")
        return pd.DataFrame()


def load_tspulse_zs(
    metrics_dir: str, data_dir: str, metric: str
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Loads results for TSPulseZS by running Scenario 1 from the analysis script.
    Returns the scores and the split file list used.
    """
    print("\n--- Loading TSPulseZS Results (via analyze_model_strategies) ---")
    zs_heads_only = {
        "ensemble": "TSPulse_ZS_ensemble.csv",
        "fft": "TSPulse_ZS_fft.csv",
        "forecast": "TSPulse_ZS_forecast.csv",
        "time": "TSPulse_ZS_time.csv",
    }
    all_results = []
    all_raw_scores = []
    split_files: dict = {}

    for ds_type in ["multi"]:
        print(f"Running ZS analysis for '{ds_type}' dataset type...")
        try:
            split_files = load_split_files(data_dir, ds_type)
            result_s1 = evaluate_best_head_strategy(
                root_directory=metrics_dir,
                metric=metric,
                split_files=split_files,
                dataset_type=ds_type,
                heads_to_load=zs_heads_only,
                fallback_head="time",
            )
            if (
                "detailed_evaluation" in result_s1
                and not result_s1["detailed_evaluation"].empty
            ):
                detailed_df = result_s1["detailed_evaluation"]
                all_results.append(detailed_df[[metric, "selected_head"]])
                if "raw_eval_scores" in result_s1:
                    all_raw_scores.append(result_s1["raw_eval_scores"])
        except Exception as e:
            print(f"Could not run ZS analysis for '{ds_type}': {e}")
            continue

    if not all_results:
        print("Warning: Failed to get any TSPulseZS results. Skipping.")
        return pd.DataFrame(), {}, pd.DataFrame()

    combined_zs_df = pd.concat(all_results)
    combined_zs_df.rename(
        columns={metric: "TSPulseZS", "selected_head": "TSPulseZS_Head"}, inplace=True
    )
    combined_zs_df.index = combined_zs_df.index.str.replace(".csv", "", regex=False)

    raw_scores_df = pd.DataFrame()
    if all_raw_scores:
        raw_scores_df = pd.concat(all_raw_scores)
        raw_scores_df.index = raw_scores_df.index.str.replace(".csv", "", regex=False)
        raw_scores_df.columns = [f"TSPulseZS_{c}" for c in raw_scores_df.columns]

    print("Successfully loaded TSPulseZS results.")
    return combined_zs_df, split_files, raw_scores_df


def load_model_variant_results(
    model_name_prefix: str, metrics_dir: str, metric: str, split_files: dict
) -> pd.DataFrame:
    """
    Loads all variant results for a given model prefix (e.g., TSPulse2),
    filtered to include only evaluation files.
    """
    print(f"\n--- Loading {model_name_prefix} Variant Results ---")
    all_files = glob.glob(
        os.path.join(metrics_dir, "multi", f"{model_name_prefix}*.csv")
    )
    result_files = [f for f in all_files if not os.path.basename(f).endswith("_.csv")]
    eval_files = split_files.get("eval", set())

    if not eval_files:
        print(
            f"Warning: Evaluation file list is empty. Cannot filter {model_name_prefix} variants."
        )

    if not result_files:
        print(f"Warning: No {model_name_prefix} result files found. Skipping.")
        return pd.DataFrame()

    all_dfs = []
    for file_path in sorted(list(set(result_files))):
        try:
            algo_name = os.path.basename(file_path).replace(".csv", "")
            df = pd.read_csv(file_path)

            if "file" not in df.columns:
                df.rename(columns={"Unnamed: 0": "file"}, inplace=True)

            if "file" not in df.columns or metric not in df.columns:
                print(f"Warning: Skipping {file_path}, missing required columns.")
                continue

            # Filter to keep only files present in the evaluation set
            file_col = cast(pd.Series, df["file"])
            df_files_with_ext = clean_filenames(file_col) + ".csv"
            df = df[df_files_with_ext.isin(eval_files)]

            if df.empty:
                continue

            # Re-cast after filtering to help linter
            file_col = cast(pd.Series, df["file"])
            scores = df[["file", metric]].copy()
            scores["file"] = clean_filenames(file_col)
            final_scores = scores.set_index("file")  # type: ignore
            final_scores.rename(columns={metric: algo_name}, inplace=True)
            all_dfs.append(final_scores)
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    if not all_dfs:
        print(
            f"Warning: No valid {model_name_prefix} dataframes to merge after filtering."
        )
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, axis=1, join="outer")
    print(
        f"Successfully loaded {len(merged_df.columns)} {model_name_prefix} variants on {len(merged_df)} eval files."
    )
    return merged_df


def load_tspulse2_head_triangulation(
    metrics_dir: str, data_dir: str, metric: str, split_files: dict
) -> pd.DataFrame:
    """
    Loads TSPulse2 variants and applies the 'best head' strategy for all 4 fallback heads.
    """
    print("\n--- Loading TSPulse2 Head Triangulation Results ---")

    triangulation_heads = [
        "TSPulse2_llm_selection_ablated_fft",
        "TSPulse2_llm_selection_ablated_time",
        "TSPulse2_llm_selection_ablated_ensemble",
        "TSPulse2_llm_selection_ablated_forecast",
    ]
    heads_to_load = {head: f"{head}.csv" for head in triangulation_heads}

    # Define fallback heads and their corresponding column names
    fallback_heads = {
        "TSPulse2_llm_selection_ablated_ensemble": "TSPulse2Triangulation_ensemble",
        "TSPulse2_llm_selection_ablated_fft": "TSPulse2Triangulation_fft",
        "TSPulse2_llm_selection_ablated_forecast": "TSPulse2Triangulation_forecast",
        "TSPulse2_llm_selection_ablated_time": "TSPulse2Triangulation_time",
    }

    all_results = []
    ds_type = "multi"  # Based on directory

    # Run triangulation for each fallback head
    for fallback_head, column_name in fallback_heads.items():
        print(f"Running TSPulse2 Head Triangulation with fallback '{fallback_head}'...")
        try:
            result = evaluate_best_head_strategy(
                root_directory=metrics_dir,
                metric=metric,
                split_files=split_files,
                dataset_type=ds_type,
                heads_to_load=heads_to_load,
                clean_filenames_flag=True,  # Enable cleaning for TSPulse2 files
                fallback_head=fallback_head,
            )
            if (
                "detailed_evaluation" in result
                and not result["detailed_evaluation"].empty
            ):
                detailed_df = result["detailed_evaluation"][
                    [metric, "selected_head"]
                ].copy()
                detailed_df.rename(
                    columns={
                        metric: column_name,
                        "selected_head": f"{column_name}_Head",
                    },
                    inplace=True,
                )
                detailed_df.index = detailed_df.index.str.replace(
                    ".csv", "", regex=False
                )
                all_results.append(detailed_df)
        except Exception as e:
            print(
                f"Could not run TSPulse2 Head Triangulation with fallback '{fallback_head}': {e}"
            )
            continue

    if not all_results:
        print(
            "Warning: Failed to get any TSPulse2 Head Triangulation results. Skipping."
        )
        return pd.DataFrame()

    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, axis=1, join="outer")
    print(
        f"Successfully loaded TSPulse2 Head Triangulation results for {len(fallback_heads)} fallback heads."
    )
    return combined_df


def print_tspulse_triangulation_analysis(
    summary_df: pd.DataFrame, mean_scores_df: Optional[pd.DataFrame], metric: str
) -> None:
    """Prints TSPulse triangulation analysis from in-memory data.

    - Selected head per dataset using `TSPulseZS_Head` (mode with counts)
    - Actual best head counts per dataset using `TSPulseZS_Best_Actual_Head`
    - Optionally prints the overall mean for TSPulseZS from the provided mean table
    """
    print("\n--- TSPulse Triangulation Analysis ---")

    if "Dataset" not in summary_df.columns:
        print("No 'Dataset' column found; cannot compute per-dataset analysis.")
        return

    datasets = sorted(summary_df["Dataset"].dropna().unique().tolist())

    # Selected head per dataset (mode of TSPulseZS_Head, ignoring 'N/A')
    selected_head_by_dataset: dict[str, str] = {}
    if "TSPulseZS_Head" in summary_df.columns:
        print("Selected head per dataset:")
        for d in datasets:
            sub = summary_df[summary_df["Dataset"] == d]
            heads = sub["TSPulseZS_Head"].astype(str)
            heads = heads[heads != "N/A"]
            if heads.empty:
                selected_head_by_dataset[d] = "N/A"
                print(f"- {d}: N/A ({len(sub)})")
                continue
            vc = heads.value_counts()
            head = str(vc.idxmax())
            selected_head_by_dataset[d] = head
            print(f"- {d}: {head} ")
    else:
        print("Column 'TSPulseZS_Head' not found; skipping selected-head summary.")

    # Actual best head counts per dataset with alignment stats
    print("\nActual best head counts per dataset:")
    total_files_considered = 0
    total_aligned = 0
    if "TSPulseZS_Best_Actual_Head" in summary_df.columns:
        for d in datasets:
            selected = selected_head_by_dataset.get(d, "N/A")
            if selected == "N/A":
                # Count all files in this dataset as misaligned since triangulation couldn't decide
                sub = summary_df[summary_df["Dataset"] == d]
                bc = sub["TSPulseZS_Best_Actual_Head"].value_counts(dropna=False)
                total = int(bc.sum()) if len(bc) > 0 else 0
                total_files_considered += total
                # aligned stays 0 for this dataset
                print(f"- {d}: N/A ({total})")
                continue
            sub = summary_df[summary_df["Dataset"] == d]
            bc = sub["TSPulseZS_Best_Actual_Head"].value_counts(dropna=False)
            time_c = int(bc.get("TSPulseZS_time", 0))
            ensemble_c = int(bc.get("TSPulseZS_ensemble", 0))
            forecast_c = int(bc.get("TSPulseZS_forecast", 0))
            fft_c = int(bc.get("TSPulseZS_fft", 0))
            total = time_c + ensemble_c + forecast_c + fft_c

            # map selected -> corresponding col name
            sel_map = {
                "time": "TSPulseZS_time",
                "ensemble": "TSPulseZS_ensemble",
                "forecast": "TSPulseZS_forecast",
                "fft": "TSPulseZS_fft",
            }
            sel_col = sel_map.get(selected, None)
            aligned = int(bc.get(sel_col, 0)) if sel_col else 0

            total_files_considered += total
            total_aligned += aligned

            pct = (aligned / total * 100.0) if total > 0 else 0.0
            print(
                f"- {d}: time {time_c}, ensemble {ensemble_c}, forecast {forecast_c}, fft {fft_c}, "
                f"{aligned}/{total} ({pct:.2f}%) aligns with triangulation"
            )
    else:
        print("Column 'TSPulseZS_Best_Actual_Head' not found; skipping best-head counts.")

    if total_files_considered > 0:
        overall_pct = total_aligned / total_files_considered * 100.0
        print(f"- SUM: {total_aligned}/{total_files_considered} ({overall_pct:.2f}%)")

    # Print the consolidated TSPulseZS overall mean if provided
    if mean_scores_df is not None and not mean_scores_df.empty:
        algo_col = "Algorithm"
        mean_col = f"Mean_{metric}"
        if algo_col in mean_scores_df.columns and mean_col in mean_scores_df.columns:
            row = mean_scores_df[mean_scores_df[algo_col] == "TSPulseZS"]
            if not row.empty:
                val = float(row.iloc[0][mean_col])
                print("\n--- Overall Mean Scores (CONSOLIDATED) ---")
                print(f"TSPulseZS: {val:.16f}")

def generate_and_save_reports(
    scores_df: pd.DataFrame, output_dir: str, metric: str, name: str
):
    """
    Creates consolidated comparison reports from the merged dataframe,
    saves them to disk, and prints summaries to the console.
    """
    print(f"\n--- Generating Final Reports for {name.upper()} ---")
    os.makedirs(output_dir, exist_ok=True)

    scores_df.fillna(0.0, inplace=True)

    # Define columns that are metadata and not algorithm scores
    metadata_cols = [
        "ts_len",
        "anomaly_len",
        "avg_anomaly_len",
        "num_anomaly",
        "seq_anomaly",
        "point_anomaly",
        "anomaly_ratio",
    ]

    # --- 1. Create a Detailed Per-File Comparison DataFrame ---
    numeric_cols = scores_df.select_dtypes(include=np.number).columns.tolist()
    score_cols = [col for col in numeric_cols if col not in metadata_cols]

    best_scores = scores_df[score_cols].max(axis=1)
    best_algos = scores_df[score_cols].idxmax(axis=1)

    # Add Series-Level Optimal score, which is the best score from a specific subset of algorithms
    oracle_heads = [
        "TSPulse2_llm_selection_ablated_fft",
        "TSPulse2_llm_selection_ablated_time",
        "TSPulse2_llm_selection_ablated_ensemble",
        "TSPulse2_llm_selection_ablated_forecast",
    ]
    valid_oracle_heads = [h for h in oracle_heads if h in scores_df.columns]
    if valid_oracle_heads:
        scores_df["Series-Level Optimal"] = scores_df[valid_oracle_heads].max(axis=1)
        if "Series-Level Optimal" not in score_cols:
            score_cols.append("Series-Level Optimal")
    else:
        print("Warning: Could not calculate Series-Level Optimal score. No specified heads found.")
        scores_df["Series-Level Optimal"] = 0.0

    summary_df = scores_df.copy()
    summary_df["Best_Algo"] = best_algos
    summary_df["Best_Score"] = best_scores
    summary_df["Dataset"] = summary_df.index.to_series().apply(
        lambda x: x.split("_")[1] if "_" in x and len(x.split("_")) > 1 else x
    )

    # Determine the best actual head for each series from the raw ZS scores
    zs_head_cols = [
        "TSPulseZS_ensemble",
        "TSPulseZS_fft",
        "TSPulseZS_forecast",
        "TSPulseZS_time",
    ]
    valid_zs_head_cols = [h for h in zs_head_cols if h in summary_df.columns]
    if valid_zs_head_cols:
        summary_df["TSPulseZS_Best_Actual_Head"] = summary_df[
            valid_zs_head_cols
        ].idxmax(axis=1)

    # Define columns that should come early in the preferred order
    base_preferred_order = [
        "TSPulse3_non_forecast_biased",
        "TSPulse2Triangulation_ensemble",
        "TSPulse2Triangulation_fft",
        "TSPulse2Triangulation_forecast",
        "TSPulse2Triangulation_time",
        "Series-Level Optimal",
        "Best_Algo",
        "Best_Score",
        "Dataset",
        "TSPulseZS",
        "TSPulseZS_Head",
        "TSPulseZS_Best_Actual_Head",
        "TSPulseFT",
        "CNN",
    ]

    # Add all triangulation head columns
    triangulation_head_cols = [
        col
        for col in summary_df.columns
        if col.startswith("TSPulse2Triangulation_") and col.endswith("_Head")
    ]

    # Add all difference columns
    diff_cols = [col for col in summary_df.columns if "_Diff" in col]

    # Create the full preferred order
    preferred_order = base_preferred_order + triangulation_head_cols + diff_cols
    # Get remaining columns and add them to the order
    remaining_cols = [col for col in summary_df.columns if col not in preferred_order]
    final_order = preferred_order + sorted(remaining_cols)
    summary_df = summary_df.reindex(columns=final_order)

    detailed_output_path = os.path.join(
        output_dir, f"{name}_detailed_comparison_{metric}.csv"
    )
    summary_df.to_csv(detailed_output_path)
    print(f"Detailed per-file comparison saved to: {detailed_output_path}")

    # --- 2. Create and Save Per-Dataset Mean Score Summary ---
    dataset_mean_scores = scores_df.copy()
    dataset_mean_scores["Dataset"] = summary_df["Dataset"]
    dataset_mean_scores = (
        dataset_mean_scores.groupby("Dataset")[score_cols].mean().sort_index()
    )

    # Reorder columns for dataset mean scores
    dataset_mean_scores = dataset_mean_scores.reindex(
        columns=[col for col in preferred_order if col in dataset_mean_scores.columns]
        + [
            col
            for col in sorted(dataset_mean_scores.columns)
            if col not in preferred_order
        ]
    )

    dataset_mean_output_path = os.path.join(
        output_dir, f"{name}_dataset_mean_scores_{metric}.csv"
    )
    dataset_mean_scores.to_csv(dataset_mean_output_path)
    print(f"Per-dataset mean score summary saved to: {dataset_mean_output_path}")

    # --- 3. Create and Save Overall Mean Score Summary ---
    numeric_scores = scores_df.select_dtypes(include=np.number)
    if isinstance(numeric_scores, pd.DataFrame):
        mean_scores = cast(pd.Series, numeric_scores[score_cols].mean())
    else:
        # Handle case where numeric_scores might be a Series
        mean_scores = cast(
            pd.Series, numeric_scores.drop(labels=metadata_cols, errors="ignore")
        )

    mean_scores_df = mean_scores.reset_index()
    mean_scores_df.columns = ["Algorithm", f"Mean_{metric}"]

    # Sort the mean scores table in descending order
    mean_scores_df = mean_scores_df.sort_values(by=f"Mean_{metric}", ascending=False)

    mean_output_path = os.path.join(
        output_dir, f"{name}_mean_score_summary_{metric}.csv"
    )
    mean_scores_df.to_csv(mean_output_path, index=False)
    print(f"Mean score summary saved to: {mean_output_path}")

    # --- 4. Print Summaries to Console ---
    # print(f"\n--- Detailed Per-Dataset Comparison ({name.upper()}) ---")
    # with pd.option_context(
    #     "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    # ):
    #     print(summary_df.reset_index().to_string(index=False, float_format="{:.4f}".format))

    print(f"\n--- Per-Dataset Mean Scores ({name.upper()}) ---")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(dataset_mean_scores.to_string(float_format="{:.4f}".format))  # type: ignore

    print(f"\n--- Overall Mean Scores ({name.upper()}) ---")
    print(mean_scores_df.to_string(index=False, float_format="{:.16f}".format))  # type: ignore
    return mean_scores_df, score_cols, summary_df


def generate_strategy_comparison_plot(
    mean_scores_df: pd.DataFrame, metric: str, output_dir: str
):
    """Generates and saves a bar chart comparing different model strategies."""
    print("\n--- Generating Strategy Comparison Plot ---")

    # 1. Determine the best triangulation algorithm from the results
    triangulation_cols = [
        c for c in mean_scores_df["Algorithm"] if c.startswith("TSPulse2Triangulation_")
    ]
    best_triangulation_algo = None
    if triangulation_cols:
        triangulation_scores = mean_scores_df[
            mean_scores_df["Algorithm"].isin(triangulation_cols)
        ]
        if not triangulation_scores.empty:
            best_triangulation_algo = triangulation_scores.loc[
                triangulation_scores[f"Mean_{metric}"].idxmax()
            ]["Algorithm"]

    # 2. Define the algorithms and their labels for the plot
    algorithms_to_plot = {
        "TSPulse2_llm_selection_ablated_ensemble": "Static: Ensemble",
        "TSPulse2_llm_selection_ablated_fft": "Static: FFT",
        "TSPulse2_llm_selection_ablated_forecast": "Static: Forecast",
        "TSPulse2_llm_selection_ablated_time": "Static: Time",
        "TSPulse3_non_forecast_biased": "Ours (Few-Shot)",
        "TSPulse2": "Ours (Zero-Shot)",
        "Series-Level Optimal": "Series-Level Optimal",
    }
    if best_triangulation_algo:
        algorithms_to_plot[best_triangulation_algo] = "Triangulation"

    # 3. Filter and prepare the DataFrame for plotting
    plot_df = mean_scores_df[
        mean_scores_df["Algorithm"].isin(algorithms_to_plot.keys())
    ].copy()
    plot_df["Strategy"] = plot_df["Algorithm"].map(algorithms_to_plot)
    plot_df = plot_df.sort_values(by=f"Mean_{metric}", ascending=False)

    if plot_df.empty:
        print("Warning: No data available for the strategy comparison plot. Skipping.")
        return

    # 4. Create and save the plot
    def get_color(strategy: str) -> str:
        if strategy.startswith("Ours"):
            return "darkorange"  # Emphasize our methods
        if strategy.startswith("Triangulation"):
            return "forestgreen"  # Color for the next best
        if strategy == "Series-Level Optimal":
            return "gold"  # Special color for Series-Level Optimal
        return "steelblue"  # Standard color for static methods

    bar_colors = [get_color(s) for s in plot_df["Strategy"]]

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_bar_on_ax(
        ax,
        plot_df,
        title="",  # No main title for this plot
        x_col="Strategy",
        y_col=f"Mean_{metric}",
        y_label=f"Mean {metric}",
        colors=bar_colors,
    )

    # Customize text for specific bars to be bold
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text.startswith("Ours") or text.startswith("Triangulation") or text == "Series-Level Optimal":
            label.set_weight("bold")

    fig.tight_layout(pad=1.0)

    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    output_path = os.path.join(charts_dir, "consolidated_strategy_comparison")

    fig.savefig(f"{output_path}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{output_path}.png", format="png", bbox_inches="tight", dpi=300)
    print(f"Strategy comparison plot saved to {output_path}.pdf and .png")
    plt.close(fig)


def analyze_dimensionality_reduction(
    metrics_dir: str, metric: str, split_files: dict
):
    """
    Analyzes and compares dimensionality reduction methods by loading data from the
    same evaluation split as the main report to ensure consistent comparisons.
    """
    print("\n" + "=" * 50)
    print("--- Starting Dimensionality Reduction Analysis ---")
    print("=" * 50)

    # 1. Use the same evaluation file list as the main report for consistency.
    eval_files = split_files.get("eval", set())
    if not eval_files:
        print(
            "Warning: Evaluation file list is empty for dim-redux analysis. Skipping."
        )
        return

    # 2. Load and process results from the specific files for this analysis.
    all_dfs = []
    main_results_file = os.path.join(
        metrics_dir, "multi", "TSPulse3_non_forecast_biased.csv"
    )
    ablated_file_path = os.path.join(
        metrics_dir, "multi", "TSPulse3_non_forecast_biased_dim_redux_ablated.csv"
    )

    for file_path in [main_results_file, ablated_file_path]:
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(file_path)
            if "file" not in df.columns or metric not in df.columns:
                continue

            # Filter by the official evaluation file list *before* processing.
            df_files_with_ext = clean_filenames(cast(pd.Series, df["file"])) + ".csv"
            df = df[df_files_with_ext.isin(eval_files)].copy()
            if df.empty:
                continue

            # Assign method based on the source file.
            if file_path == ablated_file_path:
                df["method"] = "Ablated"
            else:  # Main results file

                def get_standard_method(data_filename: str) -> str:
                    name_part = os.path.splitext(data_filename)[0]
                    if "-" in name_part and not name_part.split("-")[-1].isdigit():
                        return "Channel Selection"
                    return "PCA"

                df["method"] = pd.Series(df["file"]).apply(get_standard_method)

            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not all_dfs:
        print("Warning: No valid TSPulse2 dataframes to merge for this analysis.")
        return

    # 3. Combine, normalize, and create the per-file score dataframe.
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["file"] = clean_filenames(combined_df["file"].astype(str)) + ".csv"

    combined_df["dataset"] = combined_df["file"].apply(
        lambda x: x.split("_")[1] if "_" in x and len(x.split("_")) > 1 else "unknown"
    )

    # 4. Run and print specific, side-by-side comparisons using MICRO-averaging.
    def _run_and_print_comparison(
        title: str,
        per_file_df: pd.DataFrame,
        methods: list[str],
        metric_name: str,
    ):
        """Helper to run and print a specific comparison using micro-averaging."""
        print("\n" + "#" * 50)
        print(f"### {title} ###")
        print("#" * 50)

        # Pivot to easily find common files across methods
        pivoted_df = per_file_df.pivot_table(
            index=["file", "dataset"], columns="method", values=metric_name
        ).reset_index()

        # Find files that have scores for ALL specified methods
        valid_methods = [m for m in methods if m in pivoted_df.columns]
        if len(valid_methods) < len(methods):
            missing = set(methods) - set(valid_methods)
            print(f"Warning: Methods {list(missing)} not found. Skipping comparison.")
            return

        comparison_df = pivoted_df.dropna(subset=valid_methods, how="any")

        if comparison_df.empty:
            print(f"No common files found for methods: {methods}. Skipping.")
            return

        print(f"Found {len(comparison_df)} common files for comparison.")

        # Per-Dataset Summary (Micro-Average within each dataset)
        print(f"\n--- Per-Dataset Average {metric_name} ---")

        # Overall Average (Micro-Average across all common files)
        print(f"\n--- Overall Average {metric_name} ---")
        overall_scores_series = comparison_df[valid_methods].mean()
        if isinstance(overall_scores_series, pd.Series):
            overall_scores = overall_scores_series.sort_values(
                ascending=False
            ).reset_index()
            overall_scores.columns = ["Method", f"Average_{metric_name}"]
            print(overall_scores.to_string(index=False, float_format="{:.4f}".format))

            # --- Plot on the provided axis ---
            # plot_bar_on_ax(
            #     ax=ax,
            #     df=overall_scores,
            #     title=title,
            #     xlabel="Method",
            #     ylabel=f"Average_{metric_name}",
            # )
        else:
            print(
                f"Could not compute overall scores. Mean result: {overall_scores_series}"
            )
        return overall_scores if isinstance(overall_scores_series, pd.Series) else pd.DataFrame()

    comparison1_df = _run_and_print_comparison(
        "Ablated vs. Channel Selection (on their common series)",
        combined_df,
        ["Ablated", "Channel Selection"],
        metric,
    )
    comparison2_df = _run_and_print_comparison(
        "Ablated vs. PCA (on their common series)",
        combined_df,
        ["Ablated", "PCA"],
        metric,
    )

    print("\n" + "=" * 50)
    print("--- Dimensionality Reduction Analysis Complete ---")
    print("=" * 50 + "\n")
    return comparison1_df, comparison2_df


def main():
    """Main function to run all analysis and generate consolidated plots."""
    print("\n" + "=" * 50)
    print("--- Starting Comparison Generation ---")
    print("=" * 50)

    # 1. Load results from all sources
    ft_scores = load_tspulse_ft()
    benchmark_scores = load_benchmark_results()
    zs_scores, split_files, zs_raw_scores = load_tspulse_zs(
        METRICS_ROOT_DIR, DATA_ROOT_DIR, METRIC_TO_COMPARE
    )
    stl_scores = load_stl_ad_results(METRICS_ROOT_DIR, METRIC_TO_COMPARE, split_files)
    ts2_scores = load_model_variant_results(
        "TSPulse2", METRICS_ROOT_DIR, METRIC_TO_COMPARE, split_files
    )
    ts3_scores = load_model_variant_results(
        "TSPulse3", METRICS_ROOT_DIR, METRIC_TO_COMPARE, split_files
    )
    ts2_triangulation_scores = load_tspulse2_head_triangulation(
        METRICS_ROOT_DIR, DATA_ROOT_DIR, METRIC_TO_COMPARE, split_files
    )

    # 2. Merge all file-based scores
    file_scores_df = pd.DataFrame()
    for df in [
        benchmark_scores,
        zs_scores,
        stl_scores,
        ts2_scores,
        ts3_scores,
        ts2_triangulation_scores,
        zs_raw_scores,
    ]:
        if not df.empty:
            if file_scores_df.empty:
                file_scores_df = df
            else:
                file_scores_df = file_scores_df.join(df, how="outer")

    if file_scores_df.empty:
        print("\nNo file-based data could be loaded. Aborting report generation.")
        return

    # 3. Map the per-dataset TSPulseFT scores
    if not ft_scores.empty:
        file_scores_df["Dataset_temp"] = file_scores_df.index.to_series().apply(
            lambda x: x.split("_")[1] if "_" in x and len(x.split("_")) > 1 else x
        )
        ft_map = ft_scores["TSPulseFT"].to_dict()
        file_scores_df["TSPulseFT"] = file_scores_df["Dataset_temp"].map(ft_map)  # type: ignore
        file_scores_df.drop(columns=["Dataset_temp"], inplace=True)

    # 4. Generate reports and get data for plotting
    mean_scores_df, _, summary_df = generate_and_save_reports(
        scores_df=file_scores_df,
        output_dir=OUTPUT_DIR,
        metric=METRIC_TO_COMPARE,
        name="consolidated",
    )
    print("\n" + "=" * 50)
    print("--- Comparison Generation Complete ---")
    print("=" * 50 + "\n")

    comp1_df, comp2_df = analyze_dimensionality_reduction(
        METRICS_ROOT_DIR, METRIC_TO_COMPARE, split_files
    )

    # 5. Create consolidated plot
    fig, axes = plt.subplots(1, 3, figsize=(33, 8))

    # Plot 1: MAD Variant Comparison
    tspulse3_variants = [
        "TSPulse3_non_forecast_biased",
        "TSPulse3_non_forecast_biased_dim_redux_ablated",
    ]
    specific_df = mean_scores_df[
        mean_scores_df["Algorithm"].isin(tspulse3_variants)
    ].copy()
    new_labels = {
        "TSPulse3_non_forecast_biased": "MAD (Full)",
        "TSPulse3_non_forecast_biased_dim_redux_ablated": "MAD (Ablated)",
    }
    specific_df["Algorithm"] = specific_df["Algorithm"].map(new_labels)
    specific_df.rename(columns={"Algorithm": "Model Variant"}, inplace=True)
    plot_bar_on_ax(
        axes[0],
        specific_df,
        "",
        "Model Variant",
        f"Mean_{METRIC_TO_COMPARE}",
        y_label=f"Mean_{METRIC_TO_COMPARE}",
    )

    # Plot 2 & 3: Dimensionality Reduction Comparisons
    plot_bar_on_ax(
        axes[1],
        comp1_df,
        "",
        "Method",
        f"Average_{METRIC_TO_COMPARE}",
    )
    plot_bar_on_ax(
        axes[2],
        comp2_df,
        "",
        "Method",
        f"Average_{METRIC_TO_COMPARE}",
    )

    # Print triangulation analysis from in-memory data (no CSV reads)
    try:
        print_tspulse_triangulation_analysis(summary_df, mean_scores_df, METRIC_TO_COMPARE)
    except Exception as e:
        print(f"Warning: Could not print triangulation analysis: {e}")

    # 6. Save the final figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "charts", "consolidated_summary")
    os.makedirs(os.path.join(OUTPUT_DIR, "charts"), exist_ok=True)
    fig.savefig(f"{output_path}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{output_path}.png", format="png", bbox_inches="tight", dpi=300)
    print(f"\nConsolidated summary plot saved to {output_path}.pdf and .png")
    plt.close(fig)

    # 6. Generate the separate strategy comparison plot
    generate_strategy_comparison_plot(mean_scores_df, METRIC_TO_COMPARE, OUTPUT_DIR)


if __name__ == "__main__":
    main()
