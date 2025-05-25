from __future__ import division, print_function

import argparse
import os
import time
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MAD import MAD_constants

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def plot_raw_data(raw_data, base_filename, save_dir):
    """
    Generates and saves a plot showing the full raw time series data for ALL dimensions.
    Does NOT highlight anomalies or indicate a training split.

    Args:
        raw_data (np.ndarray): The raw time series data (n_samples, n_features) or (n_samples,).
        base_filename (str): The base name of the original data file (without extension).
        save_dir (str): The directory to save the plot in.
    """
    expected_plot_filename = os.path.join(
        save_dir, f"{base_filename}_RawData_AllDims.png"
    )
    if os.path.exists(expected_plot_filename):
        print(
            f"  Skipping plot (already exists): {os.path.basename(expected_plot_filename)}"
        )
        return True
    print(f"  Generating full raw data plot (all dims) for: {base_filename}")
    os.makedirs(save_dir, exist_ok=True)

    n_samples = raw_data.shape[0]
    if n_samples == 0:
        print("    Warning: No data points to plot.")
        return False

    time_index = np.arange(n_samples)

    if raw_data.ndim == 1:
        n_dims = 1
        data_to_plot = raw_data.reshape(-1, 1)
    elif raw_data.ndim == 2:
        n_dims = raw_data.shape[1]
        data_to_plot = raw_data
    else:
        print(
            f"    Warning: Raw data has unexpected dimensions ({raw_data.ndim}). Cannot plot."
        )
        return False

    num_subplots = n_dims
    fig_height = max(6, 1.0 * num_subplots)
    fig_width = MAD_constants.PLOT_FIG_WIDTH

    fig = None
    try:
        fig, axes = plt.subplots(
            num_subplots, 1, figsize=(fig_width, fig_height), sharex=True, squeeze=False
        )
        axes = axes.ravel()

        legend_handles = []

        for d in range(n_dims):
            ax_raw = axes[d]
            dim_data = data_to_plot[:, d]
            dim_label_str = f"Dim {d}"
            ax_raw.set_ylabel(
                dim_label_str, color="steelblue", fontsize=6, rotation=0, ha="right"
            )
            ax_raw.tick_params(axis="y", labelcolor="steelblue", labelsize=6)
            ax_raw.grid(True, linestyle=":", alpha=0.4)
            ln_data = ax_raw.plot(
                time_index, dim_data, color="steelblue", linewidth=0.6, label="Data"
            )
            if d == 0:
                legend_handles.extend(ln_data)
        if legend_handles:
            plot_labs = [h.get_label() for h in legend_handles]
            axes[0].legend(
                legend_handles, plot_labs, loc="upper right", fontsize="xx-small"
            )

        axes[-1].set_xlabel("Time Index", fontsize=8)
        fig.suptitle(f"Raw Data - {base_filename}", fontsize=10)
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(expected_plot_filename, dpi=300)
        print(f"    Plot saved to: {os.path.basename(expected_plot_filename)}")
        return True

    except MemoryError as me:
        print(f"    MemoryError generating raw data plot: {me}. Skipping this plot.")
        return False
    except Exception as e:
        print(f"    Error generating raw data plot: {e}")
        traceback.print_exc()
        return False
    finally:
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    overall_start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Plot Raw Time Series Data (All Dims) (Skips Existing)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Datasets/TSB-AD-U/",
        help="Directory containing the dataset CSV files.",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default="Datasets/File_List/TSB-AD-U.csv",
        help="CSV file listing the filenames to process.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="Raw_Data_Plots_Only",
        help="Directory to save the output plots.",
    )
    args = parser.parse_args()

    RAW_PLOT_SAVE_DIR = args.save_dir
    FILE_LIST_PATH = args.file_list
    DATA_DIR = args.dataset_dir
    os.makedirs(RAW_PLOT_SAVE_DIR, exist_ok=True)
    try:
        file_list_df = pd.read_csv(FILE_LIST_PATH)
        if file_list_df.empty or file_list_df.shape[1] == 0:
            print(f"Warning: File list {FILE_LIST_PATH} is empty or has no columns.")
            filenames = []
        else:
            filenames = file_list_df.iloc[:, 0].dropna().unique().tolist()
        total_files = len(filenames)
        print(f"Found {total_files} unique valid filenames listed in {FILE_LIST_PATH}")
    except FileNotFoundError:
        print(f"Error: File list {FILE_LIST_PATH} not found.")
        filenames = []
        total_files = 0
    except Exception as e:
        print(f"Error reading file list {FILE_LIST_PATH}: {e}")
        filenames = []
        total_files = 0
    processed_files_count = 0
    skipped_files_count = 0
    failed_plot_count = 0

    for i, filename in enumerate(filenames):
        print(f"\nProcessing file {i+1}/{total_files}: {filename}")
        file_path = os.path.join(DATA_DIR, filename)
        base_fname_for_plot = os.path.splitext(filename)[0]
        expected_plot_path = os.path.join(
            RAW_PLOT_SAVE_DIR, f"{base_fname_for_plot}_RawData_AllDims.png"
        )
        if os.path.exists(expected_plot_path):
            print(
                f"  Skipping plot generation (already exists): {os.path.basename(expected_plot_path)}"
            )
            skipped_files_count += 1
            continue

        try:

            if not os.path.exists(file_path):
                print(
                    f"  Error: File not found: {file_path}. Skipping plot generation."
                )
                failed_plot_count += 1
                continue

            try:

                header_df = pd.read_csv(file_path, nrows=0)
                all_columns = header_df.columns.tolist()
                feature_columns = [col for col in all_columns if col != "Label"]
                if not feature_columns:
                    print(f"  Warning: Skipping {filename} (no feature columns found).")
                    failed_plot_count += 1
                    continue
                df = pd.read_csv(file_path, usecols=feature_columns)
            except ValueError as ve:  # Handles issues like duplicate column names
                print(
                    f"  Error reading specific columns for {filename}: {ve}. Trying basic load."
                )
                df = pd.read_csv(file_path)
                if "Label" in df.columns:
                    df = df.drop(columns=["Label"])
                if df.shape[1] == 0:
                    print(
                        f"  Warning: Skipping {filename} (no feature columns after fallback)."
                    )
                    failed_plot_count += 1
                    continue

            print(f"  Step 1: Read CSV done")
            df = df.fillna(0)
            if df.empty:
                print(f"  Warning: Skipping {filename} (empty after loading/fillna).")
                failed_plot_count += 1
                continue

            data = df.values.astype(float)
            n_samples = data.shape[0]
            print(f"  Data Shape for plotting: {data.shape}")
            plot_generated = plot_raw_data(data, base_fname_for_plot, RAW_PLOT_SAVE_DIR)
            if plot_generated:
                processed_files_count += 1
            else:
                failed_plot_count += 1

        except FileNotFoundError:
            print(f"  Error: File not found (inside loop): {file_path}. Skipping.")
            failed_plot_count += 1
        except pd.errors.EmptyDataError:
            print(f"  Error: File {filename} is empty. Skipping.")
            failed_plot_count += 1
        except ValueError as ve:
            print(f"  Error processing file {filename} (ValueError): {ve}. Skipping.")
            failed_plot_count += 1
        except MemoryError as me:
            print(f"  MemoryError processing file {filename}: {me}. Skipping.")
            failed_plot_count += 1
        except Exception as e:
            print(f"  Unexpected critical error processing {filename}: {e}")
            traceback.print_exc()
            failed_plot_count += 1
    print(f"\n--- Run Summary ---")
    print(f"Attempted processing for {total_files} listed files.")
    print(f"Generated plots for {processed_files_count} files.")
    print(f"Skipped {skipped_files_count} files because plots already existed.")
    print(
        f"Failed to generate plots for {failed_plot_count} files due to errors or invalid data."
    )
    print(f"Plots saved in directory: {RAW_PLOT_SAVE_DIR}")
    print(f"\nTotal execution time: {time.time() - overall_start_time:.2f} seconds")
