# coding: utf-8
# Author: Qinghua Liu liu.11085@osu.edu
# License: Apache-2.0 License
# Modified by AI Assistant to plot all dimensions with higher resolution

from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import time
import traceback
import warnings

# --- Set Matplotlib Backend ---
# Try setting a non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
# -----------------------------


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# --- Helper function needed for plotting ---
def _find_clusters(indices):
    """Finds contiguous clusters of indices."""
    if not isinstance(indices, (list, np.ndarray)) or len(indices) == 0:
        return []
    # Ensure indices are sorted and unique integers
    try:
        indices = np.unique(np.asarray(indices, dtype=int))
    except ValueError:
        return []

    if len(indices) == 0:
        return []

    clusters = []
    if len(indices) == 1:
        return [(indices[0], indices[0])]

    diffs = np.diff(indices)
    split_points = np.where(diffs > 1)[0]
    start_indices = np.insert(indices[split_points + 1], 0, indices[0])
    end_indices = np.append(indices[split_points], indices[-1])
    clusters = list(zip(start_indices, end_indices))
    return clusters


# --- Raw Data Plotting Function ---
# Removed max_dims_to_plot parameter
def plot_raw_data_and_label(raw_data, label, base_filename, save_dir, train_index):
    """
    Generates and saves a plot showing the full raw time series data for ALL dimensions,
    coloring the data red during anomalous periods (label=1) and
    adding a vertical line at the train_index.

    Args:
        raw_data (np.ndarray): The raw time series data (n_samples, n_features) or (n_samples,).
        label (np.ndarray): The ground truth labels (n_samples,).
        base_filename (str): The base name of the original data file (without extension).
        save_dir (str): The directory to save the plot in.
        train_index (int): The index indicating the end of the training data.
    """
    # Construct the expected output filename BEFORE plotting
    expected_plot_filename = os.path.join(
        save_dir, f"{base_filename}_Raw_RedAnomaly_AllDims.png")

    # --- Check if plot already exists ---
    if os.path.exists(expected_plot_filename):
        print(
            f"  Skipping plot (already exists): {os.path.basename(expected_plot_filename)}")
        return True  # Indicate skipped
    # ------------------------------------

    print(
        f"  Generating full raw data plot (all dims, anomalies in red) for: {base_filename}")
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
            f"    Warning: Raw data has unexpected dimensions ({raw_data.ndim}). Cannot plot.")
        return False

    if len(label) != n_samples:
        print(
            f"    Warning: Label length ({len(label)}) mismatch with data ({n_samples}). Cannot plot.")
        return False

    num_subplots = n_dims
    fig_height = max(6, 1.0 * num_subplots)  # Adjusted height scaling
    fig_width = 16 if num_subplots < 10 else 18

    fig = None
    try:
        fig, axes = plt.subplots(num_subplots, 1, figsize=(
            fig_width, fig_height), sharex=True, squeeze=False)
        axes = axes.ravel()

        anomaly_indices = np.where(label == 1)[0]
        anomaly_segments = _find_clusters(anomaly_indices)

        legend_handles = []

        for d in range(n_dims):
            ax_raw = axes[d]
            dim_data = data_to_plot[:, d]
            dim_label_str = f'Dim {d}'
            ax_raw.set_ylabel(dim_label_str, color='steelblue',
                              fontsize=6, rotation=0, ha='right')
            ax_raw.tick_params(axis='y', labelcolor='steelblue', labelsize=6)
            ax_raw.grid(True, linestyle=':', alpha=0.4)

            # Plot normal data segments
            ln_normal = ax_raw.plot(
                time_index, dim_data, color='steelblue', linewidth=0.6, label='Normal Data')
            if d == 0:
                legend_handles.extend(ln_normal)

            # Overlay anomalous data segments in red
            ln_anomaly_handle_added = False
            for start, end in anomaly_segments:
                segment_indices = np.arange(start, end + 1)
                segment_indices = segment_indices[segment_indices < n_samples]
                if len(segment_indices) > 0:
                    plot_indices = segment_indices
                    plot_data = dim_data[plot_indices]
                    if start > 0:
                        plot_indices = np.insert(plot_indices, 0, start - 1)
                        plot_data = np.insert(
                            plot_data, 0, dim_data[start - 1])
                    if end < n_samples - 1:
                        plot_indices = np.append(plot_indices, end + 1)
                        plot_data = np.append(plot_data, dim_data[end + 1])

                    ln_seg = ax_raw.plot(
                        time_index[plot_indices], plot_data, color='red', linewidth=0.7, label='Anomalous Data')
                    if d == 0 and not ln_anomaly_handle_added:
                        legend_handles.extend(ln_seg)
                        ln_anomaly_handle_added = True

            # --- Draw Vertical Line for Train Index ---
            vline_label = 'Train End' if d == 0 else "_nolegend_"
            vl = ax_raw.axvline(x=train_index, color='purple', linestyle='--',
                                linewidth=0.8, alpha=0.7, label=vline_label)
            if d == 0:
                legend_handles.append(vl)

        # Add combined legend only to the first subplot
        plot_labs = [h.get_label() for h in legend_handles]
        axes[0].legend(legend_handles, plot_labs,
                       loc='upper right', fontsize='xx-small')

        axes[-1].set_xlabel('Time Index', fontsize=8)
        fig.suptitle(
            f'Raw Data (Anomalies Highlighted) & Train Split - {base_filename}', fontsize=10)
        plt.subplots_adjust(hspace=0.1)

        # --- Save the figure with HIGHER DPI ---
        # Changed DPI from 75 to 300
        plt.savefig(expected_plot_filename, dpi=300)
        # ---------------------------------------
        print(f"    Plot saved to: {os.path.basename(expected_plot_filename)}")
        return True

    except MemoryError as me:
        print(
            f"    MemoryError generating raw data plot: {me}. Skipping this plot.")
        return False
    except Exception as e:
        print(f"    Error generating raw data plot: {e}")
        return False
    finally:
        if fig is not None:
            plt.close(fig)

# --- End of plotting function ---


# --- Main Execution Block ---
if __name__ == '__main__':
    overall_start_time = time.time()

    # --- Configuration ---
    parser = argparse.ArgumentParser(
        description='Plot Raw Time Series Data (All Dims) with Anomalies Highlighted (Skips Existing)')
    parser.add_argument('--dataset_dir', type=str,
                        default='Datasets/TSB-AD-M/', help='Directory containing the dataset CSV files.')
    parser.add_argument('--file_list', type=str,
                        default='Datasets/File_List/TSB-AD-M.csv', help='CSV file listing the filenames to process.')
    parser.add_argument('--save_dir', type=str,
                        default='Raw_Data_Plots_AllDims', help='Directory to save the output plots.')
    args = parser.parse_args()

    RAW_PLOT_SAVE_DIR = args.save_dir
    FILE_LIST_PATH = args.file_list
    DATA_DIR = args.dataset_dir

    # --- Create Save Directory ---
    os.makedirs(RAW_PLOT_SAVE_DIR, exist_ok=True)

    # --- Get File List ---
    try:
        file_list_df = pd.read_csv(FILE_LIST_PATH)
        if file_list_df.empty or file_list_df.shape[1] == 0:
            print(
                f"Warning: File list {FILE_LIST_PATH} is empty or has no columns.")
            filenames = []
        else:
            filenames = file_list_df.iloc[:, 0].dropna().unique().tolist()
        total_files = len(filenames)
        print(
            f"Found {total_files} unique valid filenames listed in {FILE_LIST_PATH}")
    except FileNotFoundError:
        print(f"Error: File list {FILE_LIST_PATH} not found.")
        filenames = []
        total_files = 0
    except Exception as e:
        print(f"Error reading file list {FILE_LIST_PATH}: {e}")
        filenames = []
        total_files = 0

    # --- Main Loop ---
    processed_files_count = 0
    skipped_files_count = 0
    failed_plot_count = 0

    for i, filename in enumerate(filenames):
        print(f"\nProcessing file {i+1}/{total_files}: {filename}")
        file_path = os.path.join(DATA_DIR, filename)
        base_fname_for_plot = os.path.splitext(filename)[0]
        expected_plot_path = os.path.join(
            RAW_PLOT_SAVE_DIR, f"{base_fname_for_plot}_Raw_RedAnomaly_AllDims.png")

        # --- Check if plot already exists BEFORE loading data ---
        if os.path.exists(expected_plot_path):
            print(
                f"  Skipping plot generation (already exists): {os.path.basename(expected_plot_path)}")
            skipped_files_count += 1
            continue
        # ------------------------------------------------------

        try:
            # --- Data Loading and Validation ---
            if not os.path.exists(file_path):
                print(
                    f"  Error: File not found: {file_path}. Skipping plot generation.")
                failed_plot_count += 1
                continue

            df = pd.read_csv(file_path)
            if 'Label' not in df.columns:
                print(
                    f"  Warning: Skipping {filename} ('Label' column missing).")
                failed_plot_count += 1
                continue
            if df.shape[1] < 2:
                print(
                    f"  Warning: Skipping {filename} (needs >=1 feature column besides 'Label').")
                failed_plot_count += 1
                continue
            print(f"  Step 1: Read CSV done")
            df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
            df = df.dropna()
            if df.empty:
                print(
                    f"  Warning: Skipping {filename} (empty after dropping NaNs).")
                failed_plot_count += 1
                continue
            data = df.iloc[:, :-1].values.astype(float)
            label = df['Label'].values.astype(int)
            n_samples = data.shape[0]
            print(f"  Data Shape: {data.shape}, Label Shape: {label.shape}")

            # --- Determine Train Index from Filename ---
            train_index = -1
            try:
                filename_parts = base_fname_for_plot.split('_')
                if len(filename_parts) >= 3:
                    train_index_str = filename_parts[-3]
                    try:
                        train_index = int(train_index_str)
                        if not (0 < train_index < n_samples):
                            print(
                                f"    Warning: Parsed train index {train_index} out of bounds (0, {n_samples}). Using 30% default.")
                            train_index = max(1, int(0.3 * n_samples))
                    except ValueError:
                        print(
                            f"    Warning: Cannot parse train index '{train_index_str}' from filename. Using 30% default.")
                        train_index = max(1, int(0.3 * n_samples))
                else:
                    print(
                        f"    Warning: Filename '{base_fname_for_plot}' has too few parts. Using 30% default for train index.")
                    train_index = max(1, int(0.3 * n_samples))
            except Exception as e_parse:
                print(
                    f"    Error parsing train index from filename: {e_parse}. Using 30% default.")
                train_index = max(1, int(0.3 * n_samples))
            print(f"    Train Index determined as: {train_index}")
            # ------------------------------------------

            # --- Generate Raw Data Plot ---
            plot_generated = plot_raw_data_and_label(
                data, label, base_fname_for_plot, RAW_PLOT_SAVE_DIR,
                train_index
            )
            if plot_generated:
                processed_files_count += 1
            else:
                failed_plot_count += 1

        except FileNotFoundError:
            print(
                f"  Error: File not found (inside loop): {file_path}. Skipping.")
            failed_plot_count += 1
        except pd.errors.EmptyDataError:
            print(f"  Error: File {filename} is empty. Skipping.")
            failed_plot_count += 1
        except ValueError as ve:
            print(
                f"  Error processing file {filename} (ValueError): {ve}. Skipping.")
            failed_plot_count += 1
        except MemoryError as me:
            print(f"  MemoryError processing file {filename}: {me}. Skipping.")
            failed_plot_count += 1
        except Exception as e:
            print(f"  Unexpected critical error processing {filename}: {e}")
            traceback.print_exc()
            failed_plot_count += 1

    # --- End File Loop ---
    print(f"\n--- Run Summary ---")
    print(f"Attempted processing for {total_files} listed files.")
    print(f"Generated plots for {processed_files_count} files.")
    print(
        f"Skipped {skipped_files_count} files because plots already existed.")
    print(
        f"Failed to generate plots for {failed_plot_count} files due to errors or invalid data.")
    print(f"Plots saved in directory: {RAW_PLOT_SAVE_DIR}")
    print(
        f"\nTotal execution time: {time.time() - overall_start_time:.2f} seconds")
