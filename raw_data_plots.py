# coding: utf-8
# Author: Qinghua Liu liu.11085@osu.edu
# License: Apache-2.0 License
# Modified by AI Assistant to plot all dims, indicate anomaly indices in filename & plot, move legend

from __future__ import division, print_function
import pandas as pd
import numpy as np
# Import pyplot only after setting backend if needed
# import matplotlib.pyplot as plt

import argparse
import os
import time
import traceback
import warnings

# --- Set Matplotlib Backend ---
import matplotlib
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: Matplotlib backend 'Agg' failed. Falling back to default.")
    import matplotlib.pyplot as plt
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
    try:
        indices = np.unique(np.asarray(indices, dtype=int))
    except ValueError:
        return []

    if len(indices) == 0:
        return []
    if len(indices) == 1:
        return [(indices[0], indices[0])]

    diffs = np.diff(indices)
    split_points = np.where(diffs > 1)[0]
    start_indices = np.insert(indices[split_points + 1], 0, indices[0])
    end_indices = np.append(indices[split_points], indices[-1])
    clusters = list(zip(start_indices, end_indices))
    return clusters


# --- Raw Data Plotting Function ---
def plot_raw_data_and_label(raw_data, label, base_filename, save_dir, train_index):
    """
    Generates and saves a plot showing the full raw time series data for ALL dimensions,
    coloring the data red during anomalous periods (label=1), adding a vertical line
    at the train_index, including anomaly indices in the filename, adding text
    annotations for anomaly start/end on the first subplot, and positioning the legend outside.

    Args:
        raw_data (np.ndarray): Raw time series data (n_samples, n_features) or (n_samples,).
        label (np.ndarray): Ground truth labels (n_samples,).
        base_filename (str): Base name of the original data file (without extension).
        save_dir (str): Directory to save the plot in.
        train_index (int): Index indicating the end of the training data.

    Returns:
        bool: True if plot was generated or skipped (exists), False if an error occurred.
    """
    n_samples = raw_data.shape[0]
    if n_samples == 0:
        print("    Warning: No data points to plot.")
        return False

    if len(label) != n_samples:
        print(
            f"    Warning: Label length ({len(label)}) mismatch with data ({n_samples}). Cannot plot.")
        return False

    # --- Calculate anomaly segments FIRST ---
    anomaly_indices = np.where(label == 1)[0]
    anomaly_segments = _find_clusters(anomaly_indices)
    print(
        f"    Detected anomaly segments (start-end): {anomaly_segments if anomaly_segments else 'None'}")

    # --- Construct anomaly info string for filename ---
    anomaly_info_str = ""
    max_anomaly_info_len = 150
    if anomaly_segments:
        segment_strs = []
        current_len = 0
        for s, e in anomaly_segments:
            seg_str = f"_{s}-{e}"
            if current_len + len(seg_str) < max_anomaly_info_len:
                segment_strs.append(seg_str)
                current_len += len(seg_str)
            else:
                if not segment_strs or segment_strs[-1] != "_etc":
                    segment_strs.append("_etc")
                break
        if segment_strs:
            anomaly_info_str = "_Anom" + "".join(segment_strs)

    # --- Construct the expected output filename ---
    expected_plot_filename = os.path.join(
        save_dir, f"{base_filename}_Raw_RedAnomaly_AllDims{anomaly_info_str}.png")

    # --- Check if plot already exists ---
    if os.path.exists(expected_plot_filename):
        print(
            f"  Skipping plot (already exists): {os.path.basename(expected_plot_filename)}")
        return True
    # ------------------------------------

    print(
        f"  Generating raw data plot (all dims, anomalies in red): {os.path.basename(expected_plot_filename)}")
    os.makedirs(save_dir, exist_ok=True)

    # --- Prepare data for plotting ---
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

    # --- Plotting ---
    num_subplots = n_dims
    # Increase height slightly more to accommodate legend better if outside
    fig_height = max(6, 1.1 * num_subplots)
    fig_width = 16 if num_subplots < 10 else 18
    fig = None
    try:
        fig, axes = plt.subplots(num_subplots, 1, figsize=(
            fig_width, fig_height), sharex=True, squeeze=False)
        axes = axes.ravel()
        legend_handles = []  # Collect handles for the final legend

        # Determine y-limits for the first dimension to help position text
        y_min_dim0, y_max_dim0 = np.min(
            data_to_plot[:, 0]), np.max(data_to_plot[:, 0])
        y_range_dim0 = y_max_dim0 - y_min_dim0
        if y_range_dim0 == 0:
            y_range_dim0 = 1.0  # Avoid division by zero if flat
        text_y_offset = y_range_dim0 * 0.05

        for d in range(n_dims):
            ax_raw = axes[d]
            dim_data = data_to_plot[:, d]
            dim_label_str = f'Dim {d}'
            # Adjust ylabel position and alignment
            ax_raw.set_ylabel(dim_label_str, color='steelblue', fontsize=6,
                              rotation=0, ha='right', va='center', labelpad=10)  # Added labelpad
            ax_raw.tick_params(axis='y', labelcolor='steelblue', labelsize=6)
            ax_raw.grid(True, linestyle=':', alpha=0.4)

            # Plot normal data segments
            ln_normal = ax_raw.plot(
                time_index, dim_data, color='steelblue', linewidth=0.6, label='Normal Data')
            # Only add handles once for the legend
            if d == 0:
                legend_handles.extend(ln_normal)

            # Overlay anomalous data segments in red
            ln_anomaly_handle_added = False
            for start, end in anomaly_segments:
                # Robust plotting of red segment
                seg_indices = np.arange(start, end + 1)
                valid_seg_indices = seg_indices[(
                    seg_indices >= 0) & (seg_indices < n_samples)]
                if len(valid_seg_indices) == 0:
                    continue
                segment_data = dim_data[valid_seg_indices]
                segment_time = time_index[valid_seg_indices]
                plot_time = segment_time
                plot_data_seg = segment_data
                if valid_seg_indices[0] > 0:
                    plot_time = np.insert(
                        plot_time, 0, time_index[valid_seg_indices[0] - 1])
                    plot_data_seg = np.insert(
                        plot_data_seg, 0, dim_data[valid_seg_indices[0] - 1])
                if valid_seg_indices[-1] < n_samples - 1:
                    plot_time = np.append(
                        plot_time, time_index[valid_seg_indices[-1] + 1])
                    plot_data_seg = np.append(
                        plot_data_seg, dim_data[valid_seg_indices[-1] + 1])

                # Use label only once for the legend
                anom_label = 'Anomalous Data' if d == 0 and not ln_anomaly_handle_added else "_nolegend_"
                ln_seg = ax_raw.plot(
                    plot_time, plot_data_seg, color='red', linewidth=0.7, label=anom_label)
                if d == 0 and not ln_anomaly_handle_added:
                    legend_handles.extend(ln_seg)
                    ln_anomaly_handle_added = True

                # Add Text Annotation (Only on Dim 0)
                if d == 0 and len(valid_seg_indices) > 0:
                    x_pos = (time_index[valid_seg_indices[0]] +
                             time_index[valid_seg_indices[-1]]) / 2.0
                    y_peak_in_segment = np.max(
                        data_to_plot[valid_seg_indices, 0])
                    y_pos = y_peak_in_segment + text_y_offset
                    text_label = f"[{start}-{end}]"
                    ax_raw.text(x_pos, y_pos, text_label, color='darkred',
                                fontsize='xx-small', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))

            # Draw Vertical Line for Train Index
            vline_label = 'Train End' if d == 0 else "_nolegend_"
            vl = ax_raw.axvline(x=train_index, color='purple', linestyle='--',
                                linewidth=0.8, alpha=0.7, label=vline_label)
            if d == 0:
                legend_handles.append(vl)

        # --- Create Legend OUTSIDE the plot area ---
        # Filter out duplicates from handles/labels
        plot_labs = [h.get_label() for h in legend_handles]
        unique_handles, unique_labels = [], []
        seen_labels = set()
        for h, l in zip(legend_handles, plot_labs):
            if l != "_nolegend_" and l not in seen_labels:
                unique_handles.append(h)
                unique_labels.append(l)
                seen_labels.add(l)
        if unique_handles:
            # Place legend above the top plot, centered horizontally
            # bbox_to_anchor=(0.5, 1.02) means 50% across horizontally, 2% above the axes bounding box
            # loc='lower center' ensures the bottom-center of the legend is anchored at that point
            fig.legend(unique_handles, unique_labels, loc='lower center', ncol=len(unique_handles),  # Arrange horizontally
                       # Adjust y value (e.g., 0.99 or 1.0)
                       fontsize='small', bbox_to_anchor=(0.5, 0.99),
                       borderaxespad=0.1)  # Padding between legend and axes

        axes[-1].set_xlabel('Time Index', fontsize=8)
        # Adjust title position slightly if legend is above
        # Lowered y slightly
        fig.suptitle(
            f'Raw Data (Anomalies Highlighted) & Train Split - {base_filename}', fontsize=10, y=0.97)

        # Adjust layout to prevent overlap - crucial when legend is outside
        # rect=[left, bottom, right, top]
        # Leave space at top for legend and title
        plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

        # --- Save the figure ---
        # Save before potentially showing
        plt.savefig(expected_plot_filename, dpi=300)
        print(
            f"    Plot saved successfully: {os.path.basename(expected_plot_filename)}")
        return True

    except MemoryError as me:
        print(f"    MemoryError generating raw data plot: {me}. Skipping.")
        return False
    except Exception as e:
        print(f"    Error generating raw data plot: {e}")
        traceback.print_exc()
        return False
    finally:
        if fig is not None:
            plt.close(fig)  # Ensure figure is closed

# --- End of plotting function ---


# --- Main Execution Block (No changes needed here from previous version) ---
if __name__ == '__main__':
    overall_start_time = time.time()

    # --- Configuration ---
    parser = argparse.ArgumentParser(
        description='Plot Raw Time Series Data (All Dims) with Anomalies Highlighted & Annotated (Skips Existing)')
    parser.add_argument('--dataset_dir', type=str, default='Datasets/TSB-AD-M/',
                        help='Directory containing the dataset CSV files.')
    parser.add_argument('--file_list', type=str, default='Datasets/File_List/TSB-AD-M.csv',
                        help='CSV file listing the filenames to process.')
    parser.add_argument('--save_dir', type=str, default='Raw_Data_Plots_AllDims_AnomAnnotated',
                        help='Directory to save the output plots.')
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
            filenames = file_list_df.iloc[:, 0].astype(
                str).str.strip().dropna()
            filenames = filenames[filenames != ''].unique().tolist()
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
    failed_plots_count = 0

    for i, filename in enumerate(filenames):
        print(f"\nProcessing file {i+1}/{total_files}: {filename}")
        file_path = os.path.join(DATA_DIR, filename)
        base_fname_for_plot = os.path.splitext(filename)[0]

        # --- Data Loading and Validation ---
        try:
            if not os.path.exists(file_path):
                print(f"  Error: Data file not found: {file_path}. Skipping.")
                failed_plots_count += 1
                continue

            df = pd.read_csv(file_path)
            if 'Label' not in df.columns:
                print(
                    f"  Warning: Skipping {filename} ('Label' column missing).")
                failed_plots_count += 1
                continue
            if df.shape[1] < 2:
                print(
                    f"  Warning: Skipping {filename} (needs >=1 feature column besides 'Label').")
                failed_plots_count += 1
                continue

            df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
            df_cleaned = df.dropna()
            if df_cleaned.empty:
                print(
                    f"  Warning: Skipping {filename} (empty after dropping NaNs or invalid labels).")
                failed_plots_count += 1
                continue

            data = df_cleaned.iloc[:, :-1].values.astype(float)
            label = df_cleaned['Label'].values.astype(int)
            n_samples = data.shape[0]
            print(
                f"  Data Shape (after cleaning): {data.shape}, Label Shape: {label.shape}")
            if n_samples == 0:
                print(
                    "  Warning: Skipping {filename} (no valid data points after cleaning).")
                failed_plots_count += 1
                continue

            # --- Determine Train Index ---
            train_index = -1
            try:
                filename_parts = base_fname_for_plot.split('_')
                if len(filename_parts) >= 3:
                    train_index_str = filename_parts[-3]
                    try:
                        train_index = int(train_index_str)
                        if not (0 < train_index < n_samples):
                            print(
                                f"    Warning: Parsed train index {train_index} out of bounds [1, {n_samples-1}]. Using 30% default.")
                            train_index = max(1, int(0.3 * n_samples))
                    except ValueError:
                        print(
                            f"    Warning: Cannot parse train index '{train_index_str}' from filename. Using 30% default.")
                        train_index = max(1, int(0.3 * n_samples))
                else:
                    print(
                        f"    Warning: Filename '{base_fname_for_plot}' format unexpected for train index parsing. Using 30% default.")
                    train_index = max(1, int(0.3 * n_samples))
            except Exception as e_parse:
                print(
                    f"    Error parsing train index from filename: {e_parse}. Using 30% default.")
                train_index = max(1, int(0.3 * n_samples))
            print(f"    Train Index determined as: {train_index}")

            # --- Generate Raw Data Plot ---
            plot_status = plot_raw_data_and_label(
                data, label, base_fname_for_plot, RAW_PLOT_SAVE_DIR,
                train_index
            )
            if not plot_status:
                failed_plots_count += 1

        except FileNotFoundError:
            print(
                f"  Error: File not found (outer loop): {file_path}. Skipping.")
            failed_plots_count += 1
        except pd.errors.EmptyDataError:
            print(f"  Error: File {filename} is empty. Skipping.")
            failed_plots_count += 1
        except ValueError as ve:
            print(
                f"  Error processing file {filename} (ValueError): {ve}. Skipping.")
            traceback.print_exc()
            failed_plots_count += 1
        except MemoryError as me:
            print(f"  MemoryError processing file {filename}: {me}. Skipping.")
            failed_plots_count += 1
        except Exception as e:
            print(f"  Unexpected critical error processing {filename}: {e}")
            traceback.print_exc()
            failed_plots_count += 1

    # --- End File Loop ---
    # Recalculate final counts
    final_plotted_count = 0
    for filename in filenames:
        base_fname = os.path.splitext(filename)[0]
        temp_file_path = os.path.join(DATA_DIR, filename)
        temp_label = None
        if os.path.exists(temp_file_path):
            try:
                temp_df = pd.read_csv(temp_file_path)
                temp_df['Label'] = pd.to_numeric(
                    temp_df['Label'], errors='coerce')
                temp_df_cleaned = temp_df.dropna()
                if not temp_df_cleaned.empty:
                    temp_label = temp_df_cleaned['Label'].values.astype(int)
            except Exception:
                temp_label = None
        temp_anomaly_info_str = ""
        if temp_label is not None:
            temp_anomaly_indices = np.where(temp_label == 1)[0]
            temp_anomaly_segments = _find_clusters(temp_anomaly_indices)
            temp_max_len = 150
            if temp_anomaly_segments:
                temp_segment_strs = []
                temp_current_len = 0
                for s, e in temp_anomaly_segments:
                    temp_seg_str = f"_{s}-{e}"
                    if temp_current_len + len(temp_seg_str) < temp_max_len:
                        temp_segment_strs.append(temp_seg_str)
                        temp_current_len += len(temp_seg_str)
                    else:
                        if not temp_segment_strs or temp_segment_strs[-1] != "_etc":
                            temp_segment_strs.append("_etc")
                        break
                if temp_segment_strs:
                    temp_anomaly_info_str = "_Anom" + \
                        "".join(temp_segment_strs)
        final_expected_path = os.path.join(
            RAW_PLOT_SAVE_DIR, f"{base_fname}_Raw_RedAnomaly_AllDims{temp_anomaly_info_str}.png")
        if os.path.exists(final_expected_path):
            final_plotted_count += 1

    print(f"\n--- Run Summary ---")
    print(f"Checked {total_files} listed files.")
    print(
        f"Found {final_plotted_count} plot files in the output directory ({RAW_PLOT_SAVE_DIR}).")
    print(
        f"Failed during data loading or plotting for {failed_plots_count} files.")
    print(
        f"\nTotal execution time: {time.time() - overall_start_time:.2f} seconds")
