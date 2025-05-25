import argparse
import json
import os
import random
import re
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from MAD.core.constants import PLOT_FIG_WIDTH, PLOT_NBINS
from MAD.core.MAD import F_May_22
from MAD.core.mad_utils import _find_clusters, thousands_formatter
from TSB_AD.evaluation.basic_metrics import basic_metricor
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank


# Custom JSON encoder to handle NumPy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def extract_train_index_from_filename(filename):
    """Extracts the Train Index from TSB-AD filenames using regex.
    Format: [index]_[Dataset Name]_id_[id]_[Domain]_tr_[Train Index]_1st_[First Anomaly Index].csv
    """
    match = re.search(r"_tr_(\d+)_", filename)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Could not parse Train Index from filename: {filename}. Expected format like '_tr_[NUMBER]_."
    )


# --- Model Runner Functions ---
def run_F_May_22(data_train, data_full_for_decision, y_train_for_fit, HP):
    """Runs F_May_22.
    Fit is called with training data (and optional training labels for hints).
    Decision_function is called with the full data (train + test context).
    Returns scaled scores for the full data, plus other artifacts.
    """
    clf = F_May_22(HP=HP)

    clf.fit(X_train=data_train, y_train=y_train_for_fit)

    score_full = clf.decision_function(
        X_test=data_full_for_decision
    )  # Changed X to X_test

    # Aggregate S1 identified ranges and S2 JSON input segments from per-feature artifacts
    all_s1_ranges_agg = set()  # Use set to store unique tuples
    all_s2_segments_agg = set()

    if clf.per_feature_artifacts_:
        for feat_idx, artifacts in clf.per_feature_artifacts_.items():
            if artifacts and artifacts.get("error") is None:
                s1_ranges = artifacts.get("identified_ranges", [])
                s2_segments = artifacts.get("s2_json_input_segments", [])
                if s1_ranges:
                    for r in s1_ranges:
                        if isinstance(r, list) and len(r) == 2:
                            all_s1_ranges_agg.add(tuple(r))
                if s2_segments:
                    for s in s2_segments:
                        if isinstance(s, list) and len(s) == 2:
                            all_s2_segments_agg.add(tuple(s))

    if (
        score_full is None
        or not isinstance(score_full, np.ndarray)
        or score_full.size == 0
        or score_full.shape[0] != data_full_for_decision.shape[0]
    ):
        # Return zeros for the expected full length if scoring failed
        return (
            np.zeros(data_full_for_decision.shape[0]),
            {},  # Return empty dict for artifacts if scoring failed
            clf,
        )

    # Scale the full scores
    scaled_score_full = np.nan_to_num(
        score_full,
        nan=0.0,
        posinf=(
            np.nanmax(score_full[np.isfinite(score_full)])
            if np.any(np.isfinite(score_full))
            else 1.0
        ),
        neginf=0.0,
    )
    score_ptp_full = np.ptp(
        scaled_score_full
    )  # Use scaled_score_full for ptp after nan_to_num
    if score_ptp_full == 0:
        # scaled_score_full is already nan_to_num, so if ptp is 0, it's constant (likely all zeros)
        pass  # No further scaling needed if already constant
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_score_full = scaler.fit_transform(
            scaled_score_full.reshape(-1, 1)
        ).ravel()

    return (
        scaled_score_full,
        clf.per_feature_artifacts_,  # Return the full dictionary
        clf,
    )


# --- Visualization Function (NEW VERSION) ---
def visualize_errors(
    data,  # New: (n_samples, n_features) NumPy array of raw data
    original_label,  # (n_samples,) NumPy array of true labels
    score,  # (n_samples,) NumPy array of anomaly scores (0-1 normalized)
    per_feature_artifacts,  # Dict: {feat_idx: {plot_path, json_path, identified_ranges, anomalous_indices, s2_json_input_segments, error}}
    base_plot_filename,  # String: base name for the plot file
    plot_dir,  # String: directory to save the plot
    train_idx_val=None,  # New: index to distinguish training data
    model_prefix="",  # String: prefix for titles/filenames (e.g., model name)
):
    """Generates a multi-feature plot showing raw data, anomaly scores, true labels, and model-identified ranges."""
    plt.style.use("seaborn-v0_8-whitegrid")  # Using a style

    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[0] == 0:
        return
    n_samples, n_features = data.shape

    if not (
        isinstance(original_label, np.ndarray)
        and original_label.ndim == 1
        and original_label.shape[0] == n_samples
    ):
        return
    if not (
        isinstance(score, np.ndarray)
        and score.ndim == 1
        and score.shape[0] == n_samples
    ):
        return
    # Validate aggregated range inputs
    if not isinstance(per_feature_artifacts, dict):
        per_feature_artifacts = {}

    os.makedirs(plot_dir, exist_ok=True)

    # Determine plot dimensions
    fig_width_viz, fig_height_per_feature_viz, dpi_val_viz = (
        PLOT_FIG_WIDTH,
        3,
        150,
    )  # Width = 320
    fig_height_viz = max(6, n_features * fig_height_per_feature_viz)

    fig, axes = plt.subplots(
        nrows=n_features,
        ncols=1,
        figsize=(fig_width_viz, fig_height_viz),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.flatten()
    time_range = np.arange(n_samples)

    true_anomaly_indices = np.where(original_label == 1)[0]
    true_anomaly_label_added = False  # ADD: For true anomalies plotted in red
    score_label_added = False  # ADD: For the score plot on secondary axis

    s1_range_label_added = False
    s2_segment_label_added = False
    data_label_added = False  # For raw data
    train_data_label_added = False  # For train data
    test_data_label_added = False
    per_feature_s2_anomaly_label_added = {}

    for i_feat in range(n_features):
        ax = axes_flat[i_feat]

        # Increase tick font size
        ax.tick_params(axis="both", which="major", labelsize=12)

        # 1. Plot raw data for the feature
        current_data_label = (
            None  # Reset for each feature if needed, or manage globally
        )

        show_train_test_split_viz = (
            train_idx_val is not None and 0 < train_idx_val < n_samples
        )

        if show_train_test_split_viz:
            # Plot training data part
            train_label_viz = None
            if not train_data_label_added:
                train_label_viz = "Training Data"
                train_data_label_added = True
            ax.plot(
                time_range[:train_idx_val],
                data[:train_idx_val, i_feat],
                color="darkgreen",
                linestyle="-",
                lw=1.0,
                label=train_label_viz,
            )
            # Plot test data part, starting from the last point of train data for continuity
            test_label_viz = None
            if not test_data_label_added:
                test_label_viz = "Test Data"
                test_data_label_added = True
            ax.plot(
                time_range[train_idx_val:],
                data[train_idx_val:, i_feat],
                color="darkblue",
                linestyle="-",
                lw=1.0,
                label=test_label_viz,
            )

            # Add vertical lines for training data start and end
            ax.axvline(
                x=0,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label="Train Start" if i_feat == 0 else "",
            )  # Label only once
            ax.axvline(
                x=train_idx_val - 1,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label="Train End" if i_feat == 0 else "",
            )  # Label only once

            # Add ticks for training data start and end
            existing_ticks = list(ax.get_xticks())
            new_ticks = sorted(list(set(existing_ticks + [0, train_idx_val - 1])))
            ax.set_xticks(new_ticks)
        else:
            # Default: plot all data as one series
            if not data_label_added:  # If not split, use the general data label
                current_data_label = f"Feature Data"
                data_label_added = True
            ax.plot(
                time_range,
                data[:, i_feat],
                color="dimgray",
                lw=1.0,
                label=current_data_label,
            )

        ax.set_ylabel(f"F{i_feat} Val", fontsize="small")

        # Apply consistent tick locator AFTER potential ylim adjustments
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=PLOT_NBINS, integer=True, prune="both")
        )
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))  # ADDED

        # 2. New: Plot TP, FP, FN segments (Based on final aggregated score) # REMOVE THIS SECTION
        # True Positives (Red)
        # if tp_ranges_viz:
        #     for start_tp, end_tp in tp_ranges_viz:
        #         plot_start, plot_end = max(0, start_tp), min(n_samples, end_tp + 1)
        #         if plot_start < plot_end:
        #             label_text_tp = "True Positive" if not tp_label_added else None
        #             # --- Apply new plotting style for TP ---
        #             segment_indices_tp = time_range[plot_start:plot_end]
        #             segment_data_tp = data[plot_start:plot_end, i_feat]
        #
        #             plot_time_segment_tp = segment_indices_tp.copy()
        #             plot_data_segment_tp = segment_data_tp.copy()

        #             if plot_start > 0:
        #                 plot_time_segment_tp = np.insert(plot_time_segment_tp, 0, time_range[plot_start - 1])
        #                 plot_data_segment_tp = np.insert(plot_data_segment_tp, 0, data[plot_start - 1, i_feat])
        #             if end_tp +1 < n_samples: # Use end_tp from loop, plot_end might be n_samples
        #                 plot_time_segment_tp = np.append(plot_time_segment_tp, time_range[end_tp + 1])
        #                 plot_data_segment_tp = np.append(plot_data_segment_tp, data[end_tp + 1, i_feat])
        #             # --- End new plotting style for TP ---

        #             ax.plot(
        #                 plot_time_segment_tp, # Use modified time
        #                 plot_data_segment_tp, # Use modified data
        #                 color="darkred",
        #                 lw=1.3,
        #                 label=label_text_tp,
        #                 zorder=5,
        #             )
        #             if label_text_tp:
        #                 tp_label_added = True

        # False Positives (Magenta)
        # if fp_ranges_viz:
        #     for start_fp, end_fp in fp_ranges_viz:
        #         plot_start, plot_end = max(0, start_fp), min(n_samples, end_fp + 1)
        #         if plot_start < plot_end:
        #             label_text_fp = "False Positive" if not fp_label_added else None
        #             # --- Apply new plotting style for FP ---
        #             segment_indices_fp = time_range[plot_start:plot_end]
        #             segment_data_fp = data[plot_start:plot_end, i_feat]

        #             plot_time_segment_fp = segment_indices_fp.copy()
        #             plot_data_segment_fp = segment_data_fp.copy()

        #             if plot_start > 0:
        #                 plot_time_segment_fp = np.insert(plot_time_segment_fp, 0, time_range[plot_start - 1])
        #                 plot_data_segment_fp = np.insert(plot_data_segment_fp, 0, data[plot_start - 1, i_feat])
        #             if end_fp + 1 < n_samples: # Use end_fp from loop
        #                 plot_time_segment_fp = np.append(plot_time_segment_fp, time_range[end_fp + 1])
        #                 plot_data_segment_fp = np.append(plot_data_segment_fp, data[end_fp + 1, i_feat])
        #             # --- End new plotting style for FP ---
        #
        #             ax.plot(
        #                 plot_time_segment_fp, # Use modified time
        #                 plot_data_segment_fp, # Use modified data
        #                 color="darkmagenta",
        #                 lw=1.3,
        #                 label=label_text_fp,
        #                 linestyle="-",
        #                 zorder=4,
        #             )
        #             if label_text_fp:
        #                 fp_label_added = True

        # False Negatives (Cyan)
        # if fn_ranges_viz:
        #     for start_fn, end_fn in fn_ranges_viz:
        #         plot_start, plot_end = max(0, start_fn), min(n_samples, end_fn + 1)
        #         if plot_start < plot_end:
        #             label_text_fn = "False Negative" if not fn_label_added else None
        #             # --- Apply new plotting style for FN ---
        #             segment_indices_fn = time_range[plot_start:plot_end]
        #             segment_data_fn = data[plot_start:plot_end, i_feat]

        #             plot_time_segment_fn = segment_indices_fn.copy()
        #             plot_data_segment_fn = segment_data_fn.copy()

        #             if plot_start > 0:
        #                 plot_time_segment_fn = np.insert(plot_time_segment_fn, 0, time_range[plot_start - 1])
        #                 plot_data_segment_fn = np.insert(plot_data_segment_fn, 0, data[plot_start - 1, i_feat])
        #             if end_fn + 1 < n_samples: # Use end_fn from loop
        #                 plot_time_segment_fn = np.append(plot_time_segment_fn, time_range[end_fn + 1])
        #                 plot_data_segment_fn = np.append(plot_data_segment_fn, data[end_fn + 1, i_feat])
        #             # --- End new plotting style for FN ---

        #             ax.plot(
        #                 plot_time_segment_fn, # Use modified time
        #                 plot_data_segment_fn, # Use modified data
        #                 color="darkcyan",
        #                 lw=1.3,
        #                 label=label_text_fn,
        #                 linestyle="-",
        #                 zorder=4,
        #             )
        #             if label_text_fn:
        #                 fn_label_added = True
        # --- END REMOVE THIS SECTION ---

        # --- ADD: Plot True Anomalies (from original_label) ---
        true_anomaly_ranges = _find_clusters(true_anomaly_indices)
        if true_anomaly_ranges:
            for start_true, end_true in true_anomaly_ranges:
                plot_start, plot_end = max(0, start_true), min(n_samples, end_true + 1)
                if plot_start < plot_end:
                    label_text_true = (
                        "True Anomaly" if not true_anomaly_label_added else None
                    )
                    segment_indices_true = time_range[plot_start:plot_end]
                    segment_data_true = data[plot_start:plot_end, i_feat]

                    plot_time_segment_true = segment_indices_true.copy()
                    plot_data_segment_true = segment_data_true.copy()

                    if plot_start > 0:
                        plot_time_segment_true = np.insert(
                            plot_time_segment_true, 0, time_range[plot_start - 1]
                        )
                        plot_data_segment_true = np.insert(
                            plot_data_segment_true, 0, data[plot_start - 1, i_feat]
                        )
                    if end_true + 1 < n_samples:
                        plot_time_segment_true = np.append(
                            plot_time_segment_true, time_range[end_true + 1]
                        )
                        plot_data_segment_true = np.append(
                            plot_data_segment_true, data[end_true + 1, i_feat]
                        )

                    ax.plot(
                        plot_time_segment_true,
                        plot_data_segment_true,
                        color="darkred",
                        lw=1.3,
                        label=label_text_true,
                        zorder=5,
                    )
                    if label_text_true:
                        true_anomaly_label_added = True
        # --- END ADD: Plot True Anomalies ---

        # --- ADD: Secondary Y-axis for Anomaly Scores ---
        # Create a secondary y-axis for the score, only if not already added for this subplot
        # Check if ax has an attribute that indicates a twinx has been made (not straightforward)
        # For now, assume we create it once per feature plot (i_feat)
        if not hasattr(ax, "_score_axis_added_visualize_errors"):
            ax_score = ax.twinx()
            score_plot_label = "Anomaly Score" if not score_label_added else None
            ax_score.plot(
                time_range,
                score,  # This is the (n_samples,) normalized score from the model
                color="red",
                linestyle="-",
                lw=1.0,
                label=score_plot_label,
                alpha=0.6,
            )
            ax_score.set_ylabel("Anomaly Score", color="red", fontsize="small")
            ax_score.tick_params(axis="y", labelcolor="red", labelsize="x-small")
            ax_score.set_ylim(-0.1, 1.1)  # Scores are 0-1 normalized
            ax._score_axis_added_visualize_errors = (
                True  # Mark that score axis is added
            )
            if score_plot_label:
                score_label_added = (
                    True  # Ensure legend item is added only once globally
                )
        # --- END ADD: Secondary Y-axis for Anomaly Scores ---

        # --- Per-Feature Artifacts Visualization ---
        feature_artifacts = per_feature_artifacts.get(i_feat, {})
        if feature_artifacts and not feature_artifacts.get("error"):

            # 4. Highlight S1 identified ranges for THIS feature (gray shaded regions)
            s1_ranges_feat = feature_artifacts.get("identified_ranges_s1", [])
            valid_s1_ranges = [
                item
                for item in s1_ranges_feat
                if isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(n, (int, np.integer)) for n in item)
            ]
            if valid_s1_ranges:
                for start_s1, end_s1 in valid_s1_ranges:
                    safe_start_s1, safe_end_s1 = max(0, start_s1), min(
                        n_samples - 1, end_s1
                    )
                    if safe_start_s1 <= safe_end_s1:
                        label_text_s1 = None
                        if not s1_range_label_added:
                            label_text_s1 = "S1 Identified Ranges (Feat.)"
                            s1_range_label_added = True
                        # --- Apply new plotting style for S1 ranges ---
                        plot_time_segment_s1 = time_range[
                            safe_start_s1 : safe_end_s1 + 1
                        ].copy()
                        plot_data_segment_s1 = data[
                            safe_start_s1 : safe_end_s1 + 1, i_feat
                        ].copy()

                        if safe_start_s1 > 0:
                            plot_time_segment_s1 = np.insert(
                                plot_time_segment_s1, 0, time_range[safe_start_s1 - 1]
                            )
                            plot_data_segment_s1 = np.insert(
                                plot_data_segment_s1, 0, data[safe_start_s1 - 1, i_feat]
                            )
                        if safe_end_s1 < n_samples - 1:
                            plot_time_segment_s1 = np.append(
                                plot_time_segment_s1, time_range[safe_end_s1 + 1]
                            )
                            plot_data_segment_s1 = np.append(
                                plot_data_segment_s1, data[safe_end_s1 + 1, i_feat]
                            )

                        if len(plot_time_segment_s1) > 0:
                            ax.plot(
                                plot_time_segment_s1,
                                plot_data_segment_s1,
                                color="darkslategray",  # Changed from darkgray for plot()
                                lw=1.8,  # Slightly thicker than base data line
                                label=label_text_s1,
                                zorder=3,  # Ensure visible but behind anomalies
                            )
                        # --- End new plotting style ---

            # 5. Highlight S2 JSON input segments for THIS feature (light sky blue shaded regions)
            s2_segments_feat = feature_artifacts.get("s2_included_snippet_ranges", [])
            valid_step2_ranges = [
                item
                for item in s2_segments_feat
                if isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(n, (int, np.integer)) for n in item)
            ]
            if valid_step2_ranges:
                for start_s2, end_s2 in valid_step2_ranges:
                    safe_start_s2, safe_end_s2 = max(0, start_s2), min(
                        n_samples - 1, end_s2
                    )
                    if safe_start_s2 <= safe_end_s2:
                        label_text_s2 = None
                        if not s2_segment_label_added:
                            label_text_s2 = "S2 Snippet Ranges (Feat.)"
                            s2_segment_label_added = True
                        # --- Apply new plotting style for S2 segments ---
                        plot_time_segment_s2 = time_range[
                            safe_start_s2 : safe_end_s2 + 1
                        ].copy()
                        plot_data_segment_s2 = data[
                            safe_start_s2 : safe_end_s2 + 1, i_feat
                        ].copy()

                        if safe_start_s2 > 0:
                            plot_time_segment_s2 = np.insert(
                                plot_time_segment_s2, 0, time_range[safe_start_s2 - 1]
                            )
                            plot_data_segment_s2 = np.insert(
                                plot_data_segment_s2, 0, data[safe_start_s2 - 1, i_feat]
                            )
                        if safe_end_s2 < n_samples - 1:
                            plot_time_segment_s2 = np.append(
                                plot_time_segment_s2, time_range[safe_end_s2 + 1]
                            )
                            plot_data_segment_s2 = np.append(
                                plot_data_segment_s2, data[safe_end_s2 + 1, i_feat]
                            )

                        if len(plot_time_segment_s2) > 0:
                            ax.plot(
                                plot_time_segment_s2,
                                plot_data_segment_s2,
                                color="steelblue",  # Changed from deepskyblue
                                lw=1.8,
                                linestyle="--",  # Dashed to differentiate from S1
                                label=label_text_s2,
                                zorder=3.1,  # Slightly above S1
                            )
                        # --- End new plotting style ---

            # 6. Plot S2 identified anomalous indices for THIS feature (e.g., green 'x' markers)
            # MODIFIED to use anomaly_scores
            s2_anomaly_scores_feat = feature_artifacts.get(
                "feature_scores_s2", []
            )  # Changed key to feature_scores_s2
            if (
                isinstance(s2_anomaly_scores_feat, list)
                and len(s2_anomaly_scores_feat) == n_samples
            ):
                s2_scores_np = np.array(s2_anomaly_scores_feat)
                # Define a threshold for what constitutes a "high score" worth marking - S2 scores are binary (0 or 1)
                # HIGH_SCORE_THRESHOLD = 0.75
                s2_anomalous_indices_for_feat = np.where(s2_scores_np == 1)[0]

                if s2_anomalous_indices_for_feat.size > 0:
                    y_min, y_max = ax.get_ylim()  # Get y-limits of the feature plot
                    # Plot markers slightly above the minimum y-value
                    marker_y_position = y_min + 0.05 * (y_max - y_min)

                    current_s2_label_text = None
                    if not per_feature_s2_anomaly_label_added.get(i_feat, False):
                        current_s2_label_text = (
                            f"S2 Anomaly (F{i_feat})"  # More direct label
                        )
                        per_feature_s2_anomaly_label_added[i_feat] = True

                    ax.plot(
                        s2_anomalous_indices_for_feat,
                        [marker_y_position] * len(s2_anomalous_indices_for_feat),
                        marker="o",  # Changed marker to 'o' for differentiation
                        markersize=4,  # Slightly smaller markers
                        linestyle="None",  # No line connecting markers
                        color="darkorange",  # Changed color to orange
                        alpha=0.9,
                        label=current_s2_label_text,
                        zorder=7,  # Ensure markers are very visible
                    )
        # --- End Per-Feature Artifacts ---

        ax.tick_params(axis="y", labelsize="x-small")  # Primary Y for feature data
        ax.tick_params(axis="x", labelsize="small")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(
            -0.01 * n_samples, n_samples * 1.01
        )  # Add a little padding to x-limits

        # Show x-axis labels and ticks on all subplots
        ax.set_xlabel("Time Index", fontsize="medium")
        # ax.tick_params(axis='x', labelbottom=True) # Ensure x-axis labels are on for all

    # Consolidate legends from all axes (primary feature axes and secondary score axes)
    handles, labels = [], []
    for ax_loop in fig.axes:  # Iterate over all axes instances in the figure
        h, l = ax_loop.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if (
                label not in labels and label is not None
            ):  # Avoid duplicate and None labels
                handles.append(handle)
                labels.append(label)

    if (
        handles
    ):  # Place legend outside the plot area if many features, or adjust position
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=min(len(handles), 8),  # Increased ncol for legend further
            fontsize="small",
        )  # Increased ncol for legend

    title = f"{model_prefix} - {base_plot_filename}\nRaw Data, True Anomaly(Red), Score(Red Line), S1(Gray), S2 Seg(SkyBlue) - Per Feature"  # Updated title
    fig.suptitle(title, fontsize="large", y=0.99)

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # ADDED to minimize whitespace, rect adjusted for suptitle

    # Save the figure
    plot_filename_new = os.path.join(
        plot_dir, f"{base_plot_filename}_{model_prefix}_RawData_Score_Labels_Ranges.png"
    )
    try:
        plt.savefig(plot_filename_new, dpi=dpi_val_viz)
    except Exception as e_save_plot:
        pass
    finally:
        if plt.fignum_exists(fig.number):
            plt.close(fig)
    plt.close("all")  # Ensure all figures are closed


# --- Setup & Global Configuration ---
seed = 2024
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BASE_METRIC_ORDER = [
    "VUS-ROC",
    "AUC-PR",
    "AUC-ROC",
    "Affiliation-F",
    "R-based-F1",
    "Event-based-F1",
    "PA-F1",
    "Standard-F1",
]
MODEL_NAMES = ["F_May_22"]


def get_ordered_columns(current_columns):
    """Orders columns for the results DataFrame for better readability."""
    model_prefixes = {name: name for name in MODEL_NAMES}
    final_order = []
    processed_columns = set()

    if "filename" in current_columns:
        final_order.append("filename")
        processed_columns.add("filename")

    for prefix in model_prefixes.values():
        for col in [f"{prefix}_VUS-PR", f"{prefix}_VUS-ROC"]:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)

    # Add Benchmark Comparison Columns (VUS-PR Diffs) earlier
    for prefix in model_prefixes.values():
        benchmark_diff_cols = [
            f"{prefix}_F_May_22_vs_Avg_VUS-PR_Diff",
            f"{prefix}_F_May_22_vs_Max_VUS-PR_Diff",
        ]
        for col in benchmark_diff_cols:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)
        # Add the other benchmark scores
        benchmark_score_cols = [
            f"{prefix}_Benchmark_Avg_VUS-PR",
            f"{prefix}_Benchmark_Max_VUS-PR",
        ]
        for col in benchmark_score_cols:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)

    for prefix in model_prefixes.values():
        cols = [
            f"{prefix}_{m}" for m in BASE_METRIC_ORDER if m not in ["VUS-PR", "VUS-ROC"]
        ]
        for col in cols:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)

    for prefix in model_prefixes.values():
        other_info_cols = [
            f"{prefix}_runtime",
            f"{prefix}_FP_count_ext",
            f"{prefix}_FN_count_ext",
        ]
        error_cols = [
            f"{prefix}_{e}"
            for e in ["Error", "Scaling_Error", "Metrics_Error", "Visualization_Error"]
        ]
        result_cols = [f"{prefix}_Metrics_Result"]
        cols_to_add = other_info_cols + error_cols + result_cols
        for col in cols_to_add:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)

    # Add per_feature_artifacts_path if it exists
    per_feat_artifact_path_col = f"{prefix}_per_feature_artifacts_path"
    if (
        per_feat_artifact_path_col in current_columns
        and per_feat_artifact_path_col not in processed_columns
    ):
        final_order.append(per_feat_artifact_path_col)
        processed_columns.add(per_feat_artifact_path_col)

    remaining = sorted(list(set(current_columns) - processed_columns))
    final_order.extend(remaining)
    return final_order


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run F_May_22 Anomaly Detector.")
    parser.add_argument(
        "--file_list_path",
        type=str,
        default=os.path.join("Datasets", "File_List", "TSB-AD-U-Eva-Debug.csv"),
        help="Path to the CSV file containing the list of datasets to process.",
    )
    parser.add_argument(
        "--unsupervised",
        action="store_true",
    )
    args = parser.parse_args()

    overall_start_time = time.time()
    RESULTS_CSV_PATH = os.path.join("MAD", "F_May_22_detailed_results.csv")
    PLOT_DIR_BASE = os.path.join("MAD", "F_May_22_ScoreLabel_Plots")
    DATA_DIR = os.path.join("Datasets", "TSB-AD-U")

    VISUALIZE_ANOMALIES = True
    FILE_LIST_PATH = args.file_list_path

    for d_path_main in [
        PLOT_DIR_BASE,
        DATA_DIR,
    ]:
        os.makedirs(d_path_main, exist_ok=True)
    if FILE_LIST_PATH:
        os.makedirs(os.path.dirname(FILE_LIST_PATH), exist_ok=True)
    metricor = basic_metricor()
    metrics_calculator = get_metrics
    window_finder = find_length_rank
    processed_files, all_results = set(), []
    if os.path.exists(RESULTS_CSV_PATH):
        existing_df_main = pd.read_csv(RESULTS_CSV_PATH)
        if "filename" in existing_df_main.columns:
            all_results = existing_df_main.where(
                pd.notna(existing_df_main), None
            ).to_dict("records")
            processed_files.update(
                existing_df_main["filename"].dropna().astype(str).unique()
            )
    filenames_to_process_list_main = []
    if not FILE_LIST_PATH:
        raise ValueError("FILE_LIST_PATH is not defined.")
    if not os.path.exists(FILE_LIST_PATH):
        raise FileNotFoundError(f"File list {FILE_LIST_PATH} not found.")
    df_list_main = pd.read_csv(FILE_LIST_PATH)
    if not df_list_main.empty and len(df_list_main.columns) > 0:
        filenames_to_process_list_main = (
            df_list_main.iloc[:, 0].astype(str).str.strip().dropna().unique().tolist()
        )
    if not filenames_to_process_list_main:
        raise ValueError(f"No valid filenames in {FILE_LIST_PATH}.")
    files_to_run_now_main = [
        f_run
        for f_run in filenames_to_process_list_main
        if f_run not in processed_files
    ]
    processed_count_this_run_final = 0

    # --- Load and Process Benchmark Data ---
    benchmark_metrics = {}
    BENCHMARK_CSV_PATH = os.path.join(
        "benchmark_exp", "benchmark_eval_results", "uni_mergedTable_VUS-PR.csv"
    )
    if os.path.exists(BENCHMARK_CSV_PATH):
        try:
            benchmark_df = pd.read_csv(BENCHMARK_CSV_PATH)
            if "file" not in benchmark_df.columns:
                pass
            else:
                # Identify algorithm columns (assuming they are between 'file' and 'ts_len')
                # This is a heuristic, might need adjustment if CSV structure changes.
                file_col_idx = benchmark_df.columns.get_loc("file")
                ts_len_col_idx = (
                    benchmark_df.columns.get_loc("ts_len")
                    if "ts_len" in benchmark_df.columns
                    else -1
                )

                if ts_len_col_idx != -1 and ts_len_col_idx > file_col_idx + 1:
                    algo_columns = benchmark_df.columns[
                        file_col_idx + 1 : ts_len_col_idx
                    ].tolist()

                    for _, row in benchmark_df.iterrows():
                        filename_bench = row["file"]
                        vus_pr_scores = []
                        for algo_col in algo_columns:
                            if algo_col in row:
                                score_val = pd.to_numeric(
                                    row[algo_col], errors="coerce"
                                )
                                if not pd.isna(score_val):
                                    vus_pr_scores.append(score_val)

                        if vus_pr_scores:
                            avg_vus_pr = (
                                np.nanmean(vus_pr_scores) if vus_pr_scores else np.nan
                            )
                            max_vus_pr = (
                                np.nanmax(vus_pr_scores) if vus_pr_scores else np.nan
                            )
                            benchmark_metrics[filename_bench] = {
                                "avg": avg_vus_pr,
                                "max": max_vus_pr,
                            }
                        else:
                            benchmark_metrics[filename_bench] = {
                                "avg": np.nan,
                                "max": np.nan,
                            }
                else:
                    pass
        except Exception as e:
            pass
    else:
        pass
    # --- End Load and Process Benchmark Data ---

    # --- Backfill Missing Benchmark Columns in Existing Results (all_results) ---
    if (
        all_results and benchmark_metrics
    ):  # Only backfill if there are existing results and benchmark data
        model_prefix_backfill = MODEL_NAMES[0]  # Should be "F_May_22"
        F_May_22_vus_pr_col_bf = f"{model_prefix_backfill}_VUS-PR"
        bench_avg_col_bf = f"{model_prefix_backfill}_Benchmark_Avg_VUS-PR"
        bench_max_col_bf = f"{model_prefix_backfill}_Benchmark_Max_VUS-PR"
        diff_avg_col_bf = f"{model_prefix_backfill}_F_May_22_vs_Avg_VUS-PR_Diff"
        diff_max_col_bf = f"{model_prefix_backfill}_F_May_22_vs_Max_VUS-PR_Diff"
        benchmark_cols_to_check = [
            bench_avg_col_bf,
            bench_max_col_bf,
            diff_avg_col_bf,
            diff_max_col_bf,
        ]
        updated_count_backfill = 0

        for row_dict_bf in all_results:
            filename_bf = row_dict_bf.get("filename")
            if not filename_bf:
                continue

            # Check if any of the benchmark columns are missing
            is_any_col_missing = False
            for col_bf in benchmark_cols_to_check:
                val_bf = row_dict_bf.get(col_bf)
                if pd.isna(val_bf) or (
                    isinstance(val_bf, str) and val_bf.startswith("N/A")
                ):
                    is_any_col_missing = True
                    break

            if is_any_col_missing:
                F_May_22_score_bf = pd.to_numeric(
                    row_dict_bf.get(F_May_22_vus_pr_col_bf), errors="coerce"
                )

                if not pd.isna(F_May_22_score_bf):
                    bench_data_for_file_bf = benchmark_metrics.get(filename_bf)
                    if bench_data_for_file_bf:
                        avg_bench_score_bf = bench_data_for_file_bf.get("avg", np.nan)
                        max_bench_score_bf = bench_data_for_file_bf.get("max", np.nan)

                        row_dict_bf[bench_avg_col_bf] = avg_bench_score_bf
                        row_dict_bf[bench_max_col_bf] = max_bench_score_bf
                        row_dict_bf[diff_avg_col_bf] = (
                            F_May_22_score_bf - avg_bench_score_bf
                            if not pd.isna(avg_bench_score_bf)
                            else np.nan
                        )
                        row_dict_bf[diff_max_col_bf] = (
                            F_May_22_score_bf - max_bench_score_bf
                            if not pd.isna(max_bench_score_bf)
                            else np.nan
                        )
                        updated_count_backfill += 1
                    else:
                        # No benchmark data for this specific file
                        for col_bf_fill in benchmark_cols_to_check:
                            row_dict_bf[col_bf_fill] = "N/A_Benchmark_Filled"
                        updated_count_backfill += (
                            1  # Count as updated because we filled placeholders
                        )
                else:
                    # F_May_22 score itself is missing, so can't compare
                    for col_bf_fill in benchmark_cols_to_check:
                        row_dict_bf[col_bf_fill] = "N/A_SelfNoScore_Filled"
                    updated_count_backfill += 1  # Count as updated
        if updated_count_backfill > 0:
            pass
        else:
            pass
    elif not benchmark_metrics:
        pass
    # --- End Backfill ---

    pbar_files = tqdm(files_to_run_now_main, desc="Processing files", unit="file")
    for filename_main_final in pbar_files:
        pbar_files.set_postfix_str(f"Current file: {filename_main_final}", refresh=True)
        loop_start_time_final = time.time()
        file_path_final = os.path.join(DATA_DIR, filename_main_final)
        current_results_final = {"filename": filename_main_final}
        model_prefix_final = MODEL_NAMES[0]

        df_raw = pd.read_csv(file_path_final)
        df_final_run = df_raw.dropna()

        if df_final_run.empty:
            current_results_final[f"{model_prefix_final}_Error"] = (
                f"Data empty after dropna for {filename_main_final}"
            )
            all_results.append(current_results_final)
            continue  # Skip to next file

        label_col_target_name = "Label"
        if label_col_target_name not in df_final_run.columns:
            found_label_col_case_insensitive = next(
                (col for col in df_final_run.columns if col.lower() == "label"), None
            )
            if found_label_col_case_insensitive:
                df_final_run.rename(
                    columns={found_label_col_case_insensitive: label_col_target_name},
                    inplace=True,
                )
            else:
                current_results_final[f"{model_prefix_final}_Error"] = (
                    f"Missing 'Label' column in {filename_main_final}"
                )
                all_results.append(current_results_final)
                continue

        try:
            data_full_for_run = df_final_run.iloc[:, 0:-1].values.astype(float)
            labels_for_eval = df_final_run[label_col_target_name].astype(int).to_numpy()
        except IndexError as e_idx_split:
            current_results_final[f"{model_prefix_final}_Error"] = (
                f"Error during data/label split for {filename_main_final}: {e_idx_split}"
            )
            all_results.append(current_results_final)
            continue

        if (
            data_full_for_run.shape[0] != labels_for_eval.shape[0]
            or data_full_for_run.shape[0] == 0
            or data_full_for_run.ndim != 2
            or data_full_for_run.shape[1] == 0
        ):
            current_results_final[f"{model_prefix_final}_Error"] = (
                f"Data/label mismatch or zero/invalid dimensions in {filename_main_final} after split."
            )
            all_results.append(current_results_final)
            continue

        window_final_run = 10
        if window_finder:
            try:
                estimated_window_final_run = window_finder(data_full_for_run, rank=1)
                if (
                    isinstance(estimated_window_final_run, (int, float))
                    and 5
                    < estimated_window_final_run
                    < data_full_for_run.shape[0] * 0.5
                ):
                    window_final_run = int(estimated_window_final_run)
            except Exception as e_win:
                pass

        # User-specified train_index logic
        try:
            train_idx_val = extract_train_index_from_filename(filename_main_final)
            if not (0 < train_idx_val < data_full_for_run.shape[0]):
                raise ValueError(
                    f"Parsed Train Index {train_idx_val} out of bounds for data length {data_full_for_run.shape[0]}."
                )
        except (IndexError, ValueError) as e_parse_idx:
            current_results_final[f"{model_prefix_final}_Error"] = (
                f"Error parsing train_index from {filename_main_final}: {e_parse_idx}"
            )
            all_results.append(current_results_final)
            if VISUALIZE_ANOMALIES:
                print(
                    f"  Skipping visualization due to train_index parsing error: {filename_main_final}"
                )
            continue

        data_train_for_run = data_full_for_run[:train_idx_val]
        # data_test_for_run = data_full_for_run[train_idx_val:] # Not explicitly needed for run_F_May_22 call structure now
        y_train_for_fit = labels_for_eval[:train_idx_val]
        # labels_test_for_eval_metrics = labels_for_eval[train_idx_val:] # Not needed if evaluating on full scores

        model_start_time_final_run = time.time()

        base_dataset_name_for_hp = os.path.splitext(filename_main_final)[0]
        run_hp_config_final = {
            "use_training_labels_for_step1_plot": not args.unsupervised,
            "dataset_name_for_artifacts": base_dataset_name_for_hp,  # Added dataset name for artifact folder naming
        }

        (
            output_scaled_full,
            per_feature_artifacts_final,
            clf_instance_final,
        ) = run_F_May_22(
            data_train=data_train_for_run,
            data_full_for_decision=data_full_for_run,
            y_train_for_fit=y_train_for_fit,
            HP=run_hp_config_final,
        )

        duration_final_run = time.time() - model_start_time_final_run
        current_results_final[f"{model_prefix_final}_runtime"] = duration_final_run

        if (
            output_scaled_full is None
            or not isinstance(output_scaled_full, np.ndarray)
            or output_scaled_full.size == 0
            or output_scaled_full.shape[0] != data_full_for_run.shape[0]
        ):
            current_results_final[f"{model_prefix_final}_Error"] = (
                "Invalid scores from run_F_May_22"
            )
        else:
            # output_scaled_full is already scaled and covers the full dataset
            if metrics_calculator:
                eval_result_final_run = metrics_calculator(
                    labels=labels_for_eval,  # Evaluate on full labels
                    score=output_scaled_full,  # Use full scores
                    slidingWindow=window_final_run,
                )
                if isinstance(eval_result_final_run, dict):
                    for k_result_final, v_result_final in eval_result_final_run.items():
                        key_final_result = f"{model_prefix_final}_{k_result_final}"
                        if isinstance(v_result_final, np.number):
                            current_results_final[key_final_result] = (
                                v_result_final.item()
                            )
                        elif (
                            isinstance(v_result_final, (int, float, str, bool))
                            or v_result_final is None
                        ):
                            current_results_final[key_final_result] = v_result_final
                        else:
                            current_results_final[key_final_result] = str(
                                v_result_final
                            )

                if VISUALIZE_ANOMALIES:
                    base_name_vis_final_run = os.path.splitext(filename_main_final)[0]
                    plot_subdir_final_run = os.path.join(
                        PLOT_DIR_BASE, model_prefix_final
                    )
                    visualize_errors(
                        data=data_full_for_run,  # Pass the full data for visualization
                        original_label=labels_for_eval,  # Pass full labels
                        score=output_scaled_full,  # Pass full scores
                        per_feature_artifacts=per_feature_artifacts_final,  # Pass the full dictionary
                        base_plot_filename=base_name_vis_final_run + "_full_data_plot",
                        plot_dir=plot_subdir_final_run,
                        train_idx_val=train_idx_val,  # Pass train_idx_val
                        model_prefix=model_prefix_final,
                    )
            if f"{model_prefix_final}_Error" not in current_results_final:
                processed_count_this_run_final += 1

            # Store path to per_feature_artifacts JSON if saved
            # This assumes clf_instance_final.current_run_artifact_folder_name is correctly set
            if (
                clf_instance_final
                and hasattr(clf_instance_final, "current_run_artifact_folder_name")
                and clf_instance_final.current_run_artifact_folder_name
            ):
                if per_feature_artifacts_final:  # Check if artifacts exist
                    artifacts_json_filename = (
                        f"{model_prefix_final}_per_feature_artifacts_details.json"
                    )
                    artifacts_json_path = os.path.join(
                        clf_instance_final.plot_save_dir,
                        clf_instance_final.current_run_artifact_folder_name,
                        artifacts_json_filename,
                    )
                    try:
                        with open(artifacts_json_path, "w") as f_json:
                            json.dump(
                                per_feature_artifacts_final,
                                f_json,
                                indent=2,
                                cls=NpEncoder,
                            )  # Added NpEncoder
                        current_results_final[
                            f"{model_prefix_final}_per_feature_artifacts_path"
                        ] = artifacts_json_path
                    except Exception as e_json_save:
                        print(
                            f"  Error saving per_feature_artifacts to JSON: {e_json_save}"
                        )
                        current_results_final[
                            f"{model_prefix_final}_per_feature_artifacts_path"
                        ] = "Error saving"
                else:
                    current_results_final[
                        f"{model_prefix_final}_per_feature_artifacts_path"
                    ] = "No artifacts generated"
            else:
                current_results_final[
                    f"{model_prefix_final}_per_feature_artifacts_path"
                ] = "Artifact folder name not set"

            # Add Benchmark Comparison
            F_May_22_vus_pr_key = f"{model_prefix_final}_VUS-PR"
            if F_May_22_vus_pr_key in current_results_final:
                F_May_22_score = current_results_final[F_May_22_vus_pr_key]
                if isinstance(F_May_22_score, (float, int)) and not pd.isna(
                    F_May_22_score
                ):
                    bench_data_for_file = benchmark_metrics.get(filename_main_final)
                    if bench_data_for_file:
                        avg_bench_score = bench_data_for_file.get("avg", np.nan)
                        max_bench_score = bench_data_for_file.get("max", np.nan)

                        current_results_final[
                            f"{model_prefix_final}_Benchmark_Avg_VUS-PR"
                        ] = avg_bench_score
                        current_results_final[
                            f"{model_prefix_final}_Benchmark_Max_VUS-PR"
                        ] = max_bench_score

                        if not pd.isna(avg_bench_score):
                            current_results_final[
                                f"{model_prefix_final}_F_May_22_vs_Avg_VUS-PR_Diff"
                            ] = (F_May_22_score - avg_bench_score)
                        else:
                            current_results_final[
                                f"{model_prefix_final}_F_May_22_vs_Avg_VUS-PR_Diff"
                            ] = np.nan

                        if not pd.isna(max_bench_score):
                            current_results_final[
                                f"{model_prefix_final}_F_May_22_vs_Max_VUS-PR_Diff"
                            ] = (F_May_22_score - max_bench_score)
                        else:
                            current_results_final[
                                f"{model_prefix_final}_F_May_22_vs_Max_VUS-PR_Diff"
                            ] = np.nan
                    else:
                        current_results_final[
                            f"{model_prefix_final}_Benchmark_Avg_VUS-PR"
                        ] = "N/A_Benchmark"
                        current_results_final[
                            f"{model_prefix_final}_Benchmark_Max_VUS-PR"
                        ] = "N/A_Benchmark"
                        current_results_final[
                            f"{model_prefix_final}_F_May_22_vs_Avg_VUS-PR_Diff"
                        ] = "N/A_Benchmark"
                        current_results_final[
                            f"{model_prefix_final}_F_May_22_vs_Max_VUS-PR_Diff"
                        ] = "N/A_Benchmark"
                else:
                    # F_May_22 VUS-PR score is not numeric or is NaN
                    current_results_final[
                        f"{model_prefix_final}_Benchmark_Avg_VUS-PR"
                    ] = "N/A_SelfNoScore"
                    current_results_final[
                        f"{model_prefix_final}_Benchmark_Max_VUS-PR"
                    ] = "N/A_SelfNoScore"
                    current_results_final[
                        f"{model_prefix_final}_F_May_22_vs_Avg_VUS-PR_Diff"
                    ] = "N/A_SelfNoScore"
                    current_results_final[
                        f"{model_prefix_final}_F_May_22_vs_Max_VUS-PR_Diff"
                    ] = "N/A_SelfNoScore"

            print(f"  Metrics for {filename_main_final}:")
            vus_pr_score = current_results_final.get(
                f"{model_prefix_final}_VUS-PR", "N/A"
            )
            diff_avg = current_results_final.get(
                f"{model_prefix_final}_F_May_22_vs_Avg_VUS-PR_Diff", "N/A"
            )
            diff_max = current_results_final.get(
                f"{model_prefix_final}_F_May_22_vs_Max_VUS-PR_Diff", "N/A"
            )

            print(f"    - VUS-PR: {vus_pr_score}")
            print(f"    - Diff vs Avg VUS-PR: {diff_avg}")
            print(f"    - Diff vs Max VUS-PR: {diff_max}")

        existing_index_final_res = next(
            (
                idx_final_res
                for idx_final_res, r_final_res in enumerate(all_results)
                if r_final_res.get("filename") == filename_main_final
            ),
            -1,
        )
        if existing_index_final_res != -1:
            all_results[existing_index_final_res].update(current_results_final)
        else:
            all_results.append(current_results_final)
        processed_files.add(filename_main_final)

        if all_results:
            df_inc_final_save = pd.DataFrame(all_results)
            df_inc_final_save.drop_duplicates(
                subset="filename", keep="last", inplace=True
            )
            df_inc_final_save = df_inc_final_save[
                get_ordered_columns(df_inc_final_save.columns.tolist())
            ]
            df_inc_final_save.to_csv(RESULTS_CSV_PATH, index=False)
    print(
        f"\nFinal results ({len(df_inc_final_save) if 'df_inc_final_save' in locals() else 0} files) saved: {RESULTS_CSV_PATH}"
    )
    if "df_inc_final_save" in locals() and not df_inc_final_save.empty:
        print("\n--- Average Metrics Calculation ---")
        error_cols_check_end_avg = [f"{m}_Error" for m in MODEL_NAMES] + [
            f"{m}_Metrics_Error" for m in MODEL_NAMES
        ]
        existing_err_cols_end_avg = [
            c for c in error_cols_check_end_avg if c in df_inc_final_save.columns
        ]
        ok_filter_end_avg = pd.Series(True, index=df_inc_final_save.index)
        for col_err_end_avg in existing_err_cols_end_avg:
            ok_filter_end_avg &= df_inc_final_save[col_err_end_avg].isnull() | (
                df_inc_final_save[col_err_end_avg] == ""
            )
        ok_df_end_avg = df_inc_final_save[ok_filter_end_avg]
        num_ok_end_avg, num_total_end_avg = len(ok_df_end_avg), len(df_inc_final_save)
        if not ok_df_end_avg.empty:
            print(
                f"\n--- Avg Metrics Across {num_ok_end_avg} OK Files (of {num_total_end_avg} total) ---"
            )

            def print_average_metric_final_val(
                df_metric_val_end, metric_key_val_end, display_name_val_end=None
            ):
                if metric_key_val_end in df_metric_val_end.columns:
                    numeric_col_val_end = pd.to_numeric(
                        df_metric_val_end[metric_key_val_end], errors="coerce"
                    )
                    if numeric_col_val_end.notna().any():
                        mean_val_calc_end = np.nanmean(numeric_col_val_end)
                        name_to_print_end = (
                            display_name_val_end
                            if display_name_val_end
                            else metric_key_val_end.split("_", 1)[-1]
                        )
                        print(f"      - {name_to_print_end}: {mean_val_calc_end:.6f}")
                        return True
                return False

            for model_iter_end_avg in MODEL_NAMES:
                print(
                    f"\n  --- Model: {model_iter_end_avg} ({num_ok_end_avg} successful files) ---"
                )
                print(f"    -- Key Performance Metrics --")
                vus_pr_fnd_end = print_average_metric_final_val(
                    ok_df_end_avg, f"{model_iter_end_avg}_VUS-PR", "VUS-PR"
                )
                vus_roc_fnd_end = print_average_metric_final_val(
                    ok_df_end_avg, f"{model_iter_end_avg}_VUS-ROC", "VUS-ROC"
                )
                if not vus_pr_fnd_end and not vus_roc_fnd_end:
                    print("      - VUS-PR / VUS-ROC: N/A")
                print(f"\n    -- Other Standard Metrics --")
                other_mets_fnd_count_end = 0
                for met_iter_end_run in BASE_METRIC_ORDER:
                    if met_iter_end_run not in ["VUS-PR", "VUS-ROC"]:
                        if print_average_metric_final_val(
                            ok_df_end_avg,
                            f"{model_iter_end_avg}_{met_iter_end_run}",
                            met_iter_end_run,
                        ):
                            other_mets_fnd_count_end += 1
                if other_mets_fnd_count_end == 0:
                    print("      - Other standard metrics: N/A")
                print(f"\n    -- Other Information --")
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_runtime",
                    "Avg Runtime (s)",
                )
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_FP_count_ext",
                    "Avg FP Count (Extended)",
                )
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_FN_count_ext",
                    "Avg FN Count (Extended)",
                )
                # Print averages for new benchmark comparison columns
                print(f"\n    -- Benchmark VUS-PR Comparison --")
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_Benchmark_Avg_VUS-PR",
                    "Benchmark Avg VUS-PR",
                )
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_Benchmark_Max_VUS-PR",
                    "Benchmark Max VUS-PR",
                )
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_F_May_22_vs_Avg_VUS-PR_Diff",
                    "Diff vs Avg VUS-PR",
                )
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_F_May_22_vs_Max_VUS-PR_Diff",
                    "Diff vs Max VUS-PR",
                )

            print("\n  ------------------------------------")
        else:
            print(
                f"\nNo fully successful runs for averaging from {num_total_end_avg} files."
            )
    else:
        print("\nFinal results DataFrame is empty or not created.")
    overall_duration_script_final_end = time.time() - overall_start_time
    print(
        f"\nTotal script execution time: {overall_duration_script_final_end:.2f} seconds ({overall_duration_script_final_end/60:.2f} minutes)"
    )
    print("Script finished.")
