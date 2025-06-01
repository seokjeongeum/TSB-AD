import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from MAD.MAD_0530.constants_0530 import PLOT_FIG_WIDTH

from .mad_utils_0530 import _find_clusters, _format_exception_for_logging

# _find_clusters_func will be passed from mad_utils (e.g., from MAD.mad_utils import _find_clusters)
# _format_exception_func will be passed from mad_utils (e.g., from MAD.mad_utils import _format_exception_for_logging)
# PLOT_FIG_WIDTH will be passed (e.g., from MAD.constants import PLOT_FIG_WIDTH)


def generate_interest_id_plot(
    feature_data_train,
    feature_labels_train,
    i_feat,
    n_train_samples,
    target_range_size,
    plot_path,  # Full path to save the plot
    logger,
    full_feature_data_for_context,
    actual_train_len_for_context,
):
    """
    Generates and saves the plot for the interest identification step.
    Plots the full data context, highlights the training span, and shows training anomalies.
    Returns: (bool: success, str: error_info if any)
    """
    fig_id_step, ax_id_step = None, None
    plot_generated_successfully = False
    error_info = None
    try:
        plt.close("all")
        fig_id_step, ax_id_step = plt.subplots(figsize=(PLOT_FIG_WIDTH, 5))

        n_full_samples = len(full_feature_data_for_context)
        time_indices_full = np.arange(n_full_samples)

        # Plot the full feature data series
        ax_id_step.plot(
            time_indices_full,
            full_feature_data_for_context,
            color="dimgray",  # Base color for full context
            lw=0.8,
            label=f"Full Context Ft {i_feat} (Non-Training Part)",  # Adjusted label
            alpha=0.7,  # Slightly more transparent
        )

        # Highlight the training data span on the plot of full data using ax.plot()
        explicit_ticks = []
        if (
            actual_train_len_for_context > 0
            and actual_train_len_for_context <= n_full_samples
        ):
            # --- Apply new plotting style for Training Data Span ---
            # feature_data_train is the actual training data (Y values)
            # time_indices_train_segment are the X values for feature_data_train (0 to actual_train_len_for_context-1)
            time_indices_train_segment = np.arange(actual_train_len_for_context)

            plot_time_segment_train_span = time_indices_train_segment.copy()
            plot_data_segment_train_span = feature_data_train.copy()

            # Prepend point is not applicable as training span starts at index 0 of its own context.
            # Append point: Connect to the rest of the full_feature_data_for_context if it exists
            if (
                actual_train_len_for_context < n_full_samples
            ):  # If training data is not the entire full context
                plot_time_segment_train_span = np.append(
                    plot_time_segment_train_span,
                    time_indices_full[actual_train_len_for_context],
                )
                plot_data_segment_train_span = np.append(
                    plot_data_segment_train_span,
                    full_feature_data_for_context[actual_train_len_for_context],
                )

            if len(plot_time_segment_train_span) > 0:
                ax_id_step.plot(
                    plot_time_segment_train_span,
                    plot_data_segment_train_span,
                    color="darkblue",  # UPDATED from blue
                    lw=1.2,  # Slightly thicker than base full context
                    label=f"Training Data Span [0-{actual_train_len_for_context-1}]",
                    zorder=1,  # Below anomalies but above base full context if needed
                )
            # --- End new plotting style ---
            explicit_ticks.extend([0, actual_train_len_for_context - 1])

        # Plot anomalies from feature_labels_train (which corresponds to feature_data_train)
        if (
            feature_labels_train is not None and n_train_samples > 0
        ):  # n_train_samples is len(feature_data_train)
            train_anomalies_indices = np.where(feature_labels_train == 1)[0]
            if train_anomalies_indices.size > 0:
                anomaly_ranges_plot = _find_clusters(train_anomalies_indices)
                plotted_anomaly_label = False
                for r_start, r_end in anomaly_ranges_plot:
                    plot_r_start = max(0, r_start)
                    plot_r_end = min(n_train_samples - 1, r_end)
                    if plot_r_start <= plot_r_end:
                        # Use plot() instead of axvspan()
                        segment_indices = time_indices_train_segment[
                            plot_r_start : plot_r_end + 1
                        ]
                        segment_data = feature_data_train[plot_r_start : plot_r_end + 1]
                        if len(segment_indices) > 0:  # Ensure segment is not empty
                            # --- Apply new plotting style ---
                            plot_time_segment = segment_indices.copy()
                            plot_data_segment = segment_data.copy()

                            if plot_r_start > 0:
                                plot_time_segment = np.insert(
                                    plot_time_segment,
                                    0,
                                    time_indices_train_segment[plot_r_start - 1],
                                )
                                plot_data_segment = np.insert(
                                    plot_data_segment,
                                    0,
                                    feature_data_train[plot_r_start - 1],
                                )
                            if plot_r_end < n_train_samples - 1:
                                plot_time_segment = np.append(
                                    plot_time_segment,
                                    time_indices_train_segment[plot_r_end + 1],
                                )
                                plot_data_segment = np.append(
                                    plot_data_segment,
                                    feature_data_train[plot_r_end + 1],
                                )
                            # --- End new plotting style ---

                            ax_id_step.plot(
                                plot_time_segment,  # Use modified time
                                plot_data_segment,  # Use modified data
                                color="darkred",  # UPDATED from red
                                lw=1.5,  # Make anomaly line thicker
                                linestyle="-",  # Explicitly solid line, or choose another for anomalies
                                label=(
                                    "Labeled Training Anomalies"
                                    if not plotted_anomaly_label
                                    else None
                                ),
                                zorder=5,  # Ensure it plots on top
                            )
                            plotted_anomaly_label = True

        # New title
        title_str = f"Feature {i_feat}: Interest Identification Input\n"
        title_str += f"Full Series (N={n_full_samples}), Training Segment (N={n_train_samples}), Target Range Size ~{target_range_size})"
        ax_id_step.set_title(title_str, fontsize=14)  # Added fontsize for title

        ax_id_step.set_xlabel(
            f"Time Step (Overall Index, 0 to {n_full_samples-1 if n_full_samples > 0 else 0})"
        )
        ax_id_step.set_ylabel("Value")
        ax_id_step.tick_params(
            axis="both", which="major", labelsize=16
        )  # Increased tick font size

        handles, labels = ax_id_step.get_legend_handles_labels()
        if handles:
            unique_labels_map = {}
            final_handles, final_labels = [], []
            for handle, label in zip(handles, labels):
                if label and label not in unique_labels_map:  # Ensure label is not None
                    unique_labels_map[label] = handle
                    final_handles.append(handle)
                    final_labels.append(label)
            if final_handles:
                ax_id_step.legend(
                    final_handles, final_labels, loc="best", fontsize="small"
                )

        ax_id_step.grid(True, alpha=0.3, linestyle=":")
        # Add explicit ticks, ensuring they are within bounds and unique
        if n_full_samples > 0:
            explicit_ticks.append(n_full_samples - 1)  # Add end of full data
            explicit_ticks.append(0)  # Add start of full data

        final_ticks = sorted(
            list(set(int(t) for t in explicit_ticks if 0 <= t < n_full_samples))
        )
        if (
            not final_ticks and n_full_samples > 0
        ):  # Fallback if no specific ticks were added but data exists
            final_ticks = [0, n_full_samples - 1]
        elif not final_ticks and n_full_samples == 0:
            final_ticks = [0]

        if final_ticks:  # Set ticks if any are determined
            ax_id_step.set_xticks(final_ticks)

        ax_id_step.set_xlim(
            -0.01 * n_full_samples, n_full_samples * 1.01 if n_full_samples > 0 else 1
        )
        plt.tight_layout()
        fig_id_step.savefig(plot_path, dpi=100)  # Standard DPI for this utility plot
        plot_generated_successfully = True
        if logger:
            logger.debug(f"Interest ID plot for feature {i_feat} saved to {plot_path}")

    except Exception as e_plot:
        formatted_exception = _format_exception_for_logging(e_plot)
        if logger:
            logger.error(
                f"Feature {i_feat}: Error generating Interest ID plot: {formatted_exception}"
            )
        error_info = formatted_exception
    finally:
        if fig_id_step:
            plt.close(fig_id_step)

    return plot_generated_successfully, error_info


def generate_batch_llm_input_plot(
    X_data_col,
    i_feat,
    batch_num,
    n_samples,  # Total samples in X_data_col (could be train or test)
    identified_train_interest_range,
    use_training_labels_for_hint,
    y_labels_for_overall_hint,  # Can be y_train if available and hint is enabled
    train_len_for_overall_hint,  # Original length of training data, corresponding to y_labels_for_overall_hint
    current_batch_training_snippets_list,
    current_batch_analysis_snippets_list,
    plot_path,
    logger,
    plot_fig_height,
    plot_dpi,
):
    """
    Generates the detailed plot for each batch input to the main LLM.
    Highlights training focus range, overall anomaly examples (from training), and current batch snippets.
    Returns: (bool: success, str: error_info if any)
    """
    _batch_plot_fig, ax_p_batch_plot = None, None
    plot_generated_successfully = False
    error_info = None

    try:
        plt.close("all")
        fig_width_plot, fig_height_plot, dpi_val_plot = (
            PLOT_FIG_WIDTH,
            plot_fig_height,
            plot_dpi,
        )

        _batch_plot_fig, ax_p_batch_plot = plt.subplots(
            figsize=(fig_width_plot, fig_height_plot)
        )

        plot_title_main_part = (
            f"LLM Input Batch Plot: Feature {i_feat}, Batch {batch_num}"
        )
        batch_plot_subtitle_parts = []

        ax_p_batch_plot.set_ylabel("Value", fontsize="medium")
        ax_p_batch_plot.grid(True, alpha=0.3, linestyle=":")
        ax_p_batch_plot.tick_params(axis="both", labelsize=16)
        ax_p_batch_plot.set_xlabel("Time Step (Overall Data Index)", fontsize="medium")

        time_range_plot_full = np.arange(n_samples)
        ax_p_batch_plot.plot(
            time_range_plot_full,
            X_data_col,
            color="dimgray",
            linestyle="-",
            lw=0.8,
            label="Full Data Series for Feature",
        )

        # Highlight Identified Training Interest Range (if this batch is processing training data or for context)
        train_focus_range_plotted_batch_plot = False
        if identified_train_interest_range:
            min_plot_idx_interest = identified_train_interest_range[0]
            max_plot_idx_interest = identified_train_interest_range[1]
            if (
                max_plot_idx_interest < n_samples
                and min_plot_idx_interest < n_samples
                and min_plot_idx_interest <= max_plot_idx_interest  # Ensure valid range
            ):
                # --- Apply new plotting style for Train Focus Range ---
                plot_time_segment_focus = time_range_plot_full[
                    min_plot_idx_interest : max_plot_idx_interest + 1
                ].copy()
                plot_data_segment_focus = X_data_col[
                    min_plot_idx_interest : max_plot_idx_interest + 1
                ].copy()

                if min_plot_idx_interest > 0:
                    plot_time_segment_focus = np.insert(
                        plot_time_segment_focus,
                        0,
                        time_range_plot_full[min_plot_idx_interest - 1],
                    )
                    plot_data_segment_focus = np.insert(
                        plot_data_segment_focus,
                        0,
                        X_data_col[min_plot_idx_interest - 1],
                    )
                if max_plot_idx_interest < n_samples - 1:
                    plot_time_segment_focus = np.append(
                        plot_time_segment_focus,
                        time_range_plot_full[max_plot_idx_interest + 1],
                    )
                    plot_data_segment_focus = np.append(
                        plot_data_segment_focus, X_data_col[max_plot_idx_interest + 1]
                    )

                if len(plot_time_segment_focus) > 0:
                    ax_p_batch_plot.plot(
                        plot_time_segment_focus,
                        plot_data_segment_focus,
                        color="magenta",  # UPDATED from darkblue
                        lw=1.5,  # Slightly thicker
                        label=f"Train Focus Range [{min_plot_idx_interest}-{max_plot_idx_interest}]",
                        zorder=3,  # Adjusted zorder
                    )
                    train_focus_range_plotted_batch_plot = True
                # --- End new plotting style ---

                subtitle_text_interest = (
                    f"Train Focus [{min_plot_idx_interest}-{max_plot_idx_interest}]"
                )
                if subtitle_text_interest not in batch_plot_subtitle_parts:
                    batch_plot_subtitle_parts.append(subtitle_text_interest)

        example_anomalies_plotted_on_batch_plot = False
        if (
            use_training_labels_for_hint
            and y_labels_for_overall_hint is not None
            and train_len_for_overall_hint is not None
            and train_len_for_overall_hint > 0
        ):
            # Anomalies from y_labels_for_overall_hint (typically y_train) up to train_len_for_overall_hint
            indices_to_check_for_anomalies = y_labels_for_overall_hint[
                :train_len_for_overall_hint
            ]
            train_anomalies_indices_for_hint = np.where(
                indices_to_check_for_anomalies == 1
            )[0]
            if train_anomalies_indices_for_hint.size > 0:
                anomaly_ranges_to_plot_hint = _find_clusters(
                    train_anomalies_indices_for_hint
                )
                for r_start, r_end in anomaly_ranges_to_plot_hint:
                    # Ensure these anomaly examples are within the current plot's x-axis limits (n_samples)
                    plot_r_start = max(0, r_start)
                    plot_r_end = min(n_samples - 1, r_end)
                    if plot_r_start <= plot_r_end:
                        # Use plot() instead of axvspan()
                        segment_indices = time_range_plot_full[
                            plot_r_start : plot_r_end + 1
                        ]
                        segment_data = X_data_col[plot_r_start : plot_r_end + 1]
                        if len(segment_indices) > 0:
                            # --- Apply new plotting style ---
                            plot_time_segment = segment_indices.copy()
                            plot_data_segment = segment_data.copy()

                            if plot_r_start > 0:
                                plot_time_segment = np.insert(
                                    plot_time_segment,
                                    0,
                                    time_range_plot_full[plot_r_start - 1],
                                )
                                plot_data_segment = np.insert(
                                    plot_data_segment, 0, X_data_col[plot_r_start - 1]
                                )
                            if plot_r_end < n_samples - 1:
                                plot_time_segment = np.append(
                                    plot_time_segment,
                                    time_range_plot_full[plot_r_end + 1],
                                )
                                plot_data_segment = np.append(
                                    plot_data_segment, X_data_col[plot_r_end + 1]
                                )
                            # --- End new plotting style ---

                            ax_p_batch_plot.plot(
                                plot_time_segment,  # Use modified time
                                plot_data_segment,  # Use modified data
                                color="darkred",  # UPDATED from lightcoral
                                lw=1.5,  # Thicker line for anomalies
                                linestyle="-",
                                label=(
                                    "Overall Training Anomaly Examples"
                                    if not example_anomalies_plotted_on_batch_plot
                                    else None
                                ),
                                zorder=6,  # Ensure it plots on top of other region highlights if necessary
                            )
                            example_anomalies_plotted_on_batch_plot = True
                if example_anomalies_plotted_on_batch_plot:
                    if (
                        "Overall Training Anomaly Examples Shown"
                        not in batch_plot_subtitle_parts
                    ):
                        batch_plot_subtitle_parts.append(
                            "Overall Training Anomaly Examples Shown"
                        )

        # Highlight Current Batch Training Snippets (if any)
        if current_batch_training_snippets_list:
            # These snippets have indices relative to the original training data.
            # Their relevance here is to show what training examples LLM is seeing in this batch.
            min_idx_tr_b = min(s["index"] for s in current_batch_training_snippets_list)
            max_idx_tr_b = max(s["index"] for s in current_batch_training_snippets_list)
            if (
                max_idx_tr_b < n_samples
                and min_idx_tr_b <= max_idx_tr_b  # Ensure valid range
            ):  # Check if these training snippets are within current plot bounds
                # --- Apply new plotting style for Batch Train Snips ---
                plot_time_segment_train_snip = time_range_plot_full[
                    min_idx_tr_b : max_idx_tr_b + 1
                ].copy()
                plot_data_segment_train_snip = X_data_col[
                    min_idx_tr_b : max_idx_tr_b + 1
                ].copy()

                if min_idx_tr_b > 0:
                    plot_time_segment_train_snip = np.insert(
                        plot_time_segment_train_snip,
                        0,
                        time_range_plot_full[min_idx_tr_b - 1],
                    )
                    plot_data_segment_train_snip = np.insert(
                        plot_data_segment_train_snip, 0, X_data_col[min_idx_tr_b - 1]
                    )
                if max_idx_tr_b < n_samples - 1:
                    plot_time_segment_train_snip = np.append(
                        plot_time_segment_train_snip,
                        time_range_plot_full[max_idx_tr_b + 1],
                    )
                    plot_data_segment_train_snip = np.append(
                        plot_data_segment_train_snip, X_data_col[max_idx_tr_b + 1]
                    )

                if len(plot_time_segment_train_snip) > 0:
                    ax_p_batch_plot.plot(
                        plot_time_segment_train_snip,
                        plot_data_segment_train_snip,
                        color="darkgreen",  # UPDATED from limegreen
                        lw=1.5,
                        label=f"Batch Train Snips Sent [{min_idx_tr_b}-{max_idx_tr_b}]",
                        zorder=4,  # Adjusted zorder
                    )
                # --- End new plotting style ---
                subtitle_part = f"Batch Train Snips [{min_idx_tr_b}-{max_idx_tr_b}]"
                if subtitle_part not in batch_plot_subtitle_parts:
                    batch_plot_subtitle_parts.append(subtitle_part)

        # Highlight Current Batch Analysis Snippets
        if current_batch_analysis_snippets_list:
            # These snippets have indices relative to X_data_col
            min_idx_an_b = min(s["index"] for s in current_batch_analysis_snippets_list)
            max_idx_an_b = max(s["index"] for s in current_batch_analysis_snippets_list)
            if min_idx_an_b <= max_idx_an_b:  # Ensure valid range
                # --- Apply new plotting style for Batch Analysis Snips ---
                plot_time_segment_analysis_snip = time_range_plot_full[
                    min_idx_an_b : max_idx_an_b + 1
                ].copy()
                plot_data_segment_analysis_snip = X_data_col[
                    min_idx_an_b : max_idx_an_b + 1
                ].copy()

                if min_idx_an_b > 0:
                    plot_time_segment_analysis_snip = np.insert(
                        plot_time_segment_analysis_snip,
                        0,
                        time_range_plot_full[min_idx_an_b - 1],
                    )
                    plot_data_segment_analysis_snip = np.insert(
                        plot_data_segment_analysis_snip, 0, X_data_col[min_idx_an_b - 1]
                    )
                if max_idx_an_b < n_samples - 1:
                    plot_time_segment_analysis_snip = np.append(
                        plot_time_segment_analysis_snip,
                        time_range_plot_full[max_idx_an_b + 1],
                    )
                    plot_data_segment_analysis_snip = np.append(
                        plot_data_segment_analysis_snip, X_data_col[max_idx_an_b + 1]
                    )

                if len(plot_time_segment_analysis_snip) > 0:
                    ax_p_batch_plot.plot(
                        plot_time_segment_analysis_snip,
                        plot_data_segment_analysis_snip,
                        color="darkgoldenrod",  # UPDATED from orange
                        lw=1.7,
                        label=f"Batch Analysis Snips [{min_idx_an_b}-{max_idx_an_b}]",
                        zorder=2.2,  # Adjusted zorder
                    )
                # --- End new plotting style ---
            subtitle_part = f"Batch Analysis Snips [{min_idx_an_b}-{max_idx_an_b}]"
            if subtitle_part not in batch_plot_subtitle_parts:
                batch_plot_subtitle_parts.append(subtitle_part)

        # Set X-axis ticks based on important ranges shown
        explicit_plot_ticks_batch = []
        if n_samples > 0:
            explicit_plot_ticks_batch.extend([0, n_samples - 1])
        if identified_train_interest_range:
            explicit_plot_ticks_batch.extend(
                [identified_train_interest_range[0], identified_train_interest_range[1]]
            )
        if current_batch_training_snippets_list:
            explicit_plot_ticks_batch.extend(
                [
                    min(s["index"] for s in current_batch_training_snippets_list),
                    max(s["index"] for s in current_batch_training_snippets_list),
                ]
            )
        if current_batch_analysis_snippets_list:
            explicit_plot_ticks_batch.extend(
                [
                    min(s["index"] for s in current_batch_analysis_snippets_list),
                    max(s["index"] for s in current_batch_analysis_snippets_list),
                ]
            )

        final_plot_ticks_b = sorted(
            list(
                set(tick for tick in explicit_plot_ticks_batch if 0 <= tick < n_samples)
            )
        )
        if not final_plot_ticks_b and n_samples > 0:
            final_plot_ticks_b = sorted(list(set([0, n_samples - 1])))
        elif not final_plot_ticks_b and n_samples == 0:
            final_plot_ticks_b = [0]
        if final_plot_ticks_b:
            ax_p_batch_plot.set_xticks(final_plot_ticks_b)

        ax_p_batch_plot.set_xlim(
            -0.01 * n_samples, n_samples * 1.01 if n_samples > 0 else 1
        )

        final_batch_plot_title_str = plot_title_main_part
        if batch_plot_subtitle_parts:
            final_batch_plot_title_str += "\n" + " | ".join(batch_plot_subtitle_parts)

        _batch_plot_fig.suptitle(
            final_batch_plot_title_str, fontsize="medium", y=0.985
        )  # Adjusted y for suptitle

        handles_plot_b, labels_plot_b = ax_p_batch_plot.get_legend_handles_labels()
        if handles_plot_b:
            unique_labels_map_plot_b = {}
            final_handles_plot_b, final_labels_plot_b = [], []
            for handle, label in zip(handles_plot_b, labels_plot_b):
                if (
                    label and label not in unique_labels_map_plot_b
                ):  # Ensure label is not None
                    unique_labels_map_plot_b[label] = handle
                    final_handles_plot_b.append(handle)
                    final_labels_plot_b.append(label)
            if final_handles_plot_b:
                ax_p_batch_plot.legend(
                    final_handles_plot_b,
                    final_labels_plot_b,
                    fontsize="x-small",
                    loc="best",
                )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        _batch_plot_fig.savefig(plot_path, dpi=dpi_val_plot)
        plot_generated_successfully = True
        if logger:
            logger.debug(
                f"Batch LLM input plot for feature {i_feat}, batch {batch_num} saved to {plot_path}"
            )

    except Exception as e_batch_plot_final:
        formatted_exception = _format_exception_for_logging(e_batch_plot_final)
        if logger:
            logger.error(
                f"Error generating/saving batch LLM input plot for Feature {i_feat}, Batch {batch_num}: {formatted_exception}"
            )
        error_info = formatted_exception
    finally:
        if _batch_plot_fig:
            plt.close(_batch_plot_fig)

    return plot_generated_successfully, error_info


def generate_llm_identified_anomalies_plot(
    X_data_col,
    llm_anomaly_scores,  # This is the full n_samples length boolean/binary array from LLM code for this batch
    i_feat,
    batch_num,
    api_key_index,
    plot_path,  # Full path to save the plot
    logger,
    plot_fig_height,
    plot_dpi,
    analysis_snippets_to_highlight=None,  # Renamed and clarified
):
    """
    Generates a plot showing the feature data and anomalies identified by the LLM
    for a specific batch and API key attempt. Optionally highlights analysis snippet range.
    The `analysis_snippets_to_highlight` should be the list of analysis snippet dicts
    that were actually used in the LLM call that produced the `llm_anomaly_scores` being plotted.
    Returns: (bool: success, str: error_info if any)
    """
    _fig_llm_anom, _ax_llm_anom = None, None
    plot_generated_successfully = False
    error_info = None
    n_samples = len(X_data_col)
    time_range_plot = np.arange(n_samples)

    try:
        plt.close("all")
        _fig_llm_anom, _ax_llm_anom = plt.subplots(
            figsize=(PLOT_FIG_WIDTH, plot_fig_height)
        )

        # Plot the full feature data series for context
        _ax_llm_anom.plot(
            time_range_plot,
            X_data_col,
            color="dimgray",
            lw=0.8,
            label=f"Feature {i_feat} Data",
            alpha=0.9,
        )

        # --- Add secondary Y axis for anomaly scores ---
        _ax2_llm_anom = _ax_llm_anom.twinx()
        _ax2_llm_anom.plot(
            time_range_plot,
            llm_anomaly_scores,
            color="red",
            lw=1.2,
            label="LLM Anomaly Score",
            linestyle="-",  # or "steps-post"
            alpha=0.7,
        )
        _ax2_llm_anom.set_ylabel(
            "LLM Anomaly Score (0 or 1)", color="red", fontsize="small"
        )
        _ax2_llm_anom.tick_params(axis="y", labelcolor="red", labelsize="x-small")
        _ax2_llm_anom.set_ylim(-0.1, 1.1)  # Ensure 0 and 1 are clearly visible
        # --- End secondary Y axis ---

        # --- Highlight Current Batch Analysis Snippets --- (Item 1)
        if analysis_snippets_to_highlight:
            analysis_indices = [
                s.get("index")
                for s in analysis_snippets_to_highlight
                if s.get("index") is not None
            ]
            if analysis_indices:
                min_idx_an_b = min(analysis_indices)
                max_idx_an_b = max(analysis_indices)
                if min_idx_an_b <= max_idx_an_b:  # Ensure valid range
                    # --- Apply new plotting style for Batch Analysis Snips ---
                    plot_time_segment_analysis_llm_anom = time_range_plot[
                        min_idx_an_b : max_idx_an_b + 1
                    ].copy()
                    plot_data_segment_analysis_llm_anom = X_data_col[
                        min_idx_an_b : max_idx_an_b + 1
                    ].copy()

                    if min_idx_an_b > 0:
                        plot_time_segment_analysis_llm_anom = np.insert(
                            plot_time_segment_analysis_llm_anom,
                            0,
                            time_range_plot[min_idx_an_b - 1],
                        )
                        plot_data_segment_analysis_llm_anom = np.insert(
                            plot_data_segment_analysis_llm_anom,
                            0,
                            X_data_col[min_idx_an_b - 1],
                        )
                    if max_idx_an_b < n_samples - 1:
                        plot_time_segment_analysis_llm_anom = np.append(
                            plot_time_segment_analysis_llm_anom,
                            time_range_plot[max_idx_an_b + 1],
                        )
                        plot_data_segment_analysis_llm_anom = np.append(
                            plot_data_segment_analysis_llm_anom,
                            X_data_col[max_idx_an_b + 1],
                        )

                    if len(plot_time_segment_analysis_llm_anom) > 0:
                        _ax_llm_anom.plot(
                            plot_time_segment_analysis_llm_anom,
                            plot_data_segment_analysis_llm_anom,
                            color="darkgoldenrod",  # CORRECT
                            lw=1.5,
                            label=f"Batch Analysis Snips [{min_idx_an_b}-{max_idx_an_b}]",
                            zorder=1.0,  # Lower zorder than anomalies themselves
                        )
                    # --- End new plotting style ---
        # --- End Highlight ---

        title_str = f"LLM Identified Anomalies - Feature {i_feat}\nBatch {batch_num}, API Key Idx {api_key_index}"
        _ax_llm_anom.set_title(title_str, fontsize=12)
        _ax_llm_anom.set_xlabel(
            f"Time Step (0 to {n_samples-1 if n_samples > 0 else 0})"
        )
        _ax_llm_anom.set_ylabel("Value")
        _ax_llm_anom.tick_params(axis="both", which="major", labelsize=16)

        if _ax_llm_anom.has_data():
            handles, labels = _ax_llm_anom.get_legend_handles_labels()
            handles2, labels2 = (
                _ax2_llm_anom.get_legend_handles_labels()
            )  # Get from secondary axis
            # Combine legends
            all_handles = handles + handles2
            all_labels = labels + labels2

            if all_handles:
                # Use a unique legend for clarity if multiple elements are plotted
                unique_legend_items = {}
                for handle, label in zip(all_handles, all_labels):
                    if label not in unique_legend_items:
                        unique_legend_items[label] = handle
                if unique_legend_items:
                    _ax_llm_anom.legend(
                        unique_legend_items.values(),
                        unique_legend_items.keys(),
                        loc="best",
                        fontsize="small",
                    )

        _ax_llm_anom.grid(True, alpha=0.3, linestyle=":")
        _ax_llm_anom.set_xlim(
            -0.01 * n_samples, n_samples * 1.01 if n_samples > 0 else 1
        )

        plt.tight_layout()
        _fig_llm_anom.savefig(plot_path, dpi=plot_dpi)
        plot_generated_successfully = True
        if logger:
            logger.debug(
                f"LLM identified anomalies plot for F{i_feat}, B{batch_num}, K{api_key_index} saved to {plot_path}"
            )

    except Exception as e_plot_llm_anom:
        formatted_exception = _format_exception_for_logging(e_plot_llm_anom)
        if logger:
            logger.error(
                f"F{i_feat}, B{batch_num}, K{api_key_index}: Error generating LLM identified anomalies plot: {formatted_exception}"
            )
        error_info = formatted_exception
    finally:
        if _fig_llm_anom:
            plt.close(_fig_llm_anom)

    return plot_generated_successfully, error_info


def generate_feature_final_anomalies_plot(
    X_data_col,
    aggregated_anomaly_scores,  # This is the full n_samples length boolean/binary array from LLM code for this feature
    i_feat,
    plot_path,  # Full path to save the plot
    logger,
    plot_fig_height,
    plot_dpi,
):
    """
    Generates a plot showing the full feature data and the final aggregated LLM identified anomalies for that feature.
    Returns: (bool: success, str: error_info if any)
    """
    _fig_final_anom, _ax_final_anom = None, None
    plot_generated_successfully = False
    error_info = None
    n_samples = len(X_data_col)
    time_range_plot = np.arange(n_samples)

    try:
        plt.close("all")
        _fig_final_anom, _ax_final_anom = plt.subplots(
            figsize=(PLOT_FIG_WIDTH, plot_fig_height)
        )

        # Plot the full feature data series for context
        _ax_final_anom.plot(
            time_range_plot,
            X_data_col,
            color="dimgray",
            lw=0.8,
            label=f"Feature {i_feat} Data",
            alpha=0.9,
        )

        # --- Add secondary Y axis for aggregated anomaly scores ---
        _ax2_final_anom = _ax_final_anom.twinx()
        _ax2_final_anom.plot(
            time_range_plot,
            aggregated_anomaly_scores,
            color="red",
            lw=1.2,
            label="Final Aggregated LLM Anomaly Score",
            linestyle="-",  # or "steps-post"
            alpha=0.7,
        )
        _ax2_final_anom.set_ylabel(
            "Agg. Anomaly Score (0 or 1)", color="red", fontsize="small"
        )
        _ax2_final_anom.tick_params(axis="y", labelcolor="red", labelsize="x-small")
        _ax2_final_anom.set_ylim(-0.1, 1.1)  # Ensure 0 and 1 are clearly visible
        # --- End secondary Y axis ---

        title_str = f"Feature {i_feat} - Final Aggregated LLM Anomalies"
        _ax_final_anom.set_title(title_str, fontsize=12)
        _ax_final_anom.set_xlabel(
            f"Time Step (0 to {n_samples-1 if n_samples > 0 else 0})"
        )
        _ax_final_anom.set_ylabel("Value")
        _ax_final_anom.tick_params(axis="both", which="major", labelsize=10)

        if _ax_final_anom.has_data():
            handles, labels = _ax_final_anom.get_legend_handles_labels()
            handles2, labels2 = (
                _ax2_final_anom.get_legend_handles_labels()
            )  # Get from secondary axis
            # Combine legends
            all_handles = handles + handles2
            all_labels = labels + labels2

            if all_handles:
                # Use a unique legend for clarity if multiple elements are plotted
                unique_legend_items = {}
                for handle, label in zip(all_handles, all_labels):
                    if label not in unique_legend_items:
                        unique_legend_items[label] = handle
                if unique_legend_items:
                    _ax_final_anom.legend(
                        unique_legend_items.values(),
                        unique_legend_items.keys(),
                        loc="best",
                        fontsize="small",
                    )

        _ax_final_anom.grid(True, alpha=0.3, linestyle=":")
        _ax_final_anom.set_xlim(
            -0.01 * n_samples, n_samples * 1.01 if n_samples > 0 else 1
        )

        plt.tight_layout()
        _fig_final_anom.savefig(plot_path, dpi=plot_dpi)
        plot_generated_successfully = True
        if logger:
            logger.debug(
                f"Final aggregated anomalies plot for Feature {i_feat} saved to {plot_path}"
            )

    except Exception as e_plot_final_feature:
        formatted_exception = _format_exception_for_logging(e_plot_final_feature)
        if logger:
            logger.error(
                f"Feature {i_feat}: Error generating final aggregated anomalies plot: {formatted_exception}"
            )
        error_info = formatted_exception
    finally:
        if _fig_final_anom:
            plt.close(_fig_final_anom)

    return plot_generated_successfully, error_info


def generate_unpredictability_scores_plot(
    X_data_col,
    unpredictability_scores,  # (n_samples,) array of unpredictability scores (0-1)
    i_feat,
    batch_num,  # Or some other identifier for the plot context
    plot_path,  # Full path to save the plot
    logger,
    plot_fig_height,
    plot_dpi,
    analysis_snippets_to_highlight=None,  # Optional: list of analysis snippet dicts used for this score generation
):
    """
    Generates a plot showing the feature data and its corresponding unpredictability scores.
    Optionally highlights the range of analysis snippets used to generate these scores.
    Returns: (bool: success, str: error_info if any)
    """
    _fig_unpred, _ax_unpred_raw = None, None  # Renamed figure and axis variables
    plot_generated_successfully = False
    error_info = None
    n_samples = len(X_data_col)
    time_range_plot = np.arange(n_samples)

    try:
        plt.close("all")
        _fig_unpred, _ax_unpred_raw = plt.subplots(
            figsize=(PLOT_FIG_WIDTH, plot_fig_height)
        )

        # Plot the raw feature data series on the primary y-axis
        _ax_unpred_raw.plot(
            time_range_plot,
            X_data_col,
            color="dimgray",
            lw=0.9,
            label=f"Feature {i_feat} Raw Data",
            alpha=0.8,
        )
        _ax_unpred_raw.set_xlabel(
            f"Time Step (0 to {n_samples-1 if n_samples > 0 else 0})"
        )
        _ax_unpred_raw.set_ylabel(f"Feature {i_feat} Value", color="dimgray")
        _ax_unpred_raw.tick_params(axis="y", labelcolor="dimgray", labelsize="small")
        _ax_unpred_raw.tick_params(axis="x", labelsize="small")

        # Create a secondary y-axis for unpredictability scores
        _ax_unpred_score = _ax_unpred_raw.twinx()  # Renamed axis variable
        _ax_unpred_score.plot(
            time_range_plot,
            unpredictability_scores,  # Renamed variable
            color="blueviolet",  # Distinct color for unpredictability
            lw=1.1,
            label="Unpredictability Score (0=Easy, 1=Hard)",  # Updated label
            linestyle="-",
            alpha=0.75,
        )
        _ax_unpred_score.set_ylabel(
            "Unpredictability Score", color="blueviolet"
        )  # Updated label
        _ax_unpred_score.tick_params(
            axis="y", labelcolor="blueviolet", labelsize="small"
        )
        _ax_unpred_score.set_ylim(-0.05, 1.05)  # Scores are 0-1

        # Highlight the analysis snippet range if provided
        if analysis_snippets_to_highlight:
            analysis_indices = [
                s.get("index")
                for s in analysis_snippets_to_highlight
                if s.get("index") is not None
            ]
            if analysis_indices:
                min_idx_an = min(analysis_indices)
                max_idx_an = max(analysis_indices)
                if min_idx_an <= max_idx_an:
                    # To make the highlight on the raw data plot clearer against the line itself,
                    # we can plot the segment again with a different style.
                    # We take the segment of X_data_col corresponding to these analysis snippets.
                    plot_time_segment_analysis = time_range_plot[
                        min_idx_an : max_idx_an + 1
                    ].copy()
                    plot_data_segment_analysis = X_data_col[
                        min_idx_an : max_idx_an + 1
                    ].copy()

                    # Add points before and after for continuous line segment appearance if applicable
                    if min_idx_an > 0:
                        plot_time_segment_analysis = np.insert(
                            plot_time_segment_analysis,
                            0,
                            time_range_plot[min_idx_an - 1],
                        )
                        plot_data_segment_analysis = np.insert(
                            plot_data_segment_analysis, 0, X_data_col[min_idx_an - 1]
                        )
                    if max_idx_an < n_samples - 1:
                        plot_time_segment_analysis = np.append(
                            plot_time_segment_analysis, time_range_plot[max_idx_an + 1]
                        )
                        plot_data_segment_analysis = np.append(
                            plot_data_segment_analysis, X_data_col[max_idx_an + 1]
                        )

                    if len(plot_time_segment_analysis) > 0:
                        _ax_unpred_raw.plot(
                            plot_time_segment_analysis,
                            plot_data_segment_analysis,
                            color="darkgoldenrod",  # Consistent with other analysis snippet highlights
                            lw=1.7,  # Slightly thicker to stand out
                            label=f"Analysis Snippets Range [{min_idx_an}-{max_idx_an}]",
                            zorder=2,  # Ensure it's visible
                        )

        title_str = f"Feature {i_feat} - Batch {batch_num}: Raw Data & Unpredictability Scores"  # Updated title
        _ax_unpred_raw.set_title(title_str, fontsize=11)

        # Combine legends from both axes
        lines_raw, labels_raw = _ax_unpred_raw.get_legend_handles_labels()
        lines_score, labels_score = _ax_unpred_score.get_legend_handles_labels()

        # To avoid duplicate labels if any, though unlikely here
        unique_legend_items = {}
        for handle, label in zip(lines_raw + lines_score, labels_raw + labels_score):
            if label not in unique_legend_items:
                unique_legend_items[label] = handle

        if unique_legend_items:
            _ax_unpred_raw.legend(
                unique_legend_items.values(),
                unique_legend_items.keys(),
                loc="best",
                fontsize="x-small",
            )

        _ax_unpred_raw.grid(True, alpha=0.3, linestyle=":")
        _ax_unpred_raw.set_xlim(
            -0.01 * n_samples, n_samples * 1.01 if n_samples > 0 else 1
        )

        plt.tight_layout()
        _fig_unpred.savefig(plot_path, dpi=plot_dpi)  # Renamed figure variable
        plot_generated_successfully = True
        if logger:
            logger.debug(
                f"Unpredictability scores plot for F{i_feat}, B{batch_num} saved to {plot_path}"  # Updated log message
            )

    except Exception as e_plot_unpred:  # Renamed exception variable
        formatted_exception = _format_exception_for_logging(e_plot_unpred)
        if logger:
            logger.error(
                f"F{i_feat}, B{batch_num}: Error generating unpredictability scores plot: {formatted_exception}"  # Updated log message
            )
        error_info = formatted_exception
    finally:
        if _fig_unpred:  # Renamed figure variable
            plt.close(_fig_unpred)

    return plot_generated_successfully, error_info
