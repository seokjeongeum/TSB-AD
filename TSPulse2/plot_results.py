import matplotlib
import numpy as np
import pandas as pd
import re

matplotlib.use("Agg")
import logging
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to get the project root
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# --- Plot Configuration ---
# These parameters are optimized for LLM vision model input:
# - Width (40 inches): Large width to display long time series with good temporal resolution
# - Heights (6 * num_plots inches): Scales with number of subplots for readability
# - Font sizes (96/84/72 pt): Large fonts for better OCR/vision model recognition
# - DPI (150): Balances quality with file size for vision model processing
# Note: These values were empirically chosen to work well with vision-language models
# for anomaly score interpretation.

# Style map for consistent plotting - ALL LINES ARE NOW SOLID
STYLE_MAP = {
    "TSPulse_ZS_ensemble": {
        "color": "orangered",
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.9,
        "zorder": 5,
    },
    "TSPulse_ZS_time": {
        "color": "dodgerblue",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 0.9,
        "zorder": 4,
    },
    "TSPulse_ZS_fft": {
        "color": "forestgreen",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 0.9,
        "zorder": 3,
    },
    "TSPulse_ZS_forecast": {
        "color": "darkviolet",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 0.9,
        "zorder": 3,
    },
}
DEFAULT_STYLE = {
    "color": "darkgoldenrod",
    "linestyle": "-",
    "linewidth": 1.5,
    "alpha": 0.8,
    "zorder": 3,
}


def plot_anomaly_scores(file_basename, dataset_dir, score_dir, plot_dir, variant: str):
    logging.info(f"--- Starting processing for: {file_basename} ---")

    # --- 1. Load Data ---
    data_path = os.path.join(dataset_dir, f"{file_basename}.csv")
    try:
        data_df = pd.read_csv(data_path)
        labels = data_df.iloc[:, -1].values
    except Exception as e:
        logging.error(f"Error loading data from '{data_path}': {e}", exc_info=True)
        return

    # --- Determine channels ---
    # If more than one data column, treat as multivariate
    is_multivariate = data_df.shape[1] > 2
    if is_multivariate:
        channel_cols = data_df.columns[:-1]
    else:
        channel_cols = data_df.columns[:1]

    # --- 2. Iterate through each channel and plot ---
    for chan_idx, channel_name in enumerate(channel_cols):
        time_series = data_df[channel_name].values

        # --- 2a. Load Scores for this Channel ---
        # Handle different score file naming conventions based on variant.
        if variant == "multi":
            # For 'multi_as_uni', the score filename is the same as the data filename (basename).
            score_file_basename = file_basename
        else:
            # For 'uni', sanitize basename to find the correct score file.
            score_file_basename = re.sub(r"-.+$", "", file_basename)

        plot_order = [
            "TSPulse_ZS_ensemble",
            "TSPulse_ZS_time",
            "TSPulse_ZS_fft",
            "TSPulse_ZS_forecast",
        ]
        title_map = {
            "TSPulse_ZS_ensemble": "Head_ensemble",
            "TSPulse_ZS_time": "Head_time",
            "TSPulse_ZS_fft": "Head_FFT",
            "TSPulse_ZS_forecast": "Head_forecast",
        }
        scores_to_plot = {}
        for algo_name in plot_order:
            score_path = os.path.join(score_dir, algo_name, f"{score_file_basename}.npy")
            if os.path.exists(score_path):
                try:
                    all_scores = np.load(score_path)

                    # Select scores for the current channel if available
                    if all_scores.ndim > 1 and all_scores.shape[1] > chan_idx:
                        scores = all_scores[:, chan_idx]
                    else:
                        scores = all_scores  # Use 1D score as is

                    if len(scores) != len(time_series):
                        scores = np.resize(scores, len(time_series))

                    scores_to_plot[algo_name] = scores
                except Exception as e:
                    logging.error(
                        f"Could not load scores from {score_path} for channel {channel_name}: {e}",
                        exc_info=True,
                    )

        if not scores_to_plot:
            logging.warning(
                f"No valid score files found for '{file_basename}' channel '{channel_name}'. Skipping channel."
            )
            continue

        # --- 2b. Create Figure with a subplot for each score ---
        num_plots = len(scores_to_plot)
        fig, axes = plt.subplots(
            num_plots, 1, figsize=(40, 6 * num_plots), sharex=True, squeeze=False
        )
        axes = axes.flatten()  # Ensure axes is always a 1D array

        # --- 2c. Generate each subplot ---
        for i, (algo_name, scores) in enumerate(scores_to_plot.items()):
            ax1 = axes[i]
            ax2 = ax1.twinx()

            # Plot raw data
            ax1.plot(
                time_series,
                color="steelblue",
                linewidth=1.5,
                label="Time Series Data",
                zorder=1,
            )
            ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax1.tick_params(axis="y", labelsize=72)

            # Plot ground truth anomaly regions
            ax2.fill_between(
                range(len(labels)),
                0,
                1,
                where=labels == 1,
                color="lightcoral",
                alpha=0.4,
                transform=ax2.get_xaxis_transform(),
                label="Ground Truth Anomaly",
                zorder=2,
            )

            # Plot the specific score for this subplot
            style = STYLE_MAP.get(algo_name, DEFAULT_STYLE)
            ax2.plot(scores, label=f"Score (MinMax Scaled)", **style)
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis="y", labelsize=72)

            # Use new titles and increased font size
            subplot_title = title_map.get(algo_name, algo_name)
            ax1.set_title(subplot_title, fontsize=96)

        # --- 2d. Final Touches ---
        # Add a single, centered y-label for the whole figure for visual balance.
        fig.supylabel("Data Value", fontsize=84, x=0.01)
        fig.text(
            0.99,
            0.5,
            "Anomaly Score",
            va="center",
            rotation=-90,
            fontsize=84,
        )
        plt.tight_layout(rect=(0, 0, 0.95, 0.99))

        # --- 2e. Save the Plot ---
        os.makedirs(plot_dir, exist_ok=True)
        plot_basename = f"{file_basename}"
        if is_multivariate:
            plot_basename += f"_{channel_name}"

        png_path = os.path.join(plot_dir, f"{plot_basename}_plot_subplots.png")
        pdf_path = os.path.join(plot_dir, f"{plot_basename}_plot_subplots.pdf")

        try:
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
        except Exception as e:
            logging.error(
                f"Failed to save plot for {plot_basename}: {e}", exc_info=True
            )
        finally:
            plt.close(fig)


# --- Main execution block ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    variants = [
        "uni",
        "multi",
    ]

    for variant in variants:
        logging.info(f"--- Starting {variant.upper()} Plot Generation ---")

        # --- Configure paths for the current variant ---
        if variant == "uni":
            dataset_dir = os.path.join(project_root, "Datasets", "TSB-AD-U")
            file_list_path = os.path.join(
                project_root, "Datasets", "File_List", "TSB-AD-U.csv"
            )
            score_dir = os.path.join(project_root, "eval", "score", "uni")
        elif variant == "multi":
            dataset_dir = os.path.join(project_root, "Datasets", "TSB-AD-M")
            file_list_path = os.path.join(
                project_root, "Datasets", "File_List", "TSB-AD-M-univariate.csv"
            )
            # Use the specific score directory for the multi-as-uni case
            score_dir = os.path.join(project_root, "eval", "score", "multi_as_uni")
        else:
            logging.warning(f"Unknown variant '{variant}'. Skipping.")
            continue

        plot_dir = os.path.join(project_root, "plots", variant)

        try:
            file_list_df = pd.read_csv(file_list_path)
        except FileNotFoundError:
            logging.error(
                f"Fatal: File list not found at '{file_list_path}'. Skipping {variant} variant."
            )
            continue

        files_to_plot = file_list_df["file_name"].tolist()
        logging.info(
            f"Found {len(files_to_plot)} files to process for {variant} variant."
        )

        for csv_filename in tqdm(
            files_to_plot, desc=f"Generating {variant.upper()} Plots"
        ):
            base_name = os.path.splitext(csv_filename)[0]
            plot_anomaly_scores(base_name, dataset_dir, score_dir, plot_dir, variant)

    logging.info("--- All plot generation complete. ---")
