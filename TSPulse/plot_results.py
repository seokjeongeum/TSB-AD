import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import logging

# --- Configuration ---
DATASET_DIR = "/workspaces/TSB-AD/Datasets/TSB-AD-U/"
SCORE_DIR_BASE = "/workspaces/TSB-AD/eval/score/uni/"
PLOT_DIR = "/workspaces/TSB-AD/plots/uni/"
FILE_LIST_PATH = "/workspaces/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv"

# Define a color map for different TSPulse variants for consistent plotting
COLOR_MAP = {
    "TSPulse_ZS_ensemble": "orangered",
    "TSPulse_ZS_time": "dodgerblue",
    "TSPulse_ZS_fft": "forestgreen",
    "TSPulse_ZS_future": "darkviolet",
}
DEFAULT_COLOR = "darkgoldenrod"


def plot_anomaly_scores(file_basename):
    """
    Generates and saves a plot showing the raw time series data, ground truth anomalies,
    and anomaly scores from various TSPulse methods.

    Args:
        file_basename (str): The base name of the data file (e.g., '001_NAB_id_1..._2014').
    """
    logging.info(f"--- Starting processing for: {file_basename} ---")

    # --- 1. Load Raw Data and Labels ---
    data_path = os.path.join(DATASET_DIR, f"{file_basename}.csv")
    try:
        data_df = pd.read_csv(data_path)
        time_series = data_df.iloc[:, 0].values
        labels = data_df.iloc[:, -1].values
        logging.info(
            f"Successfully loaded data with {len(time_series)} points from '{data_path}'"
        )
    except FileNotFoundError:
        logging.warning(f"Data file not found, skipping: {data_path}")
        return
    except Exception as e:
        logging.error(f"Error loading data from '{data_path}': {e}", exc_info=True)
        return

    # --- 2. Load All Corresponding Anomaly Scores ---
    score_search_path = os.path.join(SCORE_DIR_BASE, "*", f"{file_basename}.npy")
    score_files = glob.glob(score_search_path)
    logging.info(
        f"Found {len(score_files)} score files matching pattern: {score_search_path}"
    )

    if not score_files:
        logging.warning(
            f"No score files found for '{file_basename}'. Skipping plot generation."
        )
        return

    # --- 3. Create the Plot ---
    logging.debug("Creating matplotlib figure and axes...")
    fig, ax1 = plt.subplots(figsize=(40, 6))
    ax2 = ax1.twinx()

    # Plot raw time series data
    ax1.plot(time_series, color="gray", linewidth=1, label="Time Series Data", zorder=1)
    ax1.set_ylabel("Data Value", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    # --- 4. Plot Scores and Labels ---
    for score_path in sorted(score_files):
        algo_name = os.path.basename(os.path.dirname(score_path))
        try:
            logging.debug(f"Loading scores for '{algo_name}' from '{score_path}'")
            scores = np.load(score_path)

            # Critical check for length mismatch
            if len(scores) != len(time_series):
                logging.warning(
                    f"Length mismatch for {algo_name}: scores ({len(scores)}) vs "
                    f"time_series ({len(time_series)}). Resizing scores. "
                    "This may indicate a data processing issue."
                )
                scores = np.resize(scores, len(time_series))

            color = COLOR_MAP.get(algo_name, DEFAULT_COLOR)
            ax2.plot(
                scores,
                label=f"Score ({algo_name})",
                color=color,
                linewidth=1.5,
                zorder=3,
            )
        except Exception as e:
            logging.error(
                f"Could not load or plot scores from {score_path}: {e}", exc_info=True
            )

    # Plot ground truth anomaly regions
    logging.debug("Plotting ground truth anomaly regions...")
    ax2.fill_between(
        range(len(labels)),
        0,
        1,
        where=labels == 1,
        color="red",
        alpha=0.3,
        transform=ax2.get_xaxis_transform(),
        label="Ground Truth Anomaly",
        zorder=2,
    )

    ax2.set_ylabel("Anomaly Score", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0, 1.05)

    # --- 5. Final Touches ---
    fig.suptitle(f"Anomaly Detection Results for: {file_basename}", fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- 6. Save the Plot ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    save_path = os.path.join(PLOT_DIR, f"{file_basename}_plot.png")
    try:
        logging.info(f"Saving plot to: {save_path}...")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Successfully saved plot for {file_basename}")
    except Exception as e:
        logging.error(f"Failed to save plot {save_path}: {e}", exc_info=True)
    finally:
        plt.close(fig)  # Always close the figure to free up memory


if __name__ == "__main__":
    # --- THIS IS THE MODIFIED PART ---
    # Configure the logging with path and line number
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # --- END OF MODIFICATION ---

    logging.info("--- Starting Plot Generation Script ---")

    # --- Main Automation Logic ---
    try:
        logging.info(f"Loading file list from: {FILE_LIST_PATH}")
        file_list_df = pd.read_csv(FILE_LIST_PATH)
    except FileNotFoundError:
        logging.error(f"Fatal: File list not found at '{FILE_LIST_PATH}'. Exiting.")
        exit()

    files_to_plot = file_list_df["file_name"].tolist()
    logging.info(
        f"Found {len(files_to_plot)} files to process from '{os.path.basename(FILE_LIST_PATH)}'."
    )

    # Use tqdm for a nice progress bar while iterating
    for csv_filename in tqdm(files_to_plot, desc="Generating Plots"):
        base_name = os.path.splitext(csv_filename)[0]
        plot_anomaly_scores(base_name)

    logging.info("--- Plot generation complete. ---")
    logging.info(f"All plots have been saved to: {PLOT_DIR}")
