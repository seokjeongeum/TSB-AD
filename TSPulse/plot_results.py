import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging

# --- Configuration ---
DATASET_DIR = "/workspaces/TSB-AD/Datasets/TSB-AD-U/"
SCORE_DIR_BASE = "/workspaces/TSB-AD/eval/score/uni/"
PLOT_DIR = "/workspaces/TSB-AD/plots/uni/"
FILE_LIST_PATH = "/workspaces/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv"

# Style map for consistent plotting - ALL LINES ARE NOW SOLID
STYLE_MAP = {
    "TSPulse_ZS_ensemble": {"color": "orangered", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9, "zorder": 5},
    "TSPulse_ZS_time": {"color": "dodgerblue", "linestyle": "-", "linewidth": 1.8, "alpha": 0.9, "zorder": 4},
    "TSPulse_ZS_fft": {"color": "forestgreen", "linestyle": "-", "linewidth": 1.8, "alpha": 0.9, "zorder": 3},
    "TSPulse_ZS_future": {"color": "darkviolet", "linestyle": "-", "linewidth": 1.8, "alpha": 0.9, "zorder": 3},
}
DEFAULT_STYLE = {"color": "darkgoldenrod", "linestyle": "-", "linewidth": 1.5, "alpha": 0.8, "zorder": 3}

def plot_anomaly_scores(file_basename):
    logging.info(f"--- Starting processing for: {file_basename} ---")

    # --- 1. Load Data and All Scores ---
    data_path = os.path.join(DATASET_DIR, f"{file_basename}.csv")
    try:
        data_df = pd.read_csv(data_path)
        time_series = data_df.iloc[:, 0].values
        labels = data_df.iloc[:, -1].values
    except Exception as e:
        logging.error(f"Error loading data from '{data_path}': {e}", exc_info=True)
        return

    # Define the desired order of plots
    plot_order = ["TSPulse_ZS_ensemble", "TSPulse_ZS_time", "TSPulse_ZS_fft", "TSPulse_ZS_future"]
    
    # Load only the scores we intend to plot
    scores_to_plot = {}
    for algo_name in plot_order:
        score_path = os.path.join(SCORE_DIR_BASE, algo_name, f"{file_basename}.npy")
        if os.path.exists(score_path):
            try:
                scores = np.load(score_path)
                if len(scores) != len(time_series):
                    scores = np.resize(scores, len(time_series))
                    
                scores_to_plot[algo_name] = scores
            except Exception as e:
                logging.error(f"Could not load scores from {score_path}: {e}", exc_info=True)

    if not scores_to_plot:
        logging.warning(f"No valid score files found for '{file_basename}'. Skipping.")
        return

    # --- 2. Create Figure with a subplot for each score ---
    num_plots = len(scores_to_plot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(40, 6 * num_plots), sharex=True, squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    # --- 3. Generate each subplot ---
    for i, (algo_name, scores) in enumerate(scores_to_plot.items()):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        # Plot raw data
        ax1.plot(time_series, color="gray", linewidth=1.5, label="Time Series Data", zorder=1)
        ax1.set_ylabel("Data Value", color="gray", fontsize=14)
        ax1.tick_params(axis="y", labelcolor="gray", labelsize=12)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot ground truth anomaly regions
        ax2.fill_between(range(len(labels)), 0, 1, where=labels == 1, color="lightcoral",
                         alpha=0.4, transform=ax2.get_xaxis_transform(), label="Ground Truth Anomaly", zorder=2)

        # Plot the specific score for this subplot
        style = STYLE_MAP.get(algo_name, DEFAULT_STYLE)
        ax2.plot(scores, label=f"Score (MinMax Scaled)", **style)
        ax2.set_ylabel("Anomaly Score", fontsize=14)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelsize=12)

        # Create a combined legend for data, anomaly, and score
        lines, labels_1 = ax1.get_legend_handles_labels()
        lines2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels_1 + labels_2, loc='upper right', fontsize=12)

        ax1.set_title(f"{algo_name}", fontsize=16)

    # --- 4. Final Touches ---
    fig.suptitle(f"Anomaly Detection Results for: {file_basename}", fontsize=20, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.99)) # Use tuple to fix linter error and adjust for suptitle

    # --- 5. Save the Plot ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    save_path = os.path.join(PLOT_DIR, f"{file_basename}_plot_subplots.png")
    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    except Exception as e:
        logging.error(f"Failed to save plot {save_path}: {e}", exc_info=True)
    finally:
        plt.close(fig)

# --- Main execution block remains the same ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.info("--- Starting Plot Generation Script ---")
    try:
        file_list_df = pd.read_csv(FILE_LIST_PATH)
    except FileNotFoundError:
        logging.error(f"Fatal: File list not found at '{FILE_LIST_PATH}'. Exiting.")
        exit()
    files_to_plot = file_list_df["file_name"].tolist()
    logging.info(f"Found {len(files_to_plot)} files to process.")
    for csv_filename in tqdm(files_to_plot, desc="Generating Plots"):
        base_name = os.path.splitext(csv_filename)[0]
        plot_anomaly_scores(base_name)
    logging.info("--- Plot generation complete. ---")
