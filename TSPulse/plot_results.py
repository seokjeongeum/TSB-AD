import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
from tqdm import tqdm  # Import tqdm for a progress bar

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
    # --- 1. Load Raw Data and Labels ---
    data_path = os.path.join(DATASET_DIR, f"{file_basename}.csv")
    try:
        data_df = pd.read_csv(data_path)
        time_series = data_df.iloc[:, 0].values
        labels = data_df.iloc[:, -1].values
    except FileNotFoundError:
        # Silently skip if the data file doesn't exist for some reason
        return
    except Exception as e:
        print(f"Error loading data from '{data_path}': {e}")
        return

    # --- 2. Load All Corresponding Anomaly Scores ---
    score_files = glob.glob(os.path.join(SCORE_DIR_BASE, "*", f"{file_basename}.npy"))

    if not score_files:
        # Silently skip if no scores are found for this file
        return

    # --- 3. Create the Plot ---
    fig, ax1 = plt.subplots(figsize=(40, 6))
    ax2 = ax1.twinx()

    # Plot raw time series data
    ax1.plot(time_series, color="gray", linewidth=1, label="Time Series Data", zorder=1)
    ax1.set_ylabel("Data Value", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    # --- 4. Plot Scores and Labels ---
    for score_path in sorted(score_files):
        try:
            scores = np.load(score_path)
            if len(scores) != len(time_series):
                scores = np.resize(scores, len(time_series))

            algo_name = os.path.basename(os.path.dirname(score_path))
            color = COLOR_MAP.get(algo_name, DEFAULT_COLOR)
            ax2.plot(
                scores,
                label=f"Score ({algo_name})",
                color=color,
                linewidth=1.5,
                zorder=3,
            )
        except Exception as e:
            print(f"Could not load or plot scores from {score_path}: {e}")

    # Plot ground truth anomaly regions
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
    plt.savefig(
        save_path, dpi=150, bbox_inches="tight"
    )  # Reduced DPI slightly for speed
    plt.close(fig)


if __name__ == "__main__":
    # --- Main Automation Logic ---
    try:
        file_list_df = pd.read_csv(FILE_LIST_PATH)
    except FileNotFoundError:
        print(f"Error: File list not found at '{FILE_LIST_PATH}'")
        exit()

    # Get the list of file names from the 'file_name' column
    files_to_plot = file_list_df["file_name"].tolist()

    print(
        f"Found {len(files_to_plot)} files to process from '{os.path.basename(FILE_LIST_PATH)}'."
    )

    # Use tqdm for a nice progress bar while iterating
    for csv_filename in tqdm(files_to_plot, desc="Generating Plots"):
        # Remove the '.csv' extension to get the base name
        base_name = os.path.splitext(csv_filename)[0]
        plot_anomaly_scores(base_name)

    print("\nPlot generation complete.")
    print(f"All plots have been saved to: {PLOT_DIR}")
