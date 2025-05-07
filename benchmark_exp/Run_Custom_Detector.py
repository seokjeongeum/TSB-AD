import argparse  # Added import
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
from google import genai
from google.genai import types
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.basic_metrics import basic_metricor
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.slidingWindows import find_length_rank

model_name = "gemini-2.5-pro-exp-03-25"
api_keys = [
    "AIzaSyCHc-jYxJM_ZY56JgLstsNEU_AOGS1Jy9M",
    "AIzaSyD_K8KdyceLRsdGjiDpSdpEGEP4c2jzNkQ",
    "AIzaSyBw_XM5LTLykhfmOQYfAZacNB0kqWEA930",
    "AIzaSyCShbrFsylOr7IrsZEAHFeNTYFpjrWV3hE",
    "AIzaSyB4_okiRsYRKQqRXDB4YXHTNsN43wqninY",
    "AIzaSyA_z-q40aLVL9KcQXNFy9Cl6-LNM63PyV0",
    "AIzaSyD-Z4plUhNJ2zEsEHQcOzoV-ySB7RCkLGc",
    "AIzaSyBPEuvm0BvoSNLPclw-bs5CxsU5nrj5xc8",
    "AIzaSyAJ1ewluAP_CZugA3Fjr-gE3p17p2iHN38",
    "AIzaSyAWFrUAJXqAEEsdf3nQBFC4r83TAxYW8W0",
    "AIzaSyCuK5WWC8QRC7rtr3oKSp751lT22ewPS5Q",
    "AIzaSyCET6Ew3zj9IGwo-S6Y-_gw1SjaRq1NPuY",
    "AIzaSyCXrExiL5aaHrigFPIz7WILblGvOohiXjs",
    "AIzaSyBu7_3G2Zne1SZIzYZGO2-bBilE9d8zIB8",
    "AIzaSyDH2PFj5ktGvMov-erxJekCwEkKP5RM6Ko",
    "AIzaSyCG-eklid7qRWKG19ny-152jz9uIFalBNA",
    "AIzaSyD34QmIi1MXhUAsHSI3hW_zlXNS5dpOwFQ",
    "AIzaSyA5Vgg0Oxl_FnZbmaI3hxl2fy_utPPixWU",
    "AIzaSyBsH5YbiQU88nSM0ZyQsWWuHG4pAaSLOq0",
    "AIzaSyDjJm1IfV_X6frznOFrtPqttlWPEZEY2UI",
    "AIzaSyBgWIDQ2BDwkOAu4jEn6KwuYDIa07FZ-KQ",
    "AIzaSyBQ7S8TNRqyIn0nJkFHHkc7bjME3udflI0",
    "AIzaSyBnWeouM8SUJfS7VuzKwAW9z9e5CjkhCaQ",
    "AIzaSyD7UQ9TANoqK-P2ArntYxLFC2p9ZXlpjLI",
    "AIzaSyCZHCBavhPUeQT_C2kYqqdbJ6vddkj3ls8",
]


# --- Utility Functions ---
def strip_markdown_code_fences(code_string):
    """Removes markdown code fences (```) from a string."""
    if not isinstance(code_string, str):
        return ""
    pattern_python = r"\s*```python\n(.*?)\n```\s*$"
    match = re.match(pattern_python, code_string, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    pattern_generic = r"\s*```\n(.*?)\n```\s*$"
    match = re.match(pattern_generic, code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    code = re.sub(r"^\s*```[a-zA-Z]*\n?", "", code_string)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def extract_domain_from_filename(filename):
    """Extracts the domain name (e.g., 'SMAP', 'MSL') from TSB-AD filenames."""
    match = re.search(r"_id_.*?_(.*?)_tr_", filename)
    if match:
        return match.group(1)
    parts = filename.split("_")
    if len(parts) > 1:
        return parts[1]
    return "unknown_domain"


def _find_clusters(indices):
    """Groups consecutive or near-consecutive integer indices into start-end ranges."""
    if not isinstance(indices, (list, np.ndarray)) or len(indices) == 0:
        return []
    indices = np.unique(np.asarray(indices, dtype=int))
    if len(indices) == 0:
        return []
    if len(indices) == 1:
        return [(indices[0], indices[0])]

    diffs = np.diff(indices)
    split_points = np.where(diffs > 1)[0]
    starts = np.insert(indices[split_points + 1], 0, indices[0])
    ends = np.append(indices[split_points], indices[-1])
    return list(zip(starts, ends))


def generate_and_save_example_plot(
    data, label, domain, base_filename, example_plot_dir
):
    """Generates and saves an example plot. try-catch removed. Font sizes increased."""
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("Invalid data input for example plot generation.")
    if (
        not isinstance(label, np.ndarray)
        or label.ndim != 1
        or label.shape[0] != data.shape[0]
    ):
        raise ValueError(
            "Invalid label input or shape mismatch for example plot generation."
        )

    n_samples, n_features = data.shape
    domain_dir = os.path.join(example_plot_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    plot_filename = f"{base_filename}_EXAMPLE.png"
    plot_path = os.path.join(domain_dir, plot_filename)

    if os.path.exists(plot_path):
        return plot_path

    print(f"  Generating EXAMPLE plot for domain '{domain}': {plot_filename}...")
    plt.close("all")
    fig_width, fig_height_per_feature, dpi_val = (
        20,
        1.5,
        150,
    )  # Increased height per feature a bit
    fig_height = max(6, 2 + n_features * fig_height_per_feature)
    fig, axes = plt.subplots(
        nrows=n_features,
        ncols=1,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.flatten()
    time_range = np.arange(n_samples)
    true_anomaly_indices = np.where(label == 1)[0]
    true_anomaly_ranges = _find_clusters(true_anomaly_indices)
    colors = plt.cm.viridis(np.linspace(0, 0.8, min(n_features, 8)))
    linestyles = ["-", "--", ":", "-."]

    for i_ax in range(n_features):  # Renamed
        ax = axes_flat[i_ax]
        color = colors[i_ax % len(colors)]
        ls = linestyles[i_ax % len(linestyles)]
        ax.plot(time_range, data[:, i_ax], lw=1.0, color=color, linestyle=ls)
        ax.set_title(f"Feature {i_ax+1}", fontsize="medium", loc="left", pad=2)
        ax.set_ylabel("Value", fontsize="small")
        ax.grid(True, alpha=0.4, linestyle=":")
        ax.tick_params(axis="both", labelsize="small")  # Changed from xx-small
        ax.set_xlim(0, max(1, n_samples - 1))
        ax.set_xlabel("Time Step", fontsize="small")  # Always label
        ax.tick_params(axis="x", labelbottom=True)  # Ensure x-ticks are shown

        if true_anomaly_ranges:
            for (
                start_r,
                end_r,
            ) in true_anomaly_ranges:  # Renamed and ensured it's indented under the if
                ax.axvspan(start_r, end_r + 1, color="red", alpha=0.3, zorder=-1, lw=0)

    fig.suptitle(
        f"Example: {base_filename} (Domain: {domain}) - Anomalies Highlighted (Red)",
        fontsize="medium",
        y=0.99,
    )
    plt.subplots_adjust(hspace=0.25)  # Adjusted spacing slightly
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.96])  # Adjusted rect for labels
    fig.savefig(plot_path, dpi=dpi_val, bbox_inches="tight")
    print(f"  Example plot saved: {plot_path}")
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return plot_path


def extract_base_dataset_name(filename):
    """Extracts the base dataset name (e.g., MSL, SMAP) from the filename."""
    parts = filename.split("_")
    if len(parts) > 1:
        known_prefixes = ["MSL", "SMAP", "SMD", "NAB", "UCR", "MBA", "ECG", "YAHOO"]
        for part in parts:
            if part.upper() in known_prefixes:
                return part.upper()
        return parts[1]
    return "unknown_dataset"


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
def run_E_May_7_Semisupervised(data_train, data_test, HP):
    """Runs E_May_7. Handles supervised or unsupervised mode based on HP['SUPERVISION_MODE']."""
    clf = E_May_7(
        HP=HP
    )  # HP should contain dataset_filename, data_full, label_full, train_idx, SUPERVISION_MODE etc.

    train_labels = None
    current_supervision_mode = HP.get("SUPERVISION_MODE", "supervised").lower()

    if current_supervision_mode == "supervised":
        # Attributes like label_full_for_plot are set in E_May_7.__init__ from HP
        if clf.label_full_for_plot is not None and clf.train_index_for_plot is not None:
            if clf.train_index_for_plot <= len(clf.label_full_for_plot):
                train_labels = clf.label_full_for_plot[: clf.train_index_for_plot]
    else:  # This 'else' must align with the 'if' on line 211
        print(
            f"  Running E_May_7 in UNSUPERVISED mode for fit(). No training labels will be passed to fit."
        )
        # train_labels remains None

    clf.fit(X=data_train, y=train_labels)  # y will be None if unsupervised

    score = clf.decision_function(
        X=data_test
    )  # data_test is the full data series for scoring
    identified_ranges_from_fit = (
        clf.identified_ranges_
    )  # Get ranges identified during fit

    if (
        score is None
        or not isinstance(score, np.ndarray)
        or score.size == 0
        or score.shape[0] != data_test.shape[0]  # Score should match data_test
    ):
        # Return zeros matching the shape of data_test (the full data scored)
        return np.zeros(data_test.shape[0]), identified_ranges_from_fit

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=(
            np.nanmax(score[np.isfinite(score)]) if np.any(np.isfinite(score)) else 1.0
        ),
        neginf=0.0,
    )
    score_ptp = np.ptp(score)
    if score_ptp == 0:
        scaled_score = np.zeros_like(score)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_score = scaler.fit_transform(score.reshape(-1, 1)).ravel()
    return scaled_score, identified_ranges_from_fit


# --- Visualization Function ---
def visualize_errors(
    original_label,
    score,
    model_identified_ranges,
    base_plot_filename,
    plot_dir,
    model_prefix="",
):
    """Generates a plot comparing anomaly scores to true labels and highlights model ranges."""
    plt.style.use("seaborn-v0_8-whitegrid")

    n_samples = len(original_label)
    if not (
        isinstance(original_label, np.ndarray)
        and original_label.ndim == 1
        and isinstance(score, np.ndarray)
        and score.ndim == 1
        and score.shape[0] == n_samples
        and original_label.shape[0] == n_samples
    ):
        raise ValueError(f"Invalid input for visualization {base_plot_filename}")

    if not isinstance(model_identified_ranges, list):
        model_identified_ranges = []

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    time_range = np.arange(n_samples)
    ax.plot(
        time_range,
        score,
        label=f"Anomaly Score ({model_prefix})",
        color="darkorange",
        linestyle="-",
        alpha=0.9,
        lw=1.5,
    )
    ax.step(
        time_range,
        original_label,
        label="True Anomaly Label",
        color="dodgerblue",
        linestyle="--",
        alpha=0.8,
        lw=1.0,
        where="post",
    )

    range_label_added = False
    if model_identified_ranges:
        for item in model_identified_ranges:
            if not (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(n, (int, np.integer)) for n in item)
            ):
                continue
            start, end = item
            safe_start, safe_end = max(0, start), min(n_samples, end + 1)
            if safe_start < safe_end:
                label_text = (
                    "Model Identified Range" if not range_label_added else "_nolegend_"
                )
                ax.axvspan(
                    safe_start,
                    safe_end,
                    color="gray",
                    alpha=0.20,
                    zorder=-1,
                    label=label_text,
                )
                range_label_added = True

    ax.set_ylabel("Score / Label", fontsize="medium")
    ax.tick_params(axis="y", labelsize="small")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(1, n_samples - 1))
    ax.set_xlabel("Time Index", fontsize="medium")

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize="medium")

    title = f"{model_prefix} - {base_plot_filename}\\nScore vs Label (Model-Identified Ranges Highlighted)"
    fig.suptitle(title, fontsize=12)

    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.94])

    plot_filename = os.path.join(
        plot_dir, f"{base_plot_filename}_{model_prefix}_Score_vs_Label_ModelRanges.png"
    )
    plt.savefig(plot_filename, bbox_inches="tight", dpi=200)

    if fig is not None and plt.fignum_exists(fig.number):
        plt.close(fig)


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
MODEL_NAMES = ["E_May_7"]


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

    for prefix in model_prefixes.values():
        cols = [
            f"{prefix}_{m}" for m in BASE_METRIC_ORDER if m not in ["VUS-PR", "VUS-ROC"]
        ]
        for col in cols:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)

    # Add Benchmark Comparison Columns here
    for prefix in model_prefixes.values():
        benchmark_cols = [
            f"{prefix}_Benchmark_Avg_VUS-PR",
            f"{prefix}_Benchmark_Max_VUS-PR",
            f"{prefix}_E_May_7_vs_Avg_VUS-PR_Diff",
            f"{prefix}_E_May_7_vs_Max_VUS-PR_Diff",
        ]
        for col in benchmark_cols:
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

    remaining = sorted(list(set(current_columns) - processed_columns))
    final_order.extend(remaining)
    return final_order


# --- E_May_7 Model Class ---
class E_May_7(BaseDetector):
    """
    Anomaly detector using Google GenAI (Gemini). API error handling is removed.
    """

    def __init__(self, HP):
        super().__init__()
        self.api_keys = api_keys
        self.current_api_key_index = 0
        self.client = None
        self.decision_scores_ = None  # Stores scores from fit() on training data
        self.identified_ranges_ = []
        self.generated_code_ = None
        self.generated_code_dir = "E_May_7_Generated_Code"
        self.plot_save_dir = "E_May_7_Input_Plots"
        self.uploaded_files_tracking = []  # For robust cleanup

        # Extract necessary context from HP, passed by the runner
        self.dataset_filename = HP.get("dataset_filename", "unknown_dataset.csv")
        self.data_full_for_plot = HP.get(
            "data_full_for_plot"
        )  # Expected to be a NumPy array
        self.label_full_for_plot = HP.get(
            "label_full_for_plot"
        )  # Expected to be a NumPy array
        self.train_index_for_plot = HP.get(
            "train_index_for_plot"
        )  # Expected to be an int
        self.supervision_mode = HP.get("SUPERVISION_MODE", "supervised").lower()

        # Validate essential context items
        if self.data_full_for_plot is None:
            raise ValueError(
                "E_May_7 requires 'data_full_for_plot' in HP for plotting context."
            )
        if self.label_full_for_plot is None:
            raise ValueError(
                "E_May_7 requires 'label_full_for_plot' in HP for plotting context."
            )
        if self.train_index_for_plot is None:
            raise ValueError(
                "E_May_7 requires 'train_index_for_plot' in HP for plotting context."
            )

        self.domain = extract_domain_from_filename(self.dataset_filename)
        self.example_plot_dir = "Example_Plots"

        # Standard HPs for the model's behavior
        self.MAX_EXAMPLE_PLOTS = HP.get("max_example_plots", 5)
        self.use_same_domain_examples_only = HP.get(
            "use_same_domain_examples_only", True
        )
        self.provide_examples_to_api = HP.get("provide_examples_to_api", False)

        os.makedirs(self.generated_code_dir, exist_ok=True)
        os.makedirs(self.plot_save_dir, exist_ok=True)
        os.makedirs(self.example_plot_dir, exist_ok=True)
        self._initialize_client()

    def _initialize_client(self):
        """
        Initializes the GenAI client using the current API key.
        Relies on the caller to handle exceptions from genai.Client().
        """
        if not genai:  # From global import
            raise RuntimeError("Google GenAI library (genai) not available.")

        if not self.api_keys:
            raise RuntimeError("No API keys loaded into E_May_7 model instance.")

        if not (0 <= self.current_api_key_index < len(self.api_keys)):
            raise RuntimeError(
                f"Internal error: current_api_key_index ({self.current_api_key_index}) is out of bounds for {len(self.api_keys)} keys."
            )

        key = self.api_keys[self.current_api_key_index]
        # The print statement here was slightly different from the one in _call_api_with_retry, aligning them.
        # The _call_api_with_retry will print the "Attempt X/Y..." line. This will specify init.
        print(
            f"    Initializing GenAI Client with key index {self.current_api_key_index}...",
            end="",
        )

        if hasattr(genai, "Client"):
            self.client = genai.Client(api_key=key)  # This can raise exceptions
            print(" OK.")
        else:
            self.client = None
            raise RuntimeError(
                "genai.Client class not found in the imported genai library."
            )

    def _call_api_with_retry(self, api_call_func, *args, **kwargs):
        """
        Wraps a GenAI API call, implementing retry logic with API key rotation.
        If an API call fails, it tries the next key.
        """
        max_attempts = len(self.api_keys)
        if max_attempts == 0:
            raise RuntimeError("No API keys configured for E_May_7 model.")

        for attempt_num in range(max_attempts):
            # self.current_api_key_index is used by _initialize_client directly
            print(
                f"  API Call: Attempt {attempt_num + 1}/{max_attempts}. Using API Key Index: {self.current_api_key_index}"
            )

            try:
                self.client = None  # Force re-initialization for the current key
                self._initialize_client()  # Uses self.current_api_key_index

                if not self.client:
                    # This case implies _initialize_client didn't set self.client but also didn't raise an error,
                    # which its current logic shouldn't allow.
                    raise RuntimeError(
                        "_initialize_client failed to set self.client without raising an error."
                    )

                result = api_call_func(self.client, *args, **kwargs)
                # If successful, no need to print here, _initialize_client already printed "OK."
                return result
            except Exception as e:
                # Catches errors from _initialize_client (e.g., bad key) or from api_call_func (e.g., quota, server error)
                print(
                    f"    Attempt {attempt_num + 1} with key index {self.current_api_key_index} failed: {type(e).__name__} - {e}"
                )

                # Rotate to the next key
                self.current_api_key_index = (
                    self.current_api_key_index + 1
                ) % max_attempts
                self.client = None  # Ensure client is reset for the next attempt

                if attempt_num < max_attempts - 1:
                    # No explicit print for rotating, the next iteration's "API Call: Attempt..." line will show the new index.
                    pass
                else:
                    # This was the last attempt
                    last_failed_index = (
                        self.current_api_key_index - 1 + max_attempts
                    ) % max_attempts  # Index that just failed
                    print(
                        f"    All {max_attempts} API keys have been tried and failed."
                    )
                    raise RuntimeError(
                        f"All {max_attempts} API keys failed. Last error on key index {last_failed_index}: {type(e).__name__} - {e}"
                    )

    def fit(self, X, y=None):
        """Fits on training data X. Uses self.data_full_for_plot, self.label_full_for_plot,
        self.train_index_for_plot for plotting full context and Step 1 GenAI.
        y is training labels, used for plotting known anomalies in training part if provided.
        """
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Invalid input training data X to fit method.")
        if not self.api_keys:
            raise ValueError("No API keys for E_May_7 model.")

        # Validate context from __init__ needed for plotting and prompts
        if (
            not isinstance(self.data_full_for_plot, np.ndarray)
            or self.data_full_for_plot.ndim != 2
        ):
            raise ValueError("Invalid self.data_full_for_plot")
        if (
            not isinstance(self.label_full_for_plot, np.ndarray)
            or self.label_full_for_plot.ndim != 1
        ):
            raise ValueError("Invalid self.label_full_for_plot")
        if not (self.data_full_for_plot.shape[0] == self.label_full_for_plot.shape[0]):
            raise ValueError(
                "Shape mismatch: self.data_full_for_plot vs self.label_full_for_plot"
            )
        if not (0 <= self.train_index_for_plot <= self.data_full_for_plot.shape[0]):
            raise ValueError("self.train_index_for_plot out of bounds")

        print(f"Processing FIT for: {self.dataset_filename} on train data {X.shape}")
        print(
            f"Plotting context: Full data {self.data_full_for_plot.shape}, train_idx {self.train_index_for_plot}"
        )

        _main_plot_fig_to_close = None
        n_samples_fit, n_features_fit = X.shape  # From training data X
        n_samples_plot, n_features_plot = (
            self.data_full_for_plot.shape
        )  # From full data context

        if n_features_fit != n_features_plot:  # Should be same number of features
            raise ValueError(
                f"Feature count mismatch: train data X ({n_features_fit}) vs full data for plot ({n_features_plot})."
            )
        n_features = n_features_fit  # Use consistent n_features

        main_plot_api_file, example_plot_api_files = None, []
        base_fname = os.path.splitext(self.dataset_filename)[0]
        main_plot_filename = f"{base_fname}_InputPlot_FullContext.png"
        main_plot_path = os.path.join(self.plot_save_dir, main_plot_filename)
        code_filename = f"{base_fname}_generated_code.py"
        code_save_path = os.path.join(self.generated_code_dir, code_filename)
        current_base_dataset_name = extract_base_dataset_name(self.dataset_filename)

        print("  Generating main input plot (full data context)...")
        plt.close("all")
        fig_width_plot, fig_height_per_plot_val, dpi_val_plot = 40, 2.0, 150
        fig_height_plot = max(
            10, fig_height_per_plot_val * n_features
        )  # Use n_features
        _main_plot_fig_to_close, axes_plot = plt.subplots(
            nrows=n_features,
            ncols=1,
            figsize=(fig_width_plot, fig_height_plot),
            squeeze=False,
            sharex=True,
        )
        axes_flat_plot = axes_plot.flatten()
        time_range_plot = np.arange(
            n_samples_plot
        )  # Use n_samples_plot for time range of full plot

        # Use self.label_full_for_plot and self.train_index_for_plot for anomalies in TRAIN part of the plot
        train_anomaly_indices_on_full_plot = np.where(
            self.label_full_for_plot[: self.train_index_for_plot] == 1
        )[0]
        train_anomaly_ranges_on_full_plot = _find_clusters(
            train_anomaly_indices_on_full_plot
        )

        plot_title_suffix = "".join(
            [
                f"Train [0-{self.train_index_for_plot-1}] Blue.",
                (
                    " Known Train Anoms Red."
                    if self.supervision_mode == "supervised"
                    and y is not None
                    and np.any(y)
                    else ""
                ),
                " Post-Train Gray Dash.",
            ]
        )

        for i_plot_feat in range(n_features):
            ax_p = axes_flat_plot[i_plot_feat]
            ax_p.set_title(
                f"Feature {i_plot_feat+1} of {n_features} (Original Index: {i_plot_feat})",
                fontsize="medium",
                loc="left",
                pad=2,
            )
            ax_p.set_ylabel("Value", fontsize="small")
            ax_p.grid(True, alpha=0.3, linestyle=":")
            ax_p.tick_params(axis="both", labelsize="small")

            # Plot training data segment (from X) up to self.train_index_for_plot points, on the full plot scale
            # The actual training data X might be shorter than self.train_index_for_plot if train_index was set for a larger original dataset.
            # We plot X, which is data_train_for_fit. Its length is n_samples_fit.
            # self.train_index_for_plot is the boundary on the *full plot*.

            # Plot the portion of self.data_full_for_plot that corresponds to the training region
            if self.train_index_for_plot > 0:
                # Ensure we don't plot beyond the actual length of self.data_full_for_plot or X for this feature
                train_segment_len = min(
                    self.train_index_for_plot, self.data_full_for_plot.shape[0]
                )
                ax_p.plot(
                    time_range_plot[:train_segment_len],
                    self.data_full_for_plot[:train_segment_len, i_plot_feat],
                    color="blue",  # Main data color
                    linestyle="-",
                    lw=1.1,
                )

            # Plot post-training data segment (if any) from self.data_full_for_plot
            if self.train_index_for_plot < n_samples_plot:
                ax_p.plot(
                    time_range_plot[self.train_index_for_plot : n_samples_plot],
                    self.data_full_for_plot[
                        self.train_index_for_plot : n_samples_plot, i_plot_feat
                    ],
                    color="blue",  # Same color as training data
                    linestyle="-",  # Same linestyle
                    lw=1,  # Can adjust lw if needed, but keeping similar to train
                )
            # If train_index is 0 but there's full data, plot all of it as "Data" (blue)
            elif self.train_index_for_plot == 0 and n_samples_plot > 0:
                ax_p.plot(
                    time_range_plot[:n_samples_plot],
                    self.data_full_for_plot[:n_samples_plot, i_plot_feat],
                    color="blue",  # Same color
                    linestyle="-",
                    lw=1,
                )

            # Highlight known anomalies IN TRAINING REGION (red line segments on top of blue)
            # These anomalies are derived from self.label_full_for_plot[:self.train_index_for_plot]
            # Only plot if in supervised mode and y (training labels) were provided and have anomalies
            if (
                self.supervision_mode == "supervised"
                and y is not None
                and train_anomaly_ranges_on_full_plot
            ):
                for start_tr_anom, end_tr_anom in train_anomaly_ranges_on_full_plot:
                    # These indices are relative to the start of the full plot
                    plot_start_idx = start_tr_anom
                    plot_end_idx = end_tr_anom + 1

                    # Ensure we're plotting data from the training segment of self.data_full_for_plot
                    # and indices are within the bounds of this segment
                    if (
                        plot_start_idx < plot_end_idx
                        and plot_start_idx < self.train_index_for_plot
                    ):
                        eff_plot_end_idx = min(plot_end_idx, self.train_index_for_plot)
                        # Data for red line comes from the training part of self.data_full_for_plot
                        ax_p.plot(
                            time_range_plot[plot_start_idx:eff_plot_end_idx],
                            self.data_full_for_plot[
                                plot_start_idx:eff_plot_end_idx, i_plot_feat
                            ],
                            color="red",
                            linestyle="-",
                            lw=1.2,
                            zorder=10,
                        )

            # X-axis formatting for each subplot
            ax_p.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
            ax_p.set_xlim(0, max(1, n_samples_plot - 1))
            ax_p.set_xlabel("Time Step", fontsize="small")
            ax_p.tick_params(axis="x", labelbottom=True)

            # Vertical line for end of training data
            if (
                self.train_index_for_plot < n_samples_plot
                and self.train_index_for_plot > 0
            ):
                ax_p.axvline(
                    x=self.train_index_for_plot - 0.5,
                    color="green",
                    linestyle="--",
                    linewidth=1.2,
                )

        _main_plot_fig_to_close.suptitle(
            f"Input: {self.dataset_filename} ({n_features}F, {n_samples_plot}S Total). {plot_title_suffix}",
            fontsize="medium",
            y=0.99,
        )
        _main_plot_fig_to_close.tight_layout(rect=[0.03, 0.04, 0.97, 0.95])
        _main_plot_fig_to_close.savefig(main_plot_path, dpi=dpi_val_plot)
        print(f"  Main context plot saved: {main_plot_path}")

        selected_example_paths, example_plot_parts = [], []
        if self.provide_examples_to_api:
            print("  Searching for example plots...")
            all_found_example_files, search_dirs = [], []
            if self.use_same_domain_examples_only:
                domain_dir = os.path.join(self.example_plot_dir, self.domain)
                if os.path.isdir(domain_dir):
                    search_dirs.append((domain_dir, self.domain))
            else:
                for item in os.listdir(self.example_plot_dir):
                    potential_dir = os.path.join(self.example_plot_dir, item)
                    if os.path.isdir(potential_dir):
                        search_dirs.append((potential_dir, item))
            for dir_path, domain_name_iter_ex in search_dirs:
                for fname_ex in os.listdir(dir_path):
                    if fname_ex.lower().endswith("_example.png"):
                        all_found_example_files.append(
                            (os.path.join(dir_path, fname_ex), domain_name_iter_ex)
                        )
            available_examples_info_ex = [
                (fp_ex, dn_ex)
                for fp_ex, dn_ex in all_found_example_files
                if extract_base_dataset_name(os.path.basename(fp_ex))
                != current_base_dataset_name
            ]
            if available_examples_info_ex:
                num_to_select_ex = min(
                    len(available_examples_info_ex), self.MAX_EXAMPLE_PLOTS
                )
                selected_examples_info_ex = random.sample(
                    available_examples_info_ex, num_to_select_ex
                )
                selected_example_paths = [
                    info_ex[0] for info_ex in selected_examples_info_ex
                ]
                print(f"  Selected {len(selected_example_paths)} example plots.")

        upload_func_fit = lambda client, path_up_fit: client.files.upload(
            file=path_up_fit
        )
        main_plot_api_file = self._call_api_with_retry(upload_func_fit, main_plot_path)
        if main_plot_api_file:
            self.uploaded_files_tracking.append(
                (main_plot_api_file, self.current_api_key_index)
            )

        main_plot_part = types.Part(
            file_data=types.FileData(
                mime_type=main_plot_api_file.mime_type, file_uri=main_plot_api_file.uri
            )
        )
        if self.provide_examples_to_api and selected_example_paths:
            print(f"  Uploading {len(selected_example_paths)} example plots...")
            for ex_path_up in selected_example_paths:
                if os.path.exists(ex_path_up):
                    ex_api_file_up = self._call_api_with_retry(
                        upload_func_fit, ex_path_up
                    )
                    example_plot_parts.append(
                        types.Part(
                            file_data=types.FileData(
                                mime_type=ex_api_file_up.mime_type,
                                file_uri=ex_api_file_up.uri,
                            )
                        )
                    )
                    example_plot_api_files.append(
                        ex_api_file_up
                    )  # Keep this for local reference if needed, but tracking is primary
                    if ex_api_file_up:
                        self.uploaded_files_tracking.append(
                            (ex_api_file_up, self.current_api_key_index)
                        )

        if self.provide_examples_to_api and example_plot_parts:
            print(
                f"\n--- Step 1: Identifying anomalous ranges from FULL CONTEXT PLOT ({len(example_plot_parts)} examples if provided)... ---"
            )
        example_source_text_s1_fit = (
            f"from domain '{self.domain}'"
            if self.use_same_domain_examples_only
            else "from any domain"
        )
        prompt_step1_parts_list_fit = []
        if self.provide_examples_to_api and example_plot_parts:
            prompt_step1_parts_list_fit.append(
                types.Part(
                    text=f"{len(example_plot_parts)} EXAMPLE plots ({example_source_text_s1_fit}) showing anomalies (red regions):"
                )
            )
            prompt_step1_parts_list_fit.extend(example_plot_parts)
            prompt_step1_parts_list_fit.append(
                types.Part(
                    text="Now, analyze the MAIN CONTEXT PLOT (full series with training highlighted) provided below:"
                )
            )
        else:
            prompt_step1_parts_list_fit.append(
                types.Part(
                    text="Analyze the following MAIN CONTEXT PLOT (full series with training data region indicated):"
                )
            )
        prompt_step1_parts_list_fit.append(main_plot_part)

        step1_prompt_anomaly_guidance = ""
        if self.supervision_mode == "supervised" and y is not None and np.any(y):
            step1_prompt_anomaly_guidance = f"Known anomalies WITHIN THIS TRAINING REGION [0, {self.train_index_for_plot-1}] are highlighted in red. Consider these as context."
        else:
            step1_prompt_anomaly_guidance = f"The training region is [0, {self.train_index_for_plot-1}]. No specific anomalies are pre-highlighted in this training region on the plot for your Step 1 analysis, or supervision is off."

        prompt_step1_instruction_text_fit = (
            f"The MAIN plot shows {n_features} features over {n_samples_plot} total time steps for dataset '{self.dataset_filename}'. "
            f"The training data region spans indices [0, {self.train_index_for_plot-1}] and is plotted in blue. {step1_prompt_anomaly_guidance} "
            f"Based ONLY on visual inspection of THIS MAIN PLOT (entire series), identify time step index ranges [start, end] (inclusive, 0-based integers relative to the full {n_samples_plot} steps) that appear visually anomalous. "
            f"Your goal is to find anomalies in the WHOLE series, including any new ones outside or even within the training part if they look different from any already marked ones (if any were marked). "
            f"Return ONLY a valid JSON list of ranges. Example: [[100, 150], [{n_samples_plot-100}, {n_samples_plot-50}]]. If none, return []. NO other text."
        )
        prompt_step1_parts_list_fit.append(
            types.Part(text=prompt_step1_instruction_text_fit)
        )
        contents_step1_fit = [
            types.Content(role="user", parts=prompt_step1_parts_list_fit)
        ]
        config_step1_fit = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            response_schema=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.NUMBER),
                    min_items=2,
                    max_items=2,
                ),
            ),
        )
        step1_api_call_fit = lambda client, model_arg, contents_arg, config_arg: client.models.generate_content(
            model=model_arg, contents=contents_arg, config=config_arg
        )
        response_step1_fit = self._call_api_with_retry(
            step1_api_call_fit, model_name, contents_step1_fit, config_step1_fit
        )
        self.identified_ranges_ = []
        parsed_ranges_s1_fit = response_step1_fit.parsed
        if parsed_ranges_s1_fit is None and response_step1_fit.text:
            cleaned_text_s1_fit = strip_markdown_code_fences(
                response_step1_fit.text
            ).strip()
            if cleaned_text_s1_fit:
                parsed_ranges_s1_fit = json.loads(cleaned_text_s1_fit)
            else:
                parsed_ranges_s1_fit = []
        if isinstance(parsed_ranges_s1_fit, list):
            for item_s1_fit in parsed_ranges_s1_fit:
                if isinstance(item_s1_fit, (list, tuple)) and len(item_s1_fit) == 2:
                    s_val_fit, e_val_fit = int(item_s1_fit[0]), int(item_s1_fit[1])
                    if 0 <= s_val_fit <= e_val_fit < n_samples_plot:
                        self.identified_ranges_.append([s_val_fit, e_val_fit])
            if self.identified_ranges_:
                self.identified_ranges_.sort(key=lambda x_val_fit: x_val_fit[0])
        print(
            f"  LLM identified ranges (on full plotted data): {self.identified_ranges_}"
        )

        print(
            f"\n--- Step 2: Generating Python scoring function (context from TRAIN data {X.shape}, {len(example_plot_parts) if self.provide_examples_to_api else 0} examples)... ---"
        )
        numerical_data_extracts_s2_fit = {}
        for start_r_full, end_r_full in self.identified_ranges_:
            overlap_start = max(0, start_r_full)
            overlap_end = min(
                self.train_index_for_plot - 1, end_r_full
            )  # Use self.train_index_for_plot
            # If the range from full plot has some part within the training data segment
            if overlap_start < overlap_end:
                # Extract from X (training data), using indices relative to X
                eff_start_in_train = max(0, start_r_full)
                eff_end_in_train = min(self.train_index_for_plot, end_r_full + 1)
                if eff_start_in_train < eff_end_in_train:
                    # Ensure slice indices are valid for X
                    slice_start_for_X = eff_start_in_train
                    slice_end_for_X = eff_end_in_train
                    if (
                        slice_start_for_X < X.shape[0]
                        and slice_end_for_X <= X.shape[0]
                        and slice_start_for_X < slice_end_for_X
                    ):
                        current_extract_data = X[slice_start_for_X:slice_end_for_X, :]
                        if current_extract_data.shape[0] > 0:
                            if current_extract_data.shape[0] > 100:
                                current_extract_data = current_extract_data[
                                    np.linspace(
                                        0,
                                        current_extract_data.shape[0] - 1,
                                        100,
                                        dtype=int,
                                    ),
                                    :,
                                ]
                            numerical_data_extracts_s2_fit[
                                f"range_{start_r_full}-{end_r_full}_from_full_plot_showing_train_segment_{slice_start_for_X}-{slice_end_for_X-1}"
                            ] = current_extract_data.tolist()

        data_string_step2_json_fit = json.dumps(
            numerical_data_extracts_s2_fit, separators=(",", ":")
        )
        if len(data_string_step2_json_fit) > 15000:
            data_string_step2_json_fit = (
                data_string_step2_json_fit[:15000] + "...(truncated)}"
            )

        prompt_step2_parts_list_fit = []
        if self.provide_examples_to_api and example_plot_parts:
            prompt_step2_parts_list_fit.append(
                types.Part(
                    text=f"Context: {len(example_plot_parts)} Example Plot Images ({example_source_text_s1_fit}):"
                )
            )
            prompt_step2_parts_list_fit.extend(example_plot_parts)
        prompt_step2_parts_list_fit.append(
            types.Part(
                text=f"Main Context Plot (Full Series with Train Data Highlighted - Dataset: {self.dataset_filename}):"
            )
        )
        prompt_step2_parts_list_fit.append(main_plot_part)

        prompt_step2_instruction_text_updated_final = f"""You are an expert Python programmer for time series anomaly detection.

**Context for defining `calculate_anomaly_scores(current_X_data)`:**
1.  **Training Data Characteristics:** The function's logic should be primarily informed by analysis of a specific TRAINING DATA segment (shape: ({n_samples_fit}, {n_features})). The dataset is '{self.dataset_filename}'. This training data was passed as `X` to the `fit` method.
2.  **Main Context Plot:** You have also seen a plot of the FULL data series ({n_samples_plot} total steps, from `self.data_full_for_plot`), where the training segment [0, {self.train_index_for_plot-1}] was highlighted{', along with its known anomalies (from `self.label_full_for_plot`)' if self.supervision_mode == 'supervised' and y is not None and np.any(y) else ''}. 
3.  **Example Plots (if provided):** {'You were also shown ' + str(len(example_plot_parts)) + ' example plots (' + example_source_text_s1_fit + ').' if self.provide_examples_to_api and example_plot_parts else 'No separate example plots were provided or disabled.'}
4.  **Visually Identified Ranges (from full plot):** Based on the MAIN CONTEXT PLOT, these ranges across the *full series* were flagged as potentially anomalous: {self.identified_ranges_ if self.identified_ranges_ else 'None'}. These might be inside or outside the training part.
5.  **Numerical Snippets (from training data `X`, portions overlapping with identified ranges):**
    ```json
    {data_string_step2_json_fit}
    ```

**Task:**
Write a Python function `def calculate_anomaly_scores(current_X_data):`.
Your primary deliverable is the Python code string for this function. This function will be later executed by the calling system with different datasets (e.g., the training data segment itself, or entirely new test data).
- `current_X_data`: A NumPy array (the data to be scored, e.g., training data or new test data).
- Returns: A 1D NumPy array or Python list of anomaly scores (floats), one per sample in `current_X_data`. Higher score = more anomalous.

**Function Requirements:**
1.  **Signature:** `def calculate_anomaly_scores(current_X_data):`.
2.  **Data Usage:** MUST use `current_X_data` for calculations.
3.  **No Global Training Data Access:** The original training data `X` (from which context like shape ({n_samples_fit}, {n_features}) was derived) is NOT directly accessible as a global variable within the `calculate_anomaly_scores` function. All necessary parameters or logic derived from the training context must be self-contained or hardcoded within the generated function itself.
4.  **Context Integration:** Base the scoring logic on insights from the TRAINING DATA `X` characteristics (shape ({n_samples_fit}, {n_features})){', its known anomalies (contextualized by plot from `self.label_full_for_plot` if supervision was on and labels provided)' if self.supervision_mode == 'supervised' and y is not None and np.any(y) else ''}, and the `identified_ranges` (from the full plot). Your function may hardcode parameters (thresholds, window sizes, feature indices) derived from this training context. The `identified_ranges` from the full plot (context item 4) should primarily inform what types of patterns to look for, rather than being hardcoded as specific indices if `current_X_data` is not the full series.
5.  **Anomaly Logic Examples for `current_X_data`:**
    *   Calculate deviations from rolling statistics of `current_X_data`.
    *   Use statistical properties (variance, entropy) in sliding windows.
    *   {'If `current_X_data` corresponds to the training segment `X`, you might score known anomaly regions (contextualized by plot from `self.label_full_for_plot`) higher.' if self.supervision_mode == 'supervised' and y is not None and np.any(y) else 'If `current_X_data` corresponds to the training segment `X`, base your analysis on its patterns (even if specific anomalies were not pre-labeled for you).'} If it's new data, look for similar patterns.
6.  **Output:** Scores for `current_X_data`. Non-negative. Normalize (0-1) if feasible.
7.  **Imports:** Standard libraries assumed. Necessary imports (e.g., `import numpy as np`) should be placed *inside* the `calculate_anomaly_scores` function if needed, but not at the top level of the returned code string.
8.  **Strictly Self-Contained Function Code:**
    *   The Python code string you output MUST start with `def calculate_anomaly_scores(current_X_data):` and contain ONLY this single function definition.
    *   All logic, constants, parameters derived from the training context (e.g., feature lists, thresholds, window sizes based on `X` and `y` from `fit`), and any helper sub-routines (which must be nested functions) MUST be defined *inside* the `calculate_anomaly_scores` function body.
    *   Do NOT include any statements (e.g., global constant assignments, comments, or top-level imports) *outside* this single function definition in the returned code string.
9.  **Important Note on NumPy Boolean Operations:** When working with NumPy arrays that may contain boolean values, do NOT use arithmetic subtraction (`-`) for element-wise comparison or logical operations. Instead, use appropriate bitwise operators (e.g., `^` for XOR, `&` for AND, `|` for OR, `~` for NOT) or NumPy's logical functions (e.g., `np.logical_xor`, `np.logical_and`, `np.logical_or`). For example, to find elements that are in one boolean array but not another, use `array1 ^ array2` (XOR) or `array1 & ~array2`.

**Output Format:** ONLY the raw Python code string defining the single `calculate_anomaly_scores` function, starting with `def calculate_anomaly_scores(current_X_data):` and ending with the function's last line. NO explanations, NO markdown, NO surrounding text or comments outside the function definition itself.
"""
        prompt_step2_parts_list_fit.append(
            types.Part(text=prompt_step2_instruction_text_updated_final)
        )
        contents_step2_fit = [
            types.Content(role="user", parts=prompt_step2_parts_list_fit)
        ]
        config_step2_fit = types.GenerateContentConfig(temperature=0.0)
        step2_api_call_fit = lambda client, model_arg, contents_arg, config_arg: client.models.generate_content(
            model=model_arg, contents=contents_arg, config=config_arg
        )
        response_step2_fit = self._call_api_with_retry(
            step2_api_call_fit, model_name, contents_step2_fit, config_step2_fit
        )
        print("  Step 2 GenAI response received.")

        self.decision_scores_ = np.zeros(X.shape[0])
        self.generated_code_ = None
        if not response_step2_fit.text:
            raise ValueError("Step 2 GenAI returned empty text for code.")
        self.generated_code_ = strip_markdown_code_fences(
            response_step2_fit.text
        )  # Existing function should handle it
        with open(code_save_path, "w", encoding="utf-8") as f_code_save:
            f_code_save.write(
                f"# Generated for {self.dataset_filename}, train data context {X.shape}.\n"
            )
            f_code_save.write(
                f"# Visually identified ranges (from full plot): {self.identified_ranges_}\n"
            )
            f_code_save.write(
                f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f_code_save.write(self.generated_code_)
        print(f"  Generated code saved: {code_save_path}")

        exec_globals_fit_run = {"np": np, "pd": pd}
        if "scipy" not in exec_globals_fit_run:
            import scipy.stats

            exec_globals_fit_run["scipy"] = scipy
        if "sklearn" not in exec_globals_fit_run:
            import sklearn.ensemble
            import sklearn.neighbors
            import sklearn.preprocessing

            exec_globals_fit_run["sklearn"] = sklearn
        exec_locals_fit_run = {}
        exec(self.generated_code_, exec_globals_fit_run, exec_locals_fit_run)
        if "calculate_anomaly_scores" not in exec_locals_fit_run or not callable(
            exec_locals_fit_run["calculate_anomaly_scores"]
        ):
            raise NameError(
                "Generated code must define callable 'calculate_anomaly_scores(current_X_data)'."
            )
        calc_func_for_fit_data = exec_locals_fit_run["calculate_anomaly_scores"]
        scores_raw_for_fit_data = calc_func_for_fit_data(
            X
        )  # Call with X (training data)

        if not isinstance(scores_raw_for_fit_data, (list, np.ndarray)):
            raise TypeError(
                f"Generated function returned invalid type for FIT data: {type(scores_raw_for_fit_data)}."
            )
        scores_arr_for_fit_data = np.array(scores_raw_for_fit_data, dtype=float)
        if not (
            scores_arr_for_fit_data.ndim == 1
            and scores_arr_for_fit_data.shape[0] == X.shape[0]
        ):  # Compare with X
            raise ValueError(
                f"Scores shape ({scores_arr_for_fit_data.shape}) mismatch FIT data ({X.shape[0]},)."
            )
        if np.any(~np.isfinite(scores_arr_for_fit_data)):
            max_finite_fit_run = (
                np.nanmax(scores_arr_for_fit_data[np.isfinite(scores_arr_for_fit_data)])
                if np.any(np.isfinite(scores_arr_for_fit_data))
                else 1.0
            )
            scores_arr_for_fit_data = np.nan_to_num(
                scores_arr_for_fit_data, nan=0.0, posinf=max_finite_fit_run, neginf=0.0
            )
        self.decision_scores_ = scores_arr_for_fit_data
        print(
            f"  Successfully obtained {len(self.decision_scores_)} scores for training data segment."
        )

        # --- Enhanced File Cleanup Logic ---
        print("  Executing fit() cleanup for uploaded API files...")
        original_active_key_index_at_cleanup_start = (
            self.current_api_key_index
        )  # Save current state

        # Create a copy because we might modify tracking if we decide to remove successfully deleted items,
        # though for this script, __init__ clears it for the next E_May_7 instance.
        files_to_attempt_deletion = list(self.uploaded_files_tracking)
        # self.uploaded_files_tracking is cleared in __init__ for the next instance.

        if files_to_attempt_deletion and self.api_keys:
            print(
                f"  Attempting to delete {len(files_to_attempt_deletion)} tracked uploaded API file(s)..."
            )
            for (
                api_file_obj_to_delete,
                key_idx_at_upload_time,
            ) in files_to_attempt_deletion:
                if api_file_obj_to_delete and hasattr(api_file_obj_to_delete, "name"):
                    print(
                        f"    - Trying to delete '{api_file_obj_to_delete.name}' using API key index {key_idx_at_upload_time} (key used for its upload)."
                    )
                    self.current_api_key_index = (
                        key_idx_at_upload_time  # Set the key index used for upload
                    )
                    try:
                        self.client = None  # Force re-initialization
                        self._initialize_client()  # Uses self.current_api_key_index; will print "OK." or raise
                        # If _initialize_client succeeded, self.client is now set with the correct key.
                        print(
                            f"      Client re-initialized with key index {key_idx_at_upload_time}. Attempting delete..."
                        )
                        self.client.files.delete(name=api_file_obj_to_delete.name)
                        print(
                            f"      Successfully deleted '{api_file_obj_to_delete.name}'."
                        )
                    except Exception as e_cleanup_del:
                        print(
                            f"      Error deleting file '{api_file_obj_to_delete.name}' with key index {key_idx_at_upload_time}: {type(e_cleanup_del).__name__} - {e_cleanup_del}"
                        )
                else:
                    print(
                        f"    - Skipping an invalid/unnamed tracked file object originally associated with key index {key_idx_at_upload_time}."
                    )
        elif not self.api_keys:
            print("  Skipping API file cleanup: No API keys configured.")
        else:
            print("  No API files were tracked for deletion in this 'fit' call.")

        # Restore the API key index that was active before this cleanup section
        self.current_api_key_index = original_active_key_index_at_cleanup_start
        try:
            print(
                f"  Restoring client to originally active API key index {self.current_api_key_index} for consistency..."
            )
            self.client = None  # Force re-init
            self._initialize_client()  # Re-initialize with the original active key; prints "OK." or raises
        except Exception as e_restore_client:
            print(
                f"  Warning: Failed to restore client to original API key index {self.current_api_key_index}: {e_restore_client}. Client might be None if subsequent calls are made on this instance."
            )
            self.client = None  # Ensure client is None if restoration failed

        # Original local plot file cleanup (closing matplotlib figures)
        if _main_plot_fig_to_close is not None and plt.fignum_exists(
            _main_plot_fig_to_close.number
        ):
            plt.close(_main_plot_fig_to_close)
        plt.close(
            "all"
        )  # Close any other lingering example plot figures if not closed by their generator
        print("  Fit cleanup finished.")
        return self

    def decision_function(self, X):
        """Scores data X using the generated function from fit()."""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Invalid X to decision_function.")
        if self.generated_code_ is None:
            raise ValueError(
                "No generated code available. Fit may have failed or not run."
            )
        exec_globals_dec = {"np": np, "pd": pd}
        if "scipy" not in exec_globals_dec:
            import scipy.stats

            exec_globals_dec["scipy"] = scipy
        if "sklearn" not in exec_globals_dec:
            import sklearn.ensemble
            import sklearn.neighbors
            import sklearn.preprocessing

            exec_globals_dec["sklearn"] = sklearn
        exec_locals_dec = {}
        exec(self.generated_code_, exec_globals_dec, exec_locals_dec)
        if "calculate_anomaly_scores" not in exec_locals_dec or not callable(
            exec_locals_dec["calculate_anomaly_scores"]
        ):
            raise NameError(
                "'calculate_anomaly_scores(current_X_data)' not in stored code."
            )
        calc_func_dec = exec_locals_dec["calculate_anomaly_scores"]
        scores_raw_dec = calc_func_dec(X)  # Pass argument X here

        if not isinstance(scores_raw_dec, (list, np.ndarray)):
            raise TypeError(
                f"Generated function (for test data) returned type: {type(scores_raw_dec)}."
            )
        scores_arr_dec = np.array(scores_raw_dec, dtype=float)
        if not (
            scores_arr_dec.ndim == 1 and scores_arr_dec.shape[0] == X.shape[0]
        ):  # Compare with X
            raise ValueError(
                f"Scores shape ({scores_arr_dec.shape}) mismatch X ({X.shape[0]},)."
            )
        if np.any(~np.isfinite(scores_arr_dec)):
            max_finite_dec = (
                np.nanmax(scores_arr_dec[np.isfinite(scores_arr_dec)])
                if np.any(np.isfinite(scores_arr_dec))
                else 1.0
            )
            scores_arr_dec = np.nan_to_num(
                scores_arr_dec, nan=0.0, posinf=max_finite_dec, neginf=0.0
            )
        return scores_arr_dec


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E_May_7 Anomaly Detector.")
    parser.add_argument(
        "--supervision_mode",
        type=str,
        choices=["supervised", "unsupervised"],
        default="supervised",
        help="Mode of operation: supervised or unsupervised (default: supervised)",
    )
    parser.add_argument(
        "--file_list_path",
        type=str,
        default=os.path.join("Datasets", "File_List", "TSB-AD-M-Eva-Debug.csv"),
        help="Path to the CSV file containing the list of datasets to process.",
    )
    args = parser.parse_args()

    overall_start_time = time.time()
    # FILE_LIST_PATH will be set by args.file_list_path
    # SUPERVISION_MODE will be set by args.supervision_mode

    RESULTS_CSV_PATH = os.path.join("E_May_7_detailed_results.csv")
    PLOT_DIR_BASE = os.path.join("E_May_7_ScoreLabel_Plots")
    EXAMPLE_PLOT_DIR = os.path.join("Example_Plots")
    GENERATED_CODE_DIR = os.path.join("E_May_7_Generated_Code")
    INPUT_PLOT_DIR = os.path.join("E_May_7_Input_Plots")
    DATA_DIR = os.path.join(
        "Datasets", "TSB-AD-M"
    )  # Added DATA_DIR for consistency if used explicitly elsewhere

    VISUALIZE_ANOMALIES = True
    USE_SAME_DOMAIN_EXAMPLES_ONLY = False
    PROVIDE_EXAMPLES_TO_API = False
    # SUPERVISION_MODE = "supervised"  # Options: "supervised", "unsupervised" # Now from args
    SUPERVISION_MODE = args.supervision_mode
    FILE_LIST_PATH = args.file_list_path

    print(f"--- Running E_May_7 Detector ---")
    print(f"Supervision Mode: {SUPERVISION_MODE}")
    print(f"File List Path: {FILE_LIST_PATH}")

    for d_path_main in [
        PLOT_DIR_BASE,
        EXAMPLE_PLOT_DIR,
        GENERATED_CODE_DIR,
        INPUT_PLOT_DIR,
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
    if not filenames_to_process_list_main:
        exit("No filenames to process. Exiting.")
    files_to_run_now_main = [
        f_run
        for f_run in filenames_to_process_list_main
        if f_run not in processed_files
    ]
    print(f"\n--- Processing Plan ---")
    print(f"Total unique files in list: {len(filenames_to_process_list_main)}")
    print(
        f"Files already processed: {len(processed_files.intersection(filenames_to_process_list_main))}"
    )
    print(f"Files remaining this run: {len(files_to_run_now_main)}")
    if not api_keys:
        print("\nCRITICAL WARNING: No API keys found!\n")
    processed_count_this_run_final = 0

    # --- Load and Process Benchmark Data ---
    benchmark_metrics = {}
    BENCHMARK_CSV_PATH = os.path.join(
        "benchmark_exp", "benchmark_eval_results", "multi_mergedTable_VUS-PR.csv"
    )
    if os.path.exists(BENCHMARK_CSV_PATH):
        print(f"Loading benchmark data from: {BENCHMARK_CSV_PATH}")
        try:
            benchmark_df = pd.read_csv(BENCHMARK_CSV_PATH)
            if "file" not in benchmark_df.columns:
                print(
                    f"Warning: 'file' column not found in {BENCHMARK_CSV_PATH}. Cannot perform benchmark comparison."
                )
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
                    print(
                        f"  Benchmark algorithms identified for VUS-PR comparison: {algo_columns}"
                    )

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
                    print(
                        f"  Processed {len(benchmark_metrics)} files from benchmark data."
                    )
                else:
                    print(
                        f"Warning: Could not reliably identify algorithm columns in {BENCHMARK_CSV_PATH}."
                    )
        except Exception as e:
            print(
                f"Error loading or processing benchmark CSV {BENCHMARK_CSV_PATH}: {e}"
            )
    else:
        print(
            f"Warning: Benchmark data file not found at {BENCHMARK_CSV_PATH}. No comparison will be performed."
        )
    # --- End Load and Process Benchmark Data ---

    # --- Backfill Missing Benchmark Columns in Existing Results (all_results) ---
    if (
        all_results and benchmark_metrics
    ):  # Only backfill if there are existing results and benchmark data
        print("\n--- Backfilling missing benchmark data in existing results... ---")
        model_prefix_backfill = MODEL_NAMES[0]  # Should be "E_May_7"
        e_may_7_vus_pr_col_bf = f"{model_prefix_backfill}_VUS-PR"
        bench_avg_col_bf = f"{model_prefix_backfill}_Benchmark_Avg_VUS-PR"
        bench_max_col_bf = f"{model_prefix_backfill}_Benchmark_Max_VUS-PR"
        diff_avg_col_bf = f"{model_prefix_backfill}_E_May_7_vs_Avg_VUS-PR_Diff"
        diff_max_col_bf = f"{model_prefix_backfill}_E_May_7_vs_Max_VUS-PR_Diff"
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
                e_may_7_score_bf = pd.to_numeric(
                    row_dict_bf.get(e_may_7_vus_pr_col_bf), errors="coerce"
                )

                if not pd.isna(e_may_7_score_bf):
                    bench_data_for_file_bf = benchmark_metrics.get(filename_bf)
                    if bench_data_for_file_bf:
                        avg_bench_score_bf = bench_data_for_file_bf.get("avg", np.nan)
                        max_bench_score_bf = bench_data_for_file_bf.get("max", np.nan)

                        row_dict_bf[bench_avg_col_bf] = avg_bench_score_bf
                        row_dict_bf[bench_max_col_bf] = max_bench_score_bf
                        row_dict_bf[diff_avg_col_bf] = (
                            e_may_7_score_bf - avg_bench_score_bf
                            if not pd.isna(avg_bench_score_bf)
                            else np.nan
                        )
                        row_dict_bf[diff_max_col_bf] = (
                            e_may_7_score_bf - max_bench_score_bf
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
                    # E_May_7 score itself is missing, so can't compare
                    for col_bf_fill in benchmark_cols_to_check:
                        row_dict_bf[col_bf_fill] = "N/A_SelfNoScore_Filled"
                    updated_count_backfill += 1  # Count as updated
        if updated_count_backfill > 0:
            print(f"  Backfilled benchmark data for {updated_count_backfill} records.")
        else:
            print("  No records required backfilling of benchmark data.")
    elif not benchmark_metrics:
        print("\n--- Skipping backfill: Benchmark metrics dictionary is empty. ---")
    # --- End Backfill ---

    for i_main_loop_final, filename_main_final in enumerate(files_to_run_now_main, 1):
        loop_start_time_final = time.time()
        print(
            f"\n[{i_main_loop_final}/{len(files_to_run_now_main)}] Processing: {filename_main_final}"
        )
        file_path_final = os.path.join(DATA_DIR, filename_main_final)
        current_results_final = {"filename": filename_main_final}
        model_prefix_final = MODEL_NAMES[0]

        df_final_run = pd.read_csv(file_path_final)
        if df_final_run.empty:
            raise ValueError(f"CSV empty: {filename_main_final}")
        label_col_name_final = "Label"
        if label_col_name_final not in df_final_run.columns:
            found_label_col_final = next(
                (
                    c_final
                    for c_final in df_final_run.columns
                    if c_final.lower() == "label"
                ),
                None,
            )
            if not found_label_col_final:
                raise ValueError(
                    f"Missing '{label_col_name_final}' in {filename_main_final}"
                )
            df_final_run.rename(
                columns={found_label_col_final: label_col_name_final}, inplace=True
            )
        feature_cols_final = (
            [
                c_final
                for c_final in df_final_run.columns
                if c_final != label_col_name_final
            ]
            if df_final_run.columns[-1] != label_col_name_final
            else df_final_run.columns[:-1].tolist()
        )
        if not feature_cols_final:
            raise ValueError(f"No features in {filename_main_final}.")
        df_final_run.dropna(
            subset=feature_cols_final + [label_col_name_final], inplace=True
        )
        if df_final_run.empty:
            raise ValueError(f"Data empty after NaN drop: {filename_main_final}.")
        data_full_for_run = df_final_run[feature_cols_final].values.astype(float)
        label_full_for_run = df_final_run[label_col_name_final].astype(int).to_numpy()
        if (
            data_full_for_run.shape[0] != label_full_for_run.shape[0]
            or data_full_for_run.shape[0] == 0
        ):
            raise ValueError(
                f"Data/label mismatch or zero rows in {filename_main_final}."
            )

        if PROVIDE_EXAMPLES_TO_API:
            domain_final_run = extract_domain_from_filename(filename_main_final)
            base_name_final_run = os.path.splitext(filename_main_final)[0]
            generate_and_save_example_plot(
                data_full_for_run,
                label_full_for_run,
                domain_final_run,
                base_name_final_run,
                EXAMPLE_PLOT_DIR,
            )

        window_final_run = 10
        if window_finder:
            estimated_window_final_run = window_finder(data_full_for_run, rank=1)
            if (
                isinstance(estimated_window_final_run, (int, float))
                and 5 < estimated_window_final_run < data_full_for_run.shape[0] * 0.5
            ):
                window_final_run = int(estimated_window_final_run)

        train_index_val_final_run = extract_train_index_from_filename(
            filename_main_final
        )
        if not (0 < train_index_val_final_run < data_full_for_run.shape[0]):
            raise ValueError(
                f"Train Index {train_index_val_final_run} out of bounds for {data_full_for_run.shape[0]} in {filename_main_final}"
            )
        data_train_for_model_fit = data_full_for_run[:train_index_val_final_run]
        data_to_score_with_model = data_full_for_run
        labels_for_eval = label_full_for_run

        print("  Running E_May_7 model (Semi-Supervised Mode)...")
        model_start_time_final_run = time.time()
        run_hp_config_final = {
            "use_same_domain_examples_only": USE_SAME_DOMAIN_EXAMPLES_ONLY,
            "provide_examples_to_api": PROVIDE_EXAMPLES_TO_API,
            # Add context for E_May_7.__init__
            "dataset_filename": filename_main_final,
            "data_full_for_plot": data_full_for_run,
            "label_full_for_plot": label_full_for_run,
            "train_index_for_plot": train_index_val_final_run,
            "SUPERVISION_MODE": SUPERVISION_MODE,  # Pass the global mode from args
        }

        output_scaled_final_run, model_ranges_final_run = run_E_May_7_Semisupervised(
            data_train_for_model_fit,  # data_train
            data_full_for_run,  # data_test (this is the full data for scoring by decision_function)
            run_hp_config_final,  # HP containing context and other model params
        )

        duration_final_run = time.time() - model_start_time_final_run
        current_results_final[f"{model_prefix_final}_runtime"] = duration_final_run
        print(f"  Model execution finished in {duration_final_run:.2f}s.")

        if (
            output_scaled_final_run is None
            or not isinstance(output_scaled_final_run, np.ndarray)
            or output_scaled_final_run.size == 0
            or output_scaled_final_run.shape[0] != data_full_for_run.shape[0]
        ):  # Compare with data_full_for_run
            current_results_final[f"{model_prefix_final}_Error"] = (
                "Invalid scores from run_E_May_7_Semisupervised"
            )
        else:
            output_scaled_final_run = np.nan_to_num(
                output_scaled_final_run, nan=0.0, posinf=1.0, neginf=0.0
            )
            if metrics_calculator:
                eval_result_final_run = metrics_calculator(
                    labels=labels_for_eval,
                    score=output_scaled_final_run,
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
                        original_label=labels_for_eval,
                        score=output_scaled_final_run,
                        model_identified_ranges=model_ranges_final_run,
                        base_plot_filename=base_name_vis_final_run,
                        plot_dir=plot_subdir_final_run,
                        model_prefix=model_prefix_final,
                    )
            if f"{model_prefix_final}_Error" not in current_results_final:
                processed_count_this_run_final += 1

            # Add Benchmark Comparison
            e_may_7_vus_pr_key = f"{model_prefix_final}_VUS-PR"
            if e_may_7_vus_pr_key in current_results_final:
                e_may_7_score = current_results_final[e_may_7_vus_pr_key]
                if isinstance(e_may_7_score, (float, int)) and not pd.isna(
                    e_may_7_score
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
                                f"{model_prefix_final}_E_May_7_vs_Avg_VUS-PR_Diff"
                            ] = (e_may_7_score - avg_bench_score)
                        else:
                            current_results_final[
                                f"{model_prefix_final}_E_May_7_vs_Avg_VUS-PR_Diff"
                            ] = np.nan

                        if not pd.isna(max_bench_score):
                            current_results_final[
                                f"{model_prefix_final}_E_May_7_vs_Max_VUS-PR_Diff"
                            ] = (e_may_7_score - max_bench_score)
                        else:
                            current_results_final[
                                f"{model_prefix_final}_E_May_7_vs_Max_VUS-PR_Diff"
                            ] = np.nan
                    else:
                        print(
                            f"  Note: No benchmark VUS-PR data found for {filename_main_final}."
                        )
                        current_results_final[
                            f"{model_prefix_final}_Benchmark_Avg_VUS-PR"
                        ] = "N/A_Benchmark"
                        current_results_final[
                            f"{model_prefix_final}_Benchmark_Max_VUS-PR"
                        ] = "N/A_Benchmark"
                        current_results_final[
                            f"{model_prefix_final}_E_May_7_vs_Avg_VUS-PR_Diff"
                        ] = "N/A_Benchmark"
                        current_results_final[
                            f"{model_prefix_final}_E_May_7_vs_Max_VUS-PR_Diff"
                        ] = "N/A_Benchmark"
                else:
                    # E_May_7 VUS-PR score is not numeric or is NaN
                    current_results_final[
                        f"{model_prefix_final}_Benchmark_Avg_VUS-PR"
                    ] = "N/A_SelfNoScore"
                    current_results_final[
                        f"{model_prefix_final}_Benchmark_Max_VUS-PR"
                    ] = "N/A_SelfNoScore"
                    current_results_final[
                        f"{model_prefix_final}_E_May_7_vs_Avg_VUS-PR_Diff"
                    ] = "N/A_SelfNoScore"
                    current_results_final[
                        f"{model_prefix_final}_E_May_7_vs_Max_VUS-PR_Diff"
                    ] = "N/A_SelfNoScore"

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
        print(
            f"  --> RESULT: Processed (Iteration time: {time.time() - loop_start_time_final:.2f}s)"
        )

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
                    f"{model_iter_end_avg}_E_May_7_vs_Avg_VUS-PR_Diff",
                    "Diff vs Avg VUS-PR",
                )
                print_average_metric_final_val(
                    ok_df_end_avg,
                    f"{model_iter_end_avg}_E_May_7_vs_Max_VUS-PR_Diff",
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
