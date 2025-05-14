import ast  # Added for literal_eval
import json
import logging
import os
import pprint
import random
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google import genai
from google.genai import (
    types,
)  # Assuming this is still needed for error types or specific configurations
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from TSB_AD.models.base import BaseDetector

from .mad_utils import (
    PLOT_FIG_WIDTH,
    PLOT_NBINS,
    _find_clusters,
    _merge_contiguous_ranges,
    _resolve_overlapping_ranges,
    strip_markdown_code_fences,
)


# --- Custom Pretty Formatter ---
class PrettyFormatter(logging.Formatter):
    def __init__(
        self,
        fmt=None,
        datefmt=None,
        style="%",
        validate=True,
        *,
        defaults=None,
        use_json=False,
        json_indent=2,
        pprint_indent=2,
        pprint_width=100,
    ):
        super().__init__(
            fmt, datefmt, style, validate, defaults=defaults
        )  # Python 3.8+ for defaults
        self.use_json = use_json
        self.json_indent = json_indent
        self.pprint_indent = pprint_indent
        self.pprint_width = pprint_width

    def formatMessage(self, record):
        if not record.args and isinstance(record.msg, (dict, list, tuple)):
            if self.use_json:
                try:
                    record.message = json.dumps(
                        record.msg, indent=self.json_indent, sort_keys=False
                    )
                except TypeError:
                    record.message = pprint.pformat(
                        record.msg,
                        indent=self.pprint_indent,
                        width=self.pprint_width,
                        sort_dicts=False,
                    )
            else:
                record.message = pprint.pformat(
                    record.msg,
                    indent=self.pprint_indent,
                    width=self.pprint_width,
                    sort_dicts=False,
                )
        else:
            record.message = super().formatMessage(record)
        return record.message

    def format(self, record):
        # The base format() method calls formatMessage() internally.
        # Our override of formatMessage ensures record.message is correctly populated.
        return super().format(record)


# --- End Custom Pretty Formatter ---
HAS_GOOGLE_LIBS_FOR_FORMATTER = False
google_api_exceptions = None  # Define for isinstance check to not fail
MessageToDict = None


def _format_exception_for_logging(e):
    try:
        # Attempt 0: Handle GoogleAPIError.details directly
        if (
            HAS_GOOGLE_LIBS_FOR_FORMATTER
            and google_api_exceptions
            and isinstance(e, google_api_exceptions.GoogleAPIError)
        ):
            if hasattr(e, "details") and e.details and isinstance(e.details, list):
                try:
                    details_list_of_dicts = [
                        MessageToDict(pb, preserving_proto_field_name=True)
                        for pb in e.details
                    ]
                    if details_list_of_dicts:
                        error_summary_text = (
                            str(e.message).split("{'error':")[0].strip()
                            if isinstance(e.message, str) and "{'error':" in e.message
                            else str(e.message)
                        )
                        formatted_msg = (
                            f"{type(e).__name__} (Summary: {error_summary_text}):\n"
                            f"Structured Details from e.details:\n{pprint.pformat(details_list_of_dicts, indent=2, width=120, sort_dicts=False)}"
                        )
                        return formatted_msg
                except Exception as e_pb_format:
                    pass

        # Attempt 1: If e.args[0] is already a dictionary
        if hasattr(e, "args") and e.args and isinstance(e.args[0], dict):
            error_dict = e.args[0]
            if isinstance(error_dict.get("error"), dict):
                error_to_print = error_dict["error"]
                context = "from e.args[0]['error']"
            else:
                error_to_print = error_dict
                context = "from e.args[0]"
            return (
                f"{type(e).__name__} ({context} as dict):\n"
                f"{pprint.pformat(error_to_print, indent=2, width=120, sort_dicts=False)}"
            )

        # Attempt 2: (Placeholder for e.response if needed, less likely for genai.ClientError)

        # Attempt 3: Parse from the error message string

        msg_with_dict = str(e)

        dict_literal_start_key = "{'error':"

        if dict_literal_start_key not in msg_with_dict:
            pass
        else:
            dict_start_index_in_full_str = msg_with_dict.find(dict_literal_start_key)
            prefix_text_from_full_str = msg_with_dict[
                :dict_start_index_in_full_str
            ].strip()
            potential_dict_str = msg_with_dict[dict_start_index_in_full_str:]

            try:
                error_structure = ast.literal_eval(potential_dict_str)
                final_prefix = f"{type(e).__name__}"
                if prefix_text_from_full_str:
                    final_prefix += f" ({prefix_text_from_full_str})"

                formatted_msg = (
                    f"{final_prefix} (details parsed from string):\n"
                    f"{pprint.pformat(error_structure, indent=2, width=120, sort_dicts=False)}"
                )
                return formatted_msg

            except (SyntaxError, ValueError):
                try:
                    json_compatible_str = (
                        potential_dict_str.replace("'", '"')
                        .replace("None", "null")
                        .replace("True", "true")
                        .replace("False", "false")
                    )
                    error_structure_json = json.loads(json_compatible_str)

                    final_prefix_json = f"{type(e).__name__}"
                    if prefix_text_from_full_str:
                        final_prefix_json += f" ({prefix_text_from_full_str})"
                    formatted_msg_json = (
                        f"{final_prefix_json} (details parsed as JSON after ast.literal_eval failed):\n"
                        f"{json.dumps(error_structure_json, indent=2)}"
                    )
                    return formatted_msg_json
                except json.JSONDecodeError:
                    pass
            except Exception:
                pass

    except Exception:
        pass

    default_formatted_msg = f"{type(e).__name__}: {str(e)}"
    return default_formatted_msg


class E_May_14(BaseDetector):
    """
    Anomaly detector using Google Genai (Gemini). Employs a two-step process *per feature*:
    1. Visual Range Identification: Identifies potentially anomalous ranges from a single-feature plot.
    2. Anomaly Index Identification: Identifies specific anomalous indices for that feature based on raw data snippets from identified ranges and training data (if provided).
    Operates unsupervised based on the data provided to fit(), aggregating results across features.
    """

    def __init__(self, HP):
        super().__init__()
        # --- API Keys (Consider moving to a more secure configuration method) ---
        api_keys = [
            "AIzaSyDey14war9PCrghT4FahLo9G-TgaT1tYb0",
            "AIzaSyDYikrn6sa7HVkS41h6YGban-59wqQ6XhA",
            "AIzaSyBDpvgkfSNq_wkuOMsZdIr0NifKYfMt5EA",
            "AIzaSyCwCe31rmf36gX8cPbMKPP3U7kDcQgOYzw",
            "AIzaSyCcD8_eM8YZX-rvvtDyAiwSodpNH5hl3ng",
            "AIzaSyB8VJQKsPMuDpABYa3acFJ8lmyrL0AZVko",
            "AIzaSyCOKm8w4sZoc54XbqVzIuyNroID_tG4lYo",
            "AIzaSyBZr3H5aIzx4K5VYfbBWAR9CYzPwe4d8f0",
            "AIzaSyCXc4xUJZJykyhAWkSqw1tMFGexv08TzF0",
            "AIzaSyAQq-FsC-wEu5o5yhNF9ueoqj_3IdMLsLE",
            "AIzaSyAleZfvAcjT3nPgcFBxs_l277Wkh6VmD_8",
            "AIzaSyCOood9IgOPJ79XxfEMcaiiFYQlffqHFrQ",
            "AIzaSyApMXOqUTsIMuZ-GUNQFHx0cU_GPRc9FTk",
            "AIzaSyBDZOhbCsLPjV8_4xRUdx4R265iaSis0V8",
            "AIzaSyBYQUTgF93brBw9pviV_Znz0fPm6Ba9p5Y",
            "AIzaSyDlCQNIRX6GRLg9ayJQpI6yc29ilCdQ3aA",
            "AIzaSyB8JhqItkaDI-Ma8IUNg_GUTkah-6jWvsg",
            "AIzaSyCY2hL9UD_mzuf8ReJL9cIkRb5tLxjjbyU",
            "AIzaSyAKvTmoaLg-Z-UOTxl572LWRHG4UjfEtbA",
            "AIzaSyAJMi8Pzghku32tpAlZxBz00UpfqPQzg9k",
            "AIzaSyDk3r8ps4r9z-FS7Btcqydf3asMaCTqpHg",
            "AIzaSyDGC31qbp3n78eTA2AKpKsG1dNO-Y-xRVQ",
            "AIzaSyDFV_F8P6QKOML46UshWcRHIoTx_abAMQQ",
            "AIzaSyDtysI4liSD5cbuhGjuo6rz48_AYVDAJG4",
            "AIzaSyAa5vil_PobtMvv6cR-7Q_VRbsMK-PSCaA",
            "AIzaSyDj2ZgVXy4eJcwGPaOsZivIzUkvTUDRghU",
            "AIzaSyAjJNpgFT7JLi7kwcokUhgvqdH1MgaFC_o",
            "AIzaSyA6wPrMrFlHRyf_gDHMq04Jiqoe6TfxBX0",
            "AIzaSyAzF9p_YWTTKOk8ZIF6MFN4XwjzWbyuTIQ",
            "AIzaSyBxGgmltsMKdRuGcs9Nw3RKCvPbjbvDYdA",
            "AIzaSyBuVP-ehzCXo8mZ-9Lynpnkx56ax65LuQM",
            "AIzaSyAsh77VJcGt9uNiAKP7LgC2hrAKEyqcPcA",
            "AIzaSyAtoxgUgE5yqxVI8F7zNXTG4MMvZw8BDaY",
            "AIzaSyDlmD8-uxxHBMelibW7B4I04dTgScG5-6Q",
            "AIzaSyDMsi_k3tnu1mK_BCFXV73WL2VBFvU7Fwc",
            "AIzaSyCClmz965ZUc7jwy4hdfB-EJ320dycN2CI",
            "AIzaSyBh2JNC0fAleUs3-Sm_wvSxTrvcSV0iXvs",
            "AIzaSyDcIZl6SwpOKGYXnwog0Yv06WABTiF9-ZY",
            "AIzaSyD7ARPSvk6Gc1Ko_v5aAGLWBof0wqnx0do",
            "AIzaSyDAD1cNb8gvkwE8lHhz8EEMng6AqN7ZhuM",
            "AIzaSyD1YxzfIn1Fkofwr9jtXTDdn2sBMUvp_no",
            "AIzaSyCSZvFLPCvHPxbMAALACrmFN92EdmjmwlQ",
            "AIzaSyC-nbmmUl_Ct5ih1I5r3cBEZCFYzHIAsY0",
            "AIzaSyAZj03MscGXZRvzJNDi3j4uN3W14fprSR4",
            "AIzaSyBJdzSdUk_KQhCeB33MJkdojOWXnXSyY4s",
            "AIzaSyDRH9PxiEOsPaNcoc_Nq3KltPIDZq206qI",
            "AIzaSyBcdCp96dJSdYG1xbAv5gv9jhWLORuD7YA",
            "AIzaSyBGlpsuqOq3Icz3Fle4I_eTtr4CnEYsNtA",
            "AIzaSyCN3I9gRIdeBrKVdKG1DuCzbdPwFnY89-s",
            "AIzaSyDE5JOCHfL26dEDohjWhjC_wDaZBL37HhI",
            "AIzaSyBHs3SIHp9CeO-TK1lDOnwxA9G1cMifzXE",
            "AIzaSyAt7W6J_VdW-ZEn-54K82Sb0O3CQKcOTrY",
            "AIzaSyA2iOC4uvbsZ0Suznhx1fEu5mVGcevEH3s",
            "AIzaSyBoNTXAz36CsDCONYV18-moWWWoI55fyBA",
            "AIzaSyCRoquRH22JZoJnvCLl6a5JkwhOuXZ_cSg",
            "AIzaSyCnbG6yAyD5CFM0Q22nHAK9X_oV-9sRowU",
            "AIzaSyBIV621iuIsUtG4YJqL0H_G0HnZ89TPNHQ",
            "AIzaSyDuMPgzzVSY9QSXSylmvwMyuzUict5zcnU",
            "AIzaSyCFBHrRnpSAq4vqBDAsHoSKi6J8PZyObdg",
            "AIzaSyALYd_36uji91Sikmhnnio-HzU2wiwc4t0",
            "AIzaSyAPuiWzSvXK7yM_xfWaZ7i2QoNUe-BD4-E",
            "AIzaSyCTS_2um5WlZSJDhglW2sfPTUO7TJpn5tw",
            "AIzaSyCDkULrBWyNe6seIAqctCMsXKF0mD3lpoE",
            "AIzaSyCzSLgGM3cV8fPgvSj3QxSPqIY6tVeGV7k",
            "AIzaSyBTlVTMIZsMjh8MCb8ShXedCIsVNvYVjDE",
            "AIzaSyAuPrqvQcyIu1W9ZjWAftzPrQAiTMYgu30",
            "AIzaSyDRN9Pxerp5s6PrCuyNVDQ4458OqJr0cQc",
            "AIzaSyD8A6XeXwmnH0EXYnqBRt5La4pwQ7VMtvs",
            "AIzaSyBNcuN7UnbN_xEhlPeAPkA2LBwvmBsvqQA",
            "AIzaSyBwPKdVM0e3brpdSWy8Q2j4VVD4LEGk98M",
            "AIzaSyD3vYSVJ0-4SU3zD6TBk4c2kHGBDHndK2k",
            "AIzaSyClzIpuWc4PKEfBh6zb8p4VQfC7bl7ehP0",
            "AIzaSyDKxt31WqFV5xJu2I5s83OtkX2oauQcvbI",
            "AIzaSyB99p9d6qjPbBbllsSsNZLZd0Pr81RvBZA",
            "AIzaSyAwGySNxY-r508UV5PMJSCI5PBR7jP-pjo",
        ]
        self.api_keys = api_keys
        self.client = None
        self.current_api_key_index = 0
        self.uploaded_files_tracking = {}
        self.last_run_timestamp = None

        self.max_retries_per_key = 2
        self.max_overall_retries = len(self.api_keys) * self.max_retries_per_key

        self.HP = HP if HP is not None else {}
        self.use_training_labels_for_step1_plot = self.HP.get(
            "use_training_labels_for_step1_plot", True
        )

        self.plot_save_dir = "MAD/E_May_14_Feature_Plots"
        os.makedirs(self.plot_save_dir, exist_ok=True)

        self.decision_scores_ = None
        self.per_feature_artifacts_ = {}

        # Setup logging
        # Configure root logger for a default higher level (e.g., WARNING for libraries)
        logging.basicConfig(
            level=logging.ERROR,  # Default level for other loggers
            format="%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
        )
        # Get a logger specific to this module/class
        self.logger = logging.getLogger(__name__)
        # Set this specific logger to DEBUG level
        self.logger.setLevel(logging.DEBUG)

        # Ensure this logger uses the specified format and doesn't just rely on root logger config
        if (
            not self.logger.handlers
        ):  # Add handler only if none exist, to avoid duplicates in interactive sessions
            handler = logging.StreamHandler()
            # Use the new PrettyFormatter
            formatter = PrettyFormatter(
                "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                use_json=False,
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = (
            False  # Prevent messages from being passed to the root logger's handlers
        )

        self._initialize_client()

        self.y_train_labels_for_hint = None
        self.train_data_len_for_hint = None

    def _initialize_client(self):
        """Initializes the GenAI client using the current API key."""
        if not genai:
            raise RuntimeError("Google GenAI library (genai) not available.")
        if not self.api_keys:
            raise RuntimeError("No API keys loaded into E_May_14 model instance.")
        if not (0 <= self.current_api_key_index < len(self.api_keys)):
            # This case should ideally be handled by cycling logic before it gets here
            self.logger.error(
                "API key index out of bounds during client initialization attempt."
            )
            # Attempt to reset to the first key as a fallback, though this indicates a logic flaw elsewhere
            self.current_api_key_index = 0
            if not (
                0 <= self.current_api_key_index < len(self.api_keys)
            ):  # Still bad if no keys
                raise RuntimeError(
                    "API key index out of bounds even after reset, no API keys available."
                )

        key = self.api_keys[self.current_api_key_index]
        try:
            # Ensure any previous client's resources are cleaned up if possible
            if self.client and hasattr(self.client, "close"):  # Fictional close method
                try:
                    self.client.close()
                except Exception as e_close:
                    self.logger.warning(
                        f"Error closing previous client: {_format_exception_for_logging(e_close)}"
                    )

            self.client = genai.Client(api_key=key)
            self.logger.info(
                f"Successfully initialized GenAI Client with key index {self.current_api_key_index}."
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to initialize GenAI Client with key index {self.current_api_key_index}: {_format_exception_for_logging(e)}"
            )
            self.client = None  # Ensure client is None if initialization fails
            # Do not re-raise here, let the calling retry logic handle it.
            # The _execute_api_call will attempt re-initialization or fail if client remains None.

    def _execute_api_call(self, api_call_func, *args, **kwargs):
        # This function assumes self.client might be None and tries to initialize if so.
        # It's a single attempt per call; retries and key cycling are managed by the caller loops.
        if not self.client:
            try:
                self._initialize_client()  # Attempt to initialize/re-initialize
                if not self.client:  # If still None after attempt
                    raise RuntimeError(
                        "Client re-initialization failed within _execute_api_call."
                    )
            except Exception as e_init:  # Catch init errors
                # Log and re-raise as a specific type of error or just pass it up
                self.logger.error(
                    f"Client initialization failed: {_format_exception_for_logging(e_init)}"
                )
                raise  # Re-raise the initialization error to be caught by retry logic

        # At this point, if self.client exists, proceed with the API call
        try:
            result = api_call_func(self.client, *args, **kwargs)
            return result
        except Exception as e_other:  # Catch other unexpected errors during the call
            self.logger.error(
                f"An unexpected error occurred during API call: {_format_exception_for_logging(e_other)}"
            )
            self.client = None  # Force re-init
            raise  # Re-raise

    def _run_llm_pipeline(
        self, X_data, y_labels_for_plot_hint=None, train_len_for_plot_hint=None
    ):
        pipeline_start_time = time.time()

        model_name = "gemini-2.5-pro-exp-03-25"
        if (
            not isinstance(X_data, np.ndarray)
            or X_data.ndim != 2
            or X_data.shape[0] == 0
        ):
            raise ValueError(
                "Invalid input data X_data to _run_llm_pipeline. Must be 2D NumPy array."
            )
        if not self.api_keys:
            self.logger.error("No API keys configured for E_May_14 model.")
            return np.zeros(X_data.shape[0], dtype=float)

        n_samples, n_features = X_data.shape
        final_scores_all_features = np.zeros(n_samples, dtype=float)
        self.per_feature_artifacts_ = {}
        files_uploaded_this_pipeline = (
            []
        )  # To track files for cleanup across the entire pipeline run

        timestamp_str = time.strftime("%Y%m%d%H%M%S%f")
        self.last_run_timestamp = timestamp_str

        temp_plot_base_dir = os.path.join(self.plot_save_dir, f"temp_{timestamp_str}")
        os.makedirs(temp_plot_base_dir, exist_ok=True)

        is_train_run_context = (
            y_labels_for_plot_hint is not None
            and train_len_for_plot_hint is not None
            and train_len_for_plot_hint == X_data.shape[0]
        )
        run_type_prefix = "train_fit" if is_train_run_context else "decision_run"

        for i_feat in tqdm(range(n_features), desc="Processing Features"):
            feature_loop_start_time = time.time()
            step1_succeeded = False
            step2_succeeded = False
            feature_identified_ranges = []
            feature_anomaly_scores = np.zeros(
                n_samples, dtype=float
            )  # For scores from S2 generated code
            main_plot_part_step1 = None
            last_exception_s1 = None
            last_exception_s2 = None  # For S2

            feature_file_prefix = f"{run_type_prefix}_feature_{i_feat}"
            # Define artifact paths
            plot_filename_s1 = f"{feature_file_prefix}_InputPlotS1.png"
            s1_response_filename = f"{feature_file_prefix}_S1_response.json"
            s2_raw_response_filename = f"{feature_file_prefix}_S2_raw_response.txt"
            s2_generated_code_filename = f"{feature_file_prefix}_S2_generated_code.py"

            current_plot_path = os.path.join(temp_plot_base_dir, plot_filename_s1)
            s1_response_path = os.path.join(temp_plot_base_dir, s1_response_filename)
            s2_raw_response_path = os.path.join(
                temp_plot_base_dir, s2_raw_response_filename
            )
            s2_generated_code_path = os.path.join(
                temp_plot_base_dir, s2_generated_code_filename
            )

            feature_plot_path_s1 = current_plot_path  # Path for S1 plot artifact
            plot_ax_actual_ticks = []  # To store actual X-axis ticks

            _feature_plot_fig = None
            plot_s1_gen_start_time = time.time()

            # Logging for plot generation parameters
            self.logger.debug(f"--- Plotting Feature {i_feat} ---")

            try:
                plt.close("all")
                fig_width_plot, fig_height_plot, dpi_val_plot = PLOT_FIG_WIDTH, 5, 150
                base_tick_fs, plot_title_fs, axis_label_fs = 10, 14, "medium"

                _feature_plot_fig, ax_p = plt.subplots(
                    figsize=(fig_width_plot, fig_height_plot)
                )

                time_range_plot_full = np.arange(n_samples)
                # Adjusted plot title and labeling logic
                plot_title = f"Feature {i_feat}"

                ax_p.set_title(
                    f"Feature {i_feat}", fontsize=plot_title_fs, loc="left", pad=2
                )
                ax_p.set_ylabel("Value", fontsize=axis_label_fs)
                ax_p.xaxis.set_major_locator(
                    MaxNLocator(nbins=PLOT_NBINS, integer=True, prune="both")
                )
                ax_p.grid(True, alpha=0.3, linestyle=":")
                ax_p.tick_params(axis="both", labelsize=base_tick_fs)
                ax_p.tick_params(
                    axis="x",
                    which="major",
                    direction="out",
                    length=5,
                    width=0.8,
                    labelbottom=True,
                )
                ax_p.set_xlabel("Time Step", fontsize=axis_label_fs)

                show_train_test_split = (
                    train_len_for_plot_hint is not None
                    and 0 < train_len_for_plot_hint < n_samples
                )

                if is_train_run_context:  # Data passed is training data
                    ax_p.plot(
                        time_range_plot_full,
                        X_data[:, i_feat],
                        color="darkgreen",
                        linestyle="-",
                        lw=1.0,
                        label="Training Data",
                    )
                    plot_title += f" Training Data [0-{n_samples-1}]"
                elif (
                    show_train_test_split
                ):  # Data passed has a train/test split indicated
                    ax_p.plot(
                        time_range_plot_full[:train_len_for_plot_hint],
                        X_data[:train_len_for_plot_hint, i_feat],
                        color="darkgreen",
                        linestyle="-",
                        lw=1.1,
                        label="Training Data",
                    )
                    ax_p.plot(
                        time_range_plot_full[train_len_for_plot_hint - 1 :],
                        X_data[train_len_for_plot_hint - 1 :, i_feat],
                        color="royalblue",
                        linestyle="-",
                        lw=1.0,
                        label="Test/Unseen Data",
                    )
                    plot_title += f" Input [0-{n_samples-1}] (Train/Test Split)"
                else:  # Data passed is test data (or general data without specified train part)
                    ax_p.plot(
                        time_range_plot_full,
                        X_data[:, i_feat],
                        color="royalblue",
                        linestyle="-",
                        lw=1.0,
                        label="Data",
                    )
                    plot_title += f" Input Data [0-{n_samples-1}]"

                example_anomalies_plotted = False
                if (
                    self.use_training_labels_for_step1_plot
                    and y_labels_for_plot_hint is not None
                    and train_len_for_plot_hint is not None
                    and train_len_for_plot_hint > 0
                ):
                    train_anomalies = np.where(
                        y_labels_for_plot_hint[:train_len_for_plot_hint] == 1
                    )[0]
                    anomaly_ranges_train = _find_clusters(train_anomalies)
                    for r_start, r_end in anomaly_ranges_train:
                        ax_p.axvspan(
                            r_start,
                            r_end,
                            color="red",
                            alpha=0.2,
                            label=(
                                "Train Anomaly"
                                if not example_anomalies_plotted
                                else None
                            ),
                        )
                        example_anomalies_plotted = True
                    if example_anomalies_plotted:
                        plot_title += " w/ Train Anomalies"

                if (
                    show_train_test_split
                    or example_anomalies_plotted
                    or is_train_run_context
                ):
                    ax_p.legend(fontsize="small")
                _feature_plot_fig.suptitle(plot_title, fontsize="large")
                ax_p.set_xlim(0, max(1, n_samples - 1))
                # Get actual ticks before saving plot
                plot_ax_actual_ticks = list(ax_p.get_xticks())
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                _feature_plot_fig.savefig(
                    current_plot_path, dpi=dpi_val_plot, bbox_inches="tight"
                )
                self.logger.debug(f"Feature {i_feat} plot saved to {current_plot_path}")
            except Exception as e_plot:
                self.logger.error(
                    f"Error generating plot for feature {i_feat}: {_format_exception_for_logging(e_plot)}"
                )
                # Continue to LLM step without plot if it fails, or handle error as critical
            finally:
                if _feature_plot_fig:
                    plt.close(_feature_plot_fig)

            # --- LLM Step 1: Identify Candidate Ranges ---
            step1_llm_call_start_time = time.time()
            files_uploaded_this_attempt_s1 = []
            # Retry loop for Step 1
            for attempt_s1 in range(len(self.api_keys) * self.max_retries_per_key):
                if step1_succeeded:
                    break
                self.current_api_key_index = (
                    attempt_s1 // self.max_retries_per_key
                )  # Cycle key after max_retries_per_key
                if self.current_api_key_index >= len(self.api_keys):
                    break  # Exhausted all keys

                try:
                    if not os.path.exists(current_plot_path):
                        self.logger.warning(
                            f"Plot for S1 (Feature {i_feat}) not found at {current_plot_path}. Skipping S1 LLM call."
                        )
                        last_exception_s1 = FileNotFoundError(
                            f"Plot not found: {current_plot_path}"
                        )
                        break

                    self.logger.info(
                        f"S1: Feature {i_feat}, Attempt {attempt_s1 + 1}, Key Index {self.current_api_key_index}"
                    )

                    # Token Counting for S1
                    if self.client and model_name:
                        # The actual s1_contents_list is built right before the API call.
                        # We will count tokens after s1_contents_list is fully defined.
                        pass

                    main_plot_part_step1 = self._execute_api_call(
                        lambda client, path: client.files.upload(file=path),
                        current_plot_path,
                    )
                    files_uploaded_this_attempt_s1.append(
                        main_plot_part_step1.name
                    )  # Track for cleanup on error
                    files_uploaded_this_pipeline.append(
                        main_plot_part_step1.name
                    )  # Track for overall cleanup

                    # Prepare ticks for the prompt
                    valid_plot_ticks_int = []
                    if plot_ax_actual_ticks:
                        # Ensure ticks are rounded, integer, unique, sorted, and within sample range
                        pre_filter_ticks = [int(round(t)) for t in plot_ax_actual_ticks]
                        filtered_ticks_in_range = sorted(
                            list(set(t for t in pre_filter_ticks if 0 <= t < n_samples))
                        )
                        if (
                            filtered_ticks_in_range
                        ):  # Ensure we only proceed if there are valid ticks
                            valid_plot_ticks_int.extend(filtered_ticks_in_range)

                    plot_ax_actual_ticks_str = (
                        str(valid_plot_ticks_int) if valid_plot_ticks_int else "[]"
                    )

                    # Updated S1 Prompt to use ticks and request ordered output
                    s1_prompt_text = f"""Analyze the attached plot for Feature {i_feat} (time series data range [0-{n_samples-1}]).
The X-axis displays the following major tick marks: {plot_ax_actual_ticks_str}.
Your task is to identify continuous ranges of time steps that appear visually anomalous or suspicious.
These ranges MUST be defined by selecting a start tick and an end tick from the provided list of X-axis tick marks ({plot_ax_actual_ticks_str}).
The start_index of your range must be a value from this list, and the end_index must also be a value from this list.
Consider sudden spikes/dips, sustained deviations, changes in volatility, or unusual patterns.
If training anomaly examples were marked (red shaded areas), use them as a guide for what anomalies might look like, but also find other suspicious regions not marked.
Output ONLY a JSON list of [start_tick_value, end_tick_value] tuples, ordered from most suspicious to least suspicious based on your visual analysis. The most anomalous range should appear first in the list. Ensure both values in a tuple must be present in the provided tick marks list.
Example: If ticks are [0, 50, 100, 150, 200] and you see an anomaly between tick 50 and tick 100, and another less obvious one between 150 and 200, output: [[50, 100], [150, 200]].
If no ranges appear anomalous, output an empty list []."""

                    # Save S1 prompt - REMOVED
                    # try:
                    #     with open(s1_prompt_path, 'w') as f_prompt_s1:
                    #         f_prompt_s1.write(s1_prompt_text)
                    # except IOError as e_io:
                    #     self.logger.warning(f"Could not save S1 prompt to {s1_prompt_path}: {e_io}")

                    # Construct contents for S1
                    s1_contents_list = []
                    if (
                        main_plot_part_step1
                        and hasattr(main_plot_part_step1, "uri")
                        and hasattr(main_plot_part_step1, "mime_type")
                    ):
                        s1_contents_list.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_uri(
                                        file_uri=main_plot_part_step1.uri,
                                        mime_type=main_plot_part_step1.mime_type,
                                    )
                                ],
                            )
                        )
                    s1_contents_list.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=s1_prompt_text)],
                        )
                    )
                    # Token Counting for S1 (moved here after s1_contents_list is defined)
                    if self.client and model_name and s1_contents_list:
                        try:
                            s1_token_count_result = self.client.models.count_tokens(
                                model=model_name, contents=s1_contents_list
                            )
                            self.logger.info(
                                f"S1 (Feature {i_feat}): Estimated prompt tokens: {s1_token_count_result.total_tokens if s1_token_count_result else 'N/A'}"
                            )
                        except Exception as e_count_tokens_s1:
                            self.logger.warning(
                                f"S1 (Feature {i_feat}): Could not count tokens for S1 prompt. Client: {'Exists' if self.client else 'None'}. Error:\n{_format_exception_for_logging(e_count_tokens_s1)}"
                            )

                    s1_config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            items=genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                items=genai.types.Schema(
                                    type=genai.types.Type.INTEGER,
                                ),
                            ),
                        ),
                        temperature=0,
                    )

                    # API call for S1
                    response_s1 = self._execute_api_call(
                        lambda client_obj, model_str_arg, contents_arg, config_arg: client_obj.models.generate_content(
                            model=model_str_arg,
                            contents=contents_arg,
                            config=config_arg,
                        ),
                        model_name,
                        s1_contents_list,
                        s1_config,  # s1_config is the GenerationContentConfig object
                    )

                    identified_ranges_s1_raw = None
                    s1_response_text_content = ""
                    if response_s1:
                        # Log S1 usage metadata
                        if (
                            hasattr(response_s1, "usage_metadata")
                            and response_s1.usage_metadata
                        ):
                            self.logger.info(
                                f"S1 (Feature {i_feat}): Usage metadata: Prompt Tokens: {response_s1.usage_metadata.prompt_token_count}, Candidates Tokens: {response_s1.usage_metadata.candidates_token_count}, Total Tokens: {response_s1.usage_metadata.total_token_count}"
                            )
                        elif hasattr(
                            response_s1, "text"
                        ):  # Check if usage_metadata might be missing but response exists
                            self.logger.info(
                                f"S1 (Feature {i_feat}): Response received, but no usage_metadata attribute or it's empty. Response text (first 100 chars): {response_s1.text[:100] if response_s1.text else 'N/A'}"
                            )

                        # Save S1 raw response
                        if hasattr(response_s1, "text") and response_s1.text:
                            s1_response_text_content = response_s1.text
                        elif (
                            hasattr(response_s1, "parts")
                            and response_s1.parts
                            and hasattr(response_s1.parts[0], "text")
                        ):  # Fallback for some structures
                            s1_response_text_content = response_s1.parts[0].text

                        if s1_response_text_content:
                            try:
                                with open(s1_response_path, "w") as f_resp_s1:
                                    f_resp_s1.write(s1_response_text_content)
                            except IOError as e_io:
                                self.logger.warning(
                                    f"Could not save S1 response to {s1_response_path}: {_format_exception_for_logging(e_io)}"
                                )
                        try:
                            if hasattr(response_s1, "parsed"):
                                identified_ranges_s1_raw = response_s1.parsed
                                if identified_ranges_s1_raw is None:
                                    self.logger.warning(
                                        f"S1 (Feature {i_feat}): response_s1.parsed is None. This may indicate an issue if an empty list [] was expected."
                                    )
                            else:
                                self.logger.warning(
                                    f"S1 (Feature {i_feat}): Response object does not have 'parsed' attribute. Attempting to parse from text. Response: {s1_response_text_content[:200]}..."
                                )
                                if (
                                    s1_response_text_content
                                ):  # Try to parse from text if .parsed is not available
                                    try:
                                        # Attempt to strip markdown and then parse if it's JSON like
                                        cleaned_json_text = strip_markdown_code_fences(
                                            s1_response_text_content
                                        )
                                        if cleaned_json_text.startswith(
                                            "["
                                        ) and cleaned_json_text.endswith("]"):
                                            identified_ranges_s1_raw = json.loads(
                                                cleaned_json_text
                                            )
                                        else:
                                            self.logger.warning(
                                                f"S1 (Feature {i_feat}): Text content does not appear to be a JSON list after stripping fences."
                                            )
                                    except json.JSONDecodeError as e_json_parse:
                                        self.logger.warning(
                                            f"S1 (Feature {i_feat}): Failed to parse S1 response text as JSON. Error: {_format_exception_for_logging(e_json_parse)}. Text: {cleaned_json_text[:200]}..."
                                        )
                                        last_exception_s1 = e_json_parse
                                else:
                                    last_exception_s1 = AttributeError(
                                        "Response object missing 'parsed' attribute and no text content to parse."
                                    )
                        except Exception as e_access_parsed:
                            self.logger.warning(
                                f"S1 (Feature {i_feat}): Error accessing/parsing response_s1. Error: {_format_exception_for_logging(e_access_parsed)}"
                            )
                            last_exception_s1 = e_access_parsed
                    else:
                        self.logger.warning(
                            f"S1 (Feature {i_feat}): LLM returned no response object."
                        )
                        last_exception_s1 = ValueError("S1 LLM no response object")

                    if identified_ranges_s1_raw is not None:
                        if isinstance(identified_ranges_s1_raw, list):
                            if not identified_ranges_s1_raw:
                                feature_identified_ranges = []
                                step1_succeeded = True
                                self.logger.info(
                                    f"S1 successful for feature {i_feat}. Identified ranges: [] (empty list from LLM)"
                                )
                            else:
                                validated_ranges_from_llm = []
                                all_ranges_structurally_valid_and_are_ticks = (
                                    True  # New validation flag
                                )
                                for r_val in identified_ranges_s1_raw:
                                    try:
                                        start_val, end_val = int(r_val[0]), int(
                                            r_val[1]
                                        )
                                        if not (0 <= start_val <= end_val < n_samples):
                                            self.logger.warning(
                                                f"S1 (Feature {i_feat}): Invalid range values [{start_val},{end_val}] (out of bounds for 0-{n_samples-1} or start > end). Original: {r_val}"
                                            )
                                            all_ranges_structurally_valid_and_are_ticks = (
                                                False
                                            )
                                            break
                                        # Validate that LLM used the provided ticks
                                        if (
                                            not valid_plot_ticks_int
                                        ):  # Should not happen if prompt has ticks and they are valid
                                            self.logger.warning(
                                                f"S1 (Feature {i_feat}): No valid plot ticks available for validation, though LLM provided ranges. LLM range: {r_val}"
                                            )
                                            all_ranges_structurally_valid_and_are_ticks = (
                                                False
                                            )
                                            break
                                        if (
                                            start_val not in valid_plot_ticks_int
                                            or end_val not in valid_plot_ticks_int
                                        ):
                                            self.logger.warning(
                                                f"S1 (Feature {i_feat}): LLM outputted range [{start_val},{end_val}] where one or both values are not in the provided ticks {valid_plot_ticks_int}. Original: {r_val}"
                                            )
                                            all_ranges_structurally_valid_and_are_ticks = (
                                                False
                                            )
                                            break
                                        validated_ranges_from_llm.append(
                                            [start_val, end_val]
                                        )
                                    except (
                                        ValueError,
                                        TypeError,
                                        IndexError,
                                    ) as e_conv:
                                        self.logger.warning(
                                            f"S1 (Feature {i_feat}): Could not convert/unpack range values {r_val} to int [start,end]. Error: {_format_exception_for_logging(e_conv)}"
                                        )
                                        all_ranges_structurally_valid_and_are_ticks = (
                                            False
                                        )
                                        break

                                if (
                                    all_ranges_structurally_valid_and_are_ticks
                                    and validated_ranges_from_llm
                                ):
                                    extended_ranges_after_ticks = []
                                    # unique_sorted_ticks_for_extension is valid_plot_ticks_int (the list of ticks given to LLM and used for validation)
                                    unique_sorted_ticks_for_extension = (
                                        valid_plot_ticks_int
                                    )

                                    if (
                                        unique_sorted_ticks_for_extension
                                    ):  # Should be true if validation passed
                                        for (
                                            r_s,
                                            r_e,
                                        ) in (
                                            validated_ranges_from_llm
                                        ):  # r_s, r_e are tick values from the LLM
                                            try:
                                                idx_s_of_tick = unique_sorted_ticks_for_extension.index(
                                                    r_s
                                                )
                                                idx_e_of_tick = unique_sorted_ticks_for_extension.index(
                                                    r_e
                                                )
                                            except ValueError:
                                                # This should ideally not be reached if the above checks pass (start_val/end_val in valid_plot_ticks_int)
                                                self.logger.error(
                                                    f"S1 (Feature {i_feat}): Critical error - tick value {r_s} or {r_e} from LLM not found in validated tick list {unique_sorted_ticks_for_extension} during extension phase. This indicates a logic flaw. Using original range from LLM."
                                                )
                                                extended_ranges_after_ticks.append(
                                                    [r_s, r_e]
                                                )
                                                continue

                                            new_s_tick_idx = max(
                                                0, idx_s_of_tick - 1
                                            )  # Extend to previous tick index
                                            new_e_tick_idx = min(
                                                len(unique_sorted_ticks_for_extension)
                                                - 1,
                                                idx_e_of_tick + 1,
                                            )  # Extend to next tick index

                                            final_s = unique_sorted_ticks_for_extension[
                                                new_s_tick_idx
                                            ]
                                            final_e = unique_sorted_ticks_for_extension[
                                                new_e_tick_idx
                                            ]

                                            # Bounds are implicitly handled as ticks are from plot range, and new_s/e_tick_idx are capped
                                            if final_s <= final_e:
                                                extended_ranges_after_ticks.append(
                                                    [final_s, final_e]
                                                )
                                            else:
                                                # Fallback if extension somehow invalidates range (e.g., if only one tick existed and it tried to extend)
                                                self.logger.warning(
                                                    f"S1 (Feature {i_feat}): Range extension resulted in start > end ({final_s} > {final_e}). Using original LLM range [{r_s},{r_e}]."
                                                )
                                                extended_ranges_after_ticks.append(
                                                    [r_s, r_e]
                                                )
                                    else:  # No usable ticks for extension (should not be reached if LLM ranges are valid and based on ticks)
                                        self.logger.warning(
                                            f"S1 (Feature {i_feat}): No valid_plot_ticks_int available for extending LLM ranges, using LLM ranges as is: {validated_ranges_from_llm}"
                                        )
                                        extended_ranges_after_ticks = (
                                            validated_ranges_from_llm
                                        )

                                    feature_identified_ranges_before_resolve = (
                                        extended_ranges_after_ticks
                                    )

                                    feature_identified_ranges = (
                                        _resolve_overlapping_ranges(
                                            feature_identified_ranges_before_resolve,
                                            n_samples,
                                        )
                                    )
                                    feature_identified_ranges = (
                                        _merge_contiguous_ranges(
                                            feature_identified_ranges
                                        )
                                    )
                                    step1_succeeded = True
                                    self.logger.info(
                                        f"S1 successful for feature {i_feat}. LLM tick-based ranges: {validated_ranges_from_llm}, Extended/Processed ranges: {feature_identified_ranges}"
                                    )
                                elif not all_ranges_structurally_valid_and_are_ticks:
                                    self.logger.warning(
                                        f"S1 (Feature {i_feat}): Parsed JSON has invalid range values, structure, or values not from provided ticks. Original LLM output: {identified_ranges_s1_raw}, Valid Ticks: {valid_plot_ticks_int}"
                                    )
                                    last_exception_s1 = ValueError(
                                        "S1 LLM output malformed (invalid range values/structure or not from provided ticks after parsing)"
                                    )
                        else:  # identified_ranges_s1_raw (from .parsed) is not a list
                            self.logger.warning(
                                f"S1 (Feature {i_feat}): Content of response_s1.parsed is not a list as expected. Output: {identified_ranges_s1_raw}"
                            )
                            last_exception_s1 = ValueError(
                                "S1 LLM output malformed (response_s1.parsed not a list)"
                            )
                    elif (
                        identified_ranges_s1_raw is None
                        and last_exception_s1 is None
                        and response_s1
                    ):
                        # This case handles if .parsed was None and no other error occurred during access.
                        # Schema expects an array; None is not an empty array.
                        self.logger.warning(
                            f"S1 (Feature {i_feat}): response_s1.parsed was None. Expected an empty list [] for no anomalies according to schema. Treating as failure."
                        )
                        last_exception_s1 = ValueError(
                            "S1 LLM: response_s1.parsed was None, but schema expected a list (e.g., [])."
                        )
                    # If step1_succeeded is still False, last_exception_s1 should contain the reason.
                except Exception as e_s1_attempt:
                    formatted_error = _format_exception_for_logging(e_s1_attempt)
                    self.logger.warning(
                        f"S1 (Feature {i_feat}): Attempt {attempt_s1 + 1} failed. Error:\n{formatted_error}"
                    )
                    last_exception_s1 = e_s1_attempt

                    sleep_duration = 1.0  # Default sleep
                    is_quota_error = False
                    retry_delay_match_found = (
                        False  # Flag to check if specific delay was parsed
                    )

                    error_text_to_search = str(e_s1_attempt)
                    if hasattr(e_s1_attempt, "args") and e_s1_attempt.args:
                        error_text_to_search = str(e_s1_attempt.args[0])

                    # Check for common indications of quota errors
                    # google.api_core.exceptions.ResourceExhausted might have status_code or code attribute
                    # genai.types.generation_types.BlockedPromptException can also indicate quota issues (though often content-related)
                    if (
                        "RESOURCE_EXHAUSTED" in error_text_to_search.upper()
                        or (
                            hasattr(e_s1_attempt, "status_code")
                            and e_s1_attempt.status_code == 429
                        )
                        or (hasattr(e_s1_attempt, "code") and e_s1_attempt.code == 429)
                        or isinstance(
                            e_s1_attempt, types.generation_types.BlockedPromptException
                        )
                    ):  # Broaden check
                        is_quota_error = True
                        self.logger.info(
                            f"S1 (Feature {i_feat}): Detected potential quota error on attempt {attempt_s1 + 1}."
                        )

                        retry_delay_match = re.search(
                            r"'retryDelay':\s*'(\d+)s'", error_text_to_search
                        )
                        if retry_delay_match:
                            try:
                                sleep_duration = float(retry_delay_match.group(1))
                                retry_delay_match_found = True
                                self.logger.info(
                                    f"S1 (Feature {i_feat}): Using retryDelay from error: {sleep_duration}s."
                                )
                            except ValueError:
                                self.logger.warning(
                                    f"S1 (Feature {i_feat}): Could not parse retryDelay value '{retry_delay_match.group(1)}'. Using exponential backoff."
                                )
                                sleep_duration = (
                                    2 ** (attempt_s1 % self.max_retries_per_key)
                                ) + random.uniform(0, 0.5)
                        else:
                            sleep_duration = (
                                2 ** (attempt_s1 % self.max_retries_per_key)
                            ) + random.uniform(0, 0.5)
                            self.logger.info(
                                f"S1 (Feature {i_feat}): No specific retryDelay found. Using exponential backoff: {sleep_duration:.2f}s."
                            )
                    else:
                        sleep_duration = (
                            1.0  # Default for non-quota related errors during retries
                        )

                    if (
                        not retry_delay_match_found
                    ):  # Apply cap if not from specific server advice
                        sleep_duration = min(
                            sleep_duration, 60.0
                        )  # Cap exponential backoff to 60s

                    self.logger.info(
                        f"S1 (Feature {i_feat}): Sleeping for {sleep_duration:.2f}s before next attempt."
                    )
                    time.sleep(sleep_duration)
                finally:  # Ensure main_plot_part_step1 is handled if it was successfully uploaded but step failed later - NO FILE DELETION
                    main_plot_part_step1 = (
                        None  # Reset for next potential attempt or if S1 failed
                    )
                    # files_uploaded_this_attempt_s1 = [] # Removed

            if not step1_succeeded:
                self.logger.error(
                    f"S1 failed for feature {i_feat} after all retries. Last error: {_format_exception_for_logging(last_exception_s1)}"
                )
                # Store failure artifact and continue to next feature
                self.per_feature_artifacts_[i_feat] = {
                    "plot_path_s1": (
                        os.path.join(f"run_{self.last_run_timestamp}", plot_filename_s1)
                        if self.last_run_timestamp and os.path.exists(current_plot_path)
                        else None
                    ),
                    "s1_response_path": (
                        os.path.join(
                            f"run_{self.last_run_timestamp}", s1_response_filename
                        )
                        if self.last_run_timestamp and os.path.exists(s1_response_path)
                        else None
                    ),
                    "identified_ranges_s1": [],
                    "step1_success": False,
                    "step2_success": False,
                    "feature_scores_s2": feature_anomaly_scores.tolist(),  # Zeros
                    "error_s1": (
                        _format_exception_for_logging(last_exception_s1)
                        if last_exception_s1
                        else None
                    ),
                }
                continue  # Skip to next feature if S1 failed

            # --- LLM Step 2: Generate and Execute Anomaly Scoring Code ---
            step2_llm_call_start_time = time.time()
            generated_code_s2_str = None  # Store the generated code string
            files_uploaded_this_attempt_s2 = (
                []
            )  # For S2 file uploads if any (plot is from S1)

            # Retry loop for Step 2
            for attempt_s2 in range(len(self.api_keys) * self.max_retries_per_key):
                if step2_succeeded:
                    break
                self.current_api_key_index = attempt_s2 // self.max_retries_per_key
                if self.current_api_key_index >= len(self.api_keys):
                    break

                try:
                    self.logger.info(
                        f"S2: Feature {i_feat}, Attempt {attempt_s2 + 1}, Key Index {self.current_api_key_index}"
                    )

                    # Retrieve the S1 plot file part (already uploaded and tracked in files_uploaded_this_pipeline)
                    # Find the S1 plot file part by its known display name or path if needed, or re-upload if necessary.
                    # For simplicity, assume main_plot_part_step1 (from successful S1) is still valid or re-fetch/re-upload if API requires fresh.
                    # Here, we'll re-use the File object from S1 if the API allows.
                    # Let's re-upload for S2 to be safe and manage its lifecycle independently.

                    s2_plot_file_part = None
                    if os.path.exists(current_plot_path):  # S1 plot path
                        s2_plot_file_part = self._execute_api_call(
                            lambda client, path: client.files.upload(file=path),
                            current_plot_path,
                        )
                        files_uploaded_this_attempt_s2.append(s2_plot_file_part.name)
                        files_uploaded_this_pipeline.append(s2_plot_file_part.name)
                    else:
                        self.logger.warning(
                            f"S1 plot for S2 (Feature {i_feat}) not found at {current_plot_path}. Proceeding without plot for S2."
                        )

                    # Prepare raw data snippets for S2 prompt based on S1's feature_identified_ranges
                    # feature_identified_ranges are already sorted by suspiciousness due to S1 prompt change
                    raw_data_snippets_for_s2_map = {}
                    if feature_identified_ranges:
                        POINTS_TO_SHOW_PER_RANGE_LIMIT = (
                            10  # Max data points to show from a range snippet
                        )
                        TOTAL_SNIPPETS_CHAR_LIMIT_APPROX = 1500  # Approx char limit for the whole snippets block in the prompt
                        MAX_S1_RANGES_FOR_SNIPPETS = (
                            5  # Max number of S1 ranges to provide snippets for
                        )

                        current_total_chars_in_snippets_map = 0
                        num_snippets_included = 0

                        for r_s_idx, (r_start, r_end) in enumerate(
                            feature_identified_ranges
                        ):
                            if num_snippets_included >= MAX_S1_RANGES_FOR_SNIPPETS:
                                self.logger.info(
                                    f"S2: Reached MAX_S1_RANGES_FOR_SNIPPETS ({MAX_S1_RANGES_FOR_SNIPPETS}) for feature {i_feat}. Processed {num_snippets_included} ranges."
                                )
                                if (
                                    len(feature_identified_ranges)
                                    > num_snippets_included
                                ):
                                    raw_data_snippets_for_s2_map[
                                        "_TRUNCATION_NOTE_MAX_RANGES_"
                                    ] = f"Snippets for {len(feature_identified_ranges) - num_snippets_included} further S1 ranges (less suspicious) not shown due to max snippet count limit."
                                break

                            if not (0 <= r_start <= r_end < n_samples):
                                self.logger.warning(
                                    f"S2 data prep: Invalid range [{r_start},{r_end}] from S1 for feature {i_feat}. Skipping for snippet."
                                )
                                continue

                            data_segment = X_data[r_start : r_end + 1, i_feat]
                            # Use .tolist() for JSON serializable numbers, then format
                            segment_data_list = data_segment.tolist()

                            # Format numbers to have fewer decimal places for brevity
                            formatted_segment_data_list = [
                                float(f"{val:.3g}") for val in segment_data_list
                            ]
                            indices_for_segment = list(range(r_start, r_end + 1))

                            current_range_snippet_dict = {}
                            display_segment_str_json_key_val = "{}"

                            if (
                                len(formatted_segment_data_list)
                                > POINTS_TO_SHOW_PER_RANGE_LIMIT
                            ):
                                half_limit = POINTS_TO_SHOW_PER_RANGE_LIMIT // 2
                                for k in range(half_limit):
                                    current_range_snippet_dict[
                                        str(indices_for_segment[k])
                                    ] = formatted_segment_data_list[k]
                                current_range_snippet_dict["..."] = (
                                    f"truncated {len(formatted_segment_data_list) - POINTS_TO_SHOW_PER_RANGE_LIMIT} points"
                                )
                                for k in range(
                                    POINTS_TO_SHOW_PER_RANGE_LIMIT - half_limit
                                ):
                                    original_idx = (
                                        len(formatted_segment_data_list)
                                        - (POINTS_TO_SHOW_PER_RANGE_LIMIT - half_limit)
                                        + k
                                    )
                                    current_range_snippet_dict[
                                        str(indices_for_segment[original_idx])
                                    ] = formatted_segment_data_list[original_idx]
                            else:
                                for k in range(len(formatted_segment_data_list)):
                                    current_range_snippet_dict[
                                        str(indices_for_segment[k])
                                    ] = formatted_segment_data_list[k]

                            display_segment_str_json_key_val = json.dumps(
                                current_range_snippet_dict
                            )  # dict to string for one range

                            range_key_str = (
                                f"Range [{r_start}-{r_end}] (S1 rank: {r_s_idx+1})"
                            )

                            # Estimate length of this snippet when added to JSON map
                            estimated_snippet_len = (
                                len(range_key_str)
                                + len(display_segment_str_json_key_val)
                                + 20
                            )  # for quotes, commas, etc.

                            if (
                                current_total_chars_in_snippets_map
                                + estimated_snippet_len
                                > TOTAL_SNIPPETS_CHAR_LIMIT_APPROX
                                and num_snippets_included > 0
                            ):
                                self.logger.info(
                                    f"S2: Stopping snippet inclusion for feature {i_feat} due to char limit. Included {num_snippets_included} of {min(len(feature_identified_ranges), MAX_S1_RANGES_FOR_SNIPPETS)} S1 ranges considered."
                                )
                                raw_data_snippets_for_s2_map[
                                    "_TRUNCATION_NOTE_CHAR_LIMIT_"
                                ] = f"Data for {min(len(feature_identified_ranges), MAX_S1_RANGES_FOR_SNIPPETS) - num_snippets_included} further S1 ranges (less suspicious) not shown due to prompt length limits."
                                break

                            raw_data_snippets_for_s2_map[range_key_str] = (
                                current_range_snippet_dict  # Store the dict directly
                            )
                            current_total_chars_in_snippets_map += estimated_snippet_len
                            num_snippets_included += 1

                    snippets_json_str_for_prompt = "No S1 ranges identified, or data snippets could not be prepared/were empty."
                    if raw_data_snippets_for_s2_map:
                        try:
                            # Use json.dumps for proper formatting and escaping
                            snippets_json_str_for_prompt = json.dumps(
                                raw_data_snippets_for_s2_map, indent=2
                            )
                        except TypeError as e_json:
                            self.logger.error(
                                f"Error serializing S2 snippets to JSON for feature {i_feat}: {e_json}. Snippets map: {raw_data_snippets_for_s2_map}"
                            )
                            snippets_json_str_for_prompt = (
                                f"Error preparing snippets for prompt: {str(e_json)}"
                            )

                    if (
                        len(snippets_json_str_for_prompt)
                        > TOTAL_SNIPPETS_CHAR_LIMIT_APPROX * 1.2
                    ):  # Final check
                        self.logger.warning(
                            f"S2: Snippets JSON string for feature {i_feat} is very long ({len(snippets_json_str_for_prompt)} chars), might still exceed limits. Consider reducing limits."
                        )

                    # Construct the prompt for Step 2 - Updated S2 Prompt
                    s2_prompt_text = f"""You are an expert anomaly detection model.
For Feature {i_feat} (time series data [0-{n_samples-1}]), you are given:
1. A plot of the feature data (if attached).
2. Potentially interesting ranges identified from visual inspection (S1 output, ordered most to least suspicious): {str(feature_identified_ranges) if feature_identified_ranges else 'No specific pre-identified ranges.'}
3. Raw data snippets for some of these S1 identified ranges (prioritizing more suspicious ones if truncated). Snippets are in {"index": value} format:
```json
{snippets_json_str_for_prompt}
```
   (Note: Snippets might be truncated if ranges are very long, if total data volume is too large, or if max number of S1 ranges for snippets is reached. Values are formatted for brevity.)
   Use these actual values from the S1-identified ranges to *examine them in detail and pinpoint specific anomalous indices or sub-segments*.
4. Training data context (if available and used): {'Anomalies in training data were highlighted in the plot.' if self.use_training_labels_for_step1_plot and y_labels_for_plot_hint is not None and example_anomalies_plotted else 'No specific training anomaly labels were highlighted for direct use in this step.'}

Your task is to GENERATE a Python function called 'calculate_anomaly_scores'.
This function MUST:
1. Accept one argument: 'raw_feature_data' (a 1D NumPy array for this specific feature, length {n_samples}).
2. Return a 1D NumPy array of the same length as 'raw_feature_data', containing BINARY anomaly scores (1 for anomaly, 0 for normal).
3. CRITICAL: The primary goal is to assign anomaly scores (1) to *specific indices or short, confirmed sub-segments*.
   - Use the plot (if available), the S1-identified candidate ranges ({str(feature_identified_ranges) if feature_identified_ranges else 'N/A'}), and ESPECIALLY the provided raw data snippets to make your decisions.
   - The S1 ranges are candidates for detailed examination. Your main task is to analyze the values in the raw_data_snippets for these S1 ranges to *confirm and pinpoint the exact anomalous data points or very short anomalous sequences*.
   - For example, if an S1 range is [100,120], examine its snippet. If only indices 105-107 within that snippet show truly anomalous values (e.g., extreme spikes), then score `anomaly_scores[105:108] = 1`. Do NOT blindly score the entire S1 range [100,120] as 1 unless its entire snippet uniformly and clearly demonstrates anomalous behavior compared to typical data for this feature.
   - If specific indices or small, well-defined sub-ranges are clearly anomalous (e.g., based on the snippets showing extreme values or abrupt changes not typical for the feature), directly assign scores (e.g., `anomaly_scores[105:108] = 1`).
   - AVOID complex detection logic (e.g., rolling statistics, complex thresholds) UNLESS it's extremely concise, essential for a pattern not easily described by direct indexing using snippet information, and significantly saves tokens.
   - Focus on translating your detailed examination of the S1 ranges (via their snippets) into precise score assignments.
4. Include a brief explanation within the generated Python code in the designated comment block, justifying the chosen indices/sub-segments based on your analysis of the plot, S1 ranges, AND the provided raw data snippets. Clearly state which part of the snippet led to your decision.

Function template:
```python
import numpy as np

def calculate_anomaly_scores(raw_feature_data):
    # --- BEGIN EXPLANATION ---
    # [Explain your choices. Focus on specific indices/sub-segments. E.g., "S1 range [100-120] (rank 1) was a candidate. Its snippet showed normal values except for indices 105-107, which spiked to [value1, value2, value3], indicating a clear anomaly localized there. S1 range [200-250] (rank 2) snippet showed generally elevated values but index 220 had an extreme dip to [value_dip], also marked."]
    # --- END EXPLANATION ---
    
    n_points = len(raw_feature_data)
    anomaly_scores = np.zeros(n_points, dtype=int)
    
    # --- BEGIN SCORING LOGIC ---
    # [Implement your scoring logic here - primarily direct assignments of 1s to *specific anomalous indices or short sub-segments* identified from S1 candidate ranges and their raw data snippets]
    # Example: Scoring based on detailed analysis of S1 ranges {str(list(raw_data_snippets_for_s2_map.keys())[:MAX_S1_RANGES_FOR_SNIPPETS]) if raw_data_snippets_for_s2_map else "[]"} and their snippets.
    # Remember: S1 ranges are for focusing your attention. Use the snippets to find the *actual* anomalies.
    
    # Example for a specific S1 candidate range, say [100,120], whose snippet was in the prompt:
    # Based on the snippet for range [100,120]:
    # if n_points > 107: # Check bounds for the pinpointed anomaly
    #     # Assume snippet for [100,120] showed indices 105, 106, 107 as anomalous
    #     # anomaly_scores[105:108] = 1 # Scores indices 105, 106, 107

    # Example for another S1 candidate range, say [200,250], snippet also provided:
    # Based on the snippet for range [200,250]:
    # if n_points > 220: # Check bounds
    #     # Assume snippet showed only index 220 as a clear anomalous dip
    #     # anomaly_scores[220] = 1
    # --- END SCORING LOGIC ---
    
    return anomaly_scores
```
Provide ONLY the complete Python code for the function, enclosed in ```python ... ```."""

                    # Save S2 prompt - REMOVED

                    # Construct contents for S2
                    s2_contents_list = []
                    if (
                        s2_plot_file_part
                        and hasattr(s2_plot_file_part, "uri")
                        and hasattr(s2_plot_file_part, "mime_type")
                    ):
                        s2_contents_list.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_uri(
                                        file_uri=s2_plot_file_part.uri,
                                        mime_type=s2_plot_file_part.mime_type,
                                    )
                                ],
                            )
                        )
                    s2_contents_list.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=s2_prompt_text)],
                        )
                    )
                    # Token Counting for S2
                    if self.client and model_name and s2_contents_list:
                        try:
                            s2_token_count_result = self.client.models.count_tokens(
                                model=model_name, contents=s2_contents_list
                            )
                            self.logger.info(
                                f"S2 (Feature {i_feat}): Estimated prompt tokens: {s2_token_count_result.total_tokens if s2_token_count_result else 'N/A'}"
                            )
                        except Exception as e_count_tokens_s2:
                            self.logger.warning(
                                f"S2 (Feature {i_feat}): Could not count tokens for S2 prompt. Client: {'Exists' if self.client else 'None'}. Error:\n{_format_exception_for_logging(e_count_tokens_s2)}"
                            )

                    s2_config = types.GenerateContentConfig(
                        response_mime_type="text/plain",
                        temperature=0,
                    )

                    # API call for S2
                    response_s2 = self._execute_api_call(
                        lambda client_obj, model_str_arg, contents_arg, config_arg: client_obj.models.generate_content(
                            model=model_str_arg,
                            contents=contents_arg,
                            config=config_arg,
                        ),
                        model_name,
                        s2_contents_list,
                        s2_config,
                    )

                    s2_response_text_content = ""
                    if response_s2 and response_s2.text:
                        s2_response_text_content = response_s2.text.strip()
                        # Log S2 usage metadata
                        if (
                            hasattr(response_s2, "usage_metadata")
                            and response_s2.usage_metadata
                        ):
                            self.logger.info(
                                f"S2 (Feature {i_feat}): Usage metadata: Prompt Tokens: {response_s2.usage_metadata.prompt_token_count}, Candidates Tokens: {response_s2.usage_metadata.candidates_token_count}, Total Tokens: {response_s2.usage_metadata.total_token_count}"
                            )
                        else:  # Check if usage_metadata might be missing but response exists
                            self.logger.info(
                                f"S2 (Feature {i_feat}): Response received, but no usage_metadata attribute or it's empty. Response text (first 100 chars): {s2_response_text_content[:100] if s2_response_text_content else 'N/A'}"
                            )

                        # Save S2 raw response
                        try:
                            with open(s2_raw_response_path, "w") as f_resp_s2:
                                f_resp_s2.write(s2_response_text_content)
                        except IOError as e_io:
                            self.logger.warning(
                                f"Could not save S2 raw response to {s2_raw_response_path}: {_format_exception_for_logging(e_io)}"
                            )

                        generated_code_s2_str = strip_markdown_code_fences(
                            s2_response_text_content
                        )
                        if generated_code_s2_str:
                            # Save stripped S2 generated code
                            try:
                                with open(s2_generated_code_path, "w") as f_code_s2:
                                    f_code_s2.write(generated_code_s2_str)
                            except IOError as e_io:
                                self.logger.warning(
                                    f"Could not save S2 generated code to {s2_generated_code_path}: {_format_exception_for_logging(e_io)}"
                                )

                            local_namespace = {}
                            exec_globals = {
                                "np": np,
                                "pd": pd,
                            }  # Make numpy and pandas available
                            try:
                                exec(
                                    generated_code_s2_str, exec_globals, local_namespace
                                )
                                if "calculate_anomaly_scores" in local_namespace:
                                    score_calculator_func = local_namespace[
                                        "calculate_anomaly_scores"
                                    ]
                                    current_feature_data = X_data[
                                        :, i_feat
                                    ].copy()  # Pass a copy

                                    calculated_scores = score_calculator_func(
                                        current_feature_data
                                    )

                                    if (
                                        isinstance(calculated_scores, np.ndarray)
                                        and calculated_scores.shape == (n_samples,)
                                        and np.all(np.isin(calculated_scores, [0, 1]))
                                    ):
                                        feature_anomaly_scores = (
                                            calculated_scores.astype(float)
                                        )
                                        step2_succeeded = True
                                        self.logger.info(
                                            f"S2: Successfully executed generated code for feature {i_feat}."
                                        )
                                    else:
                                        self.logger.warning(
                                            f"S2 (Feature {i_feat}): Generated code returned invalid scores. Shape: {calculated_scores.shape if isinstance(calculated_scores, np.ndarray) else 'N/A'}, Values: {np.unique(calculated_scores) if isinstance(calculated_scores, np.ndarray) else 'N/A'}"
                                        )
                                        last_exception_s2 = ValueError(
                                            "S2 generated code output malformed"
                                        )
                                else:
                                    self.logger.warning(
                                        f"S2 (Feature {i_feat}): 'calculate_anomaly_scores' not in generated code."
                                    )
                                    last_exception_s2 = NameError(
                                        "calculate_anomaly_scores function not found"
                                    )
                            except Exception as e_exec:
                                self.logger.error(
                                    f"S2 (Feature {i_feat}): Error executing generated code: {_format_exception_for_logging(e_exec)}. Code:\n{generated_code_s2_str}"
                                )
                                last_exception_s2 = e_exec
                        else:
                            self.logger.warning(
                                f"S2 (Feature {i_feat}): LLM returned empty code string after stripping fences."
                            )
                            last_exception_s2 = ValueError("S2 LLM empty code string")
                    else:
                        self.logger.warning(
                            f"S2 (Feature {i_feat}): LLM returned empty response."
                        )
                        last_exception_s2 = ValueError("S2 LLM empty response")

                except Exception as e_s2_attempt:
                    formatted_error = _format_exception_for_logging(e_s2_attempt)
                    self.logger.warning(
                        f"S2 (Feature {i_feat}): Attempt {attempt_s2 + 1} failed. Error:\n{formatted_error}"
                    )
                    last_exception_s2 = e_s2_attempt

                    sleep_duration = 1.0  # Default sleep
                    is_quota_error = False
                    retry_delay_match_found = (
                        False  # Flag to check if specific delay was parsed
                    )

                    error_text_to_search = str(e_s2_attempt)
                    if hasattr(e_s2_attempt, "args") and e_s2_attempt.args:
                        error_text_to_search = str(e_s2_attempt.args[0])

                    if (
                        "RESOURCE_EXHAUSTED" in error_text_to_search.upper()
                        or (
                            hasattr(e_s2_attempt, "status_code")
                            and e_s2_attempt.status_code == 429
                        )
                        or (hasattr(e_s2_attempt, "code") and e_s2_attempt.code == 429)
                        or isinstance(
                            e_s2_attempt, types.generation_types.BlockedPromptException
                        )
                    ):
                        is_quota_error = True
                        self.logger.info(
                            f"S2 (Feature {i_feat}): Detected potential quota error on attempt {attempt_s2 + 1}."
                        )

                        retry_delay_match = re.search(
                            r"'retryDelay':\s*'(\d+)s'", error_text_to_search
                        )
                        if retry_delay_match:
                            try:
                                sleep_duration = float(retry_delay_match.group(1))
                                retry_delay_match_found = True
                                self.logger.info(
                                    f"S2 (Feature {i_feat}): Using retryDelay from error: {sleep_duration}s."
                                )
                            except ValueError:
                                self.logger.warning(
                                    f"S2 (Feature {i_feat}): Could not parse retryDelay value '{retry_delay_match.group(1)}'. Using exponential backoff."
                                )
                                sleep_duration = (
                                    2 ** (attempt_s2 % self.max_retries_per_key)
                                ) + random.uniform(0, 0.5)
                        else:
                            sleep_duration = (
                                2 ** (attempt_s2 % self.max_retries_per_key)
                            ) + random.uniform(0, 0.5)
                            self.logger.info(
                                f"S2 (Feature {i_feat}): No specific retryDelay found. Using exponential backoff: {sleep_duration:.2f}s."
                            )
                    else:
                        sleep_duration = 1.0

                    if not retry_delay_match_found:
                        sleep_duration = min(sleep_duration, 60.0)

                    self.logger.info(
                        f"S2 (Feature {i_feat}): Sleeping for {sleep_duration:.2f}s before next attempt."
                    )
                    time.sleep(sleep_duration)
                finally:  # Cleanup files uploaded in this S2 attempt (successful or failed) - NO FILE DELETION
                    # for file_name in files_uploaded_this_attempt_s2:
                    #     try:
                    #         self._execute_api_call(
                    #             lambda client, name: client.files.delete(name=name),
                    #             file_name,
                    #         )
                    #         if file_name in files_uploaded_this_pipeline:
                    #             files_uploaded_this_pipeline.remove(file_name)
                    #     except Exception as e_del:
                    #         self.logger.error(
                    #             f"Error deleting file {file_name} during S2 finally: {_format_exception_for_logging(e_del)}"
                    #         )
                    pass  # No action needed here anymore if not deleting files

            if step2_succeeded:
                final_scores_all_features = np.logical_or(
                    final_scores_all_features, feature_anomaly_scores.astype(bool)
                ).astype(float)
            else:
                self.logger.error(
                    f"S2 failed for feature {i_feat} after all retries. Last error: {_format_exception_for_logging(last_exception_s2)}"
                )
                # Scores for this feature remain 0 if S2 fails, contributing nothing to final_scores via logical_or

            self.per_feature_artifacts_[i_feat] = {
                "plot_path_s1": (
                    os.path.join(f"run_{self.last_run_timestamp}", plot_filename_s1)
                    if self.last_run_timestamp and os.path.exists(current_plot_path)
                    else None
                ),
                "s1_response_path": (
                    os.path.join(f"run_{self.last_run_timestamp}", s1_response_filename)
                    if self.last_run_timestamp and os.path.exists(s1_response_path)
                    else None
                ),
                "s2_raw_response_path": (
                    os.path.join(
                        f"run_{self.last_run_timestamp}", s2_raw_response_filename
                    )
                    if self.last_run_timestamp and os.path.exists(s2_raw_response_path)
                    else None
                ),
                "s2_generated_code_path": (
                    os.path.join(
                        f"run_{self.last_run_timestamp}", s2_generated_code_filename
                    )
                    if self.last_run_timestamp
                    and os.path.exists(s2_generated_code_path)
                    else None
                ),
                "identified_ranges_s1": feature_identified_ranges,
                "step1_success": step1_succeeded,
                "generated_code_s2": (
                    generated_code_s2_str if generated_code_s2_str else None
                ),
                "step2_success": step2_succeeded,
                "feature_scores_s2": feature_anomaly_scores.tolist(),  # Store actual scores (could be all zeros if S2 failed)
                "error_s1": (
                    _format_exception_for_logging(last_exception_s1)
                    if last_exception_s1
                    else None
                ),
                "error_s2": (
                    _format_exception_for_logging(last_exception_s2)
                    if last_exception_s2
                    else None
                ),
            }
            feature_duration = time.time() - feature_loop_start_time
            self.logger.debug(
                f"Feature {i_feat} processing took {feature_duration:.2f}s. S1: {'OK' if step1_succeeded else 'Fail'}, S2: {'OK' if step2_succeeded else 'Fail'}"
            )

        # --- End of feature loop ---

        self.decision_scores_ = final_scores_all_features

        # Cleanup all uploaded files for this pipeline run - REMOVED
        # self.logger.info(
        #     f"Cleaning up {len(files_uploaded_this_pipeline)} uploaded files for this run."
        # )
        # for file_name_to_delete in list(
        #     set(files_uploaded_this_pipeline)
        # ):  # Use set to avoid duplicates
        #     try:
        #         # Check if client is available, initialize if not, for cleanup
        #         if not self.client:
        #             self._initialize_client()
        #         if self.client:  # Proceed only if client is valid
        #             self._execute_api_call(
        #                 lambda client, name: client.files.delete(name=name),
        #                 file_name_to_delete,
        #             )
        #             self.logger.debug(
        #                 f"Successfully deleted uploaded file: {file_name_to_delete}"
        #             )
        #         else:
        #             self.logger.warning(
        #                 f"Cannot delete file {file_name_to_delete}, client not available after run."
        #             )
        #     except Exception as e_clean:
        #         self.logger.error(
        #             f"Error cleaning up uploaded file {file_name_to_delete}: {_format_exception_for_logging(e_clean)}"
        #         )

        # Rename temporary directories to permanent ones based on timestamp
        final_plot_dir = os.path.join(self.plot_save_dir, f"run_{timestamp_str}")
        try:
            if os.path.exists(temp_plot_base_dir):
                os.rename(temp_plot_base_dir, final_plot_dir)
                self.logger.info(
                    f"Renamed temp plot dir {temp_plot_base_dir} to {final_plot_dir}"
                )
        except OSError as e_rename:
            self.logger.error(
                f"Error renaming temporary directories: {_format_exception_for_logging(e_rename)}"
            )

        pipeline_duration = time.time() - pipeline_start_time
        self.logger.info(f"LLM Pipeline completed in {pipeline_duration:.2f} seconds.")
        return self.decision_scores_

    def fit(self, X_train, y_train=None):
        self.logger.info(
            f"Starting E_May_14 fit method. X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}"
        )
        # Store training labels and length if provided, for hinting in _run_llm_pipeline
        if y_train is not None and self.use_training_labels_for_step1_plot:
            self.y_train_labels_for_hint = y_train
            self.train_data_len_for_hint = X_train.shape[0]
            self.logger.debug(
                f"Stored y_train labels (len {len(y_train)}) and X_train length ({self.train_data_len_for_hint}) for plot hinting."
            )
        else:
            self.y_train_labels_for_hint = None
            self.train_data_len_for_hint = None
            self.logger.debug(
                "No y_train labels provided or use_training_labels_for_step1_plot is False. No training hints for plotting."
            )
        return self

    def decision_function(self, X_test):
        self.logger.info(
            f"Starting E_May_14 decision_function. X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}"
        )
        # When making decisions on new data, we might still pass the training hints
        # if the LLM's prompts are designed to leverage them generally (e.g., "this is what anomalies looked like in training")
        # However, for a pure "test" phase, y_labels_for_plot_hint would be from X_test if available for eval, or None.
        # The train_len_for_plot_hint is crucial to distinguish train/test segments if X_test is appended to X_train.
        # Here, decision_function gets X_test, so it's purely "test" data.
        # We can still provide the original training anomaly hints for context if desired.

        # If X_test is independent, train_len_for_plot_hint should be 0 or None for X_test itself.
        # The y_train_labels_for_hint and train_data_len_for_hint are from the `fit` call.
        # The _run_llm_pipeline will use these if its `y_labels_for_plot_hint` argument is populated.

        # For decision_function, we are processing X_test.
        # The plot should show X_test data. If we want to show how training data looked, that's a different plot.
        # The current _run_llm_pipeline uses y_labels_for_plot_hint for plotting *example* anomalies.
        # If we are in a true test phase, we wouldn't have X_test labels.
        # The `train_len_for_plot_hint` in `_run_llm_pipeline` refers to a split *within the data passed to it*.
        # So, for X_test, this should be None or 0 if we are not showing a "training part" *of X_test*.

        decision_scores = self._run_llm_pipeline(
            X_test,
            y_labels_for_plot_hint=self.y_train_labels_for_hint,  # General context of train anomalies
            train_len_for_plot_hint=self.train_data_len_for_hint,
        )
        self.logger.info(
            f"E_May_14 decision_function completed. Scores shape: {decision_scores.shape if decision_scores is not None else 'N/A'}"
        )
        return decision_scores

    def get_per_feature_artifacts(self):
        """Returns the artifacts collected per feature during the last run."""
        return self.per_feature_artifacts_

    def get_last_run_timestamp(self):
        """Returns the timestamp string of the last run, used for directory naming."""
        return self.last_run_timestamp

    # Consider adding a method to clear uploaded files by key_index if needed,
    # or a more robust cleanup if runs are interrupted.
    # Current cleanup is at the end of _run_llm_pipeline.
