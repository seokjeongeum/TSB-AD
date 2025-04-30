import ast
import json  # Needed for compact data format
import os
import random
import re
import tempfile
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from google import genai
from google.genai import types
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.basic_metrics import basic_metricor
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.slidingWindows import find_length_rank

model_name = "gemini-2.5-pro-exp-03-25"
api_keys = [
    "AIzaSyDjJm1IfV_X6frznOFrtPqttlWPEZEY2UI",
    "AIzaSyBgWIDQ2BDwkOAu4jEn6KwuYDIa07FZ-KQ",
    "AIzaSyBQ7S8TNRqyIn0nJkFHHkc7bjME3udflI0",
    "AIzaSyBnWeouM8SUJfS7VuzKwAW9z9e5CjkhCaQ",
    "AIzaSyD7UQ9TANoqK-P2ArntYxLFC2p9ZXlpjLI",
    "AIzaSyCZHCBavhPUeQT_C2kYqqdbJ6vddkj3ls8",
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
]
genai_types = types


class GoogleApiExceptions:
    def __init__(self):
        self.PermissionDenied = genai.errors.ClientError
        self.Unauthenticated = genai.errors.ClientError
        self.ResourceExhausted = genai.errors.ClientError


google_api_exceptions = GoogleApiExceptions()
genai.configure = lambda _: None


# Helper function (no changes)
def strip_markdown_code_fences(code_string):
    pattern_python = r"^\s*```python\n(.*?)\n```\s*$"
    match_python = re.match(pattern_python, code_string, re.DOTALL | re.IGNORECASE)
    if match_python:
        return match_python.group(1).strip()
    pattern_generic = r"^\s*```\n(.*?)\n```\s*$"
    match_generic = re.match(pattern_generic, code_string, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
    if code_string.startswith("```"):
        code_string = code_string.split("\n", 1)[1] if "\n" in code_string else ""
    if code_string.endswith("```"):
        code_string = code_string.rsplit("\n", 1)[0] if "\n" in code_string else ""
    return code_string.strip()


class D_April_30(BaseDetector):
    def __init__(self, HP):
        super().__init__()
        self.api_keys = api_keys
        self.current_api_key_index = 0
        self.client = self._initialize_client()
        self.decision_scores_ = None
        self.identified_ranges_ = []
        self.generated_code_ = None
        self.generated_code_dir = "Generated_Code"  # Define code dir here
        os.makedirs(self.generated_code_dir, exist_ok=True)

    def _initialize_client(self):
        if self.current_api_key_index >= len(self.api_keys):
            print("Error: Ran out of API keys.")
            return None
        current_key = self.api_keys[self.current_api_key_index]
        os.environ["GOOGLE_API_KEY"] = current_key
        print(
            f"Initializing GenAI Client with key index {self.current_api_key_index}..."
        )
        try:
            # Use genai.configure before creating client for clarity
            # FOLLOWING CODE DOES NOT WORK SO IT IS COMMENTED OUT
            # genai.configure(api_key=current_key)
            client = genai.Client(api_key=current_key)
            # Optional: Test client validity here if needed
            # client.list_models()
            print(
                f"Client initialized successfully with key index {self.current_api_key_index}."
            )
            return client
        except Exception as e:
            print(
                f"Error initializing GenAI Client with key index {self.current_api_key_index}: {e}"
            )
            self.current_api_key_index += 1
            return self._initialize_client()

    # --- *** MODIFIED _make_api_call *** ---
    def _make_api_call(self, model_name, plot_path, prompt_text, config):
        """
        Uploads plot, makes API call with error handling and key rotation,
        and cleans up the uploaded file.
        """
        if self.client is None:
            raise RuntimeError("GenAI Client is not initialized.")
        if not plot_path or not os.path.exists(plot_path):
            raise FileNotFoundError(f"Plot file not found at: {plot_path}")

        initial_key_index = self.current_api_key_index
        max_retries = len(self.api_keys)

        for attempt in range(max_retries):
            image_file = None  # Reset image_file for each attempt
            try:
                print(
                    f"Attempt {attempt + 1}/{max_retries} using key index {self.current_api_key_index}..."
                )

                # 1. Upload file using the current client
                print(f"  Uploading plot: {plot_path}")
                upload_start_time = time.time()
                image_file = self.client.files.upload(file=plot_path)
                upload_end_time = time.time()
                print(
                    f"  Uploaded file '{image_file.display_name}' as: {image_file.uri} (took {upload_end_time - upload_start_time:.2f}s)"
                )

                # 2. Construct contents with the new file URI
                contents = [
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                file_data=genai_types.FileData(
                                    mime_type=image_file.mime_type,
                                    file_uri=image_file.uri,
                                )
                            ),
                            genai_types.Part(text=prompt_text),
                        ],
                    )
                ]

                # 3. Make the API call
                print(f"  Sending request to model: {model_name}")
                api_call_start_time = time.time()
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                api_call_end_time = time.time()
                print(
                    f"  API call successful (took {api_call_end_time - api_call_start_time:.2f}s)."
                )
                return response  # Success

            except (
                google_api_exceptions.PermissionDenied,
                google_api_exceptions.Unauthenticated,
                google_api_exceptions.ResourceExhausted,
                # Add other potentially recoverable errors here if needed
                Exception,
            ) as e:  # Catch broader exceptions too
                print(
                    f"  API call/upload failed with key index {self.current_api_key_index}: {type(e).__name__} - {e}"
                )
                is_key_issue = isinstance(
                    e,
                    (
                        google_api_exceptions.PermissionDenied,
                        google_api_exceptions.Unauthenticated,
                    ),
                )

                if is_key_issue or attempt < max_retries - 1:
                    self.current_api_key_index = (self.current_api_key_index + 1) % len(
                        self.api_keys
                    )
                    print(
                        f"  Rotating to next API key index: {self.current_api_key_index}"
                    )
                    self.client = self._initialize_client()
                    if self.client is None:
                        raise RuntimeError("All API keys failed.") from e
                    if (
                        not is_key_issue
                        and self.current_api_key_index == initial_key_index
                    ):
                        print(
                            "  Error: Cycled through all keys, but error persists. Stopping retries."
                        )
                        raise e
                else:
                    print(
                        "  Error: Last API key attempt failed or error not key-related."
                    )
                    raise e
            finally:
                # 4. Clean up the *uploaded* file for this attempt, regardless of success/failure
                if image_file and hasattr(self.client, "files"):
                    try:
                        print(
                            f"  Deleting uploaded file from Gemini: {image_file.name}"
                        )
                        delete_start_time = time.time()
                        self.client.files.delete(name=image_file.name)
                        delete_end_time = time.time()
                        print(
                            f"  Deletion successful (took {delete_end_time - delete_start_time:.2f}s)."
                        )
                    except Exception as del_e:
                        # Log error but don't stop the process for a deletion failure
                        print(
                            f"  Warning: Could not delete uploaded file {image_file.name} during cleanup: {del_e}"
                        )

        raise RuntimeError("API call failed after exhausting all keys and retries.")

    # --- *** END MODIFIED _make_api_call *** ---

    def fit(self, X, y=None):
        """Fit detector using a two-step visual analysis approach."""
        if self.client is None:
            print("Error: GenAI Client could not be initialized.")
            self.decision_scores_ = np.zeros(X.shape[0])
            self.identified_ranges_ = []
            return self
        print("Generating plot from input data X...")
        temp_plot_path = None
        fig = None  # image_file is now handled in _make_api_call
        n_samples, n_features = X.shape

        try:
            # --- Plotting Logic (remains the same) ---
            print("  Initiating plotting logic...")
            nrows = n_features
            ncols = 1
            fig_width = 40
            fig_height_per_plot = 2.5
            fig_height = fig_height_per_plot * nrows
            print(
                f"Creating plot with {nrows} rows, {ncols} col. Size: ({fig_width}, {fig_height})"
            )
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False
            )
            axes_flat = axes.flatten()
            for i in range(n_features):
                ax = axes_flat[i]
                ax.plot(np.arange(n_samples), X[:, i])
                ax.set_title(f"Feature {i+1}")
                ax.set_ylabel("Value")
                ax.grid(True)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=40, integer=True))
                ax.tick_params(axis="x", labelsize="xx-small", rotation=45)
                ax.set_xlim(0, n_samples - 1)
                if i == n_features - 1:
                    ax.set_xlabel("Time Step")
            fig.suptitle(
                f"Time Series Data ({n_features} Features)", fontsize=16, y=0.99
            )
            fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.97])
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_plot_path = temp_file.name
                fig.savefig(temp_plot_path, dpi=200)
                print(f"  Original plot saved temporarily: {temp_plot_path}")
            plt.close(fig)
            fig = None
            # --- End Plotting ---

            # --- *** REMOVED initial file upload *** ---

            # === STEP 1: Ask Model to Identify Interesting Index Ranges ===
            print("\n--- Step 1: Asking model to identify interesting ranges ---")
            prompt_text_step1 = f"""
            Analyze the provided time series plot image, which shows {n_features} features over {n_samples} time steps.
            Based *only* on visual inspection of the plot, identify the time step index ranges (start_index, end_index) that appear most anomalous or interesting for closer inspection.
            Focus on significant spikes, shifts, or pattern deviations.
            **Use the visible x-axis ticks (e.g., ..., 15000, 15500, 16000, ...) as reference points to estimate the start and end indices for the ranges you identify.**

            Return your answer strictly as a JSON list of lists, where each inner list contains two integers: the estimated start and end index of a range based on the ticks. For example: [[15900, 16100], [15000, 16383]].
            If no specific ranges stand out significantly, return an empty list [].
            Do not include any other text, explanations, or markdown formatting.
            """
            generation_config_step1 = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=genai_types.Schema(
                    type=genai_types.Type.ARRAY,
                    items=genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.NUMBER),
                    ),
                ),
                temperature=0,
            )

            # --- Use API call wrapper (pass plot path and prompt) ---
            response_step1 = self._make_api_call(
                model_name=model_name,
                plot_path=temp_plot_path,  # Pass local path
                prompt_text=prompt_text_step1,
                config=generation_config_step1,
            )
            # -------------------------------------------------------

            # --- Parse Step 1 Response ---
            index_ranges = []
            try:
                if (
                    hasattr(response_step1, "parsed")
                    and response_step1.parsed is not None
                ):
                    parsed_ranges = response_step1.parsed
                    if isinstance(parsed_ranges, list):
                        for item in parsed_ranges:
                            if (
                                isinstance(item, (list, tuple))
                                and len(item) == 2
                                and all(isinstance(n, int) for n in item)
                            ):
                                start = max(0, item[0])
                                end = min(n_samples, item[1] + 1)
                                if start < end:
                                    index_ranges.append([start, end])
                            else:
                                print(
                                    f"Warning: Invalid range format in Step 1 parsed response item: {item}"
                                )
                        index_ranges.sort()
                        print(
                            f"Model identified ranges for closer look: {index_ranges}"
                        )
                    else:
                        print(
                            f"Warning: Step 1 parsed response was not a list, but type {type(parsed_ranges)}."
                        )
                elif response_step1.text:
                    print(
                        "Warning: Response schema parsing failed or was None, attempting manual parsing of text."
                    )
                    try:
                        parsed_ranges_manual = ast.literal_eval(response_step1.text)
                        if isinstance(parsed_ranges_manual, list):
                            for item in parsed_ranges_manual:
                                if (
                                    isinstance(item, list)
                                    and len(item) == 2
                                    and all(isinstance(n, int) for n in item)
                                ):
                                    start = max(0, item[0])
                                    end = min(n_samples, item[1] + 1)
                                    if start < end:
                                        index_ranges.append([start, end])
                                else:
                                    print(
                                        f"Warning: Invalid range format in Step 1 manual parse item: {item}"
                                    )
                            index_ranges.sort()
                            print(
                                f"Model identified ranges (manual parse): {index_ranges}"
                            )
                        else:
                            print("Warning: Step 1 manual parse result was not a list.")
                    except (SyntaxError, ValueError) as e_manual:
                        print(
                            f"Error manually parsing Step 1 JSON response: {e_manual}"
                        )
                        print("Response text was:", response_step1.text)
                else:
                    print(
                        "Warning: Step 1 returned empty response text and no parsed data."
                    )
            except Exception as e:
                print(f"Unexpected error processing Step 1 response: {e}")
                traceback.print_exc()
            if hasattr(response_step1, "text"):
                print("Response text was:", response_step1.text)
            self.identified_ranges_ = index_ranges

            # === STEP 2: Provide Numerical Data for Ranges & Ask for Final Code ===
            print(
                "\n--- Step 2: Providing numerical details and asking for final scoring code ---"
            )
            numerical_data_extracts = {}
            if self.identified_ranges_:
                print("Extracting numerical data for identified ranges...")
                for start, end in self.identified_ranges_:
                    safe_start = max(0, start)
                    safe_end = min(n_samples, end)
                    if safe_start < safe_end:
                        data_slice = X[safe_start:safe_end, :]
                        range_key = f"{safe_start}-{safe_end-1}"
                        numerical_data_extracts[range_key] = data_slice.tolist()
                    else:
                        print(
                            f"Warning: Skipping invalid range after bounds check: [{start}, {end}]"
                        )
            data_string_step2 = "{}"
            if numerical_data_extracts:
                try:
                    data_string_step2 = json.dumps(
                        numerical_data_extracts, separators=(",", ":")
                    )
                    print(
                        f"Prepared compact JSON string for {len(numerical_data_extracts)} ranges (length: {len(data_string_step2)} chars)."
                    )
                except Exception as json_e:
                    print(f"Error creating JSON string for numerical data: {json_e}")
                    data_string_step2 = "{}"

            prompt_text_step2 = f"""
            You are an expert time series anomaly detection programmer.
            You previously analyzed the time series plot provided in the image (showing {n_features} features over {n_samples} time steps).
            Based on that visual analysis, you identified specific index ranges as potentially interesting.

            Now, you are provided with the detailed numerical data for those specific ranges in the following JSON object string:
            ```json
            {data_string_step2}
            ```
            (If the above JSON object is empty `{{}}`, it means no specific ranges were highlighted previously or data extraction failed.)

            Your final task is to WRITE PYTHON CODE that calculates anomaly scores for the *entire* time series ({n_samples} points).
            Use the overall visual context from the plot image AND the detailed numerical data provided above for the specified ranges to **embed the anomaly detection logic directly into the function**.

            The Python code you generate must:
            1. Define a function named `calculate_anomaly_scores`. **This function MUST accept NO arguments.**
            2. The logic inside this function should implement your anomaly detection reasoning, integrating insights from both the full plot and the detailed numerical data segments. Use standard Python and NumPy if needed (assume `import numpy as np` is available). **The function should contain the logic itself, not expect the raw data as input.**
            3. This function MUST return a list or NumPy array of numerical anomaly scores.
            4. The length of the returned list/array MUST be exactly {n_samples}.
            5. Higher values in the returned list/array should indicate a higher likelihood of the corresponding time step being anomalous.

            IMPORTANT: Your output for this request must be ONLY the Python code string itself.
            Do not include any explanations, comments outside the code block, or markdown formatting like ```python ... ```.
            Just provide the raw Python code defining the `calculate_anomaly_scores` function.
            """
            generation_config_step2 = genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=24576,
                ),
                response_mime_type="text/plain",
                temperature=0,
            )

            # --- Use API call wrapper (pass plot path and prompt) ---
            print(f"Sending Step 2 request to Gemini model: {model_name}...")
            response_step2 = self._make_api_call(
                model_name=model_name,
                plot_path=temp_plot_path,  # Pass same local path again
                prompt_text=prompt_text_step2,
                config=generation_config_step2,
            )
            # -------------------------------------------------------

            # --- Execute Generated Code (from Step 2) ---
            generated_code_step2 = ""
            anomaly_scores_from_code = None
            try:
                if not response_step2.text:
                    print(
                        "ERROR: Step 2 received empty response text (no code generated)."
                    )
                    self.decision_scores_ = np.zeros(n_samples)
                else:
                    generated_code_step2 = strip_markdown_code_fences(
                        response_step2.text
                    )
                    self.generated_code_ = generated_code_step2  # Store code
                    try:
                        # --- *** Use filename in saved code name *** ---
                        base_fname_code = os.path.splitext(
                            os.path.basename(temp_plot_path)
                        )[
                            0
                        ]  # Get base name from plot path
                        code_filename = f"{base_fname_code}_generated_code.py"
                        code_save_path = os.path.join(
                            self.generated_code_dir, code_filename
                        )
                        # --- *************************************** ---
                        with open(code_save_path, "w") as f_code:
                            f_code.write(
                                "# Generated by D_April_30.fit\n"
                            )  # Use class name
                            f_code.write(
                                "# Identified ranges: "
                                + str(self.identified_ranges_)
                                + "\n\n"
                            )
                            f_code.write(generated_code_step2)
                        print(f"  Generated code saved to: {code_save_path}")
                    except Exception as save_code_e:
                        print(
                            f"  Warning: Failed to save generated code: {save_code_e}"
                        )

                    print("--- Generated Python Code (Step 2, Cleaned) ---")
                    print(generated_code_step2)
                    print("--- End Generated Code ---")
                    execution_globals = {"np": np}
                    execution_locals = {}
                    print("Executing generated code (Step 2)...")
                    exec(generated_code_step2, execution_globals, execution_locals)

                    if "calculate_anomaly_scores" in execution_locals and callable(
                        execution_locals["calculate_anomaly_scores"]
                    ):
                        print(
                            "Calling generated function 'calculate_anomaly_scores'..."
                        )
                        anomaly_scores_from_code = execution_locals[
                            "calculate_anomaly_scores"
                        ]()
                        if isinstance(anomaly_scores_from_code, (list, np.ndarray)):
                            self.decision_scores_ = np.array(
                                anomaly_scores_from_code, dtype=float
                            )
                            print(
                                f"Successfully got {len(self.decision_scores_)} scores from generated code."
                            )
                            if len(self.decision_scores_) != n_samples:
                                print(
                                    f"ERROR: Generated code returned scores of incorrect length! Expected {n_samples}, Got {len(self.decision_scores_)}"
                                )
                                self.decision_scores_ = np.zeros(n_samples)
                            elif not np.issubdtype(
                                self.decision_scores_.dtype, np.number
                            ):
                                print(
                                    f"ERROR: Generated code returned non-numeric scores. Type: {self.decision_scores_.dtype}"
                                )
                                self.decision_scores_ = np.zeros(n_samples)
                        else:
                            print(
                                "ERROR: Generated function 'calculate_anomaly_scores' did not return a list or NumPy array."
                            )
                            self.decision_scores_ = np.zeros(n_samples)
                    else:
                        print(
                            "ERROR: Generated code did not define 'calculate_anomaly_scores' correctly."
                        )
                        self.decision_scores_ = np.zeros(n_samples)

            except Exception as exec_e:
                print(f"ERROR executing generated code (Step 2): {exec_e}")
                traceback.print_exc()
                print("--- Faulty Generated Code (Step 2) ---")
                print(generated_code_step2)
                print("--- End Faulty Code ---")
                self.decision_scores_ = np.zeros(n_samples)

            # --- Final Check ---
            if self.decision_scores_ is None:
                print("Assigning default scores (zeros) because execution failed.")
                self.decision_scores_ = np.zeros(n_samples)

        # --- General Exception Handling ---
        except (
            google_api_exceptions.PermissionDenied,
            google_api_exceptions.Unauthenticated,
            google_api_exceptions.ResourceExhausted,
        ) as api_key_e:
            print(
                f"API Key related error encountered: {api_key_e}. This might have been handled by rotation if other keys are available."
            )
            if self.decision_scores_ is None:
                self.decision_scores_ = np.zeros(n_samples)
                self.identified_ranges_ = []
        except AttributeError as ae:
            print(f"AttributeError during API interaction: {ae}")
            traceback.print_exc()
            if self.decision_scores_ is None:
                self.decision_scores_ = np.zeros(n_samples)
                self.identified_ranges_ = []
        except Exception as e:
            print(f"An error occurred during the fit process: {e}")
            traceback.print_exc()
            if self.decision_scores_ is None:
                print("Assigning default scores (zeros) due to an error during fit.")
                self.decision_scores_ = np.zeros(n_samples)
                self.identified_ranges_ = []

        finally:
            if fig is not None and plt.fignum_exists(fig.number):
                plt.close(fig)

        return self

    # --- decision_function remains the same ---
    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        if self.decision_scores_ is None:
            print(
                "Warning: Detector not fitted or fit failed. Returning empty array for scores."
            )
            return np.array([])
        if not isinstance(self.decision_scores_, np.ndarray):
            print(
                "Warning: decision_scores_ is not a numpy array. Attempting conversion."
            )
            try:
                return np.array(self.decision_scores_, dtype=float)
            except Exception as e:
                print(f"Error converting scores to numpy array: {e}")
                return np.array([])
        if len(self.decision_scores_) != X.shape[0]:
            print(
                f"Warning: Length of stored scores ({len(self.decision_scores_)}) does not match input X ({X.shape[0]}) to decision_function. Returning empty array."
            )
            return np.array([])
        return self.decision_scores_


def run_D_April_30_Unsupervised(data, HP):  # Renamed function
    clf = D_April_30(HP=HP)  # Use renamed class
    clf.fit(data)
    score = clf.decision_scores_
    identified_ranges = (
        clf.identified_ranges_ if hasattr(clf, "identified_ranges_") else []
    )
    if score is None or score.size == 0:
        print("Warning: No valid scores generated by fit. Returning empty score array.")
        return np.zeros(data.shape[0]), identified_ranges
    score = (
        MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    )
    return score, identified_ranges


# Renamed function
def run_D_April_30_Semisupervised(data_train, data_test, HP):
    clf = D_April_30(HP=HP)  # Use renamed class
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    # identified_ranges = clf.identified_ranges_ if hasattr(clf, 'identified_ranges_') else [] # Get ranges if needed
    score = (
        MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    )
    # return score, identified_ranges # Return ranges if needed
    return score


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
    return list(zip(start_indices, end_indices))


# --- MODIFIED visualize_errors function ---
def visualize_errors(
    original_label,
    score,
    model_identified_ranges,  # New parameter for ranges identified by model
    base_plot_filename,
    plot_dir,
    model_prefix="",
    chunk_size=1500,
    chunk_indices_to_plot=None,
):
    """
    Visualizes calculated anomaly score against the ground truth label for specified chunks.
    Adds shaded regions for ranges identified by the model.
    Plots only chunks containing true anomalies.
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")

    n_samples = len(original_label)
    valid_input = True

    # --- Input Validation ---
    if not isinstance(original_label, np.ndarray) or original_label.ndim != 1:
        valid_input = False
        print("Vis Error: original_label format")
    if not isinstance(score, np.ndarray) or score.ndim != 1:
        valid_input = False
        print("Vis Error: score format")
    elif valid_input and score.shape[0] != n_samples:
        valid_input = False
        print(f"Vis Error: score length ({score.shape[0]}) vs label ({n_samples})")
    elif valid_input and original_label.shape[0] != n_samples:
        valid_input = False
        print(
            f"Vis Error: label length ({original_label.shape[0]}) vs expected ({n_samples})"
        )
    if not isinstance(model_identified_ranges, list):
        print(
            f"Vis Warning: model_identified_ranges is not a list (type: {type(model_identified_ranges)}). Annotations might fail."
        )

    if not valid_input:
        print(f"Error ({model_prefix}): Invalid input for score/label visualization.")
        return
    if not chunk_indices_to_plot:
        if np.any(original_label == 1):
            print(
                f"Info ({model_prefix}): No specific anomaly chunks requested for plotting, although anomalies exist."
            )
        else:
            print(
                f"Info ({model_prefix}): No true anomalies in label, skipping chunk plotting."
            )
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig_height = 5
    fig_width = 14
    chunks_to_iterate = sorted(list(set(chunk_indices_to_plot)))
    print(
        f"Generating Score vs Label plot(s) for anomaly chunk(s): {chunks_to_iterate}..."
    )

    for i_chunk in chunks_to_iterate:
        chunk_start = i_chunk * chunk_size
        chunk_end = min(n_samples, (i_chunk + 1) * chunk_size)
        time_range_chunk = np.arange(chunk_start, chunk_end)
        if len(time_range_chunk) == 0:
            continue

        score_chunk = score[chunk_start:chunk_end]
        original_label_chunk = original_label[chunk_start:chunk_end]
        if len(score_chunk) != len(time_range_chunk) or len(
            original_label_chunk
        ) != len(time_range_chunk):
            print(f"Vis Error (Chunk {i_chunk}): Score/Label length mismatch.")
            continue

        fig_ts = None
        try:
            fig_ts, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        except Exception as e:
            print(f"Error creating subplot for chunk {i_chunk}: {e}")
            if fig_ts:
                plt.close(fig_ts)
                continue

        try:
            ln_score = ax.plot(
                time_range_chunk,
                score_chunk,
                label=f"Anomaly Score ({model_prefix})",
                color="darkorange",
                linestyle="-",
                alpha=0.9,
                linewidth=1.5,
            )
            ln_label = ax.step(
                time_range_chunk,
                original_label_chunk,
                label="True Anomaly Label",
                color="red",
                linestyle="--",
                alpha=0.8,
                linewidth=1.2,
                where="post",
            )

            model_range_label_added = False
            if isinstance(model_identified_ranges, list):
                for start, end in model_identified_ranges:
                    overlap_start = max(chunk_start, start)
                    overlap_end = min(chunk_end, end)
                    if overlap_start < overlap_end:
                        local_start = overlap_start
                        local_end = overlap_end
                        span_label = (
                            "Model Identified Range"
                            if not model_range_label_added
                            else "_nolegend_"
                        )
                        ax.axvspan(
                            local_start,
                            local_end,
                            color="yellow",
                            alpha=0.3,
                            zorder=-1,
                            label=span_label,
                        )
                        model_range_label_added = True

            ax.set_ylabel("Score / Label")
            ax.tick_params(axis="y")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(chunk_start, chunk_end - 1)
            ax.legend(loc="upper left", fontsize="medium")

        except Exception as e:
            print(f"Error plotting data for chunk {i_chunk}: {e}")
            traceback.print_exc()
            plt.close(fig_ts)
            continue

        try:
            ax.set_xlabel("Time Index")
            title = f"{model_prefix} - {base_plot_filename}\nChunk Index {i_chunk} (Indices {chunk_start}-{chunk_end-1}) - Score vs Label (Model Ranges Highlighted)"
            fig_ts.suptitle(title, fontsize=12)
            fig_ts.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
            plot_filename_ts = os.path.join(
                plot_dir,
                f"{base_plot_filename}_{model_prefix}_ChunkIdx_{i_chunk}_{chunk_start}-{chunk_end-1}_Score_vs_Label_ModelRanges.png",
            )
            plt.savefig(plot_filename_ts, bbox_inches="tight", dpi=200)
            print(f"  Score vs Label plot saved: {os.path.basename(plot_filename_ts)}")
        except Exception as e:
            print(f"Error adjusting/saving plot chunk {i_chunk}: {e}")
            traceback.print_exc()
        finally:
            if fig_ts is not None:
                plt.close(fig_ts)

    print(
        f"Score vs Label chunk plotting complete for {base_plot_filename} ({model_prefix})."
    )


# --- End of visualize_errors function ---


# --- End of visualize_errors function ---
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# BASE_METRIC_ORDER (Unchanged)
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
# MODEL_NAMES defined in main now
MODEL_NAMES = ["D_April_30"]


def get_ordered_columns(current_columns):
    """Orders columns for the results CSV."""
    model_prefixes = {name: name for name in MODEL_NAMES}
    final_order = []
    processed_columns = set()
    if "filename" in current_columns:
        final_order.append("filename")
        processed_columns.add("filename")
    for prefix in model_prefixes.values():
        col = f"{prefix}_VUS-PR"
        if col in current_columns:
            final_order.append(col)
            processed_columns.add(col)
    for prefix in model_prefixes.values():
        other_metrics = [f"{prefix}_{m}" for m in BASE_METRIC_ORDER]
        for col in other_metrics:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)
        other_info = [
            f"{prefix}_runtime",
            f"{prefix}_FP_count_ext",
            f"{prefix}_FN_count_ext",
            f"{prefix}_Error",
            f"{prefix}_Scaling_Error",
            f"{prefix}_Metrics_Threshold_Error",
            f"{prefix}_Analysis_Error",
            f"{prefix}_Metrics_Error",
            f"{prefix}_Metrics_Result",
        ]
        for col in other_info:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)
    remaining_columns = sorted(list(set(current_columns) - processed_columns))
    final_order.extend(remaining_columns)
    return final_order


# --- Main Script ---
if __name__ == "__main__":
    overall_start_time = time.time()
    # Configuration
    FILE_LIST_PATH = os.path.join("Datasets", "File_List", "TSB-AD-M-Eva.csv")
    DATA_DIR = os.path.join("Datasets", "TSB-AD-M")
    RESULTS_CSV_PATH = "D_April_30_detailed_results.csv"  # Updated CSV name
    MODEL_NAMES = ["D_April_30"]  # Updated Model Name
    VISUALIZE_ANOMALIES = True
    PLOT_DIR_BASE = "D_April_30_Anomaly_Highlights"  # Updated Plot Dir
    CHUNK_SIZE_VIS = 1500
    # --- *** Define directory for saving generated code *** ---
    GENERATED_CODE_DIR = "D_April_30_Generated_Code"  # Updated Code Dir
    os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
    # --- ************************************************ ---

    metricor = basic_metricor() if basic_metricor else None

    # --- Load Progress from CSV ---
    processed_files = set()
    all_results = []
    if os.path.exists(RESULTS_CSV_PATH):
        try:
            print(
                f"Loading existing results from {RESULTS_CSV_PATH} to track progress..."
            )
            existing_results_df = pd.read_csv(RESULTS_CSV_PATH)
            if "filename" in existing_results_df.columns:
                all_results = existing_results_df.where(
                    pd.notnull(existing_results_df), None
                ).to_dict("records")
                processed_files.update(
                    existing_results_df["filename"].dropna().unique().tolist()
                )
                print(
                    f"Found {len(processed_files)} previously processed files in {RESULTS_CSV_PATH}."
                )
            else:
                print(
                    f"Warning: Existing results file {RESULTS_CSV_PATH} missing 'filename' column."
                )
                all_results = []
        except pd.errors.EmptyDataError:
            print(f"Existing results file {RESULTS_CSV_PATH} is empty.")
            all_results = []
        except Exception as e:
            print(
                f"Warning: Could not load existing results from {RESULTS_CSV_PATH}: {e}."
            )
            all_results = []
    else:
        print(f"No existing results file found at {RESULTS_CSV_PATH}.")

    # --- Load File List ---
    try:
        filenames_df = pd.read_csv(FILE_LIST_PATH)
        if filenames_df.shape[1] > 0:
            filenames = filenames_df.iloc[:, 0].astype(str).str.strip().dropna()
            filenames = filenames[filenames != ""].unique().tolist()
        else:
            print(f"Warning: File list {FILE_LIST_PATH} seems empty.")
            filenames = []
    except Exception as e:
        print(f"Error reading file list {FILE_LIST_PATH}: {e}")
        filenames = []

    total_files = len(filenames)
    files_to_process = [
        f
        for f in filenames
        if isinstance(f, str) and f.strip() and f not in processed_files
    ]
    total_to_process_this_run = len(files_to_process)
    processed_files_count_this_run = 0
    print(
        f"Total files: {total_files}. Processed according to CSV: {len(processed_files)}. To process now: {total_to_process_this_run}."
    )

    # --- Main File Loop ---
    for i, filename in enumerate(files_to_process):
        print(
            f"\nProcessing file {i+1}/{total_to_process_this_run} ({processed_files_count_this_run} successful this run): {filename}"
        )
        file_path = os.path.join(DATA_DIR, filename)
        file_success = False
        current_file_results = {"filename": filename}
        model_prefix = MODEL_NAMES[0]

        try:
            # --- Data Loading and Validation ---
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path).dropna()
            if df.empty:
                raise ValueError("Data is empty after dropping NaNs")
            if "Label" not in df.columns:
                raise ValueError("'Label' column missing")
            if df.shape[1] < 2:
                raise ValueError("Needs at least one feature column besides 'Label'")
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df["Label"].astype(int).to_numpy()
            if data.shape[0] != label.shape[0]:
                raise ValueError("Data and Label shape mismatch after loading")

            slidingWindow = find_length_rank(data, rank=1)

            # --- Run Model ---
            run_start_time = time.time()
            # --- *** Use correct function name *** ---
            output, model_ranges = run_D_April_30_Unsupervised(
                data, {}
            )  # Use renamed function
            # --- ******************************* ---
            run_end_time = time.time()
            current_file_results[f"{model_prefix}_runtime"] = (
                run_end_time - run_start_time
            )

            # --- Scale & Evaluate ---
            if output is None or output.size == 0:
                print(
                    "  Warning: Model fitting returned empty scores. Skipping evaluation and visualization."
                )
                current_file_results[f"{model_prefix}_Error"] = "Empty scores from fit"
            else:
                output_scaled = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(output.reshape(-1, 1))
                    .ravel()
                )
                evaluation_result = get_metrics(
                    output_scaled, label, slidingWindow=slidingWindow
                )
                print("Evaluation Result: ", evaluation_result)
                if evaluation_result:
                    for key, value in evaluation_result.items():
                        result_key = f"{model_prefix}_{key}"
                        if isinstance(value, (np.integer, np.floating)):
                            current_file_results[result_key] = value.item()
                        elif (
                            isinstance(value, (int, float, str, bool)) or value is None
                        ):
                            current_file_results[result_key] = value
                        else:
                            current_file_results[result_key] = str(value)

                # --- Visualization Call ---
                if VISUALIZE_ANOMALIES:
                    base_fname_vis = os.path.splitext(os.path.basename(filename))[0]
                    model_name_vis = model_prefix
                    print(f"  Generating visualizations for {model_name_vis}...")
                    score_for_vis = output_scaled
                    original_label_for_vis = label

                    vis_data_valid = True
                    if score_for_vis is None or original_label_for_vis is None:
                        vis_data_valid = False
                        print(
                            f"    Skipping visualization for {model_name_vis}: Missing essential data (score/label)."
                        )
                    elif (
                        not isinstance(score_for_vis, np.ndarray)
                        or score_for_vis.ndim != 1
                        or not isinstance(original_label_for_vis, np.ndarray)
                        or original_label_for_vis.ndim != 1
                        or score_for_vis.shape[0] != original_label_for_vis.shape[0]
                    ):
                        vis_data_valid = False
                        print(
                            f"    Skipping visualization for {model_name_vis}: Score/Label shape mismatch or type error."
                        )

                    if vis_data_valid:
                        chunk_indices_to_plot = set()
                        if np.any(original_label_for_vis == 1):
                            anomaly_clusters = _find_clusters(
                                np.where(original_label_for_vis == 1)[0]
                            )
                            for start, end in anomaly_clusters:
                                start_chunk = start // CHUNK_SIZE_VIS
                                end_chunk = end // CHUNK_SIZE_VIS
                                chunk_indices_to_plot.update(
                                    range(start_chunk, end_chunk + 1)
                                )
                        else:
                            print(
                                "    No true anomalies found in label. Skipping anomaly chunk visualization."
                            )

                        chunk_indices_list = sorted(list(chunk_indices_to_plot))
                        if chunk_indices_list:
                            print(f"    Plotting anomaly chunks: {chunk_indices_list}")
                            model_plot_dir = os.path.join(PLOT_DIR_BASE, model_name_vis)
                            try:
                                visualize_errors(
                                    original_label=original_label_for_vis,
                                    score=score_for_vis,
                                    model_identified_ranges=model_ranges,
                                    base_plot_filename=base_fname_vis,
                                    plot_dir=model_plot_dir,
                                    model_prefix=model_name_vis,
                                    chunk_size=CHUNK_SIZE_VIS,
                                    chunk_indices_to_plot=chunk_indices_list,
                                )
                            except Exception as vis_e:
                                print(
                                    f"    Error during visualize_errors call for {model_name_vis}: {vis_e}"
                                )
                                current_file_results[
                                    f"{model_prefix}_Visualization_Error"
                                ] = f"Visualize Error: {vis_e}"
                                traceback.print_exc()
                        elif np.any(original_label_for_vis == 1):
                            print(
                                "    Anomalies present, but no chunks identified for plotting."
                            )

            file_success = True

        # --- Outer Loop Exception Handling ---
        except FileNotFoundError as fnf_e:
            print(f"Error: {fnf_e}")
            current_file_results[f"{model_prefix}_Error"] = "File Not Found"
        except (ValueError, RuntimeError, TypeError) as data_load_e:
            print(f"Data Loading/Validation Error: {data_load_e}")
            traceback.print_exc()
            current_file_results[f"{model_prefix}_Error"] = (
                f"Data Load/Validation Error: {data_load_e}"
            )
        except pd.errors.EmptyDataError as ede:
            print(f"Pandas EmptyDataError: {ede}. Skipping.")
            current_file_results[f"{model_prefix}_Error"] = "Pandas EmptyDataError"
        except MemoryError as me:
            print(f"CRITICAL MemoryError (Outer Loop): {me}. Stopping file.")
            traceback.print_exc()
            current_file_results[f"{model_prefix}_Error"] = "MemoryError (Outer Loop)"
            file_success = False
        except Exception as e:
            print(f"Unexpected critical error processing {filename}: {e}")
            traceback.print_exc()
            current_file_results[f"{model_prefix}_Error"] = f"Critical fail: {e}"
            file_success = False

        # --- Append result AND Save Incrementally ---
        if "filename" not in current_file_results:
            current_file_results["filename"] = filename
        existing_file_index = next(
            (
                idx
                for idx, res in enumerate(all_results)
                if res.get("filename") == filename
            ),
            -1,
        )
        if existing_file_index != -1:
            all_results[existing_file_index].update(current_file_results)
        else:
            all_results.append(current_file_results)

        # Mark file as processed in memory set for this run's count
        processed_files.add(filename)
        if file_success:
            processed_files_count_this_run += 1
            print(f"  File processed successfully.")
        else:
            print(f"  File processing marked as unsuccessful/skipped.")

        # Save intermediate CSV (this now IS the progress tracking)
        try:
            if all_results:
                temp_df = pd.DataFrame(all_results)
                temp_df = temp_df.drop_duplicates(subset=["filename"], keep="last")
                if not temp_df.empty:
                    ordered_cols = get_ordered_columns(temp_df.columns.tolist())
                    temp_df = temp_df.reindex(columns=ordered_cols)
                temp_df.to_csv(RESULTS_CSV_PATH, index=False)
        except Exception as e:
            print(f"Warning: Could not save intermediate CSV: {e}")

    # --- End File Loop ---

    # --- Final Summary ---
    print(f"\n--- Run Summary ---")
    final_df_check = pd.DataFrame(all_results).drop_duplicates(
        subset=["filename"], keep="last"
    )
    total_processed_final = len(final_df_check)
    print(
        f"Attempted: {total_to_process_this_run} new files this run. Succeeded this run: {processed_files_count_this_run}. Total files in results CSV: {total_processed_final}."
    )

    # --- Final Save & Averaging ---
    if all_results:
        final_results_df = pd.DataFrame(all_results)
        final_results_df = final_results_df.drop_duplicates(
            subset=["filename"], keep="last"
        )
        try:
            if not final_results_df.empty:
                ordered_cols = get_ordered_columns(final_results_df.columns.tolist())
                final_results_df = final_results_df.reindex(columns=ordered_cols)
            final_results_df.to_csv(RESULTS_CSV_PATH, index=False)
            print(f"Final results saved to {RESULTS_CSV_PATH}")
        except Exception as e:
            print(f"Error saving final results CSV: {e}")

        error_columns = []
        for m in MODEL_NAMES:
            error_columns.extend(
                [
                    f"{m}_Error",
                    f"{m}_Scaling_Error",
                    f"{m}_Analysis_Error",
                    f"{m}_Metrics_Error",
                    f"{m}_Visualization_Error",
                ]
            )
        existing_error_columns = [
            col for col in error_columns if col in final_results_df.columns
        ]
        successful_filter = pd.Series(
            [True] * len(final_results_df), index=final_results_df.index
        )
        for err_col in existing_error_columns:
            successful_filter &= ~(
                final_results_df[err_col].notna() & (final_results_df[err_col] != "")
            )
        successful_df = final_results_df[successful_filter]
        num_successful = len(successful_df)
        num_failed = len(final_results_df) - num_successful
        if not successful_df.empty:
            print(
                f"\n--- Average Metrics Across {num_successful} Successfully Processed Files --- ({num_failed} failed/error files excluded)"
            )

            def print_average(df, col_key, prefix="Average"):
                if col_key in df.columns:
                    numeric_col = pd.to_numeric(df[col_key], errors="coerce")
                    if numeric_col.notna().any():
                        avg_val = np.nanmean(numeric_col.astype(float))
                        metric_name = (
                            col_key.split("_", 1)[1] if "_" in col_key else col_key
                        )
                        print(f'      "{prefix} {metric_name}": {avg_val:.6f},')
                        return True
                return False

            for model_key in MODEL_NAMES:
                print(f"\n  --- {model_key} Averages ({num_successful} files) ---")
                print(f"    -- Key Metrics --")
                if not print_average(
                    successful_df, f"{model_key}_VUS-PR"
                ) and not print_average(successful_df, f"{model_key}_VUS-ROC"):
                    print("      VUS-PR/VUS-ROC not available.")
                print(f"\n    -- Other TSB-AD Metrics --")
                metrics_printed = 0
                for metric in BASE_METRIC_ORDER:
                    if metric not in ["VUS-PR", "VUS-ROC"] and print_average(
                        successful_df, f"{model_key}_{metric}"
                    ):
                        metrics_printed += 1
                if metrics_printed == 0:
                    print("      No other standard metrics available.")
                print(f"\n    -- Other Info --")
                print_average(
                    successful_df, f"{model_key}_runtime", prefix="Avg Runtime (s)"
                )
                print_average(
                    successful_df, f"{model_key}_FP_count_ext", prefix="Avg FP (ext)"
                )
                print_average(
                    successful_df, f"{model_key}_FN_count_ext", prefix="Avg FN (ext)"
                )
            print("\n  ------------------------------------")
        else:
            print(
                f"\nNo successfully processed files found for averaging metrics ({len(final_results_df)} total files in results)."
            )
    else:
        print("\nNo results generated or loaded.")
    overall_end_time = time.time()
    print(
        f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds"
    )
