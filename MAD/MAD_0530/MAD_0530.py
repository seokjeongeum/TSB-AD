import inspect
import json
import logging
import os
import time

import numpy as np
from google.genai import types
from tqdm import tqdm

from MAD.MAD_0530.constants_0530 import (
    API_KEYS,
    DEFAULT_FALLBACK_TOKEN_LIMIT,
    DEFAULT_OUTPUT_TOKENS_INTEREST_ID,
    DEFAULT_RETRY_DELAY_SECONDS,
    FIXED_MODEL_NAME_GEMINI_FLASH,
    MAX_RETRIES_WITH_DELAY_PER_KEY,
    THINKING_BUDGET_INTEREST_ID,
    THINKING_BUDGET_MAIN_LLM,
    TOKEN_BUDGET_SAFETY_MARGIN_INTEREST_ID,
    TOKEN_LIMIT_SAFETY_FACTOR,
)
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.slidingWindows import find_length_rank

from . import (
    gemini_api_utils_0530,
    mad_utils_0530,
    plotting_utils_0530,
    prompt_utils_0530,
)
from .interest_identifier_0530 import perform_interest_identification_step
from .llm_batch_processor_0530 import process_single_batch_with_llm
from .unpredictability_processor_0530 import perform_unpredictability_step


class MAD_0530(BaseDetector):
    """
    Anomaly detector using Google Genai (Gemini). Employs a single-step process *per feature*:
    1. Anomaly Index Identification: Identifies specific anomalous indices for a feature based on
       a plot, raw data snippets from the feature itself (analysis data), and optionally,
       snippets from training data with labels (training examples).
       Operates in batches if the data snippets exceed token limits.
    Operates unsupervised based on the data provided to fit(), or semi-supervised if y_train is provided,
    aggregating results across features.
    """

    def __init__(self, HP):
        super().__init__()
        # --- API Keys (Consider moving to a more secure configuration method) ---

        self.api_keys = API_KEYS
        self.current_api_key_index = 0  # Managed by outer batch loop's retry
        self.last_run_timestamp = None
        self.dataset_name_for_artifacts = HP.get("dataset_name_for_artifacts", None)
        self.current_run_artifact_folder_name = None

        self.max_retries_per_key = 1  # This now means: try each key once for a given level of failure. The new logic handles same-key retries internally.
        self.max_overall_retries = len(
            self.api_keys
        )  # Total attempts is number of keys

        # New HP for same-key retries with delay (Request 2)
        self.max_retries_with_delay_per_key = MAX_RETRIES_WITH_DELAY_PER_KEY
        self.default_retry_delay_seconds = DEFAULT_RETRY_DELAY_SECONDS

        self.HP = HP if HP is not None else {}
        # This HP controls if training anomalies are shown on the input plot to the LLM
        self.use_training_labels_for_plot_hint_on_main_plot = True

        self.plot_save_dir = "MAD/Feature_Plots/MAD_0530"  # MODIFIED: Main directory for all feature plots
        self.score_label_plot_save_dir = "MAD/ScoreLabel_Plots/MAD_0530"  # MODIFIED: Main directory for score label plots
        os.makedirs(self.plot_save_dir, exist_ok=True)
        os.makedirs(
            self.score_label_plot_save_dir, exist_ok=True
        )  # Create score label plot dir

        self.decision_scores_ = None
        self.per_feature_artifacts_ = {}

        # Setup logging
        # Create a logs directory if it doesn't exist, within the main plot_save_dir
        self.log_dir = os.path.join(self.plot_save_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        log_file_path = os.path.join(self.log_dir, "mad_run.log")
        with open(log_file_path, "a") as f_log_spacer:
            for _ in range(2):
                f_log_spacer.write("\n")

        logging.basicConfig(
            level=logging.ERROR,  # Root logger level
            format="%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set level for this specific logger

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file_path)
            # You can use the same formatter or a different one for the file
            # For file logs, a more standard, non-pretty format might be better for parsing
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(
                logging.DEBUG
            )  # Set level for file handler separately if needed
            self.logger.addHandler(file_handler)

        self.logger.propagate = False

        self.X_train_fit_data = None
        self.y_train_fit_labels = None
        self.logger.debug("""MAD_0530 Initialized with HPs: %s""", HP)
        # train_data_len_for_hint is now implicitly self.X_train_fit_data.shape[0] if it exists
        self.per_feature_interest_ranges_ = (
            {}
        )  # Stores (start, end) for each feature's training data interest
        self.dynamic_token_limit_overrides = {}  # For dynamic quota adjustment

    def _initialize_client_for_attempt(self, api_key_index_to_use):
        """
        Initializes and returns a Gemini client for a specific API key index.
        Uses the utility function from gemini_api_utils.
        """
        if not (0 <= api_key_index_to_use < len(self.api_keys)):
            self.logger.error(
                f"API key index {api_key_index_to_use} is out of bounds. Cannot initialize client."
            )
            return None

        key_to_use = self.api_keys[api_key_index_to_use]
        client = gemini_api_utils_0530.initialize_gemini_client(key_to_use, self.logger)
        self.logger.debug(
            f"Client initialized with key index {api_key_index_to_use}: {'Success' if client else 'Failed'}"
        )
        return client

    def _run_llm_pipeline(self, X_data):
        pipeline_start_time = time.time()
        if (
            not isinstance(X_data, np.ndarray)
            or X_data.ndim != 2
            or X_data.shape[0] == 0
        ):
            raise ValueError("Invalid input data X_data. Must be 2D NumPy array.")
        if not self.api_keys:
            self.logger.error("No API keys configured.")
            return np.zeros(X_data.shape[0], dtype=float)

        _log_msg = f"Starting LLM pipeline. X_data shape: {X_data.shape}. n_features: {X_data.shape[1]}"
        self.logger.info(_log_msg)

        n_samples, n_features = X_data.shape
        final_scores_all_features = np.zeros(n_samples, dtype=float)
        self.per_feature_artifacts_ = {}
        timestamp_str = time.strftime("%Y%m%d%H%M%S")
        self.last_run_timestamp = timestamp_str
        run_folder_name_candidate = (
            self.dataset_name_for_artifacts or f"run_{self.last_run_timestamp}"
        )
        self.current_run_artifact_folder_name = run_folder_name_candidate
        prospective_final_run_dir = os.path.join(
            self.plot_save_dir, self.current_run_artifact_folder_name
        )
        if os.path.exists(prospective_final_run_dir):
            self.current_run_artifact_folder_name = (
                f"{run_folder_name_candidate}_{self.last_run_timestamp}"
            )
        temp_artifact_base_dir = os.path.join(
            self.plot_save_dir, f"temp_artifacts_{timestamp_str}"
        )
        os.makedirs(temp_artifact_base_dir, exist_ok=True)

        for i_feat in tqdm(range(n_features), desc="Processing Features (LLM)"):
            feature_loop_start_time = time.time()
            _log_msg = f"Processing feature {i_feat}/{n_features-1}"
            self.logger.info(_log_msg)

            feature_artifact_dir = os.path.join(
                temp_artifact_base_dir, f"feature_{i_feat}"
            )
            os.makedirs(feature_artifact_dir, exist_ok=True)
            self.logger.info(
                f"Feature {i_feat}: Artifacts will be stored in: {feature_artifact_dir}"
            )

            # Initialize feature-level results
            feature_anomaly_scores_from_llm = np.zeros(
                n_samples, dtype=float
            )  # For main anomaly scores
            unpredictability_scores_for_feature = np.zeros(
                n_samples, dtype=float
            )  # For unpredictability scores
            unpredictability_plot_uri_for_feature = None
            unpredictability_step_succeeded_for_feature = False
            llm_main_step_succeeded_for_feature = False  # For main anomaly scoring step
            last_exception_for_feature_unpred_step = None
            last_exception_for_feature_main_step = None

            if i_feat not in self.per_feature_artifacts_:
                self.per_feature_artifacts_[i_feat] = {}

            current_feature_analysis_data = X_data[:, i_feat]
            # Convert all analysis data for the feature to the list of dicts format once
            all_analysis_data_snippets_for_this_feature_list_of_dicts = []
            for idx_snip_feat_prep in range(len(current_feature_analysis_data)):
                all_analysis_data_snippets_for_this_feature_list_of_dicts.append(
                    {
                        "index": idx_snip_feat_prep,
                        "value": float(
                            f"{current_feature_analysis_data[idx_snip_feat_prep]:.3g}"
                        ),
                    }
                )

            # --- Prepare Training Snippets (once per feature, before any LLM stage) ---
            identified_train_interest_range = None  # Initialize before API key loop
            last_exception_for_interest_id_overall = (
                None  # Store last error if all keys fail
            )
            # Initialize interest_identification artifacts to a default error state or empty
            self.per_feature_artifacts_[i_feat]["interest_identification"] = {
                "status": "Not run or failed for all keys"
            }

            if (
                self.X_train_fit_data is not None
                and self.y_train_fit_labels is not None
                and i_feat < self.X_train_fit_data.shape[1]
                and self.X_train_fit_data.shape[0] > 0
            ):
                interest_id_artifact_sub_dir = os.path.join(
                    feature_artifact_dir, "interest_identification"
                )
                os.makedirs(interest_id_artifact_sub_dir, exist_ok=True)

                for interest_id_api_key_attempt_num in range(len(self.api_keys)):
                    if identified_train_interest_range:  # If already found, break
                        break

                    current_key_for_interest_id_idx = interest_id_api_key_attempt_num
                    self.logger.info(
                        f"Ft {i_feat} Interest ID Stage: Attempting with API Key Index {current_key_for_interest_id_idx}"
                    )

                    interest_id_client = self._initialize_client_for_attempt(
                        current_key_for_interest_id_idx
                    )

                    if not interest_id_client:
                        self.logger.error(
                            f"Ft {i_feat} Interest ID Stage: Failed to initialize client for API key index {current_key_for_interest_id_idx}. Skipping to next key."
                        )
                        last_exception_for_interest_id_overall = RuntimeError(
                            f"Interest ID client init failed for key index {current_key_for_interest_id_idx}"
                        )
                        self.per_feature_artifacts_[i_feat][
                            "interest_identification"
                        ] = {
                            "status": "Failed",
                            "error": f"Client init failed for key {current_key_for_interest_id_idx}",
                            "last_recorded_error_overall": mad_utils_0530._format_exception_for_logging(
                                last_exception_for_interest_id_overall
                            ),
                        }
                        continue  # To the next key in the loop

                    # Calculation of calculated_target_range_size_for_interest_id (needs client)
                    calculated_target_range_size_for_interest_id = 200  # Default
                    if (
                        self.X_train_fit_data is not None
                        and self.X_train_fit_data.shape[0] > 0
                    ):
                        _empty_snippets_for_interest_id_estimation = (
                            mad_utils_0530.convert_snippet_list_to_final_json(
                                [], True, self.logger
                            )
                        )
                        _empty_snippets_json_str_for_interest_id_estimation = (
                            json.dumps(_empty_snippets_for_interest_id_estimation)
                        )
                        _base_prompt_interest_id_struct = prompt_utils_0530.construct_interest_id_prompt(
                            i_feat=i_feat,
                            num_train_samples=self.X_train_fit_data.shape[0],
                            training_snippets_json_str_for_id_step=_empty_snippets_json_str_for_interest_id_estimation,
                            target_range_size=1,
                        )
                        base_prompt_tokens_for_interest_id_step = (
                            gemini_api_utils_0530.count_gemini_tokens(
                                interest_id_client,
                                FIXED_MODEL_NAME_GEMINI_FLASH,
                                [_base_prompt_interest_id_struct],
                                self.logger,
                            ).total_tokens
                        )
                        model_info_interest_id = (
                            gemini_api_utils_0530.get_gemini_model_info(
                                interest_id_client,
                                DEFAULT_OUTPUT_TOKENS_INTEREST_ID,
                                self.dynamic_token_limit_overrides.get(
                                    current_key_for_interest_id_idx
                                ),
                                self.logger,
                            )
                        )
                        input_limit_for_interest_id_model = model_info_interest_id.get(
                            "input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT
                        )
                        output_limit_for_interest_id_model = model_info_interest_id.get(
                            "output_token_limit", DEFAULT_OUTPUT_TOKENS_INTEREST_ID
                        )
                        available_for_snippets_interest_id = (
                            (
                                input_limit_for_interest_id_model
                                * TOKEN_LIMIT_SAFETY_FACTOR
                            )
                            - base_prompt_tokens_for_interest_id_step
                            - output_limit_for_interest_id_model
                            - THINKING_BUDGET_INTEREST_ID
                            - TOKEN_BUDGET_SAFETY_MARGIN_INTEREST_ID
                        )
                        if available_for_snippets_interest_id > 10:
                            all_training_snippets_for_id_step_list_of_dicts_local = []
                            current_train_feature_data = self.X_train_fit_data[
                                :, i_feat
                            ]
                            current_train_labels = self.y_train_fit_labels
                            for idx_local_prep in range(self.X_train_fit_data.shape[0]):
                                all_training_snippets_for_id_step_list_of_dicts_local.append(
                                    {
                                        "index": idx_local_prep,
                                        "value": float(
                                            f"{current_train_feature_data[idx_local_prep]:.3g}"
                                        ),
                                        "label": int(
                                            current_train_labels[idx_local_prep]
                                        ),
                                    }
                                )
                            anomalous_snippets_id_local = [
                                s
                                for s in all_training_snippets_for_id_step_list_of_dicts_local
                                if s.get("label") == 1
                            ]
                            normal_snippets_id_local = [
                                s
                                for s in all_training_snippets_for_id_step_list_of_dicts_local
                                if s.get("label") != 1
                            ]
                            centered_normal_snippets_id_local = (
                                mad_utils_0530.prepare_centered_list(
                                    normal_snippets_id_local, logger=self.logger
                                )
                            )
                            prioritized_for_target_range_calc = (
                                anomalous_snippets_id_local
                                + centered_normal_snippets_id_local
                            )
                            if prioritized_for_target_range_calc:
                                budget_for_target_range_calc_snippets = (
                                    available_for_snippets_interest_id / 2
                                )
                                fittable_snippets_for_range_calc = mad_utils_0530.fill_snippets_by_token_budget(
                                    client=interest_id_client,
                                    model_name_for_counting=FIXED_MODEL_NAME_GEMINI_FLASH,
                                    prioritized_snippets_list=prioritized_for_target_range_calc,
                                    available_tokens_for_snippets_content=budget_for_target_range_calc_snippets,
                                    json_wrapper_template_func=lambda sl: json.dumps(
                                        mad_utils_0530.convert_snippet_list_to_final_json(
                                            sl, True, self.logger
                                        )
                                    ),
                                    logger=self.logger,
                                    context_log_prefix=f"Ft {i_feat} InterestIDTargetRangeCalc: ",
                                )
                                calculated_target_range_size_for_interest_id = len(
                                    fittable_snippets_for_range_calc
                                )
                            else:
                                calculated_target_range_size_for_interest_id = 50
                        else:
                            calculated_target_range_size_for_interest_id = 50
                        if (
                            calculated_target_range_size_for_interest_id == 0
                            and self.X_train_fit_data.shape[0] > 0
                        ):
                            calculated_target_range_size_for_interest_id = 1
                        self.logger.info(
                            f"Ft {i_feat} Interest ID (Key {current_key_for_interest_id_idx}): Calculated target_range_size = {calculated_target_range_size_for_interest_id} (Avail for snips: {available_for_snippets_interest_id:.0f}, Base prompt: {base_prompt_tokens_for_interest_id_step})"
                        )

                    current_key_token_override_interest_id = (
                        self.dynamic_token_limit_overrides.get(
                            current_key_for_interest_id_idx
                        )
                    )

                    (
                        parsed_range_from_id_this_key,
                        artifacts_from_id_step_this_key,
                        _,
                        key_fatal_this_key,
                        learned_quota_this_key,
                    ) = perform_interest_identification_step(
                        feature_data_train=self.X_train_fit_data[:, i_feat],
                        feature_labels_train=self.y_train_fit_labels,
                        i_feat=i_feat,
                        n_train_samples=self.X_train_fit_data.shape[0],
                        temp_artifact_base_dir_for_step=interest_id_artifact_sub_dir,
                        target_range_size=calculated_target_range_size_for_interest_id,
                        logger=self.logger,
                        api_keys_list=[
                            self.api_keys[current_key_for_interest_id_idx]
                        ],  # Pass only current key
                        MAX_RETRIES_WITH_DELAY_PER_KEY_const=self.max_retries_with_delay_per_key,
                        DEFAULT_RETRY_DELAY_SECONDS_const=self.default_retry_delay_seconds,
                        full_feature_data_for_plot=X_data[:, i_feat],
                        actual_train_len_for_plot=self.X_train_fit_data.shape[0],
                        primary_client_for_budgeting=interest_id_client,  # Pass current client
                        input_token_limit_override=current_key_token_override_interest_id,
                    )

                    # Store artifacts from this attempt, potentially overwriting previous failed ones
                    self.per_feature_artifacts_[i_feat][
                        "interest_identification"
                    ] = artifacts_from_id_step_this_key

                    if learned_quota_this_key is not None:
                        self.logger.info(
                            f"Feature {i_feat}, Key {current_key_for_interest_id_idx}: Updating dynamic token limit from Interest ID step to {learned_quota_this_key}."
                        )
                        self.dynamic_token_limit_overrides[
                            current_key_for_interest_id_idx
                        ] = learned_quota_this_key

                    if parsed_range_from_id_this_key:
                        identified_train_interest_range = parsed_range_from_id_this_key
                        self.logger.info(
                            f"Ft {i_feat} Interest ID Stage: Success with Key Index {current_key_for_interest_id_idx}. Range: {identified_train_interest_range}"
                        )
                        # Success, already broke from inner perform_interest_identification_step retries for this key. Now break API key loop.
                        break
                    else:  # parsed_range_from_id_this_key is None, this key attempt failed.
                        # Extract last error from this key's attempt for overall logging
                        if (
                            artifacts_from_id_step_this_key
                            and "error_llm_call_interest_id"
                            in artifacts_from_id_step_this_key
                        ):
                            last_exception_for_interest_id_overall = (
                                artifacts_from_id_step_this_key[
                                    "error_llm_call_interest_id"
                                ]
                            )
                        elif (
                            artifacts_from_id_step_this_key
                            and "error_plotting" in artifacts_from_id_step_this_key
                        ):
                            last_exception_for_interest_id_overall = (
                                artifacts_from_id_step_this_key["error_plotting"]
                            )
                        else:
                            last_exception_for_interest_id_overall = RuntimeError(
                                f"Interest ID failed for key {current_key_for_interest_id_idx} with no specific error in artifacts."
                            )
                        self.logger.warning(
                            f"Ft {i_feat} Interest ID Stage: Failed with Key Index {current_key_for_interest_id_idx}. Last error for this key: {mad_utils_0530._format_exception_for_logging(last_exception_for_interest_id_overall)}"
                        )

                    if key_fatal_this_key:
                        self.logger.warning(
                            f"Feature {i_feat} Interest ID Stage: API key {current_key_for_interest_id_idx} failed fatally. Continuing to next key if any."
                        )
                        # Loop will continue to the next key automatically

            # After all API keys for Interest ID stage
            if not identified_train_interest_range:
                log_msg_fail_all_keys = f"Ft {i_feat} Interest ID Stage: FAILED for all API keys. Last overall error: {mad_utils_0530._format_exception_for_logging(last_exception_for_interest_id_overall)}"
                self.logger.error(log_msg_fail_all_keys)
                # Ensure artifacts reflect the overall failure if it occurred
                self.per_feature_artifacts_[i_feat]["interest_identification"] = {
                    "status": "Failed for all keys",
                    "error": "Interest ID failed for all API keys.",
                    "last_recorded_error_overall": mad_utils_0530._format_exception_for_logging(
                        last_exception_for_interest_id_overall
                    ),
                }
            elif (
                self.per_feature_artifacts_[i_feat]
                .get("interest_identification", {})
                .get("status")
                != "Failed for all keys"
            ):
                # If successful, ensure status reflects it, especially if a previous key failed and set it to error
                if (
                    "parsed_interest_range"
                    not in self.per_feature_artifacts_[i_feat][
                        "interest_identification"
                    ]
                ):
                    # If success was determined by identified_train_interest_range but artifacts were not updated by perform_step directly with range
                    # This is a safeguard. perform_interest_identification_step should populate its own artifacts correctly on success.
                    self.per_feature_artifacts_[i_feat]["interest_identification"][
                        "parsed_interest_range"
                    ] = identified_train_interest_range
                self.per_feature_artifacts_[i_feat]["interest_identification"][
                    "status"
                ] = "Success"

            all_training_example_snippets_for_feature = (
                mad_utils_0530.prepare_training_snippets_for_main_step(
                    X_train_fit_data=self.X_train_fit_data,
                    y_train_fit_labels=self.y_train_fit_labels,
                    i_feat=i_feat,
                    identified_train_interest_range=identified_train_interest_range,
                    logger=self.logger,
                )
            )
            # --- End Prepare Training Snippets ---

            # === STAGE 1: Unpredictability Score Generation ===
            self.logger.info(f"Ft {i_feat}: Calling STAGE 1: Unpredictability Score Generation function.")
            unpred_feature_artifact_dir = os.path.join(feature_artifact_dir, "unpredictability_generation")
            os.makedirs(unpred_feature_artifact_dir, exist_ok=True)

            (
                unpredictability_scores_for_feature,
                unpredictability_plot_uri_for_feature,
                unpredictability_step_succeeded_for_feature,
                last_exception_for_feature_unpred_step,
                temp_unpred_batch_artifacts_collection
            ) = perform_unpredictability_step(
                X_data_col_for_feature=X_data[:, i_feat],
                i_feat=i_feat,
                n_samples=n_samples,
                all_analysis_data_snippets_list=all_analysis_data_snippets_for_this_feature_list_of_dicts,
                all_training_example_snippets_list=all_training_example_snippets_for_feature,
                identified_train_interest_range=identified_train_interest_range,
                api_keys_list=self.api_keys,
                dynamic_token_limit_overrides=self.dynamic_token_limit_overrides,
                logger=self.logger,
                feature_artifact_dir_for_unpred=unpred_feature_artifact_dir,
                max_retries_with_delay_per_key_const=self.max_retries_with_delay_per_key,
                default_retry_delay_seconds_const=self.default_retry_delay_seconds,
                X_train_fit_data=self.X_train_fit_data,
                y_train_fit_labels=self.y_train_fit_labels,
            )
            
            # --- Log before artifact assignment ---
            self.logger.info(f"Ft {i_feat}: Returned from perform_unpredictability_step. Succeeded: {unpredictability_step_succeeded_for_feature}")

            try:
                self.per_feature_artifacts_[i_feat]["unpredictability_generation_artifacts"] = {
                    "succeeded_overall": unpredictability_step_succeeded_for_feature,
                    "final_unpredictability_scores_array": unpredictability_scores_for_feature.tolist(),
                    "final_feature_unpredictability_plot_uri": unpredictability_plot_uri_for_feature,
                    "batch_run_details": temp_unpred_batch_artifacts_collection,
                    "last_recorded_error_overall": (
                        mad_utils_0530._format_exception_for_logging(last_exception_for_feature_unpred_step)
                        if last_exception_for_feature_unpred_step and not unpredictability_step_succeeded_for_feature
                        else None
                    )
                }
                # --- Log after artifact assignment ---
                self.logger.info(f"Ft {i_feat}: Successfully updated per_feature_artifacts_ with unpredictability results.")

            except Exception as e_artifact_assign:
                self.logger.error(f"Ft {i_feat}: EXCEPTION during unpredictability artifact assignment: {mad_utils_0530._format_exception_for_logging(e_artifact_assign)}")
                # Decide how to handle this - e.g., mark feature as failed and continue, or re-raise to stop processing.
                # For now, let's log and allow the flow to continue to see if it reaches stage 2 logging.
                # If this exception is the cause, stage 2 logs will still be missing, but we'll see this error.

            if not unpredictability_step_succeeded_for_feature:
                 self.logger.error(f"Ft {i_feat}: STAGE 1 (Unpredictability Score Generation) FAILED. Last error: {mad_utils_0530._format_exception_for_logging(last_exception_for_feature_unpred_step)}")
            else:
                self.logger.info(f"Ft {i_feat}: STAGE 1 (Unpredictability Score Generation) SUCCEEDED.")

            # --- STAGE 2: Main Anomaly Scoring for the entire feature ---
            self.logger.info(f"Ft {i_feat}: Starting STAGE 2: Main Anomaly Scoring.")
            current_offset_for_analysis_snippets_main = len(self.X_train_fit_data)
            feature_batch_outputs_llm_main = (
                []
            )  # For artifacts from successful main LLM batches

            with tqdm(
                total=len(all_analysis_data_snippets_for_this_feature_list_of_dicts),
                desc=f"Ft {i_feat} Main Anomaly Snippets",
                unit="snippet",
                leave=False,
            ) as main_anomaly_pbar:
                for main_api_key_attempt_num in range(len(self.api_keys)):
                    if llm_main_step_succeeded_for_feature:
                        break

                    self.current_api_key_index = main_api_key_attempt_num
                    self.logger.info(
                        f"Ft {i_feat} Main Stage: Attempting with API Key Index {self.current_api_key_index}"
                    )

                    main_llm_client = self._initialize_client_for_attempt(
                        self.current_api_key_index
                    )
                    if not main_llm_client:
                        self.logger.error(
                            f"Ft {i_feat} Main Stage: Failed to initialize client for API key index {self.current_api_key_index}. Skipping to next key."
                        )
                        last_exception_for_feature_main_step = RuntimeError(
                            f"Main LLM client init failed for key index {self.current_api_key_index}"
                        )
                        continue

                    main_batch_num_this_key = 0
                    initial_offset_for_this_main_key_attempt = (
                        current_offset_for_analysis_snippets_main
                    )

                    while current_offset_for_analysis_snippets_main < len(
                        all_analysis_data_snippets_for_this_feature_list_of_dicts
                    ):
                        main_batch_num_this_key += 1
                        self.logger.info(
                            f"Ft {i_feat}, Main KeyIdx {self.current_api_key_index}, MainBatch {main_batch_num_this_key} (Offset {current_offset_for_analysis_snippets_main}): Starting."
                        )

                        candidate_analysis_snippets_for_main_batch_list = (
                            all_analysis_data_snippets_for_this_feature_list_of_dicts[
                                current_offset_for_analysis_snippets_main:
                            ]
                        )

                        try:
                            # --- Calculate characteristic length k for this main batch/feature ---
                            characteristic_k_for_main_llm = None
                            data_for_k_calc_main = (
                                X_data[:, i_feat]
                                if (
                                    X_data is not None
                                    and X_data.ndim == 2
                                    and X_data.shape[1] > i_feat
                                    and len(X_data[:, i_feat]) > 0
                                )
                                else None
                            )
                            if data_for_k_calc_main is not None:
                                try:
                                    characteristic_k_for_main_llm = find_length_rank(
                                        data_for_k_calc_main, rank=1
                                    )
                                    if characteristic_k_for_main_llm is not None:
                                        self.logger.info(
                                            f"Ft {i_feat}, MainB{main_batch_num_this_key}: Calculated characteristic_length_k = {characteristic_k_for_main_llm}"
                                        )
                                except (
                                    Exception
                                ) as e_k_calc_main:  # Be specific about exception if possible
                                    self.logger.warning(
                                        f"Ft {i_feat}, MainB{main_batch_num_this_key}: Error calculating characteristic_length_k: {mad_utils_0530._format_exception_for_logging(e_k_calc_main)}. Proceeding without it."
                                    )
                                    characteristic_k_for_main_llm = (
                                        None  # Ensure it's None if calc fails
                                    )
                            else:
                                self.logger.info(
                                    f"Ft {i_feat}, MainB{main_batch_num_this_key}: No suitable data for characteristic_length_k calculation."
                                )
                            # --- End k calculation ---

                            # Token Budgeting for Main LLM Call (get_gemini_model_info, then budget for snippets)
                            main_model_info = (
                                gemini_api_utils_0530.get_gemini_model_info(
                                    main_llm_client,
                                    65536,
                                    self.dynamic_token_limit_overrides.get(
                                        self.current_api_key_index
                                    ),
                                    self.logger,
                                )
                            )
                            main_input_token_limit = main_model_info.get(
                                "input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT
                            )
                            main_qualified_model_name = main_model_info.get(
                                "qualified_model_name", FIXED_MODEL_NAME_GEMINI_FLASH
                            )

                            # Simplified base prompt token calculation for main LLM
                            _base_prompt_main_struct_for_tokens = prompt_utils_0530.construct_llm_batch_prompt(
                                i_feat=i_feat,
                                n_samples=n_samples,
                                current_batch_training_snippets_map={},
                                current_batch_analysis_snippets_map={},
                                training_data_indices_info_str="",
                                analysis_data_indices_info_str="",
                                X_train_fit_data_param=self.X_train_fit_data,
                                y_train_fit_labels_param=self.y_train_fit_labels,
                                image_uri_for_llm=None,  # No image for base token count
                                is_refinement_attempt=False,
                                previous_code_str=None,
                                previous_execution_error_details=None,
                                generate_func_with_data_param=False,  # For base token count
                                unpredictability_plot_uri=unpredictability_plot_uri_for_feature,  # Pass feature-level unpred plot URI
                                k=characteristic_k_for_main_llm,  # Pass k
                            )
                            base_tokens_main = (
                                gemini_api_utils_0530.count_gemini_tokens(
                                    main_llm_client,
                                    main_qualified_model_name,
                                    [_base_prompt_main_struct_for_tokens],
                                    self.logger,
                                ).total_tokens
                            )

                            # Budget for training snippets (e.g., 50% of available after base prompt)
                            avail_tokens_total_for_snippets_main = (
                                main_input_token_limit * TOKEN_LIMIT_SAFETY_FACTOR
                            ) - base_tokens_main
                            tokens_for_main_train_budget = (
                                avail_tokens_total_for_snippets_main * 0.5
                            )  # Example ratio

                            current_batch_main_training_snippets_list_for_prompt = []
                            if (
                                all_training_example_snippets_for_feature
                                and tokens_for_main_train_budget > 10
                            ):
                                current_batch_main_training_snippets_list_for_prompt = mad_utils_0530.fill_snippets_by_token_budget(
                                    main_llm_client,
                                    main_qualified_model_name,
                                    all_training_example_snippets_for_feature,
                                    tokens_for_main_train_budget,
                                    lambda snips: json.dumps(
                                        mad_utils_0530.convert_snippet_list_to_final_json(
                                            snips, True, self.logger
                                        )
                                    ),
                                    self.logger,
                                    context_log_prefix=f"Ft {i_feat} MainTrainFill: ",
                                )
                            current_batch_main_training_snippets_final_dict_for_prompt = mad_utils_0530.convert_snippet_list_to_final_json(
                                current_batch_main_training_snippets_list_for_prompt,
                                True,
                                self.logger,
                            )
                            actual_train_tokens_main = 0
                            if current_batch_main_training_snippets_list_for_prompt:
                                actual_train_tokens_main = gemini_api_utils_0530.count_gemini_tokens(
                                    main_llm_client,
                                    main_qualified_model_name,
                                    [
                                        json.dumps(
                                            current_batch_main_training_snippets_final_dict_for_prompt
                                        )
                                    ],
                                    self.logger,
                                ).total_tokens

                            tokens_for_main_analysis_budget = (
                                avail_tokens_total_for_snippets_main
                                - actual_train_tokens_main
                            )
                            if tokens_for_main_analysis_budget < 0:
                                tokens_for_main_analysis_budget = 0

                            current_batch_main_analysis_snippets_list_for_prompt = (
                                mad_utils_0530.prepare_analysis_snippets_for_batch(
                                    candidate_analysis_snippets_for_main_batch_list,
                                    main_llm_client,
                                    main_qualified_model_name,
                                    tokens_for_main_analysis_budget,
                                    self.logger,
                                )
                            )

                            if not current_batch_main_analysis_snippets_list_for_prompt:
                                self.logger.warning(
                                    f"Ft {i_feat}, MainB{main_batch_num_this_key}: No analysis snippets for main LLM after budgeting. Skipping batch for this key."
                                )
                                break

                            current_batch_main_analysis_snippets_final_dict_for_prompt = mad_utils_0530.convert_snippet_list_to_final_json(
                                current_batch_main_analysis_snippets_list_for_prompt,
                                False,
                                self.logger,
                            )
                            # Populate with full feature unpredictability scores
                            if (
                                unpredictability_step_succeeded_for_feature
                            ):  # Check if unpred scores are available
                                temp_scores_for_main_prompt_population = {}
                                for (
                                    snippet_dict_main_populate
                                ) in (
                                    current_batch_main_analysis_snippets_list_for_prompt
                                ):
                                    snippet_idx_main_populate = (
                                        snippet_dict_main_populate.get("index")
                                    )
                                    if (
                                        snippet_idx_main_populate is not None
                                        and 0 <= snippet_idx_main_populate < n_samples
                                    ):
                                        score_val_main_populate = (
                                            unpredictability_scores_for_feature[
                                                snippet_idx_main_populate
                                            ]
                                        )
                                        temp_scores_for_main_prompt_population[
                                            str(snippet_idx_main_populate)
                                        ] = round(float(score_val_main_populate), 4)
                                current_batch_main_analysis_snippets_final_dict_for_prompt[
                                    "unpredictability_score"
                                ] = temp_scores_for_main_prompt_population
                            else:
                                current_batch_main_analysis_snippets_final_dict_for_prompt[
                                    "unpredictability_score"
                                ] = {}

                            analysis_indices_str_main_prompt = (
                                f"Indices {current_batch_main_analysis_snippets_list_for_prompt[0]['index']} to {current_batch_main_analysis_snippets_list_for_prompt[-1]['index']}"
                                if current_batch_main_analysis_snippets_list_for_prompt
                                else "N/A"
                            )
                            training_indices_info_str_main_prompt = (
                                f"Selected from training (focus: {identified_train_interest_range})"
                                if identified_train_interest_range
                                else "Selected from training"
                            )

                            main_batch_extras_save_dir = os.path.join(
                                feature_artifact_dir,
                                f"main_batch_{main_batch_num_this_key}",
                            )
                            os.makedirs(main_batch_extras_save_dir, exist_ok=True)

                            (
                                batch_successful_main,
                                batch_scores_array_main,
                                batch_run_artifacts_main,
                                batch_exception_main,
                                num_analysis_snippets_processed_main,
                                learned_quota_main,
                            ) = process_single_batch_with_llm(  # Call existing batch processor
                                current_attempt_client=main_llm_client,
                                i_feat=i_feat,
                                n_samples=n_samples,
                                X_data_col=X_data[:, i_feat],
                                current_batch_collected_training_snippets_list=current_batch_main_training_snippets_list_for_prompt,
                                current_batch_training_snippets_final_dict=current_batch_main_training_snippets_final_dict_for_prompt,
                                current_batch_collected_analysis_snippets_list=current_batch_main_analysis_snippets_list_for_prompt,
                                current_batch_analysis_snippets_final_dict=current_batch_main_analysis_snippets_final_dict_for_prompt,
                                identified_train_interest_range=identified_train_interest_range,
                                X_train_fit_data=self.X_train_fit_data,
                                y_train_fit_labels=self.y_train_fit_labels,
                                training_data_indices_info_str=training_indices_info_str_main_prompt,
                                analysis_data_indices_info_str=analysis_indices_str_main_prompt,
                                batch_num=main_batch_num_this_key,
                                batch_extras_save_dir=main_batch_extras_save_dir,
                                logger=self.logger,
                                current_api_key_index=self.current_api_key_index,
                                max_retries_with_delay_per_key=self.max_retries_with_delay_per_key,
                                default_retry_delay_seconds=self.default_retry_delay_seconds,
                                input_token_limit_override=self.dynamic_token_limit_overrides.get(
                                    self.current_api_key_index
                                ),
                                unpredictability_plot_input_uri=unpredictability_plot_uri_for_feature,
                                k=characteristic_k_for_main_llm,
                            )
                            last_exception_for_feature_main_step = batch_exception_main

                            if (
                                batch_successful_main
                                and batch_scores_array_main is not None
                            ):
                                feature_anomaly_scores_from_llm = np.maximum(
                                    feature_anomaly_scores_from_llm,
                                    batch_scores_array_main,
                                )
                                if batch_run_artifacts_main:
                                    feature_batch_outputs_llm_main.append(
                                        batch_run_artifacts_main
                                    )

                                main_anomaly_pbar.update(
                                    num_analysis_snippets_processed_main
                                )
                                current_offset_for_analysis_snippets_main += (
                                    num_analysis_snippets_processed_main
                                )
                                self.logger.info(
                                    f"Ft {i_feat}, MainB{main_batch_num_this_key}, Key {self.current_api_key_index}: OK. Processed {num_analysis_snippets_processed_main}. Offset now {current_offset_for_analysis_snippets_main}"
                                )

                                if current_offset_for_analysis_snippets_main >= len(
                                    all_analysis_data_snippets_for_this_feature_list_of_dicts
                                ):
                                    llm_main_step_succeeded_for_feature = True
                                    self.logger.info(
                                        f"Ft {i_feat} Main Stage: All snippets processed with Key {self.current_api_key_index}."
                                    )
                                    break
                            else:
                                self.logger.error(
                                    f"Ft {i_feat}, MainB{main_batch_num_this_key} FAILED for Key {self.current_api_key_index}. Last batch error: {mad_utils_0530._format_exception_for_logging(batch_exception_main)}"
                                )
                                if learned_quota_main is not None:
                                    self.logger.info(
                                        f"Ft {i_feat}, Main Stage Key {self.current_api_key_index}: Updating dynamic token limit from failed MainBatch {main_batch_num_this_key} to {learned_quota_main}."
                                    )
                                    self.dynamic_token_limit_overrides[
                                        self.current_api_key_index
                                    ] = learned_quota_main
                                current_offset_for_analysis_snippets_main = (
                                    initial_offset_for_this_main_key_attempt
                                )
                                break
                        except Exception as e_main_batch_outer:
                            self.logger.error(
                                f"Ft {i_feat}, MainB{main_batch_num_this_key}, Key {self.current_api_key_index}: Outer exception in main batch loop: {mad_utils_0530._format_exception_for_logging(e_main_batch_outer)}"
                            )
                            last_exception_for_feature_main_step = e_main_batch_outer
                            current_offset_for_analysis_snippets_main = (
                                initial_offset_for_this_main_key_attempt
                            )
                            break

                    if llm_main_step_succeeded_for_feature:
                        self.logger.info(
                            f"Ft {i_feat} Main Stage, KeyIdx {self.current_api_key_index}: Main anomaly scoring completed with this key."
                        )
                        break
                    else:
                        self.logger.warning(
                            f"Ft {i_feat} Main Stage: API Key Index {self.current_api_key_index} did not complete all main snippets. Last error: {mad_utils_0530._format_exception_for_logging(last_exception_for_feature_main_step)}"
                        )

            if not llm_main_step_succeeded_for_feature:
                self.logger.error(
                    f"Ft {i_feat}: STAGE 2 (Main Anomaly Scoring) FAILED after all keys. Last error: {mad_utils_0530._format_exception_for_logging(last_exception_for_feature_main_step)}"
                )
            else:
                self.logger.info(
                    f"Ft {i_feat}: STAGE 2 (Main Anomaly Scoring) SUCCEEDED for the feature."
                )
                final_scores_all_features = np.maximum(
                    final_scores_all_features, feature_anomaly_scores_from_llm
                )  # Aggregate final scores for the dataset
            # --- END STAGE 2: Main Anomaly Scoring ---

            # --- Final Plotting and Artifacts for the Feature ---
            if llm_main_step_succeeded_for_feature:
                final_feature_plot_path = os.path.join(
                    feature_artifact_dir,
                    f"feature_{i_feat}_final_aggregated_anomalies.png",
                )
                try:
                    plotting_utils_0530.generate_feature_final_anomalies_plot(
                        X_data[:, i_feat],
                        feature_anomaly_scores_from_llm,
                        i_feat,
                        final_feature_plot_path,
                        self.logger,
                        6,
                        150,
                    )
                    if i_feat in self.per_feature_artifacts_:  # Should always be true
                        self.per_feature_artifacts_[i_feat][
                            "final_feature_plot_path"
                        ] = final_feature_plot_path
                except Exception as e_plot_final_feat:
                    self.logger.error(
                        f"Feature {i_feat}: Failed to generate final plot: {mad_utils_0530._format_exception_for_logging(e_plot_final_feat)}"
                    )

            # Update per_feature_artifacts_ with results from both stages
            self.per_feature_artifacts_[i_feat].update(
                {
                    # Unpredictability artifacts are already stored under "unpredictability_generation_artifacts"
                    "main_llm_outputs_all_successful_batches": feature_batch_outputs_llm_main,
                    "main_llm_step_success_overall": llm_main_step_succeeded_for_feature,
                    "feature_scores_main_llm_aggregated": feature_anomaly_scores_from_llm.tolist(),
                    "error_main_llm_step_last_recorded": (
                        mad_utils_0530._format_exception_for_logging(
                            last_exception_for_feature_main_step
                        )
                        if last_exception_for_feature_main_step
                        and not llm_main_step_succeeded_for_feature
                        else None
                    ),
                }
            )
            self.logger.debug(
                f"Feature {i_feat} processing took {time.time() - feature_loop_start_time:.2f}s. Main Success: {llm_main_step_succeeded_for_feature}, Unpred Success: {unpredictability_step_succeeded_for_feature}"
            )

        self.decision_scores_ = (
            final_scores_all_features  # Assign final scores for the entire dataset
        )
        len_to_overwrite = min(
            len(self.decision_scores_), len(self.y_train_fit_labels)
        )
        self.decision_scores_[:len_to_overwrite] = self.y_train_fit_labels[
            :len_to_overwrite
        ].astype(float)

        final_artifact_dir_target = os.path.join(
            self.plot_save_dir, self.current_run_artifact_folder_name
        )
        try:
            if os.path.exists(temp_artifact_base_dir):
                os.rename(temp_artifact_base_dir, final_artifact_dir_target)
                self.logger.info(
                    f"Moved temp artifacts to final feature plot run dir: {final_artifact_dir_target}"
                )
        except OSError as e_rename:
            self.logger.error(
                f"Error renaming temp artifact dir: {mad_utils_0530._format_exception_for_logging(e_rename)}."
            )

        self.logger.info(
            f"LLM Pipeline completed in {time.time() - pipeline_start_time:.2f} seconds."
        )
        return self.decision_scores_

    def fit(self, X_train, y_train=None):
        self.logger.info(
            f"Starting MAD_0530 fit. X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}"
        )
        self.logger.debug(f"y_train is provided: {y_train is not None}")

        self.X_train_fit_data = None
        self.y_train_fit_labels = None

        if y_train is not None:
            self.X_train_fit_data = X_train.copy()
            self.y_train_fit_labels = y_train.copy()
            self.logger.debug(
                f"Stored X_train (shape {self.X_train_fit_data.shape}) and y_train (len {len(self.y_train_fit_labels)}) for LLM examples and plot hinting during fit."
            )
        else:
            self.logger.debug(
                "No y_train labels provided for fit. LLM will run unsupervised on fit data, without labeled training examples."
            )
        return self

    def decision_function(self, X_test):
        self.logger.info(
            f"Starting MAD_0530 decision_function. X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}"
        )
        self.logger.debug(
            f"Using X_train_fit_data (shape: {self.X_train_fit_data.shape if self.X_train_fit_data is not None else 'None'}) and y_train_fit_labels (len: {len(self.y_train_fit_labels) if self.y_train_fit_labels is not None else 'None'}) for LLM examples."
        )
        decision_scores = self._run_llm_pipeline(X_test)
        self.logger.info(
            f"MAD_0530 decision_function completed. Scores shape: {decision_scores.shape if decision_scores is not None else 'N/A'}"
        )
        return decision_scores

    def get_per_feature_artifacts(self):
        return self.per_feature_artifacts_

    def get_last_run_timestamp(self):
        self.logger.debug(f"Returning last_run_timestamp: {self.last_run_timestamp}")
        return self.last_run_timestamp
