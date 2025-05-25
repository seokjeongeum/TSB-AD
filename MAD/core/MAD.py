import json
import logging
import os
import time

import numpy as np
from tqdm import tqdm

from MAD.core.constants import (
    API_KEYS,
    DEFAULT_FALLBACK_TOKEN_LIMIT,
    DEFAULT_RETRY_DELAY_SECONDS,
    FIXED_MODEL_NAME_GEMINI_FLASH,
    MAX_RETRIES_WITH_DELAY_PER_KEY,
    TOKEN_LIMIT_SAFETY_FACTOR,
)
from TSB_AD.models.base import BaseDetector

from . import gemini_api_utils, mad_utils, plotting_utils, prompt_utils
from .interest_identifier import perform_interest_identification_step
from .llm_batch_processor import process_single_batch_with_llm


class F_May_22(BaseDetector):
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

        self.plot_save_dir = "MAD/F_May_22_Feature_Plots"  # Main directory for all runs
        os.makedirs(self.plot_save_dir, exist_ok=True)

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
        self.logger.debug("""F_May_22 Initialized with HPs: %s""", HP)
        # train_data_len_for_hint is now implicitly self.X_train_fit_data.shape[0] if it exists
        self.per_feature_interest_ranges_ = (
            {}
        )  # Stores (start, end) for each feature's training data interest
        self.dynamic_token_limit_overrides = {} # For dynamic quota adjustment

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
        client = gemini_api_utils.initialize_gemini_client(key_to_use, self.logger)
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
        timestamp_str = time.strftime("%Y%m%d%H%M%S%f")
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
        is_processing_training_data_context = (
            self.X_train_fit_data is not None
            and X_data.shape == self.X_train_fit_data.shape
            and np.all(X_data == self.X_train_fit_data)
        )
        run_type_prefix = (
            "train_fit_run" if is_processing_training_data_context else "decision_run"
        )
        _log_msg = (
            f"Run type: {run_type_prefix}. Artifact dir: {temp_artifact_base_dir}"
        )
        self.logger.debug(_log_msg)

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

            llm_step_succeeded_for_feature = False
            feature_anomaly_scores_from_llm = np.zeros(n_samples, dtype=float)
            last_llm_exception_for_feature = None
            feature_batch_outputs_llm = []
            current_offset_for_analysis_snippets = 0

            current_feature_analysis_data = X_data[:, i_feat]
            all_analysis_data_snippets_for_this_feature = []
            for idx_snip in range(len(current_feature_analysis_data)):
                all_analysis_data_snippets_for_this_feature.append(
                    {
                        "index": idx_snip,
                        "value": float(
                            f"{current_feature_analysis_data[idx_snip]:.3g}"
                        ),
                    }
                )

            if i_feat not in self.per_feature_artifacts_:
                self.per_feature_artifacts_[i_feat] = {}

            all_training_example_snippets = []

            # Initialize tqdm for analysis snippet offset for this feature
            with tqdm(
                total=len(all_analysis_data_snippets_for_this_feature),
                desc=f"Ft {i_feat} Analysis Snippets",
                unit="snippet",
                leave=False,
            ) as offset_pbar:
                for api_key_attempt_num in range(len(self.api_keys)):
                    if llm_step_succeeded_for_feature:
                        break

                    self.current_api_key_index = api_key_attempt_num
                    _log_msg = f"Feature {i_feat}: Attempting with API Key Index {self.current_api_key_index}"
                    self.logger.info(_log_msg)

                    current_attempt_client = self._initialize_client_for_attempt(
                        self.current_api_key_index
                    )
                    if not current_attempt_client:
                        self.logger.error(
                            f"Feature {i_feat}: Failed to initialize client for API key index {self.current_api_key_index}. Skipping to next key."
                        )
                        last_llm_exception_for_feature = RuntimeError(
                            f"Client init failed for key index {self.current_api_key_index}"
                        )
                        continue

                    # --- Interest Identification and Training Snippet Prep (once per key potentially) ---
                    calculated_target_range_size = 200
                    if (
                        self.X_train_fit_data is not None
                        and self.X_train_fit_data.shape[0] > 0
                    ):
                        try:
                            # This block calculates `calculated_target_range_size` based on token limits
                            # It was a large block, ensure it's correctly summarized or fully included if critical
                            # For this refactor, we assume it correctly sets `calculated_target_range_size`
                            empty_snippets_dict = (
                                mad_utils.convert_snippet_list_to_final_json(
                                    [], True, self.logger
                                )
                            )
                            _temp_base_prompt_main_step = prompt_utils.construct_llm_batch_prompt(
                                i_feat=i_feat,
                                n_samples=n_samples,
                                current_batch_training_snippets_map=empty_snippets_dict,
                                current_batch_analysis_snippets_map=empty_snippets_dict,
                                training_data_indices_info_str="Placeholder",
                                analysis_data_indices_info_str="Placeholder",
                                is_analyzing_training_data_itself=is_processing_training_data_context,
                                X_train_fit_data_param=self.X_train_fit_data,
                                y_train_fit_labels_param=self.y_train_fit_labels,
                            )
                            base_prompt_tokens_main_step = (
                                gemini_api_utils.count_gemini_tokens(
                                    current_attempt_client,
                                    FIXED_MODEL_NAME_GEMINI_FLASH,
                                    [_temp_base_prompt_main_step],
                                    self.logger,
                                ).total_tokens
                            )

                            input_limit_main_step = gemini_api_utils.get_gemini_model_info(
                                current_attempt_client,
                                65536,
                                self.dynamic_token_limit_overrides.get(self.current_api_key_index), # Pass override
                                self.logger,
                            ).get(
                                "input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT
                            )
                            available_for_snippets_main = (
                                input_limit_main_step * TOKEN_LIMIT_SAFETY_FACTOR
                            ) - base_prompt_tokens_main_step
                            budget_for_training_snippets_main = (
                                available_for_snippets_main
                            )

                            all_train_snippets_for_calc_list = []
                            if (
                                self.X_train_fit_data is not None
                                and self.y_train_fit_labels is not None
                            ):
                                train_feature_data_calc = self.X_train_fit_data[
                                    :, i_feat
                                ]
                                train_labels_calc = self.y_train_fit_labels
                                for idx_tr in range(len(train_feature_data_calc)):
                                    all_train_snippets_for_calc_list.append(
                                        {
                                            "index": idx_tr,
                                            "value": float(
                                                f"{train_feature_data_calc[idx_tr]:.3g}"
                                            ),
                                            "label": int(train_labels_calc[idx_tr]),
                                        }
                                    )

                            if (
                                all_train_snippets_for_calc_list
                                and budget_for_training_snippets_main > 10
                            ):
                                anomalous_tr_calc = [
                                    s
                                    for s in all_train_snippets_for_calc_list
                                    if s.get("label") == 1
                                ]
                                normal_tr_calc = [
                                    s
                                    for s in all_train_snippets_for_calc_list
                                    if s.get("label") != 1
                                ]
                                centered_normal_tr_calc = (
                                    mad_utils.prepare_centered_list(
                                        normal_tr_calc, self.logger
                                    )
                                )
                                prioritized_for_calc = (
                                    anomalous_tr_calc + centered_normal_tr_calc
                                )
                                calculated_target_range_size = len(
                                    mad_utils.fill_snippets_by_token_budget(
                                        client=current_attempt_client,
                                        model_name_for_counting=FIXED_MODEL_NAME_GEMINI_FLASH,
                                        prioritized_snippets_list=prioritized_for_calc,
                                        available_tokens_for_snippets_content=budget_for_training_snippets_main
                                        / 2,  # Example division, adjust as needed
                                        json_wrapper_template_func=lambda sl: json.dumps(
                                            mad_utils.convert_snippet_list_to_final_json(
                                                sl, True, self.logger
                                            )
                                        ),
                                        logger=self.logger,
                                    )
                                )
                                if (
                                    calculated_target_range_size == 0
                                    and len(all_train_snippets_for_calc_list) > 0
                                ):
                                    calculated_target_range_size = 1
                            else:
                                fallback_len = (
                                    len(all_train_snippets_for_calc_list)
                                    if all_train_snippets_for_calc_list
                                    else 0
                                )
                                default_size = 200
                                calculated_target_range_size = (
                                    min(default_size, fallback_len)
                                    if fallback_len > 0
                                    else default_size
                                )
                                if (
                                    self.X_train_fit_data.shape[0] > 0
                                    and calculated_target_range_size == 0
                                ):
                                    calculated_target_range_size = 1
                            self.logger.info(
                                f"Feature {i_feat} (KeyIdx {self.current_api_key_index}): Calculated target_range_size for Interest ID: {calculated_target_range_size}"
                            )
                        except Exception as e_calc_range:
                            self.logger.error(
                                f"Feature {i_feat} (KeyIdx {self.current_api_key_index}): Error calculating target_range_size: {mad_utils._format_exception_for_logging(e_calc_range)}. Using default: {calculated_target_range_size}."
                            )

                    identified_train_interest_range = None
                    if (
                        self.X_train_fit_data is not None
                        and self.y_train_fit_labels is not None
                        and i_feat < self.X_train_fit_data.shape[1]
                        and self.X_train_fit_data.shape[0] > 0
                    ):
                        interest_id_artifact_sub_dir = os.path.join(
                            feature_artifact_dir,
                            f"interest_identification",
                        )
                        os.makedirs(interest_id_artifact_sub_dir, exist_ok=True)
                        current_key_token_override = self.dynamic_token_limit_overrides.get(self.current_api_key_index)
                        (
                            parsed_range,
                            artifacts_from_step,
                            _current_snippets_json_for_prompt_not_used,
                            key_fatal_in_interest_id,
                            learned_quota_from_interest_id, # Capture new return value
                        ) = perform_interest_identification_step(
                            feature_data_train=self.X_train_fit_data[:, i_feat],
                            feature_labels_train=self.y_train_fit_labels,
                            i_feat=i_feat,
                            n_train_samples=self.X_train_fit_data.shape[0],
                            temp_artifact_base_dir_for_step=interest_id_artifact_sub_dir,
                            target_range_size=calculated_target_range_size,
                            logger=self.logger,
                            api_keys_list=[self.api_keys[self.current_api_key_index]],
                            MAX_RETRIES_WITH_DELAY_PER_KEY_const=self.max_retries_with_delay_per_key,
                            DEFAULT_RETRY_DELAY_SECONDS_const=self.default_retry_delay_seconds,
                            full_feature_data_for_plot=X_data[:, i_feat],
                            actual_train_len_for_plot=(
                                self.X_train_fit_data.shape[0]
                                if self.X_train_fit_data is not None
                                else 0
                            ),
                            primary_client_for_budgeting=current_attempt_client,
                            input_token_limit_override=current_key_token_override, # Pass override
                        )
                        if parsed_range:
                            identified_train_interest_range = parsed_range
                        self.per_feature_artifacts_[i_feat][
                            f"interest_identification"
                        ] = artifacts_from_step
                        
                        if learned_quota_from_interest_id is not None:
                            self.logger.info(f"Feature {i_feat}, Key {self.current_api_key_index}: Updating dynamic token limit from Interest ID step to {learned_quota_from_interest_id}.")
                            self.dynamic_token_limit_overrides[self.current_api_key_index] = learned_quota_from_interest_id
                            # This new limit will be used by subsequent get_gemini_model_info calls for this key

                    if key_fatal_in_interest_id:
                        self.logger.error(
                            f"Feature {i_feat}: API key {self.current_api_key_index} failed fatally during Interest Identification. Skipping to next key."
                        )
                        _err_str_from_interest_id = "Key fatal in interest ID"
                        if (
                            "error_llm_call_interest_id" in artifacts_from_step
                            and artifacts_from_step["error_llm_call_interest_id"]
                        ):
                            _err_str_from_interest_id = artifacts_from_step[
                                "error_llm_call_interest_id"
                            ]
                        last_llm_exception_for_feature = RuntimeError(
                            str(_err_str_from_interest_id)
                        )  # Ensure it's an exception type
                        # Optionally mark the artifact for clarity
                        if (
                            "interest_identification"
                            in self.per_feature_artifacts_[i_feat]
                        ):
                            self.per_feature_artifacts_[i_feat][
                                "interest_identification"
                            ]["error_due_to_key_exhaustion_or_fatal"] = True
                        continue  # Skip to the next api_key_attempt_num for this feature

                    all_training_example_snippets = mad_utils.prepare_training_snippets_for_main_step(
                        X_train_fit_data=self.X_train_fit_data,
                        y_train_fit_labels=self.y_train_fit_labels,
                        i_feat=i_feat,
                        identified_train_interest_range=identified_train_interest_range,
                        logger=self.logger,
                    )
                    # --- End Interest ID and Training Snippet Prep ---

                    batch_num = 0
                    temp_batch_outputs_for_feature_this_key_attempt = []
                    initial_offset_for_this_key_attempt = (
                        current_offset_for_analysis_snippets
                    )

                    while current_offset_for_analysis_snippets < len(
                        all_analysis_data_snippets_for_this_feature
                    ):
                        batch_num += 1
                        _log_msg = f"Feature {i_feat}, KeyIdx {self.current_api_key_index}, Batch {batch_num} (Offset {current_offset_for_analysis_snippets}): Starting processing."
                        self.logger.info(_log_msg)

                        candidate_analysis_snippets_for_this_batch_call = (
                            all_analysis_data_snippets_for_this_feature[
                                current_offset_for_analysis_snippets:
                            ]
                        )
                        if not candidate_analysis_snippets_for_this_batch_call:
                            llm_step_succeeded_for_feature = (
                                current_offset_for_analysis_snippets
                                >= len(all_analysis_data_snippets_for_this_feature)
                            )
                            break

                        try:
                            model_info = gemini_api_utils.get_gemini_model_info(
                                current_attempt_client,
                                65536,
                                self.dynamic_token_limit_overrides.get(self.current_api_key_index), # Pass override
                                self.logger,
                            )
                            # This input_token_limit is now informed by any dynamic override for the current key
                            input_token_limit_for_batch_budgeting = model_info.get(
                                "input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT
                            )
                            qualified_model_name = model_info.get(
                                "qualified_model_name", FIXED_MODEL_NAME_GEMINI_FLASH
                            )

                            # --- Token Budgeting (Simplified) ---
                            _base_prompt_for_struct_count = (
                                prompt_utils.construct_llm_batch_prompt(
                                    i_feat,
                                    n_samples,
                                    {},
                                    {},
                                    "",
                                    "",
                                    is_processing_training_data_context,
                                    self.X_train_fit_data,
                                    self.y_train_fit_labels,
                                    None,
                                )
                            )
                            base_prompt_tokens = gemini_api_utils.count_gemini_tokens(
                                current_attempt_client,
                                qualified_model_name,
                                [_base_prompt_for_struct_count],
                                self.logger,
                            ).total_tokens
                            avail_tokens_snippets = (
                                input_token_limit_for_batch_budgeting * TOKEN_LIMIT_SAFETY_FACTOR # Use limit for batch
                            ) - base_prompt_tokens
                            train_budget_ratio = 0.5
                            tokens_for_train_budget = (
                                avail_tokens_snippets * train_budget_ratio
                            )

                            current_batch_collected_training_snippets_list = []
                            if (
                                all_training_example_snippets
                                and tokens_for_train_budget > 10
                            ):
                                current_batch_collected_training_snippets_list = mad_utils.fill_snippets_by_token_budget(
                                    current_attempt_client,
                                    qualified_model_name,
                                    all_training_example_snippets,
                                    tokens_for_train_budget,
                                    lambda snips: json.dumps(
                                        mad_utils.convert_snippet_list_to_final_json(
                                            snips, True, self.logger
                                        )
                                    ),
                                    self.logger,
                                )
                            current_batch_training_snippets_final_dict = (
                                mad_utils.convert_snippet_list_to_final_json(
                                    current_batch_collected_training_snippets_list,
                                    True,
                                    self.logger,
                                )
                            )

                            actual_train_tokens = 0
                            if current_batch_collected_training_snippets_list:
                                actual_train_tokens = gemini_api_utils.count_gemini_tokens(
                                    current_attempt_client,
                                    qualified_model_name,
                                    [
                                        json.dumps(
                                            current_batch_training_snippets_final_dict
                                        )
                                    ],
                                    self.logger,
                                ).total_tokens
                            tokens_for_analysis_budget = (
                                avail_tokens_snippets - actual_train_tokens
                            )
                            if tokens_for_analysis_budget < 0:
                                tokens_for_analysis_budget = 0
                            # --- End Token Budgeting ---

                            current_batch_collected_analysis_snippets_list = (
                                mad_utils.prepare_analysis_snippets_for_batch(
                                    candidate_analysis_snippets_for_this_batch_call,
                                    current_attempt_client,
                                    qualified_model_name,
                                    tokens_for_analysis_budget,
                                    self.logger,
                                )
                            )
                            current_batch_analysis_snippets_final_dict = (
                                mad_utils.convert_snippet_list_to_final_json(
                                    current_batch_collected_analysis_snippets_list,
                                    False,
                                    self.logger,
                                )
                            )

                            analysis_indices_str = (
                                f"Indices {current_batch_collected_analysis_snippets_list[0]['index']} to {current_batch_collected_analysis_snippets_list[-1]['index']}"
                                if current_batch_collected_analysis_snippets_list
                                else "N/A"
                            )
                            training_indices_info_str = (
                                f"Selected from training (focus: {identified_train_interest_range})"
                                if identified_train_interest_range
                                else "Selected from training"
                            )

                            batch_extras_save_dir_for_call = os.path.join(
                                feature_artifact_dir,
                                f"batch_{batch_num}",
                            )
                            os.makedirs(batch_extras_save_dir_for_call, exist_ok=True)

                            (
                                batch_successful,
                                batch_scores_array,  # This is (n_samples,) array or None
                                batch_run_artifacts,
                                batch_exception,
                                num_analysis_snippets_processed_by_llm,
                                learned_quota_from_batch, # Capture new return
                            ) = process_single_batch_with_llm(
                                current_attempt_client=current_attempt_client,
                                i_feat=i_feat,
                                n_samples=n_samples,
                                X_data_col=X_data[:, i_feat],
                                current_batch_collected_training_snippets_list=current_batch_collected_training_snippets_list,
                                current_batch_training_snippets_final_dict=current_batch_training_snippets_final_dict,
                                current_batch_collected_analysis_snippets_list=current_batch_collected_analysis_snippets_list,
                                current_batch_analysis_snippets_final_dict=current_batch_analysis_snippets_final_dict,
                                identified_train_interest_range=identified_train_interest_range,
                                is_processing_training_data_itself=is_processing_training_data_context,
                                X_train_fit_data=self.X_train_fit_data,
                                y_train_fit_labels=self.y_train_fit_labels,
                                training_data_indices_info_str=training_indices_info_str,
                                analysis_data_indices_info_str=analysis_indices_str,
                                batch_num=batch_num,
                                batch_extras_save_dir=batch_extras_save_dir_for_call,
                                logger=self.logger,
                                current_api_key_index=self.current_api_key_index,
                                max_retries_with_delay_per_key=self.max_retries_with_delay_per_key,
                                default_retry_delay_seconds=self.default_retry_delay_seconds,
                                input_token_limit_override=self.dynamic_token_limit_overrides.get(self.current_api_key_index), # Pass override
                            )

                            last_llm_exception_for_feature = batch_exception

                            if batch_successful and batch_scores_array is not None:
                                feature_anomaly_scores_from_llm = np.maximum(
                                    feature_anomaly_scores_from_llm, batch_scores_array
                                )
                                if batch_run_artifacts:
                                    temp_batch_outputs_for_feature_this_key_attempt.append(
                                        batch_run_artifacts
                                    )

                                # Use the actual number of snippets processed by LLM for advancing offset
                                offset_pbar.update(
                                    num_analysis_snippets_processed_by_llm
                                )
                                current_offset_for_analysis_snippets += (
                                    num_analysis_snippets_processed_by_llm
                                )
                                self.logger.info(
                                    f"Ft {i_feat}, B{batch_num}, Key {self.current_api_key_index}: OK. Processed {num_analysis_snippets_processed_by_llm} analysis snippets. Offset now {current_offset_for_analysis_snippets}"
                                )

                                if current_offset_for_analysis_snippets >= len(
                                    all_analysis_data_snippets_for_this_feature
                                ):
                                    llm_step_succeeded_for_feature = True
                                    self.logger.info(
                                        f"Ft {i_feat}: All snippets processed with Key {self.current_api_key_index}."
                                    )
                                    break  # Break from while loop (batches for this key)
                            else:  # Batch failed
                                self.logger.error(
                                    f"Ft {i_feat}, B{batch_num} (Offset {current_offset_for_analysis_snippets} from {initial_offset_for_this_key_attempt}) FAILED via processor for Key {self.current_api_key_index}. Last batch error: {mad_utils._format_exception_for_logging(batch_exception)}"
                                )
                                if learned_quota_from_batch is not None: # A quota was learned even on failure
                                    self.logger.info(f"Feature {i_feat}, Key {self.current_api_key_index}: Updating dynamic token limit from failed Batch {batch_num} to {learned_quota_from_batch}.")
                                    self.dynamic_token_limit_overrides[self.current_api_key_index] = learned_quota_from_batch
                                current_offset_for_analysis_snippets = initial_offset_for_this_key_attempt  # Reset for next key
                                break  # Break from while loop (batches for this key)

                        except Exception as e_batch_outer_refactored:
                            self.logger.error(
                                f"Ft {i_feat}, B{batch_num}, Key {self.current_api_key_index}: Outer exception in batch loop: {mad_utils._format_exception_for_logging(e_batch_outer_refactored)}"
                            )
                            last_llm_exception_for_feature = e_batch_outer_refactored
                            current_offset_for_analysis_snippets = (
                                initial_offset_for_this_key_attempt
                            )
                            break  # Break from while loop (batches for this key)

                    # After batch loop for a key
                    if llm_step_succeeded_for_feature:
                        feature_batch_outputs_llm.extend(
                            temp_batch_outputs_for_feature_this_key_attempt
                        )
                        break  # Break from API key loop, feature is done
                    elif (
                        current_offset_for_analysis_snippets
                        > initial_offset_for_this_key_attempt
                    ):  # Made partial progress
                        feature_batch_outputs_llm.extend(
                            temp_batch_outputs_for_feature_this_key_attempt
                        )
                        # Continue to next key to process remaining snippets for this feature
                    # If no progress, the outer key loop will continue to the next key for the same offset.

                # After API key loop
                if not llm_step_succeeded_for_feature:
                    self.logger.error(
                        f"LLM processing FAILED for feature {i_feat} after all keys. Offset at {current_offset_for_analysis_snippets}/{len(all_analysis_data_snippets_for_this_feature)}. Last error: {mad_utils._format_exception_for_logging(last_llm_exception_for_feature)}"
                    )
                else:
                    final_scores_all_features = np.maximum(
                        final_scores_all_features,
                        feature_anomaly_scores_from_llm,  # This is already float 0.0-1.0
                    )
                    self.logger.info(
                        f"LLM processing step fully completed for feature {i_feat}. Scores aggregated using np.maximum."
                    )

                if llm_step_succeeded_for_feature:
                    final_feature_plot_path = os.path.join(
                        feature_artifact_dir,
                        f"feature_{i_feat}_final_aggregated_anomalies.png",
                    )
                    try:
                        plotting_utils.generate_feature_final_anomalies_plot(
                            X_data[:, i_feat],
                            feature_anomaly_scores_from_llm,
                            i_feat,
                            final_feature_plot_path,
                            self.logger,
                            6,
                            150,
                        )
                        if i_feat in self.per_feature_artifacts_:
                            self.per_feature_artifacts_[i_feat][
                                "final_feature_plot_path"
                            ] = final_feature_plot_path
                    except Exception as e_plot_final:
                        self.logger.error(
                            f"Feature {i_feat}: Failed to generate final plot: {mad_utils._format_exception_for_logging(e_plot_final)}"
                        )

                self.per_feature_artifacts_[i_feat].update(
                    {
                        "llm_outputs_all_successful_batches_for_feature": feature_batch_outputs_llm,
                        "llm_step_success_overall": llm_step_succeeded_for_feature,
                        "feature_scores_llm_aggregated": feature_anomaly_scores_from_llm.tolist(),
                        "error_llm_step_last_recorded": (
                            mad_utils._format_exception_for_logging(
                                last_llm_exception_for_feature
                            )
                            if last_llm_exception_for_feature
                            and not llm_step_succeeded_for_feature
                            else None
                        ),
                    }
                )
                self.logger.debug(
                    f"Feature {i_feat} processing took {time.time() - feature_loop_start_time:.2f}s. Success: {llm_step_succeeded_for_feature}"
                )

            self.decision_scores_ = final_scores_all_features
            final_artifact_dir_target = os.path.join(
                self.plot_save_dir, self.current_run_artifact_folder_name
            )
            try:
                if os.path.exists(temp_artifact_base_dir):
                    os.rename(temp_artifact_base_dir, final_artifact_dir_target)
            except OSError as e_rename:
                self.logger.error(
                    f"Error renaming temp artifact dir: {mad_utils._format_exception_for_logging(e_rename)}."
                )

            self.logger.info(
                f"LLM Pipeline completed in {time.time() - pipeline_start_time:.2f} seconds."
            )
            return self.decision_scores_

    def fit(self, X_train, y_train=None):
        self.logger.info(
            f"Starting F_May_22 fit. X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}"
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
            f"Starting F_May_22 decision_function. X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}"
        )
        self.logger.debug(
            f"Using X_train_fit_data (shape: {self.X_train_fit_data.shape if self.X_train_fit_data is not None else 'None'}) and y_train_fit_labels (len: {len(self.y_train_fit_labels) if self.y_train_fit_labels is not None else 'None'}) for LLM examples."
        )
        decision_scores = self._run_llm_pipeline(X_test)
        self.logger.info(
            f"F_May_22 decision_function completed. Scores shape: {decision_scores.shape if decision_scores is not None else 'N/A'}"
        )
        return decision_scores

    def get_per_feature_artifacts(self):
        return self.per_feature_artifacts_

    def get_last_run_timestamp(self):
        self.logger.debug(f"Returning last_run_timestamp: {self.last_run_timestamp}")
        return self.last_run_timestamp
