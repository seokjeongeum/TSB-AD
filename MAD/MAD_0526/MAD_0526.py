import json
import logging
import os
import time

import numpy as np
from tqdm import tqdm

from MAD.MAD_May_26.constants import (
    API_KEYS,
    DEFAULT_FALLBACK_TOKEN_LIMIT,
    DEFAULT_RETRY_DELAY_SECONDS,
    FIXED_MODEL_NAME_GEMINI_FLASH,
    MAX_RETRIES_WITH_DELAY_PER_KEY,
    TOKEN_LIMIT_SAFETY_FACTOR,
)
from TSB_AD.models.base import BaseDetector

from . import gemini_api_utils_0526, mad_utils_0526, plotting_utils_0526, prompt_utils_0526
from .llm_batch_processor_0526 import process_single_batch_with_llm


class MAD_May_26(BaseDetector):
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

        self.plot_save_dir = "MAD/Feature_Plots/MAD_May_26"  # Main directory for all feature plots
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
        self.logger.debug("""MAD_May_26 Initialized with HPs: %s""", HP)
        # train_data_len_for_hint is now implicitly self.X_train_fit_data.shape[0] if it exists
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
        client = gemini_api_utils_0526.initialize_gemini_client(key_to_use, self.logger)
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
            training_metadata_for_prompt = None # New: For storing training metadata

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
                    # CALCULATE TRAINING METADATA (REPLACES INTEREST ID)
                    if self.X_train_fit_data is not None and self.y_train_fit_labels is not None:
                        training_metadata_for_prompt = mad_utils_0526.calculate_training_metadata(
                            self.X_train_fit_data[:, i_feat],
                            self.y_train_fit_labels,
                            self.logger
                        )
                    # END TRAINING METADATA CALCULATION
                    
                    # PREPARE TRAINING SNIPPETS (IF ANY, WITHOUT INTEREST ID RANGE)
                    all_training_example_snippets = mad_utils_0526.prepare_training_snippets_for_main_step(
                        X_train_fit_data=self.X_train_fit_data,
                        y_train_fit_labels=self.y_train_fit_labels,
                        i_feat=i_feat,
                        logger=self.logger,
                    )
                    # --- End Training Snippet Prep (Replaces Interest ID) ---

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
                            model_info = gemini_api_utils_0526.get_gemini_model_info(
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

                            # --- Token Budgeting (Simplified for only analysis snippets) ---
                            _base_prompt_for_struct_count = (
                                prompt_utils_0526.construct_llm_batch_prompt(
                                    i_feat,
                                    n_samples,
                                    {}, # No training snippets in base count for budgeting anlaysis
                                    {}, # Empty analysis for base count
                                    training_metadata_for_prompt, # Pass metadata
                                    "", # analysis_data_indices_info_str placeholder
                                    is_processing_training_data_context,
                                    self.X_train_fit_data,
                                    self.y_train_fit_labels,
                                    None, # image_uri_for_llm
                                )
                            )
                            base_prompt_tokens = gemini_api_utils_0526.count_gemini_tokens(
                                current_attempt_client,
                                qualified_model_name,
                                [_base_prompt_for_struct_count],
                                self.logger,
                            ).total_tokens

                            # Budget for training snippets (actual full snippets, not just metadata)
                            # This is for the actual training snippets that will be *sent* if they fit.
                            # The metadata is always sent.
                            # tokens_for_training_snippets_to_send = 0 # REMOVED
                            current_batch_collected_training_snippets_list = [] # ALWAYS EMPTY NOW
                            # if all_training_example_snippets: # If there are any training snippets to potentially send # REMOVED
                                # Estimate tokens for these training snippets # REMOVED
                                # temp_train_map_for_count = mad_utils.convert_snippet_list_to_final_json( # REMOVED
                                #     all_training_example_snippets, True, self.logger # REMOVED
                                # ) # REMOVED
                                # if temp_train_map_for_count.get("value"): # only count if there are values # REMOVED
                                #     tokens_for_training_snippets_to_send = gemini_api_utils.count_gemini_tokens( # REMOVED
                                #         current_attempt_client, # REMOVED
                                #         qualified_model_name, # REMOVED
                                #         [json.dumps(temp_train_map_for_count)], # REMOVED
                                #         self.logger # REMOVED
                                #     ).total_tokens # REMOVED

                            avail_tokens_for_analysis_and_actual_train = (
                                input_token_limit_for_batch_budgeting * TOKEN_LIMIT_SAFETY_FACTOR
                            ) - base_prompt_tokens
                            
                            # Subtract tokens for training snippets that *will* be sent, if they fit
                            # If they don't fit, they won't be sent, and analysis gets full budget.
                            tokens_for_analysis_budget = avail_tokens_for_analysis_and_actual_train
                            # if tokens_for_training_snippets_to_send <= tokens_for_analysis_budget: # REMOVED
                            #     current_batch_collected_training_snippets_list = all_training_example_snippets # REMOVED
                            #     tokens_for_analysis_budget -= tokens_for_training_snippets_to_send # REMOVED
                            # else: # REMOVED
                                # Training snippets as a whole are too large, don't send them. # REMOVED
                                # Analysis snippets get the budget previously allocated to training snippets. # REMOVED
                                # current_batch_collected_training_snippets_list = [] # REMOVED
                                # self.logger.info(f"Ft {i_feat}, B{batch_num}, Key {self.current_api_key_index}: All training snippets ({tokens_for_training_snippets_to_send} tokens) too large for budget. Sending none.") # REMOVED

                            self.logger.info(f"Ft {i_feat}, B{batch_num}, Key {self.current_api_key_index}: Training snippets will not be sent. Metadata only. Analysis token budget: {tokens_for_analysis_budget}")
                            current_batch_training_snippets_final_dict = (
                                mad_utils_0526.convert_snippet_list_to_final_json(
                                    current_batch_collected_training_snippets_list, # This will be empty
                                    True,
                                    self.logger,
                                )
                            )
                            # --- End Token Budgeting ---

                            current_batch_collected_analysis_snippets_list = (
                                mad_utils_0526.prepare_analysis_snippets_for_batch(
                                    candidate_analysis_snippets_for_this_batch_call,
                                    current_attempt_client,
                                    qualified_model_name,
                                    tokens_for_analysis_budget,
                                    self.logger,
                                )
                            )
                            current_batch_analysis_snippets_final_dict = (
                                mad_utils_0526.convert_snippet_list_to_final_json(
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
                            # training_indices_info_str no longer relevant as we send all or none, metadata is separate
                            # training_indices_info_str = (
                            #     f"Selected from training (focus: {identified_train_interest_range})"
                            #     if identified_train_interest_range
                            #     else "Selected from training"
                            # )

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
                                training_metadata_for_prompt=training_metadata_for_prompt, # Pass metadata
                                is_processing_training_data_itself=is_processing_training_data_context,
                                X_train_fit_data=self.X_train_fit_data,
                                y_train_fit_labels=self.y_train_fit_labels,
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
                                    f"Ft {i_feat}, B{batch_num} (Offset {current_offset_for_analysis_snippets} from {initial_offset_for_this_key_attempt}) FAILED via processor for Key {self.current_api_key_index}. Last batch error: {mad_utils_0526._format_exception_for_logging(batch_exception)}"
                                )
                                if learned_quota_from_batch is not None: # A quota was learned even on failure
                                    self.logger.info(f"Feature {i_feat}, Key {self.current_api_key_index}: Updating dynamic token limit from failed Batch {batch_num} to {learned_quota_from_batch}.")
                                    self.dynamic_token_limit_overrides[self.current_api_key_index] = learned_quota_from_batch
                                current_offset_for_analysis_snippets = initial_offset_for_this_key_attempt  # Reset for next key
                                break  # Break from while loop (batches for this key)

                        except Exception as e_batch_outer_refactored:
                            self.logger.error(
                                f"Ft {i_feat}, B{batch_num}, Key {self.current_api_key_index}: Outer exception in batch loop: {mad_utils_0526._format_exception_for_logging(e_batch_outer_refactored)}"
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
                        f"LLM processing FAILED for feature {i_feat} after all keys. Offset at {current_offset_for_analysis_snippets}/{len(all_analysis_data_snippets_for_this_feature)}. Last error: {mad_utils_0526._format_exception_for_logging(last_llm_exception_for_feature)}"
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
                        plotting_utils_0526.generate_feature_final_anomalies_plot(
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
                            f"Feature {i_feat}: Failed to generate final plot: {mad_utils_0526._format_exception_for_logging(e_plot_final)}"
                        )

                self.per_feature_artifacts_[i_feat].update(
                    {
                        "llm_outputs_all_successful_batches_for_feature": feature_batch_outputs_llm,
                        "llm_step_success_overall": llm_step_succeeded_for_feature,
                        "feature_scores_llm_aggregated": feature_anomaly_scores_from_llm.tolist(),
                        "error_llm_step_last_recorded": (
                            mad_utils_0526._format_exception_for_logging(
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
                    # Rename to the feature plot specific run folder
                    os.rename(temp_artifact_base_dir, final_artifact_dir_target)
                    self.logger.info(f"Moved temp artifacts to final feature plot run dir: {final_artifact_dir_target}")

            except OSError as e_rename:
                self.logger.error(
                    f"Error moving/creating artifact dirs: {mad_utils_0526._format_exception_for_logging(e_rename)}."
                )

            self.logger.info(
                f"LLM Pipeline completed in {time.time() - pipeline_start_time:.2f} seconds."
            )
            return self.decision_scores_

    def fit(self, X_train, y_train=None):
        self.logger.info(
            f"Starting MAD_May_26 fit. X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}"
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
            f"Starting MAD_May_26 decision_function. X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}"
        )
        self.logger.debug(
            f"Using X_train_fit_data (shape: {self.X_train_fit_data.shape if self.X_train_fit_data is not None else 'None'}) and y_train_fit_labels (len: {len(self.y_train_fit_labels) if self.y_train_fit_labels is not None else 'None'}) for LLM examples."
        )
        decision_scores = self._run_llm_pipeline(X_test)
        self.logger.info(
            f"MAD_May_26 decision_function completed. Scores shape: {decision_scores.shape if decision_scores is not None else 'N/A'}"
        )
        return decision_scores

    def get_per_feature_artifacts(self):
        return self.per_feature_artifacts_

    def get_last_run_timestamp(self):
        self.logger.debug(f"Returning last_run_timestamp: {self.last_run_timestamp}")
        return self.last_run_timestamp

