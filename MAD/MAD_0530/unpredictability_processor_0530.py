import inspect
import json
import os
import time

import numpy as np
from google.genai import types
from tqdm import tqdm

from . import gemini_api_utils_0530
from . import mad_utils_0530
from . import plotting_utils_0530
from . import prompt_utils_0530
from .constants_0530 import (
    DEFAULT_FALLBACK_TOKEN_LIMIT,
    FIXED_MODEL_NAME_GEMINI_FLASH,
    THINKING_BUDGET_MAIN_LLM,
    TOKEN_LIMIT_SAFETY_FACTOR,
)


def perform_unpredictability_step(
    X_data_col_for_feature: np.ndarray,
    i_feat: int,
    n_samples: int,
    all_analysis_data_snippets_list: list,
    all_training_example_snippets_list: list,
    identified_train_interest_range: tuple | None,
    api_keys_list: list[str],
    dynamic_token_limit_overrides: dict,
    logger,
    feature_artifact_dir_for_unpred: str,
    max_retries_with_delay_per_key_const: int,
    default_retry_delay_seconds_const: int,
    X_train_fit_data: np.ndarray | None,
    y_train_fit_labels: np.ndarray | None,
):
    """
    Generates unpredictability scores for a given feature.

    Returns:
        A tuple containing:
        - unpredictability_scores_for_feature (np.ndarray): Array of scores.
        - unpredictability_plot_uri_for_feature (str | None): URI of the uploaded plot.
        - unpredictability_step_succeeded (bool): True if the step succeeded.
        - last_exception_for_step (Exception | None): The last exception encountered.
        - batch_artifacts_collection (list): Collection of artifacts from each batch.
    """
    logger.info(f"Ft {i_feat}: Starting STAGE 1: Unpredictability Score Generation (in dedicated function).")
    
    unpredictability_scores_for_feature = np.zeros(n_samples, dtype=float)
    unpredictability_plot_uri_for_feature = None
    unpredictability_step_succeeded_overall = False
    last_exception_for_step = None
    batch_artifacts_collection = []

    current_offset_for_unpred_snippets = 0
    key_fatal_error_occurred_for_stage = False

    with tqdm(total=len(all_analysis_data_snippets_list), desc=f"Ft {i_feat} Unpred Snippets", unit="snippet", leave=False) as unpred_pbar:
        for unpred_api_key_attempt_num in range(len(api_keys_list)):
            if unpredictability_step_succeeded_overall:
                logger.info(f"Ft {i_feat} Unpred Stage: Overall success flag is true. Skipping remaining keys.")
                break
            
            current_api_key_index_for_stage = unpred_api_key_attempt_num
            current_api_key_value = api_keys_list[current_api_key_index_for_stage]
            logger.info(f"Ft {i_feat} Unpred Stage: Attempting with API Key Index {current_api_key_index_for_stage} (Value: ...{current_api_key_value[-4:]})")

            unpred_client = gemini_api_utils_0530.initialize_gemini_client(current_api_key_value, logger)

            if not unpred_client:
                logger.error(f"Ft {i_feat} Unpred Stage: Failed to initialize client for API key index {current_api_key_index_for_stage}. Skipping to next key.")
                last_exception_for_step = RuntimeError(f"Unpred client init failed for key index {current_api_key_index_for_stage}")
                continue
            
            unpred_batch_num_this_key = 0
            initial_offset_for_this_unpred_key_attempt = current_offset_for_unpred_snippets
            key_failed_to_process_any_batch = False

            while current_offset_for_unpred_snippets < len(all_analysis_data_snippets_list):
                if unpredictability_step_succeeded_overall: break

                unpred_batch_num_this_key += 1
                unpred_batch_save_dir = os.path.join(feature_artifact_dir_for_unpred, f"unpred_batch_{unpred_batch_num_this_key}")
                os.makedirs(unpred_batch_save_dir, exist_ok=True)
                logger.info(f"Ft {i_feat}, Unpred KeyIdx {current_api_key_index_for_stage}, UnpredBatch {unpred_batch_num_this_key} (Offset {current_offset_for_unpred_snippets}): Starting batch processing. Save dir: {unpred_batch_save_dir}")
                
                unpred_code_gen_succeeded_this_batch = False
                generated_unpred_code_str_this_batch = None
                unpred_code_execution_succeeded_this_batch = False
                key_fatal_error_occurred_for_this_api_key_this_batch = False

                has_train_anomaly_for_unpred_prompt_this_batch = False
                if y_train_fit_labels is not None and np.any(y_train_fit_labels == 1):
                    has_train_anomaly_for_unpred_prompt_this_batch = True

                candidate_analysis_snippets_for_unpred_batch_list_dicts = all_analysis_data_snippets_list[current_offset_for_unpred_snippets:]
                
                _model_info_unpred = gemini_api_utils_0530.get_gemini_model_info(
                    unpred_client, 65536, dynamic_token_limit_overrides.get(current_api_key_index_for_stage), logger
                )
                _input_limit_unpred = _model_info_unpred.get("input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT)
                _qualified_model_name_unpred = _model_info_unpred.get("qualified_model_name", FIXED_MODEL_NAME_GEMINI_FLASH)

                temp_analysis_snippets_unpred_list_for_api_call = mad_utils_0530.prepare_analysis_snippets_for_batch(
                    candidate_snippets_list=candidate_analysis_snippets_for_unpred_batch_list_dicts,
                    client=unpred_client,
                    model_name_for_counting=_qualified_model_name_unpred,
                    tokens_for_analysis_budget_this_batch=_input_limit_unpred * 0.3,
                    logger=logger
                )
                if not temp_analysis_snippets_unpred_list_for_api_call:
                    logger.warning(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: No analysis snippets for unpred LLM after budgeting. Skipping batch for this key.")
                    key_failed_to_process_any_batch = True
                    break

                temp_analysis_snippets_unpred_map_for_api_call = mad_utils_0530.convert_snippet_list_to_final_json(temp_analysis_snippets_unpred_list_for_api_call, False, logger)
                analysis_indices_str_unpred = f"Indices {temp_analysis_snippets_unpred_list_for_api_call[0]['index']} to {temp_analysis_snippets_unpred_list_for_api_call[-1]['index']}" if temp_analysis_snippets_unpred_list_for_api_call else "N/A"
                
                temp_training_snippets_unpred_list_for_api_call = []
                if all_training_example_snippets_list:
                     temp_training_snippets_unpred_list_for_api_call = mad_utils_0530.fill_snippets_by_token_budget(
                        unpred_client, _qualified_model_name_unpred, all_training_example_snippets_list,
                        _input_limit_unpred * 0.2,
                        lambda snips: json.dumps(mad_utils_0530.convert_snippet_list_to_final_json(snips, True, logger)),
                        logger, context_log_prefix=f"Ft {i_feat} UnpredTrainFill UB{unpred_batch_num_this_key}: "
                    )
                temp_training_snippets_unpred_map_for_api_call = mad_utils_0530.convert_snippet_list_to_final_json(temp_training_snippets_unpred_list_for_api_call, True, logger)
                training_indices_str_unpred = "Selected from training"

                unpred_raw_plot_filename = f"unpred_feature_{i_feat}_batch_{unpred_batch_num_this_key}_raw_input.png"
                unpred_raw_plot_path = os.path.join(unpred_batch_save_dir, unpred_raw_plot_filename)
                uploaded_unpred_raw_plot_uri_for_llm = None

                plot_success_raw, _ = plotting_utils_0530.generate_unpredictability_scores_plot(
                    X_data_col_for_feature, np.zeros(n_samples), i_feat, unpred_batch_num_this_key,
                    unpred_raw_plot_path, logger, 6, 100, analysis_snippets_to_highlight=temp_analysis_snippets_unpred_list_for_api_call
                )
                if plot_success_raw:
                    try:
                        uploaded_file_obj = gemini_api_utils_0530.upload_file_to_gemini(unpred_client, unpred_raw_plot_path, logger)
                        if uploaded_file_obj and hasattr(uploaded_file_obj, 'uri'):
                            uploaded_unpred_raw_plot_uri_for_llm = uploaded_file_obj.uri
                            if unpred_batch_num_this_key == 1 and unpredictability_plot_uri_for_feature is None:
                                unpredictability_plot_uri_for_feature = uploaded_unpred_raw_plot_uri_for_llm
                        logger.info(f"Ft {i_feat} UB{unpred_batch_num_this_key}: Raw unpred plot uploaded. URI: {uploaded_unpred_raw_plot_uri_for_llm}")
                    except Exception as e_upload:
                        logger.warning(f"Ft {i_feat} UB{unpred_batch_num_this_key}: Failed to upload raw unpred plot: {mad_utils_0530._format_exception_for_logging(e_upload)}")
                
                unpred_current_generate_with_data_param = False
                halving_iteration_this_pass_unpred = 0
                
                for unpred_llm_api_retry_num in range(max_retries_with_delay_per_key_const + 1):
                    current_api_attempt_exception = None
                    
                    current_analysis_snippets_for_halving = list(temp_analysis_snippets_unpred_list_for_api_call)
                    current_analysis_map_for_halving = dict(temp_analysis_snippets_unpred_map_for_api_call)
                    current_analysis_indices_str_for_halving = str(analysis_indices_str_unpred)

                    while True:
                        try:
                            generate_func_with_data_param_for_api_call = (halving_iteration_this_pass_unpred > 0)

                            unpred_prompt = prompt_utils_0530.construct_unpredictability_code_gen_prompt(
                                i_feat=i_feat, n_samples=n_samples,
                                current_batch_training_snippets_map=temp_training_snippets_unpred_map_for_api_call,
                                current_batch_analysis_snippets_map=current_analysis_map_for_halving,
                                training_data_indices_info_str=training_indices_str_unpred,
                                analysis_data_indices_info_str=current_analysis_indices_str_for_halving,
                                image_uri_for_llm=uploaded_unpred_raw_plot_uri_for_llm,
                                generate_func_with_data_param=generate_func_with_data_param_for_api_call,
                                training_feature_col_for_baseline_guidance=X_train_fit_data[:, i_feat] if X_train_fit_data is not None and X_train_fit_data.shape[1] > i_feat else None,
                                has_train_anomaly_for_unpred_prompt=has_train_anomaly_for_unpred_prompt_this_batch
                            )
                            mad_utils_0530.log_prompt_text_to_file(unpred_prompt, unpred_batch_save_dir, f"unpred_prompt_F{i_feat}_UB{unpred_batch_num_this_key}_A{unpred_llm_api_retry_num}_H{halving_iteration_this_pass_unpred}.txt", logger)

                            unpred_llm_contents_for_api_call = [types.Part.from_text(text=unpred_prompt)]
                            if uploaded_unpred_raw_plot_uri_for_llm:
                                unpred_llm_contents_for_api_call.insert(0, types.Part.from_uri(mime_type="image/png", file_uri=uploaded_unpred_raw_plot_uri_for_llm))

                            logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Calling LLM for unpred code (API attempt {unpred_llm_api_retry_num + 1}, Halving iter {halving_iteration_this_pass_unpred}). Analysis snips for prompt: {len(current_analysis_snippets_for_halving)}")
                            
                            unpred_response = gemini_api_utils_0530.execute_gemini_api_call(
                                unpred_client.models.generate_content, logger,
                                contents=unpred_llm_contents_for_api_call,
                                config=types.GenerateContentConfig(response_mime_type="text/plain", temperature=0.0, thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=THINKING_BUDGET_MAIN_LLM)),
                                model=_qualified_model_name_unpred,
                            )

                            if unpred_response and unpred_response.candidates and hasattr(unpred_response, "usage_metadata") and hasattr(unpred_response.usage_metadata, "thoughts_token_count"):
                                if unpred_response.usage_metadata.thoughts_token_count > 0:
                                    logger.debug(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: LLM thoughts token count: {unpred_response.usage_metadata.thoughts_token_count}")

                            if unpred_response and unpred_response.candidates and unpred_response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
                                logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: MAX_TOKENS from unpred code gen LLM (API attempt {unpred_llm_api_retry_num + 1}, Halving iter {halving_iteration_this_pass_unpred}).")
                                if hasattr(unpred_response, "usage_metadata") and hasattr(unpred_response.usage_metadata, "thoughts_token_count"):
                                     logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: MAX_TOKENS. Thoughts token count: {unpred_response.usage_metadata.thoughts_token_count}")

                                halving_iteration_this_pass_unpred += 1
                                if not unpred_current_generate_with_data_param:
                                    logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: MAX_TOKENS. Trying with generate_func_with_data_param=True.")
                                    unpred_current_generate_with_data_param = True
                                    continue
                                elif len(current_analysis_snippets_for_halving) > 1:
                                    current_analysis_snippets_for_halving = current_analysis_snippets_for_halving[:len(current_analysis_snippets_for_halving)//2]
                                    current_analysis_map_for_halving = mad_utils_0530.convert_snippet_list_to_final_json(current_analysis_snippets_for_halving, False, logger)
                                    current_analysis_indices_str_for_halving = f"Indices {current_analysis_snippets_for_halving[0]['index']} to {current_analysis_snippets_for_halving[-1]['index']}" if current_analysis_snippets_for_halving else "N/A"
                                    logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: MAX_TOKENS. Halved analysis snippets to {len(current_analysis_snippets_for_halving)}. Retrying.")
                                    unpred_current_generate_with_data_param = False
                                    continue
                                else:
                                    logger.error(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: MAX_TOKENS from unpred code gen LLM and cannot halve snippets further or change data_param. Failing this API attempt.")
                                    raise gemini_api_utils_0530.MaxTokensExtendedError("MAX_TOKENS persistent after halving/data_param for unpred code gen.")
                            
                            if not unpred_response or not unpred_response.text:
                                raise ValueError("LLM response for unpred code was empty or invalid.")
                            generated_unpred_code_str_this_batch = mad_utils_0530.strip_markdown_code_fences(unpred_response.text)
                            mad_utils_0530.log_code_to_file(generated_unpred_code_str_this_batch, unpred_batch_save_dir, f"unpred_code_F{i_feat}_UB{unpred_batch_num_this_key}_A{unpred_llm_api_retry_num}_H{halving_iteration_this_pass_unpred}.py", logger)
                            unpred_code_gen_succeeded_this_batch = True
                            break
                        
                        except Exception as e_unpred_llm_call_attempt:
                            current_api_attempt_exception = e_unpred_llm_call_attempt
                            last_exception_for_step = e_unpred_llm_call_attempt
                            logger.warning(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Unpred LLM call failed (API attempt {unpred_llm_api_retry_num + 1}, Halving iter {halving_iteration_this_pass_unpred}). Error: {mad_utils_0530._format_exception_for_logging(e_unpred_llm_call_attempt)}")
                            
                            error_lower_str_unpred = str(e_unpred_llm_call_attempt).lower()
                            parsed_retry_delay_seconds_unpred = mad_utils_0530.extract_retry_delay_from_error_details_json(error_lower_str_unpred)
                            
                            is_fatal_error_for_key = False
                            if parsed_retry_delay_seconds_unpred is None:
                                fatal_keywords_unpred = ["api key not valid", "api_key_invalid", "permissiondenied", "billing", "developer_inactive", "consumer_invalid", "api key required"]
                                if any(p in error_lower_str_unpred for p in fatal_keywords_unpred) or \
                                   (hasattr(e_unpred_llm_call_attempt, 'code') and e_unpred_llm_call_attempt.code in [401, 403]) or \
                                   (hasattr(e_unpred_llm_call_attempt, 'code') and e_unpred_llm_call_attempt.code == 429 and parsed_retry_delay_seconds_unpred is None):
                                    is_fatal_error_for_key = True
                            
                            if is_fatal_error_for_key:
                                logger.error(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: API Key {current_api_key_index_for_stage} encountered a FATAL error for unpred code gen. Error: {mad_utils_0530._format_exception_for_logging(e_unpred_llm_call_attempt)}.")
                                key_fatal_error_occurred_for_this_api_key_this_batch = True
                                key_fatal_error_occurred_for_stage = True
                                break
                            
                            break

                    if key_fatal_error_occurred_for_this_api_key_this_batch:
                        break

                    if unpred_code_gen_succeeded_this_batch:
                        logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Code generation SUCCEEDED on API attempt {unpred_llm_api_retry_num + 1} for key {current_api_key_index_for_stage}. Proceeding to execution.")
                        break

                    if unpred_llm_api_retry_num < max_retries_with_delay_per_key_const:
                        delay_s = default_retry_delay_seconds_const
                        if current_api_attempt_exception:
                             error_lower_str_unpred_retry = str(current_api_attempt_exception).lower()
                             parsed_delay_for_retry = mad_utils_0530.extract_retry_delay_from_error_details_json(error_lower_str_unpred_retry)
                             if parsed_delay_for_retry is not None and parsed_delay_for_retry > 0:
                                 delay_s = parsed_delay_for_retry
                        
                        logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Retrying unpred LLM call with key {current_api_key_index_for_stage} in {delay_s}s (attempt {unpred_llm_api_retry_num + 2}).")
                        time.sleep(delay_s)
                        unpred_current_generate_with_data_param = False
                        continue
                    else:
                        logger.error(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: All {max_retries_with_delay_per_key_const + 1} API call attempts for unpred code gen failed for key {current_api_key_index_for_stage}. Last error: {mad_utils_0530._format_exception_for_logging(current_api_attempt_exception if current_api_attempt_exception else 'N/A')}")
                        break

                if key_fatal_error_occurred_for_this_api_key_this_batch:
                    batch_artifacts_collection.append({
                        "unpred_batch_num": unpred_batch_num_this_key, "key_idx": current_api_key_index_for_stage, "status": "FailedCodeGen_FatalKey",
                        "code_gen_success": False, "code_exec_success": False, 
                        "error_details": mad_utils_0530._format_exception_for_logging(last_exception_for_step)
                    })
                    current_offset_for_unpred_snippets = initial_offset_for_this_unpred_key_attempt
                    key_failed_to_process_any_batch = True
                    break
                
                if unpred_code_gen_succeeded_this_batch and generated_unpred_code_str_this_batch:
                    unpred_exec_namespace = {"np": np, "X_data_col": X_data_col_for_feature if unpred_current_generate_with_data_param else None}
                    batch_scores_unpred = np.zeros(n_samples, dtype=float)
                    try:
                        exec(generated_unpred_code_str_this_batch, unpred_exec_namespace)
                        if 'calculate_unpredictability_scores' not in unpred_exec_namespace:
                            raise NameError("Function 'calculate_unpredictability_scores' not found in executed code.")
                        
                        calc_func_unpred = unpred_exec_namespace['calculate_unpredictability_scores']
                        if unpred_current_generate_with_data_param:
                            batch_scores_unpred = calc_func_unpred(X_data_col=X_data_col_for_feature)
                        else:
                            batch_scores_unpred = calc_func_unpred()

                        if not isinstance(batch_scores_unpred, np.ndarray) or batch_scores_unpred.shape != (n_samples,):
                            raise ValueError(f"Generated unpred code returned invalid scores: shape {batch_scores_unpred.shape if hasattr(batch_scores_unpred, 'shape') else type(batch_scores_unpred)}, expected ({n_samples},)")
                        unpredictability_scores_for_feature = np.maximum(unpredictability_scores_for_feature, batch_scores_unpred)
                        unpred_code_execution_succeeded_this_batch = True
                        logger.info(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Unpred code executed successfully.")
                        
                        batch_artifacts_collection.append({
                            "unpred_batch_num": unpred_batch_num_this_key, "key_idx": current_api_key_index_for_stage, "status": "Success",
                            "code_gen_success": True, "code_exec_success": True, "num_analysis_snippets_processed": len(temp_analysis_snippets_unpred_list_for_api_call),
                            "generated_code_path": f"unpred_code_F{i_feat}_UB{unpred_batch_num_this_key}_A{unpred_llm_api_retry_num}_H{halving_iteration_this_pass_unpred}.py",
                            "plot_uri_input": uploaded_unpred_raw_plot_uri_for_llm
                        })
                        unpred_pbar.update(len(temp_analysis_snippets_unpred_list_for_api_call))
                        current_offset_for_unpred_snippets += len(temp_analysis_snippets_unpred_list_for_api_call)
                        
                        if current_offset_for_unpred_snippets >= len(all_analysis_data_snippets_list):
                            unpredictability_step_succeeded_overall = True
                            logger.info(f"Ft {i_feat} Unpred Stage: All snippets processed successfully with Key {current_api_key_index_for_stage}.")
                            final_plot_filename = f"unpred_feature_{i_feat}_final_scores_key{current_api_key_index_for_stage}.png"
                            final_plot_path = os.path.join(feature_artifact_dir_for_unpred, final_plot_filename)
                            plot_final_success, _ = plotting_utils_0530.generate_unpredictability_scores_plot(
                                X_data_col_for_feature, unpredictability_scores_for_feature, i_feat, "final",
                                final_plot_path, logger, 6, 100
                            )
                            if plot_final_success:
                                try:
                                    final_uploaded_obj = gemini_api_utils_0530.upload_file_to_gemini(unpred_client, final_plot_path, logger)
                                    if final_uploaded_obj and hasattr(final_uploaded_obj, 'uri'):
                                        unpredictability_plot_uri_for_feature = final_uploaded_obj.uri
                                except Exception as e_fin_upload:
                                    logger.warning(f"Ft {i_feat}: Failed to upload final unpred plot: {mad_utils_0530._format_exception_for_logging(e_fin_upload)}")
                            break

                    except Exception as e_unpred_exec:
                        logger.error(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Unpred code execution failed. Error: {mad_utils_0530._format_exception_for_logging(e_unpred_exec)}")
                        last_exception_for_step = e_unpred_exec
                        batch_artifacts_collection.append({
                            "unpred_batch_num": unpred_batch_num_this_key, "key_idx": current_api_key_index_for_stage, "status": "FailedExec",
                            "code_gen_success": True, "code_exec_success": False, 
                            "error_details": mad_utils_0530._format_exception_for_logging(e_unpred_exec)
                        })
                        current_offset_for_unpred_snippets = initial_offset_for_this_unpred_key_attempt
                        key_failed_to_process_any_batch = True
                        break
                else:
                    logger.error(f"Ft {i_feat}, UB{unpred_batch_num_this_key}: Unpred code generation FAILED for all API attempts with key {current_api_key_index_for_stage}.")
                    batch_artifacts_collection.append({
                        "unpred_batch_num": unpred_batch_num_this_key, "key_idx": current_api_key_index_for_stage, "status": "FailedCodeGen_AllRetries",
                        "code_gen_success": False, "code_exec_success": False,
                        "error_details": mad_utils_0530._format_exception_for_logging(last_exception_for_step if last_exception_for_step else "Code gen failed all retries, no specific last exception captured.")
                    })
                    current_offset_for_unpred_snippets = initial_offset_for_this_unpred_key_attempt
                    key_failed_to_process_any_batch = True
                    break
            
            if unpredictability_step_succeeded_overall:
                logger.info(f"Ft {i_feat}, Unpred Stage KeyIdx {current_api_key_index_for_stage}: Unpred processing completed FOR THE FEATURE with this key.")
                break
            
            if key_fatal_error_occurred_for_stage:
                logger.warning(f"Ft {i_feat} Unpred Stage: API Key Index {current_api_key_index_for_stage} encountered a fatal error. Moving to next key if available. Offset reset to {initial_offset_for_this_unpred_key_attempt}.")
                current_offset_for_unpred_snippets = initial_offset_for_this_unpred_key_attempt
                continue

            if key_failed_to_process_any_batch:
                logger.warning(f"Ft {i_feat} Unpred Stage: API Key Index {current_api_key_index_for_stage} did not complete all unpred snippets due to batch failure(s). Offset reset to {initial_offset_for_this_unpred_key_attempt}. Last error for key: {mad_utils_0530._format_exception_for_logging(last_exception_for_step)}")
                current_offset_for_unpred_snippets = initial_offset_for_this_unpred_key_attempt
                continue
            
            logger.info(f"Ft {i_feat} Unpred Stage: Key {current_api_key_index_for_stage} completed its batch loop. Current offset: {current_offset_for_unpred_snippets}. Overall success: {unpredictability_step_succeeded_overall}")

        if not unpredictability_step_succeeded_overall:
            final_error_message = mad_utils_0530._format_exception_for_logging(last_exception_for_step if last_exception_for_step else 'Unknown error or all keys failed.')
            if key_fatal_error_occurred_for_stage: 
                final_error_message = f"A fatal key error occurred during the stage. Last recorded error: {final_error_message}"
            logger.error(f"Ft {i_feat}: STAGE 1 (Unpredictability Score Generation) FAILED for all keys or due to a fatal error. {final_error_message}")

    return (
        unpredictability_scores_for_feature,
        unpredictability_plot_uri_for_feature,
        unpredictability_step_succeeded_overall,
        last_exception_for_step,
        batch_artifacts_collection,
    ) 