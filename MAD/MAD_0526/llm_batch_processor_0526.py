import os
import time

import numpy as np
from google.genai import types
from google.genai.types import FinishReason

from . import gemini_api_utils_0526, mad_utils_0526, plotting_utils_0526, prompt_utils_0526
from .constants_0526 import DEFAULT_FALLBACK_TOKEN_LIMIT, FIXED_MODEL_NAME_GEMINI_FLASH, THINKING_BUDGET_MAIN_LLM


def process_single_batch_with_llm(
    current_attempt_client,
    i_feat,
    n_samples,
    X_data_col,  # Full data for the feature
    current_batch_collected_training_snippets_list,
    current_batch_training_snippets_final_dict,
    current_batch_collected_analysis_snippets_list,
    current_batch_analysis_snippets_final_dict,
    training_metadata_for_prompt,
    is_processing_training_data_itself,
    X_train_fit_data,  # For context in prompt
    y_train_fit_labels,  # For context in prompt
    analysis_data_indices_info_str,  # Added
    batch_num,
    batch_extras_save_dir,  # Where to save plot, prompt, code for this batch
    logger,
    current_api_key_index,  # For logging
    max_retries_with_delay_per_key,
    default_retry_delay_seconds,
    input_token_limit_override: int | None, # New parameter
):
    """
    Processes a single batch of data using the LLM, including code generation,
    execution, and an optional refinement pass.

    Returns a tuple:
        (
            bool: batch_fully_successful_for_this_key,
            np.ndarray | None: current_batch_scores_from_llm (n_samples length, or None if failed),
            dict: batch_output_artifact_data (or empty if failed),
            Exception | None: last_exception_for_batch,
            int: num_analysis_snippets_in_llm_call_for_final_code (number of analysis snippets used in the LLM call that led to the final code),
            int | None: learned_input_token_quota_for_batch (learned input token quota for the batch, or None if not learned)
        )
    """
    batch_fully_successful_for_this_key = False
    current_batch_scores_from_llm = np.zeros(n_samples, dtype=float)  # Initialize
    batch_output_artifact_data = {}
    learned_input_token_quota_for_batch = None # Initialize
    last_exception_for_batch = None
    num_analysis_snippets_in_llm_call_for_final_code = 0  # Initialize

    # --- Initial Plot Generation and Upload (for first LLM call) ---
    plot_local_path_this_batch = os.path.join(
        batch_extras_save_dir, f"plot_LLM_Input_F{i_feat}_B{batch_num}.png"
    )
    generated_plot_file_for_llm = None  # Stores the File object from Gemini API
    image_uri_for_prompt = None

    plotting_utils_0526.generate_batch_llm_input_plot(
        X_data_col=X_data_col,
        i_feat=i_feat,
        batch_num=batch_num,
        n_samples=n_samples,
        identified_train_interest_range=None,
        use_training_labels_for_hint=True,  # Assuming HP driven elsewhere if needed
        y_labels_for_overall_hint=(y_train_fit_labels if True else None),
        train_len_for_overall_hint=(
            len(X_train_fit_data) if X_train_fit_data is not None else 0
        ),
        current_batch_training_snippets_list=current_batch_collected_training_snippets_list,
        current_batch_analysis_snippets_list=current_batch_collected_analysis_snippets_list,
        plot_path=plot_local_path_this_batch,
        logger=logger,
        plot_fig_height=6,
        plot_dpi=100,
    )
    for upload_attempt in range(max_retries_with_delay_per_key):
        try:
            logger.info(
                f"Batch {batch_num}: Uploading initial plot: {plot_local_path_this_batch}"
            )
            uploaded_file_response = gemini_api_utils_0526.upload_file_to_gemini(
                client=current_attempt_client,
                file_path=plot_local_path_this_batch,
                logger=logger,
            )
            generated_plot_file_for_llm = uploaded_file_response
            image_uri_for_prompt = generated_plot_file_for_llm.uri
            logger.info(
                f"Batch {batch_num}: Initial plot uploaded successfully: {image_uri_for_prompt}"
            )
            break
        except Exception as e_upload:
            logger.warning(
                f"Batch {batch_num}, Initial Plot Upload Attempt {upload_attempt + 1} failed: {mad_utils_0526._format_exception_for_logging(e_upload)}"
            )
            if upload_attempt < max_retries_with_delay_per_key - 1:
                time.sleep(default_retry_delay_seconds)
            else:
                logger.error(
                    f"Batch {batch_num}: Initial plot upload failed after {max_retries_with_delay_per_key} attempts. Proceeding without plot for LLM."
                )
                last_exception_for_batch = e_upload

    # --- Iterative Code Generation and Execution Loop ---
    attempt_num = 0

    generated_code_str = None
    execution_error_details_for_refinement = None
    previous_code_for_refinement = None

    # Artifact paths that might be updated in the loop
    final_prompt_log_path = None
    final_code_log_path = None
    final_llm_response_raw = None
    plot_of_attempt_scores_uri_for_refinement = (
        None  # URI of the plot showing scores of the *failed* attempt
    )

    # Variables to store artifacts of the *successful* attempt, or the *last* failed one
    pass_name_of_final_code = "N/A"

    # Fetch model info once for the batch, respecting any override passed from MAD_May_26
    # This `batch_input_token_limit` will be used for budgeting analysis snippets *within* this batch processing.
    batch_model_info = gemini_api_utils_0526.get_gemini_model_info(
        current_attempt_client,
        default_output_token_limit=65536, # Example, consider if this needs to be more dynamic
        input_token_limit_override=input_token_limit_override,
        logger=logger,
    )
    batch_input_token_limit = batch_model_info.get("input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT)

    while True:  # Retry indefinitely until code generation and execution succeed
        halving_iteration_this_pass = (
            0  # Counter for re-plotting filenames if MAX_TOKENS halving occurs
        )
        is_refinement_cycle = attempt_num > 0
        current_pass_name = f"Attempt{attempt_num + 1}"
        logger.info(
            f"Batch {batch_num}, Code Gen {current_pass_name} (Key {current_api_key_index}): Starting."
        )

        # Initialize with original/full snippets for the start of a new code generation pass (attempt_num)
        current_analysis_snippets_for_this_pass = list(
            current_batch_collected_analysis_snippets_list
        )  # Use a mutable copy
        current_analysis_snippets_map_for_this_pass = dict(
            current_batch_analysis_snippets_final_dict
        )  # Use a mutable copy
        analysis_indices_info_str_for_this_pass = str(
            analysis_data_indices_info_str
        )  # Use a mutable copy

        # Construct the primary prompt for the current attempt (Pass)
        # This prompt will be dynamically rebuilt if snippet halving occurs for MAX_TOKENS *within* the same_key_retry_num loop
        # So, this is more of an initial construction for the pass.

        # Build the primary API contents for this pass (attempt_num)
        # This will also be dynamically rebuilt if snippet halving occurs.

        llm_response_text_current_attempt = None
        llm_call_succeeded_this_attempt = False
        # num_analysis_snippets_in_llm_call_for_final_code will be updated if this pass's LLM call succeeds
        
        # MAX_TOKENS specific retry state for the current "current_pass_name"
        attempted_data_param_retry_for_max_tokens = False

        for same_key_retry_num in range(
            max_retries_with_delay_per_key
        ):  # Retries for the LLM API call itself (e.g. network, non-MAX_TOKENS errors)

            # Inner loop for MAX_TOKENS handling: will try to make the API call,
            # halving snippets and re-plotting/uploading if MAX_TOKENS occurs.
            # This loop does not consume same_key_retry_num for MAX_TOKENS retries.
            while True:
                prompt_for_current_api_call = prompt_utils_0526.construct_llm_batch_prompt(
                    i_feat=i_feat,
                    n_samples=n_samples,
                    current_batch_training_snippets_map=current_batch_training_snippets_final_dict,
                    current_batch_analysis_snippets_map=current_analysis_snippets_map_for_this_pass,  # Dynamically updated by this loop
                    training_metadata_for_prompt=training_metadata_for_prompt,
                    analysis_data_indices_info_str=analysis_indices_info_str_for_this_pass,  # Dynamically updated by this loop
                    is_analyzing_training_data_itself=is_processing_training_data_itself,
                    X_train_fit_data_param=X_train_fit_data,
                    y_train_fit_labels_param=y_train_fit_labels,
                    image_uri_for_llm=image_uri_for_prompt,  # Potentially updated by this loop
                    is_refinement_attempt=is_refinement_cycle,
                    previous_code_str=previous_code_for_refinement,
                    previous_execution_error_details=execution_error_details_for_refinement,
                    generate_func_with_data_param=attempted_data_param_retry_for_max_tokens, # Modified: Always use parameterized function after first MAX_TOKENS
                )
                current_api_call_contents = [prompt_for_current_api_call]
                if image_uri_for_prompt:
                    current_api_call_contents.insert(
                        0,
                        types.Part.from_uri(
                            file_uri=image_uri_for_prompt, mime_type="image/png"
                        ),
                    )
                if is_refinement_cycle and plot_of_attempt_scores_uri_for_refinement:
                    current_api_call_contents.insert(
                        0,
                        types.Part.from_uri(
                            file_uri=plot_of_attempt_scores_uri_for_refinement,
                            mime_type="image/png",
                        ),
                    )

                # Log the prompt that will be sent for this specific API call attempt
                # Uses current_pass_name, so will overwrite if MAX_TOKENS causes iterative prompt regen for the same pass.
                temp_prompt_log_path = mad_utils_0526.log_prompt_text_to_file(
                    prompt_for_current_api_call,
                    batch_extras_save_dir,
                    f"main_prompt_F{i_feat}_B{batch_num}.txt",  # Filename based on user diff
                    logger,
                )
                # `final_prompt_log_path` logic: Store the path of the prompt that is part of a successful flow or the last attempt.
                # This will be updated if the API call succeeds OR if this is the last attempt in same_key_retry_num after this inner loop breaks.
                # For now, it is simply the path of the prompt for the current code generation pass. Update if needed.
                final_prompt_log_path = temp_prompt_log_path

                try:
                    logger.info(
                        f"Batch {batch_num}, Code Gen {current_pass_name}, API Call Attempt {same_key_retry_num + 1} (Analysis Snippets: {len(current_analysis_snippets_for_this_pass)}): Calling LLM."
                    )
                    length_of_analysis_snippets_for_this_api_call = len(
                        current_analysis_snippets_for_this_pass
                    )  # Track for this specific call

                    response = gemini_api_utils_0526.execute_gemini_api_call(
                        api_call_func=current_attempt_client.models.generate_content,
                        contents=current_api_call_contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="text/plain",
                            temperature=0,  # Deterministic
                            thinking_config=types.ThinkingConfig(
                                include_thoughts=True,
                                thinking_budget=THINKING_BUDGET_MAIN_LLM,
                            ),
                        ),
                        logger=logger,
                        model=FIXED_MODEL_NAME_GEMINI_FLASH,
                    )

                    # Save thoughts if available
                    if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        thought_log_paths_batch = []
                        for part_idx, part in enumerate(
                            response.candidates[0].content.parts
                        ):
                            if (
                                hasattr(part, "thought")
                                and part.thought
                                and hasattr(part, "text")
                                and part.text
                            ):
                                thought_filename = f"main_llm_batch_F{i_feat}_B{batch_num}_{current_pass_name}_thoughts_key{current_api_key_index}_sk_retry{same_key_retry_num}_part{part_idx}.txt"
                                thought_log_path = os.path.join(
                                    batch_extras_save_dir, thought_filename
                                )
                                try:
                                    with open(thought_log_path, "w") as f_thought:
                                        f_thought.write(
                                            f"Thought Source: Main LLM Batch, Feature {i_feat}, Batch {batch_num}, Pass {current_pass_name}, Key Index {current_api_key_index}, Same-Key Retry {same_key_retry_num + 1}, Part {part_idx}\n---BEGIN THOUGHT---\n"
                                        )
                                        f_thought.write(part.text)
                                        f_thought.write("\n---END THOUGHT---")
                                    logger.debug(
                                        f"Saved Main LLM Batch thought to {thought_log_path}"
                                    )
                                    thought_log_paths_batch.append(thought_filename)
                                except Exception as e_save_thought_batch:
                                    logger.warning(
                                        f"Failed to save Main LLM Batch thought: {mad_utils_0526._format_exception_for_logging(e_save_thought_batch)}"
                                    )
                        if thought_log_paths_batch:
                            # Add to batch_output_artifact_data, perhaps under a specific key for this attempt
                            # This will be overwritten if further retries/passes occur for this batch, which is acceptable
                            # as we only care about the thoughts of the *final* successful or last attempted LLM call for the batch artifacts.
                            batch_output_artifact_data[
                                f"thoughts_log_paths_pass_{current_pass_name}_sk_retry{same_key_retry_num}"
                            ] = thought_log_paths_batch

                    if (
                        response.candidates
                        and response.candidates[0].finish_reason
                        == FinishReason.MAX_TOKENS
                    ):
                        logger.info(
                            f"Batch {batch_num}, Code Gen {current_pass_name}, API Call (SKR {same_key_retry_num + 1}): MAX_TOKENS encountered."
                        )

                        if not attempted_data_param_retry_for_max_tokens:
                            logger.info(
                                f"Batch {batch_num}, Code Gen {current_pass_name}: First MAX_TOKENS. Attempting special retry: generate function expecting X_data_col parameter."
                            )
                            attempted_data_param_retry_for_max_tokens = True
                            # DO NOT halve snippets yet. The prompt constructor will now use generate_func_with_data_param=True.
                            continue # Retry the API call immediately with the modified prompt strategy.

                        if (
                            not current_analysis_snippets_for_this_pass
                        ):  # No snippets left to halve
                            logger.error(
                                f"Batch {batch_num}, Code Gen {current_pass_name}: MAX_TOKENS but no analysis snippets left to halve. Cannot recover for this LLM call."
                            )
                            execution_error_details_for_refinement = "LLM output truncated (MAX_TOKENS) even with zero analysis snippets. Prompt is too large."
                            llm_response_text_current_attempt = (
                                response.text
                            )  # Store truncated text
                            final_llm_response_raw = llm_response_text_current_attempt
                            llm_call_succeeded_this_attempt = False
                            last_exception_for_batch = ValueError(
                                execution_error_details_for_refinement
                            )
                            break  # Break from inner MAX_TOKENS halving loop; API call failed for this SKR attempt.

                        # If here, it means the special data_param_retry was already attempted (or wasn't applicable)
                        # and it also hit MAX_TOKENS, OR this is a subsequent MAX_TOKENS after the special retry.
                        # So, proceed with snippet halving.
                        logger.info(
                            f"Batch {batch_num}, Code Gen {current_pass_name}: MAX_TOKENS after special retry. Proceeding with snippet halving (halving iteration {halving_iteration_this_pass + 1}) while keeping parameterized function."
                        )
                        halving_iteration_this_pass += 1
                        num_analysis_current = len(
                            current_analysis_snippets_for_this_pass
                        )
                        current_analysis_snippets_for_this_pass = (
                            current_analysis_snippets_for_this_pass[
                                : num_analysis_current // 2
                            ]
                        )
                        current_analysis_snippets_map_for_this_pass = (
                            mad_utils_0526.convert_snippet_list_to_final_json(
                                current_analysis_snippets_for_this_pass, False, logger
                            )
                        )
                        analysis_indices_info_str_for_this_pass = f"Analysis snippets (halved to {len(current_analysis_snippets_for_this_pass)} items due to MAX_TOKENS, iter {halving_iteration_this_pass} for pass {current_pass_name})"

                        new_plot_local_path_after_halving = os.path.join(
                            batch_extras_save_dir,
                            f"plot_LLM_Input_F{i_feat}_B{batch_num}.png",  # Filename based on user diff (one per pass)
                        )
                        plotting_utils_0526.generate_batch_llm_input_plot(
                            X_data_col=X_data_col,
                            i_feat=i_feat,
                            batch_num=batch_num,
                            n_samples=n_samples,
                            identified_train_interest_range=None,
                            use_training_labels_for_hint=True,
                            y_labels_for_overall_hint=(
                                y_train_fit_labels if True else None
                            ),
                            train_len_for_overall_hint=(
                                len(X_train_fit_data)
                                if X_train_fit_data is not None
                                else 0
                            ),
                            current_batch_training_snippets_list=current_batch_collected_training_snippets_list,
                            current_batch_analysis_snippets_list=current_analysis_snippets_for_this_pass,
                            plot_path=new_plot_local_path_after_halving,
                            logger=logger,
                            plot_fig_height=6,
                            plot_dpi=100,
                        )
                        try:
                            logger.info(
                                f"Batch {batch_num}, MAX_TOKENS Halving Iter {halving_iteration_this_pass} for pass {current_pass_name}: Uploading new plot: {new_plot_local_path_after_halving}"
                            )
                            uploaded_halved_plot_response = (
                                gemini_api_utils_0526.upload_file_to_gemini(
                                    client=current_attempt_client,
                                    file_path=new_plot_local_path_after_halving,
                                    logger=logger,
                                )
                            )
                            if uploaded_halved_plot_response and hasattr(
                                uploaded_halved_plot_response, "uri"
                            ):
                                image_uri_for_prompt = uploaded_halved_plot_response.uri
                                logger.info(
                                    f"Batch {batch_num}, MAX_TOKENS Halving Iter {halving_iteration_this_pass} for pass {current_pass_name}: New plot uploaded: {image_uri_for_prompt}"
                                )
                            else:
                                logger.warning(
                                    f"Batch {batch_num}, MAX_TOKENS Halving Iter {halving_iteration_this_pass} for pass {current_pass_name}: New plot upload failed to return URI. Using previous or no plot URI."
                                )
                        except Exception as e_reupload:
                            logger.warning(
                                f"Batch {batch_num}, MAX_TOKENS Halving Iter {halving_iteration_this_pass} for pass {current_pass_name}: Failed to re-upload plot. Error: {mad_utils_0526._format_exception_for_logging(e_reupload)}. Using previous or no plot URI."
                            )

                        logger.info(
                            f"Batch {batch_num}, Code Gen {current_pass_name}: Retrying API call (SKR {same_key_retry_num + 1}) with {len(current_analysis_snippets_for_this_pass)} analysis snippets due to MAX_TOKENS."
                        )
                        continue  # Continue inner MAX_TOKENS loop (does not affect same_key_retry_num)

                    if not (
                        response.candidates
                        and response.candidates[0].finish_reason == FinishReason.STOP
                    ):
                        err_msg = f"LLM call for Code Gen {current_pass_name} finished with unhandled reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}."
                        logger.warning(f"Batch {batch_num}: {err_msg}")
                        raise ValueError(err_msg)

                    llm_response_text_current_attempt = response.text
                    final_llm_response_raw = llm_response_text_current_attempt
                    llm_call_succeeded_this_attempt = True
                    num_analysis_snippets_in_llm_call_for_final_code = (
                        length_of_analysis_snippets_for_this_api_call
                    )
                    logger.info(
                        f"Batch {batch_num}, Code Gen {current_pass_name}, API Call (SKR {same_key_retry_num + 1}): LLM call SUCCEEDED {'' if not attempted_data_param_retry_for_max_tokens else '(using X_data_col parameter)'}"
                        + f"{f' with halved snippets (iteration {halving_iteration_this_pass})' if halving_iteration_this_pass > 0 else ''}"
                    )
                    # If it succeeded with data_param mode, store this fact for execution
                    batch_output_artifact_data["code_expects_X_data_col_param"] = (
                        attempted_data_param_retry_for_max_tokens # Modified: Always use parameterized function after first MAX_TOKENS
                    )
                    break  # Break from inner MAX_TOKENS loop; API call was successful for this SKR attempt.

                except (
                    Exception
                ) as e_llm_call:  # Catch other API errors (network, etc.)
                    last_exception_for_batch = e_llm_call
                    logger.warning(
                        f"Batch {batch_num}, Code Gen {current_pass_name}, API Call Attempt {same_key_retry_num + 1} FAILED. Error: {mad_utils_0526._format_exception_for_logging(e_llm_call)}"
                    )
                    # --- Robust Key Fatal Error Check ---
                    key_is_truly_fatal = False
                    error_lower_str = str(e_llm_call).lower() if e_llm_call else ""
                    parsed_retry_delay_seconds = mad_utils_0526.extract_retry_delay_from_error_details_json(error_lower_str)
                    # Attempt to extract quota info, assuming mad_utils.extract_quota_info_from_error_details_json is available
                    quota_info = None
                    try:
                        quota_info = mad_utils_0526.extract_quota_info_from_error_details_json(error_lower_str)
                    except AttributeError: # If the function isn't there yet, log a warning but continue
                        logger.warning("mad_utils.extract_quota_info_from_error_details_json not found. Quota info will be unavailable for this error.")


                    if quota_info:
                        logger.info(
                            f"Batch {batch_num}, Code Gen {current_pass_name}: Quota details from error - Metric: {quota_info.get('quotaMetric', 'N/A')}, ID: {quota_info.get('quotaId', 'N/A')}, Value: {quota_info.get('quotaValue', 'N/A')}"
                        )
                        # Check if it's an input token quota and retriable
                        if quota_info and isinstance(quota_info.get("quotaValue"), (int, float)) and \
                           quota_info.get("quotaMetric") and "input_token" in quota_info.get("quotaMetric").lower() and \
                           parsed_retry_delay_seconds is not None: # Ensure it's retriable
                            
                            learned_input_token_quota_for_batch = int(float(quota_info["quotaValue"]))
                            logger.info(
                                f"Batch {batch_num}, Code Gen {current_pass_name}: Learned input token quota limit for current key: {learned_input_token_quota_for_batch} from API error. This will be returned to the main orchestrator."
                            )

                    # If a retryDelay is present, it's usually a 429 error, which is not immediately key-fatal.
                    if parsed_retry_delay_seconds is not None:
                        logger.info(
                            f"Batch {batch_num}, Code Gen {current_pass_name}: Retriable API error (e.g., 429 with retryDelay={parsed_retry_delay_seconds}s). Will retry with same key if attempts remain."
                        )
                        # This error is not key-fatal *yet*, the same_key_retry_num loop will handle it.
                    else:
                        # Check for other fatal keywords if no retryDelay was parsed
                        fatal_keywords = [
                            "api key not valid",
                            "api_key_invalid",
                            "permissiondenied",
                            # "resource_exhausted", # Not fatal if retryDelay is present
                            "billing",
                            "developer_inactive",
                            "consumer_invalid",
                            "api key required",
                        ]
                        if any(p in error_lower_str for p in fatal_keywords):
                            key_is_truly_fatal = True

                        if not key_is_truly_fatal and hasattr(e_llm_call, "code"):
                            # 401: Unauthorized (API key issue)
                            # 403: Forbidden (Permission denied, API not enabled, billing)
                            if e_llm_call.code in [401, 403]: # Only 401 and 403 are definitively fatal here
                                key_is_truly_fatal = True
                                if not any(p in error_lower_str for p in fatal_keywords): # Log if code-based detection was primary
                                    logger.info(f"Batch {batch_num}, Code Gen {current_pass_name}: Marking key as fatal due to API status code: {e_llm_call.code}")
                            elif e_llm_call.code == 429 and parsed_retry_delay_seconds is None:
                                # 429 is only fatal if there's no specific retry instruction from the API.
                                key_is_truly_fatal = True
                                logger.info(f"Batch {batch_num}, Code Gen {current_pass_name}: Marking key as fatal due to 429 error WITHOUT a retryDelay (and not caught by 'resource_exhausted' keyword).")
                    # --- End Robust Key Fatal Error Check ---

                    if key_is_truly_fatal:
                        logger.error(
                            f"API Key {current_api_key_index} invalid/exhausted. Error details: {error_lower_str}. Breaking from all retries for this batch."
                        )
                        return (
                            False,
                            None,
                            {},
                            last_exception_for_batch,
                            num_analysis_snippets_in_llm_call_for_final_code,
                            learned_input_token_quota_for_batch,
                        )  # Fatal for this key attempt
                    # This break is for the inner MAX_TOKENS loop. The outer same_key_retry_num loop will handle delay/retry.
                    llm_call_succeeded_this_attempt = (
                        False  # Ensure it's marked as failed for this SKR attempt
                    )
                    break  # Break from inner MAX_TOKENS loop, to proceed to SKR delay/retry or SKR exhaustion

            # After inner MAX_TOKENS loop finishes (either by break on success, MAX_TOKENS unrecoverable, or other API error)
            if llm_call_succeeded_this_attempt:
                break  # Break from outer same_key_retry_num loop, as we have a successful API call for this pass

            # If here, inner loop broke due to an API error (not success, not unrecoverable MAX_TOKENS handled inside it)
            # OR unrecoverable MAX_TOKENS where llm_call_succeeded_this_attempt is False.
            # The same_key_retry_num loop will provide the delay IF it's not exhausted.
            if same_key_retry_num < max_retries_with_delay_per_key - 1:
                delay_s = default_retry_delay_seconds
                # Use parsed_retry_delay_seconds if available from the API error, otherwise default.
                if parsed_retry_delay_seconds is not None and parsed_retry_delay_seconds > 0:
                    delay_s = parsed_retry_delay_seconds
                    logger.info(
                        f"Batch {batch_num}, Code Gen {current_pass_name}: Using API suggested retryDelay: {delay_s}s."
                    )
                logger.info(
                    f"Batch {batch_num}, Code Gen {current_pass_name}: General API error. Retrying SKR attempt after {delay_s}s..."
                )
                time.sleep(delay_s)
            # else: same_key_retry_num loop will exhaust, and llm_call_succeeded_this_attempt will remain False for this pass.

        # After the `for same_key_retry_num` loop (all SKR attempts for this code gen pass `attempt_num` are done)
        if not llm_call_succeeded_this_attempt or not llm_response_text_current_attempt:
            logger.error(
                f"Batch {batch_num}, Code Gen {current_pass_name}: Failed to get code from LLM after all same-key retries."
            )
            # last_exception_for_batch should already be set by the last failed API call attempt.
            # No need to reset num_analysis_snippets_in_llm_call_for_final_code here, it holds value from last successful API call *if any* for this pass.
            # If no API call ever succeeded for this pass, it remains its initial value for this pass (or from a previous pass if not reset).
            # It should reflect the snippets for the code that will be *attempted* to be executed or refined from.
            # If llm_response_text_current_attempt is None (e.g. all API calls failed with network issues before MAX_TOKENS even), then previous_code_for_refinement might be from an even earlier pass.
            # This needs to be robust.
            # If llm_response_text_current_attempt is None, we can't proceed to execution. execution_error_details_for_refinement should indicate this.
            if not llm_response_text_current_attempt:
                execution_error_details_for_refinement = "LLM call failed to produce any response text after all retries (e.g., network issues, persistent API errors not MAX_TOKENS related)."
                # last_exception_for_batch will carry the actual exception from the API call layer.

            # If we are here, it means the LLM call itself failed for this `current_pass_name`
            # We need to decide if we should try another `attempt_num` (refinement) or give up on the batch.
            # The `attempt_num` loop will handle the safety break.
            # Prepare for a potential next refinement attempt if loop continues:
            previous_code_for_refinement = (
                None  # No new code was generated to become the previous code
            )
            # execution_error_details_for_refinement is already set if MAX_TOKENS unrecoverable, or from the condition above.
            # last_exception_for_batch holds the last actual exception from the API call layer.

            # If we are here because of persistent non-MAX_TOKENS API errors,
            # previous_code_for_refinement would be from the *prior successful code generation pass*,
            # and execution_error_details_for_refinement should reflect the API call failure.
            # This seems okay, the refinement will get the old code and the new API error.

            if (
                attempt_num >= 100
            ):  # Safety break already checked later, but good for reasoning
                logger.error(
                    f"Batch {batch_num}: Reached safety limit. Aborting batch without successful code generation."
                )
                # Fall through, batch_fully_successful_for_this_key will be false.
                break  # Break from outer `while True` (attempt_num loop)

            attempt_num += 1
            continue  # To next attempt_num (refinement pass for the code generation itself)

        generated_code_str = mad_utils_0526.strip_markdown_code_fences(
            llm_response_text_current_attempt
        )
        current_code_log_path = mad_utils_0526.log_code_to_file(
            generated_code_str,
            batch_extras_save_dir,
            f"generated_code_F{i_feat}_B{batch_num}_{current_pass_name}.py",
            logger,
        )
        final_code_log_path = current_code_log_path  # Update with latest

        exec_namespace = {"np": np}
        temp_scores_this_attempt = np.zeros(n_samples, dtype=float)
        execution_succeeded_this_attempt = False
        execution_error_details_for_refinement = None  # Reset for this attempt

        try:
            exec(generated_code_str, exec_namespace)
            anomaly_func = exec_namespace.get("calculate_anomaly_scores")
            if callable(anomaly_func):
                # Check if the code was generated to expect X_data_col
                if batch_output_artifact_data.get("code_expects_X_data_col_param", False):
                    logger.info(f"Batch {batch_num}, Code Gen {current_pass_name}: Executing generated code WITH X_data_col parameter.")
                    temp_scores_this_attempt = anomaly_func(X_data_col=X_data_col.copy()) # Pass a copy
                else:
                    temp_scores_this_attempt = anomaly_func()
                if isinstance(
                    temp_scores_this_attempt, np.ndarray
                ) and temp_scores_this_attempt.shape == (n_samples,):
                    logger.info(
                        f"Batch {batch_num}, Code Gen {current_pass_name}: Code execution SUCCEEDED."
                    )
                    execution_succeeded_this_attempt = True
                    batch_fully_successful_for_this_key = (
                        True  # This batch is now considered successful
                    )
                    current_batch_scores_from_llm = temp_scores_this_attempt.copy()
                    pass_name_of_final_code = current_pass_name
                    # Store the successful code and response as the final ones for artifacts
                    # final_python_code_str = generated_code_str (already tracked by final_code_log_path)
                    # final_llm_response_raw = llm_response_text_current_attempt (already tracked)
                    break  # Break from the code generation attempt_num loop
                else:
                    err_msg_exec = f"Code Gen {current_pass_name} returned invalid scores. Shape: {temp_scores_this_attempt.shape if isinstance(temp_scores_this_attempt, np.ndarray) else 'Not ndarray'}"
                    logger.error(f"Batch {batch_num}: {err_msg_exec}")
                    execution_error_details_for_refinement = err_msg_exec
            else:
                err_msg_exec = f"'calculate_anomaly_scores' not found in Code Gen {current_pass_name}."
                logger.error(f"Batch {batch_num}: {err_msg_exec}")
                execution_error_details_for_refinement = err_msg_exec
        except Exception as e_exec:
            formatted_exec_error = mad_utils_0526._format_exception_for_logging(e_exec)
            logger.error(
                f"Batch {batch_num}, Code Gen {current_pass_name}: Code execution FAILED: {formatted_exec_error}"
            )
            execution_error_details_for_refinement = formatted_exec_error
            last_exception_for_batch = e_exec  # Update overall last exception

        # If execution failed, prepare for next refinement attempt
        if not execution_succeeded_this_attempt:
            previous_code_for_refinement = (
                generated_code_str  # This is the code that just failed
            )
            # previous_scores_for_refinement_plot is no longer needed for the prompt

            # Plot scores from this failed attempt for local artifacts
            plot_path_failed_attempt_scores = os.path.join(
                batch_extras_save_dir,
                f"llm_anomalies_F{i_feat}_B{batch_num}_{current_pass_name}_scores_FAILED.png",
            )
            plot_success_failed_scores, _ = (
                plotting_utils_0526.generate_llm_identified_anomalies_plot(
                    X_data_col=X_data_col,
                    llm_anomaly_scores=temp_scores_this_attempt,
                    i_feat=i_feat,
                    batch_num=batch_num,
                    api_key_index=current_api_key_index,
                    plot_path=plot_path_failed_attempt_scores,
                    logger=logger,
                    plot_fig_height=5,
                    plot_dpi=100,
                    analysis_snippets_to_highlight=current_analysis_snippets_for_this_pass,
                )
            )
            plot_of_attempt_scores_uri_for_refinement = None  # Reset
            if plot_success_failed_scores:
                for upload_retry in range(max_retries_with_delay_per_key):
                    try:
                        uploaded_failed_plot = gemini_api_utils_0526.upload_file_to_gemini(
                            current_attempt_client,
                            plot_path_failed_attempt_scores,
                            logger,
                        )
                        if uploaded_failed_plot and hasattr(
                            uploaded_failed_plot, "uri"
                        ):
                            plot_of_attempt_scores_uri_for_refinement = (
                                uploaded_failed_plot.uri
                            )
                            break
                    except Exception as e_upload_fail_plot:
                        logger.warning(
                            f"Batch {batch_num}, Upload of failed scores plot (for refinement) attempt {upload_retry+1} failed: {mad_utils_0526._format_exception_for_logging(e_upload_fail_plot)}"
                        )
                        if upload_retry < max_retries_with_delay_per_key - 1:
                            time.sleep(default_retry_delay_seconds)
                        # else: URI remains None

            if attempt_num >= 100:  # Safety break for indefinite loop
                logger.error(
                    f"Batch {batch_num}: Reached safety limit for code generation passes (100). Aborting batch."
                )
                # Ensure batch_fully_successful_for_this_key is False if we hit this safety break before success
                if not batch_fully_successful_for_this_key:
                    current_batch_scores_from_llm = (
                        temp_scores_this_attempt.copy()
                        if "temp_scores_this_attempt" in locals()
                        else np.zeros(n_samples, dtype=float)
                    )
                    last_exception_for_batch = last_exception_for_batch or RuntimeError(
                        "Reached max code gen passes safety limit."
                    )
                break  # Break from the while True loop

        attempt_num += 1  # Increment attempt number (pass number for code generation)

    # --- Final Artifacts and Plotting ---
    # current_batch_scores_from_llm holds the scores from the successful attempt, or the last attempt if all failed.
    # batch_fully_successful_for_this_key indicates if any attempt succeeded.

    final_plot_filename_suffix = f"Final_Scores_From_{pass_name_of_final_code}"
    if not batch_fully_successful_for_this_key:
        final_plot_filename_suffix += "_EXEC_FAILED_ALL_ATTEMPTS"

    final_scores_plot_path = os.path.join(
        batch_extras_save_dir,
        f"llm_anomalies_F{i_feat}_B{batch_num}_{final_plot_filename_suffix}.png",
    )
    plotting_utils_0526.generate_llm_identified_anomalies_plot(
        X_data_col=X_data_col,
        llm_anomaly_scores=current_batch_scores_from_llm,  # Scores from successful or last failed attempt
        i_feat=i_feat,
        batch_num=batch_num,
        api_key_index=current_api_key_index,
        plot_path=final_scores_plot_path,
        logger=logger,
        plot_fig_height=5,
        plot_dpi=100,
        analysis_snippets_to_highlight=current_batch_collected_analysis_snippets_list[
            :num_analysis_snippets_in_llm_call_for_final_code
        ],
    )

    # Retrieve the actual code string if final_code_log_path is set
    final_code_str_for_artifact = "N/A"
    if final_code_log_path and os.path.exists(final_code_log_path):
        try:
            with open(final_code_log_path, "r") as f_code:
                final_code_str_for_artifact = f_code.read()
        except Exception:  # nosec
            pass  # Keep "N/A"

    batch_output_artifact_data = {
        "feature_index": i_feat,
        "batch_number": batch_num,
        "api_key_index_used": current_api_key_index,
        "training_data_metadata_provided": training_metadata_for_prompt,
        "final_llm_response_raw": final_llm_response_raw,  # From the attempt that was chosen/last
        "final_generated_python_code": final_code_str_for_artifact,
        "final_code_pass_used": pass_name_of_final_code,
        "code_execution_successful_overall": batch_fully_successful_for_this_key,
        "input_snippets_training": sorted(
            current_batch_collected_training_snippets_list,
            key=lambda s: s.get("index", float("inf")),
        ),
        "input_snippets_analysis": sorted(
            current_batch_collected_analysis_snippets_list,
            key=lambda s: s.get("index", float("inf")),
        ),
        "num_analysis_snippets_in_llm_call_for_final_code": num_analysis_snippets_in_llm_call_for_final_code,
        "plot_local_path_llm_input_initial": plot_local_path_this_batch,
        "final_image_uri_used_for_llm_call": image_uri_for_prompt,  # URI of the plot used for the final successful/attempted LLM call
        "plot_local_path_final_scores": final_scores_plot_path,
        "prompt_text_final_saved_path": final_prompt_log_path,  # From the attempt that was chosen/last
        "code_final_saved_path": final_code_log_path,  # From the attempt that was chosen/last
        "model_used": FIXED_MODEL_NAME_GEMINI_FLASH,
        "total_code_generation_attempts": attempt_num,  # Number of iterations in the loop (will be max_total_attempts if all failed)
    }
    # last_exception_for_batch will hold the last critical exception if the whole process failed

    return (
        batch_fully_successful_for_this_key,
        current_batch_scores_from_llm,  # These are the scores from the successful or last failed attempt.
        batch_output_artifact_data,
        last_exception_for_batch,
        num_analysis_snippets_in_llm_call_for_final_code,
        learned_input_token_quota_for_batch, # New return
    )
