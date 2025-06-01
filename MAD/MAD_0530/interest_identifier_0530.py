import json
import os
import re
import time

from google.genai import types

from . import (
    gemini_api_utils_0530,
    mad_utils_0530,
    plotting_utils_0530,
    prompt_utils_0530,
)
from .constants_0530 import FIXED_MODEL_NAME_GEMINI_FLASH, THINKING_BUDGET_INTEREST_ID


def perform_interest_identification_step(
    # Data inputs
    feature_data_train,
    feature_labels_train,
    i_feat,
    n_train_samples,
    temp_artifact_base_dir_for_step,
    target_range_size,
    logger,
    api_keys_list,
    MAX_RETRIES_WITH_DELAY_PER_KEY_const,
    DEFAULT_RETRY_DELAY_SECONDS_const,
    full_feature_data_for_plot,
    actual_train_len_for_plot,
    primary_client_for_budgeting,
    input_token_limit_override: int | None,
):
    step_artifacts = {}
    last_exception_for_step = None
    parsed_range_final = None
    current_snippets_json_for_prompt = json.dumps(
        mad_utils_0530.convert_snippet_list_to_final_json(
            [], include_labels=True, logger=logger
        ),
        indent=2,
    )
    selected_snippets_list_for_id_step = []

    if n_train_samples == 0:
        logger.info(
            f"Feature {i_feat} Interest ID: No training samples. Skipping step."
        )
        step_artifacts["info"] = "No training samples, interest ID step skipped."
        logger.debug(
            f"Feature {i_feat}: Returning None, artifacts, and empty JSON due to no train samples."
        )
        return None, step_artifacts, current_snippets_json_for_prompt, False, None
    plot_filename = f"train_interest_id_feature_{i_feat}_plot.png"
    plot_path = os.path.join(temp_artifact_base_dir_for_step, plot_filename)
    plot_generated_successfully = False

    plot_generated, plot_error_info = plotting_utils_0530.generate_interest_id_plot(
        feature_data_train=feature_data_train,
        feature_labels_train=feature_labels_train,
        i_feat=i_feat,
        n_train_samples=n_train_samples,
        target_range_size=target_range_size,
        plot_path=plot_path,
        logger=logger,
        full_feature_data_for_context=full_feature_data_for_plot,
        actual_train_len_for_context=actual_train_len_for_plot,
    )
    logger.debug(
        f"Feature {i_feat}: Interest ID plot generation attempt complete. Success: {plot_generated}. Error info: {plot_error_info}"
    )
    if plot_generated:
        step_artifacts["plot_path_relative_to_step_dir"] = plot_filename
        plot_generated_successfully = True
    if plot_error_info:
        step_artifacts["error_plotting"] = plot_error_info
    client_for_snippet_prep = None
    client_for_snippet_prep = primary_client_for_budgeting

    if not client_for_snippet_prep:
        logger.error(
            f"Ft {i_feat} Interest ID: Could not initialize/use client for snippet preparation. Proceeding without snippets for LLM if data exists."
        )
        step_artifacts["error_snippet_preparation"] = (
            "Client (primary_client_for_budgeting) was None or failed for snippet prep."
        )
    else:
        logger.debug(
            f"Feature {i_feat}: Preparing training snippets for Interest ID step."
        )
        selected_snippets_list_for_id_step, _, prep_error = (
            mad_utils_0530.prepare_training_snippets_for_interest_id(
                feature_data_train=feature_data_train,
                feature_labels_train=feature_labels_train,
                i_feat=i_feat,
                n_train_samples=n_train_samples,
                target_range_size_for_prompt=target_range_size,
                client=client_for_snippet_prep,
                input_token_limit_override=input_token_limit_override,
                logger=logger,
            )
        )
        if prep_error:
            step_artifacts["error_snippet_preparation"] = prep_error
        if selected_snippets_list_for_id_step:
            selected_snippets_list_for_id_step_sorted = sorted(
                selected_snippets_list_for_id_step,
                key=lambda s: s.get("index", float("inf")),
            )
            selected_snippets_list_for_id_step = (
                selected_snippets_list_for_id_step_sorted
            )

            current_snippets_json_for_prompt = json.dumps(
                mad_utils_0530.convert_snippet_list_to_final_json(
                    selected_snippets_list_for_id_step,
                    include_labels=True,
                    logger=logger,
                ),
                indent=2,
            )
            logger.debug(
                f"Feature {i_feat}: Snippets prepared and sorted for prompt. Count: {len(selected_snippets_list_for_id_step)}. JSON length: {len(current_snippets_json_for_prompt)}"
            )
        elif not selected_snippets_list_for_id_step and not prep_error:
            current_snippets_json_for_prompt = json.dumps(
                mad_utils_0530.convert_snippet_list_to_final_json(
                    [], include_labels=True, logger=logger
                ),
                indent=2,
            )

    step_artifacts["selected_snippets_count_for_id_step"] = len(
        selected_snippets_list_for_id_step
    )
    effective_target_range_for_llm_prompt = target_range_size
    learned_input_token_quota_for_step = None
    if effective_target_range_for_llm_prompt == 0 and n_train_samples > 0:
        effective_target_range_for_llm_prompt = 1
    llm_call_succeeded_for_interest_id = False
    key_fatal_error_occurred = False
    uploaded_plot_file_obj_id = None

    for attempt_key_cycle_idx in range(len(api_keys_list)):
        uploaded_plot_file_obj_id = None
        if llm_call_succeeded_for_interest_id:
            break

        current_api_key_value = api_keys_list[attempt_key_cycle_idx]
        logger.info(
            f"Ft {i_feat} Interest ID, LLM Call Attempt with Key Index {attempt_key_cycle_idx}"
        )
        client_handle_for_key_attempt = gemini_api_utils_0530.initialize_gemini_client(
            current_api_key_value, logger
        )

        if not client_handle_for_key_attempt:
            logger.warning(
                f"Ft {i_feat} Interest ID, Key Idx {attempt_key_cycle_idx}: Client init/config failed. Skipping to next key."
            )
            last_exception_for_step = RuntimeError(
                f"Client init/config failed for key index {attempt_key_cycle_idx}."
            )
            continue
        for same_key_retry_num in range(MAX_RETRIES_WITH_DELAY_PER_KEY_const):
            if llm_call_succeeded_for_interest_id:
                break
            logger.info(
                f"Ft {i_feat} Interest ID, Key Idx {attempt_key_cycle_idx}, Same-Key Attempt {same_key_retry_num + 1}: Calling LLM."
            )

            image_part_for_llm = None
            uploaded_plot_uri_for_prompt_update = None

            try:
                if plot_generated_successfully and not uploaded_plot_file_obj_id:
                    try:
                        uploaded_plot_file_obj_id = (
                            gemini_api_utils_0530.upload_file_to_gemini(
                                client=client_handle_for_key_attempt,
                                file_path=plot_path,
                                logger=logger,
                            )
                        )
                        if uploaded_plot_file_obj_id and hasattr(
                            uploaded_plot_file_obj_id, "uri"
                        ):
                            logger.info(
                                f"Ft {i_feat} Interest ID: Plot {plot_filename} uploaded successfully (Key Idx {attempt_key_cycle_idx}): {uploaded_plot_file_obj_id.uri}"
                            )
                        else:
                            logger.warning(
                                f"Ft {i_feat} Interest ID: Plot upload (Key Idx {attempt_key_cycle_idx}) for {plot_filename} seemed to succeed but URI missing. Proceeding without plot for this API call attempt."
                            )
                            uploaded_plot_file_obj_id = None
                    except Exception as e_upload_plot_id:
                        logger.warning(
                            f"Ft {i_feat} Interest ID: Plot upload (Key Idx {attempt_key_cycle_idx}) failed for {plot_filename} on same-key attempt {same_key_retry_num + 1}. Error: {mad_utils_0530._format_exception_for_logging(e_upload_plot_id)}. Proceeding without plot for this API call attempt."
                        )
                        uploaded_plot_file_obj_id = None

                if uploaded_plot_file_obj_id and hasattr(
                    uploaded_plot_file_obj_id, "uri"
                ):
                    image_part_for_llm = types.Part.from_uri(
                        mime_type=uploaded_plot_file_obj_id.mime_type or "image/png",
                        file_uri=uploaded_plot_file_obj_id.uri,
                    )
                    uploaded_plot_uri_for_prompt_update = uploaded_plot_file_obj_id.uri
                current_prompt_text_for_api_call = prompt_utils_0530.construct_interest_id_prompt(
                    i_feat=i_feat,
                    num_train_samples=n_train_samples,
                    training_snippets_json_str_for_id_step=current_snippets_json_for_prompt,
                    target_range_size=effective_target_range_for_llm_prompt,
                    image_uri_for_llm=uploaded_plot_uri_for_prompt_update,
                )
                prompt_log_fname = f"train_interest_id_feature_{i_feat}_prompt.txt"
                prompt_log_path = os.path.join(
                    temp_artifact_base_dir_for_step, prompt_log_fname
                )
                try:
                    with open(prompt_log_path, "w") as f:
                        f.write(current_prompt_text_for_api_call)
                    step_artifacts["prompt_log_path_relative_to_step_dir"] = (
                        prompt_log_fname
                    )
                    logger.debug(
                        f"Feature {i_feat}: Logged Interest ID prompt (image URI: {uploaded_plot_uri_for_prompt_update is not None}) to {prompt_log_path}"
                    )
                except Exception as e_log_prompt:
                    logger.warning(
                        f"Ft {i_feat} Interest ID: Failed to log prompt: {e_log_prompt}"
                    )
                logger.debug(
                    f"Feature {i_feat}: Constructed prompt for API call (image URI: {uploaded_plot_uri_for_prompt_update is not None})"
                )

                llm_contents_for_api_call = []
                if image_part_for_llm:
                    llm_contents_for_api_call.append(image_part_for_llm)
                llm_contents_for_api_call.append(
                    types.Part.from_text(text=current_prompt_text_for_api_call)
                )
                generation_config_id = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "start_index": types.Schema(type=types.Type.INTEGER),
                                "end_index": types.Schema(type=types.Type.INTEGER),
                            },
                            required=["start_index", "end_index"],
                        ),
                    ),
                    temperature=0,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=THINKING_BUDGET_INTEREST_ID,
                    ),
                )
                api_call_to_run = client_handle_for_key_attempt.models.generate_content

                response_id_step = gemini_api_utils_0530.execute_gemini_api_call(
                    api_call_func=api_call_to_run,
                    config=generation_config_id,
                    contents=llm_contents_for_api_call,
                    logger=logger,
                    model=FIXED_MODEL_NAME_GEMINI_FLASH,
                )

                # Save thoughts if available
                if response_id_step and response_id_step.candidates:
                    thought_log_paths = []
                    for part_idx, part in enumerate(
                        response_id_step.candidates[0].content.parts
                    ):
                        if (
                            hasattr(part, "thought")
                            and part.thought
                            and hasattr(part, "text")
                            and part.text
                        ):
                            thought_filename = f"train_interest_id_feature_{i_feat}_thoughts_retry{same_key_retry_num}_part{part_idx}.txt"
                            thought_log_path = os.path.join(
                                temp_artifact_base_dir_for_step, thought_filename
                            )
                            try:
                                with open(thought_log_path, "w") as f_thought:
                                    f_thought.write(
                                        f"Thought Source: Interest ID Step, Feature {i_feat}, Key Index {attempt_key_cycle_idx}, Same-Key Retry {same_key_retry_num + 1}, Part {part_idx}\n---BEGIN THOUGHT---\n"
                                    )
                                    f_thought.write(part.text)
                                    f_thought.write("\n---END THOUGHT---")
                                logger.debug(
                                    f"Saved Interest ID thought to {thought_log_path}"
                                )
                                thought_log_paths.append(thought_filename)
                            except Exception as e_save_thought:
                                logger.warning(
                                    f"Failed to save Interest ID thought: {mad_utils_0530._format_exception_for_logging(e_save_thought)}"
                                )
                    if thought_log_paths:
                        step_artifacts[
                            f"thoughts_log_paths_key{attempt_key_cycle_idx}_retry{same_key_retry_num}"
                        ] = thought_log_paths

                if (
                    response_id_step
                    and hasattr(response_id_step, "parsed")
                    and response_id_step.parsed
                ):
                    step_artifacts["raw_llm_response_interest_id"] = (
                        response_id_step.parsed
                    )

                    parsed_json_list = response_id_step.parsed
                    if isinstance(parsed_json_list, list) and parsed_json_list:
                        for parsed_item in parsed_json_list:
                            if isinstance(parsed_item, dict):
                                s_idx, e_idx = parsed_item.get(
                                    "start_index"
                                ), parsed_item.get("end_index")

                                if (
                                    isinstance(s_idx, int)
                                    and isinstance(e_idx, int)
                                    and 0
                                    <= s_idx
                                    <= e_idx
                                    < (n_train_samples if n_train_samples > 0 else 1)
                                ):
                                    parsed_range_final = (s_idx, e_idx)
                                    step_artifacts["parsed_interest_range"] = (
                                        parsed_range_final
                                    )
                                    step_artifacts["all_parsed_interest_ranges"] = (
                                        parsed_json_list
                                    )
                                    llm_call_succeeded_for_interest_id = True
                                    logger.info(
                                        f"Ft {i_feat} Interest ID: Successfully parsed first valid range: {parsed_range_final} from {parsed_json_list}"
                                    )
                                    break

                        if not llm_call_succeeded_for_interest_id:
                            logger.error(
                                f"Ft {i_feat} Interest ID: LLM returned an array, but no valid range found within it. Raw: {parsed_json_list}"
                            )
                            raise ValueError(
                                "No valid range found in LLM response array."
                            )
                    elif isinstance(parsed_json_list, list) and not parsed_json_list:
                        logger.error(
                            f"Ft {i_feat} Interest ID: LLM returned an empty array. Raw: {parsed_json_list}"
                        )
                        raise ValueError(
                            "LLM returned an empty array for interest range."
                        )
                    else:
                        logger.error(
                            f"Ft {i_feat} Interest ID: LLM response was not a list as expected. Raw: {parsed_json_list}"
                        )
                        raise ValueError(
                            "LLM response for interest range was not a list."
                        )
                else:
                    last_exception_for_step = ValueError(
                        "LLM returned empty/invalid response for interest ID."
                    )
                    raise last_exception_for_step
            except Exception as e_other_id_call:
                last_exception_for_step = e_other_id_call
                logger.warning(
                    f"Ft {i_feat} Interest ID call (Google API Error {type(e_other_id_call).__name__}), Key {attempt_key_cycle_idx}, Attempt {same_key_retry_num+1}. Error: {mad_utils_0530._format_exception_for_logging(e_other_id_call)}"
                )
                error_lower_str_id = (
                    str(e_other_id_call).lower() if e_other_id_call else ""
                )
                parsed_retry_delay_seconds_id = (
                    mad_utils_0530.extract_retry_delay_from_error_details_json(
                        error_lower_str_id
                    )
                )
                # Attempt to extract quota info, assuming mad_utils.extract_quota_info_from_error_details_json is available
                quota_info_id = None
                try:
                    quota_info_id = (
                        mad_utils_0530.extract_quota_info_from_error_details_json(
                            error_lower_str_id
                        )
                    )
                except (
                    AttributeError
                ):  # If the function isn't there yet, log a warning but continue
                    logger.warning(
                        "mad_utils.extract_quota_info_from_error_details_json not found. Quota info will be unavailable for this error in Interest ID."
                    )

                if quota_info_id:
                    logger.info(
                        f"Ft {i_feat} Interest ID: Quota details from error - Metric: {quota_info_id.get('quotaMetric', 'N/A')}, ID: {quota_info_id.get('quotaId', 'N/A')}, Value: {quota_info_id.get('quotaValue', 'N/A')}"
                    )
                    # Check if it's an input token quota
                    if (
                        quota_info_id
                        and isinstance(quota_info_id.get("quotaValue"), (int, float))
                        and quota_info_id.get("quotaMetric")
                        and "input_token" in quota_info_id.get("quotaMetric").lower()
                        and parsed_retry_delay_seconds_id is not None
                    ):  # Ensure it's retriable

                        learned_input_token_quota_for_step = int(
                            float(quota_info_id["quotaValue"])
                        )
                        logger.info(
                            f"Ft {i_feat} Interest ID: Learned input token quota limit for current key: {learned_input_token_quota_for_step} from API error."
                        )

                if parsed_retry_delay_seconds_id is not None:
                    logger.info(
                        f"Ft {i_feat} Interest ID: Retriable API error (e.g., 429 with retryDelay={parsed_retry_delay_seconds_id}s). Will retry with same key if attempts remain."
                    )
                    # Not immediately fatal for the key
                else:
                    fatal_keywords_id = [
                        "api key not valid",
                        "api_key_invalid",
                        "permissiondenied",
                        # "resource_exhausted", # Not fatal if retryDelay is present
                        "billing",
                        "developer_inactive",
                        "consumer_invalid",
                        "api key required",
                    ]
                    if any(p in error_lower_str_id for p in fatal_keywords_id):
                        key_fatal_error_occurred = True

                    if not key_fatal_error_occurred and hasattr(
                        e_other_id_call, "code"
                    ):
                        if e_other_id_call.code in [
                            401,
                            403,
                        ]:  # Only 401 and 403 are definitively fatal here
                            key_fatal_error_occurred = True
                            if not any(
                                p in error_lower_str_id for p in fatal_keywords_id
                            ):
                                logger.info(
                                    f"Ft {i_feat} Interest ID: Marking key as fatal due to API status code: {e_other_id_call.code}"
                                )
                        elif (
                            e_other_id_call.code == 429
                            and parsed_retry_delay_seconds_id is None
                        ):  # Check for 429 specifically
                            # 429 is only fatal if there's no specific retry instruction from the API
                            # and not already caught by a fatal keyword (though 'resource_exhausted' was removed from fatal_keywords_id if retryDelay isn't present)
                            key_fatal_error_occurred = True
                            logger.info(
                                f"Ft {i_feat} Interest ID: Marking key as fatal due to 429 error WITHOUT a retryDelay."
                            )

                if key_fatal_error_occurred:
                    logger.error(
                        f"Ft {i_feat} Interest ID: API Key {attempt_key_cycle_idx} is invalid/exhausted. Error: {mad_utils_0530._format_exception_for_logging(e_other_id_call)}. Marking as fatal."
                    )
                    break
                if same_key_retry_num < MAX_RETRIES_WITH_DELAY_PER_KEY_const - 1:
                    delay_seconds = DEFAULT_RETRY_DELAY_SECONDS_const
                    # Use parsed_retry_delay_seconds_id if available from the API error, otherwise default
                    if (
                        parsed_retry_delay_seconds_id is not None
                        and parsed_retry_delay_seconds_id > 0
                    ):
                        delay_seconds = parsed_retry_delay_seconds_id
                        logger.info(
                            f"Ft {i_feat} Interest ID: Using API suggested retryDelay: {delay_seconds}s."
                        )
                    logger.info(f"Retrying in {delay_seconds}s.")
                    time.sleep(delay_seconds)
                else:
                    logger.error(
                        f"Ft {i_feat} Interest ID call failed after {MAX_RETRIES_WITH_DELAY_PER_KEY_const} attempts for key {attempt_key_cycle_idx}. Moving to next key if available."
                    )
                    break

        # After same_key_retry_num loop for the current API key in api_keys_list
        if llm_call_succeeded_for_interest_id:  # If successful with this key
            break  # Break from the outer `for attempt_key_cycle_idx` loop (which iterates api_keys_list)
        if key_fatal_error_occurred:  # If this key was marked fatal
            # The function will return, and MAD_May_22 will handle skipping to the next key from its main list.
            # No explicit break here needed from `for attempt_key_cycle_idx` as this loop is only one iteration
            # in the context of how perform_interest_identification_step is called by MAD_May_22.py
            # The important thing is that `key_fatal_error_occurred` is true upon return.
            pass

    if (
        not llm_call_succeeded_for_interest_id
    ):  # Covers all failure reasons for the (single) key attempt
        error_msg_log = (
            mad_utils_0530._format_exception_for_logging(last_exception_for_step)
            if last_exception_for_step
            else "LLM call failed for interest ID after all retries for the provided key."
        )
        step_artifacts["error_llm_call_interest_id"] = error_msg_log
        logger.error(
            f"Feature {i_feat}: LLM call for Interest Identification FAILED for key index {attempt_key_cycle_idx if 'attempt_key_cycle_idx' in locals() else 'N/A'}. {error_msg_log}"
        )
    elif not parsed_range_final:  # Succeeded but no valid range
        step_artifacts["error_llm_call_interest_id"] = step_artifacts.get(
            "error_llm_call_interest_id",
            "LLM call succeeded but produced no valid range.",
        )
        logger.error(
            f"Feature {i_feat}: LLM call for Interest ID Succeeded BUT no valid range was parsed. This is unexpected."
        )

    summary_artifact_fname = (
        f"train_interest_id_feature_{i_feat}_summary_artifacts.json"
    )
    summary_artifact_path = os.path.join(
        temp_artifact_base_dir_for_step, summary_artifact_fname
    )
    try:
        with open(summary_artifact_path, "w") as f_art:
            # Ensure key_fatal_error_occurred is in artifacts if it happened
            if key_fatal_error_occurred:
                step_artifacts["key_fatal_error_during_llm_call"] = True
            json.dump(step_artifacts, f_art, indent=2)
        logger.info(
            f"Feature {i_feat}: Saved Interest ID step summary artifacts to {summary_artifact_path}"
        )
    except Exception as e_save_summary:
        logger.error(
            f"Feature {i_feat}: Failed to save Interest ID step summary artifacts: {mad_utils_0530._format_exception_for_logging(e_save_summary)}"
        )

    return (
        parsed_range_final,
        step_artifacts,
        current_snippets_json_for_prompt,
        key_fatal_error_occurred,  # Ensure this is returned
        learned_input_token_quota_for_step,  # New return
    )
