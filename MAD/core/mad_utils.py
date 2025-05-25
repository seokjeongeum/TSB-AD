import ast
import json
import logging
import os
import pprint
import re

import numpy as np

from MAD.core.constants import (
    DEFAULT_FALLBACK_TOKEN_LIMIT,
    DEFAULT_OUTPUT_TOKENS_INTEREST_ID,
    FIXED_MODEL_NAME_GEMINI_FLASH,
    THINKING_BUDGET_INTEREST_ID,
    TOKEN_BUDGET_SAFETY_MARGIN_INTEREST_ID,
    TOKEN_LIMIT_SAFETY_FACTOR,
)

from . import gemini_api_utils, prompt_utils


def thousands_formatter(x, pos):
    "The two args are the value and tick position"
    if x >= 1e6:
        return "%.1fM" % (x * 1e-6)
    elif x >= 1e3:
        return "%.1fk" % (x * 1e-3)
    return "%.1f" % x


def _find_clusters(indices):
    if indices is None:  # Explicitly check for None first
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


def extract_base_dataset_name(filename):
    if not filename:
        return "unknown_dataset"
    parts = filename.split("_")
    if len(parts) > 1:
        known_prefixes = ["MSL", "SMAP", "SMD", "NAB", "UCR", "MBA", "ECG", "YAHOO"]
        for part in parts:
            if part.upper() in known_prefixes:
                return part.upper()
        potential_name = parts[1] if len(parts) > 1 else "unknown"
        if potential_name.isupper() or (potential_name and potential_name[0].isupper()):
            return potential_name
        return "unknown_dataset"
    return "unknown_dataset"


def extract_domain_from_filename(filename):
    if not filename:
        return "unknown_domain"
    match = re.search(r"_id_.*?_(.*?)_tr_", filename)
    if match:
        return match.group(1)
    parts = filename.split("_")
    if len(parts) > 3 and parts[2].lower() == "id":
        return parts[3]
    elif len(parts) > 1:
        return parts[1]
    return "unknown_domain"


def strip_markdown_code_fences(code_string):
    pattern_python = r"^\s*```python\n(.*?)\n```\s*$"
    match = re.match(pattern_python, code_string, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    pattern_generic = r"^\s*```\n(.*?)\n```\s*$"
    match = re.match(pattern_generic, code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    code = re.sub(r"^\s*```[a-zA-Z]*\n?", "", code_string)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


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
        width=100,
    ):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.use_json = use_json
        self.json_indent = json_indent
        self.pprint_indent = pprint_indent
        self.pprint_width = width

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
        return super().format(record)


HAS_GOOGLE_LIBS_FOR_FORMATTER = False
google_api_exceptions = None
MessageToDict = None


def prepare_centered_list(snippets_list, logger):
    if not snippets_list:
        return []
    if len(snippets_list) == 1:
        return snippets_list

    if not all("index" in s for s in snippets_list):
        if logger:
            logger.warning(
                "Cannot prepare centered list: some snippets lack 'index'. Returning original order."
            )
        return snippets_list

    sorted_by_index_val = sorted(snippets_list, key=lambda s: s["index"])
    min_idx = sorted_by_index_val[0]["index"]
    max_idx = sorted_by_index_val[-1]["index"]

    if min_idx == max_idx:
        return sorted_by_index_val

    center_point_val = (min_idx + max_idx) / 2.0
    return sorted(snippets_list, key=lambda s: abs(s["index"] - center_point_val))


def fill_snippets_by_token_budget(
    client,
    model_name_for_counting,
    prioritized_snippets_list,
    available_tokens_for_snippets_content,
    json_wrapper_template_func,
    logger,
    context_log_prefix="",
):
    selected_snippets = []

    if not client or not hasattr(client, "models"):
        if logger:
            logger.warning(
                f"{context_log_prefix} Invalid client provided (expected genai.Client instance with .models). Cannot fill snippets by token budget."
            )
        return []

    base_tokens_for_empty_structure = 0
    try:
        empty_json_str = json_wrapper_template_func([])
        count_response_empty = gemini_api_utils.execute_gemini_api_call(
            client.models.count_tokens,
            logger,
            model=model_name_for_counting,
            contents=[empty_json_str],
        )
        base_tokens_for_empty_structure = count_response_empty.total_tokens
    except Exception as e_count_empty:
        if logger:
            logger.warning(
                f"{context_log_prefix} Error counting tokens for empty snippet structure: {e_count_empty}. Assuming 0 base tokens."
            )

    max_tokens_for_wrapped_snippets = (
        available_tokens_for_snippets_content + base_tokens_for_empty_structure
    )

    low = 0
    high = len(prioritized_snippets_list)
    best_k = 0

    while low <= high:
        mid_k = low + (high - low) // 2
        if mid_k == 0:
            current_tokens = base_tokens_for_empty_structure
            if current_tokens <= max_tokens_for_wrapped_snippets:
                best_k = mid_k
                low = mid_k + 1
            else:
                high = mid_k - 1
            continue

        temp_selection_for_counting = prioritized_snippets_list[:mid_k]
        full_json_str_to_count = json_wrapper_template_func(temp_selection_for_counting)

        try:
            tokens_for_current_selection = gemini_api_utils.execute_gemini_api_call(
                client.models.count_tokens,
                logger,
                model=model_name_for_counting,
                contents=[full_json_str_to_count],
            ).total_tokens

            if tokens_for_current_selection <= max_tokens_for_wrapped_snippets:
                best_k = mid_k
                low = mid_k + 1
            else:
                high = mid_k - 1
        except Exception as e_count:
            if logger:
                logger.warning(
                    f"{context_log_prefix} Error counting tokens for {mid_k} snippets: {e_count}. Assuming this count ({mid_k}) is too large."
                )
            high = mid_k - 1

    selected_snippets = prioritized_snippets_list[:best_k]

    final_tokens_for_selected = 0
    if selected_snippets:
        try:
            final_json_str = json_wrapper_template_func(selected_snippets)
            final_tokens_for_selected = gemini_api_utils.execute_gemini_api_call(
                client.models.count_tokens,
                logger,
                model=model_name_for_counting,
                contents=[final_json_str],
            ).total_tokens
        except Exception:  # nosec
            final_tokens_for_selected = -1
    elif best_k == 0:
        final_tokens_for_selected = base_tokens_for_empty_structure

    if logger:
        logger.debug(
            f"{context_log_prefix} Final selected snippets: {len(selected_snippets)} (binary search determined best_k={best_k}) "
            f"with an estimated {final_tokens_for_selected} tokens "
            f"(target for wrapped snippets: {max_tokens_for_wrapped_snippets}, "
            f"budget for content: {available_tokens_for_snippets_content}, "
            f"base for empty structure: {base_tokens_for_empty_structure})."
        )
    return selected_snippets


def convert_snippet_list_to_final_json(snippet_list, include_labels=True, logger=None):
    value_dict = {}
    label_dict = {}

    if not snippet_list:
        final_struct = {"value": {}}
        if include_labels:
            final_struct["label"] = {}
        return final_struct

    for snippet in snippet_list:
        idx = snippet.get("index")
        val = snippet.get("value")

        if idx is None or val is None:
            if logger:
                logger.warning(
                    f"Skipping snippet due to missing 'index' or 'value': {snippet}"
                )
            continue

        str_idx = str(idx)
        value_dict[str_idx] = val

        if include_labels:
            lbl = snippet.get("label")
            if lbl is None:
                if logger:
                    logger.warning(
                        f"Snippet for index {idx} is missing 'label' when labels are included. Defaulting label to 0."
                    )
                label_dict[str_idx] = 0
            else:
                label_dict[str_idx] = int(lbl)

    final_struct = {"value": value_dict}
    if include_labels:
        final_struct["label"] = label_dict

    return final_struct


def _format_exception_for_logging(e):
    e_type = type(e).__name__
    e_msg = str(e)

    # Attempt to find and pretty-print the JSON part of the error message
    # Look for a pattern that often contains the detailed JSON: (details from "{'error':...}" pattern):
    json_detail_match = re.search(r"(\\{\s*\\'error\\':.*?\\})", e_msg, re.DOTALL)

    if json_detail_match:
        json_str_candidate = json_detail_match.group(1)
        # The string is often escaped, try to unescape and parse
        try:
            # Convert Python literal string to actual JSON string (e.g., ' -> ")
            # This is a common format from google-api-python-client
            actual_json_str = ast.literal_eval(json_str_candidate)
            pretty_json_details = json.dumps(actual_json_str, indent=2)
            # Replace the raw JSON string in the message with the pretty-printed one
            e_msg = e_msg.replace(json_str_candidate, "\n" + pretty_json_details)
            return f"{e_type}: {e_msg}"
        except (SyntaxError, ValueError) as parse_err:
            # If ast.literal_eval fails, it might be actual JSON already or malformed
            try:
                # Attempt to parse directly as JSON (if it was already in proper JSON format)
                parsed_direct_json = json.loads(json_str_candidate)
                pretty_json_details = json.dumps(parsed_direct_json, indent=2)
                e_msg = e_msg.replace(json_str_candidate, "\n" + pretty_json_details)
                return f"{e_type}: {e_msg}"
            except json.JSONDecodeError:
                # If both fail, just return the original message with a note
                return f"{e_type}: {e_msg} (JSON-like details found but could not be parsed: {parse_err})"

    # Fallback for general exceptions or if specific pattern not found
    return f"{e_type}: {e_msg}"


def extract_retry_delay_from_error_details_json(error_message_str: str) -> int | None:
    """
    Attempts to parse a JSON string (often found within a larger error message)
    to find a 'retryDelay' field (e.g., "34s") and return the delay in seconds.

    Args:
        error_message_str: The string potentially containing the JSON with error details.

    Returns:
        The delay in seconds as an integer if found and parsable, otherwise None.
    """
    json_candidate_str = None

    # Attempt to find Python-style dict string: {'error':...}
    py_dict_start_index = error_message_str.find("{'error':")
    # Attempt to find JSON-style dict string: {"error":...}
    json_dict_start_index = error_message_str.find('{"error":')

    dict_start_index = -1

    # Determine which pattern occurs first, if any
    if py_dict_start_index != -1 and (
        json_dict_start_index == -1 or py_dict_start_index < json_dict_start_index
    ):
        dict_start_index = py_dict_start_index
    elif json_dict_start_index != -1:
        dict_start_index = json_dict_start_index

    if dict_start_index != -1:
        balance = 0
        for i in range(dict_start_index, len(error_message_str)):
            if error_message_str[i] == "{":
                balance += 1
            elif error_message_str[i] == "}":
                balance -= 1
                if balance == 0:
                    json_candidate_str = error_message_str[dict_start_index : i + 1]
                    break

    if not json_candidate_str:
        # Fallback to original regex methods if brace balancing fails or pattern not found initially
        # This is a simplified fallback; the original regexes were complex.
        # For now, if the above fails, we assume we can't find the structured error.
        # A more robust regex could be used here if needed as a secondary strategy.
        # Original regexes:
        # match_py_escaped = re.search(r"(\\{\s*\\'error\\':.*?\\})", error_message_str, re.DOTALL)
        # match_json_end = re.search(r"(\{\s*\"error\":.*?\}\s*$)", error_message_str, re.DOTALL)
        # if match_py_escaped:
        #     json_candidate_str = match_py_escaped.group(1)
        # elif match_json_end:
        #     json_candidate_str = match_json_end.group(1)
        # else:
        return None

    error_data = None
    try:
        # Attempt 1: Try ast.literal_eval (handles Python dict strings like {'error': ...})
        error_data = ast.literal_eval(json_candidate_str)
    except (SyntaxError, ValueError):
        try:
            # Attempt 2: If ast.literal_eval fails, try json.loads (handles JSON strings like {"error": ...})
            error_data = json.loads(json_candidate_str)
        except json.JSONDecodeError:
            # If both fail, we can't parse the extracted string.
            return None

    if isinstance(error_data, dict):
        error_details = error_data.get("error", {}).get("details", [])
        if isinstance(error_details, list):
            for detail_item in error_details:
                if (
                    isinstance(detail_item, dict)
                    and detail_item.get("@type")
                    == "type.googleapis.com/google.rpc.RetryInfo"
                ):
                    retry_delay_str = detail_item.get("retryDelay")
                    if isinstance(retry_delay_str, str) and retry_delay_str.endswith(
                        "s"
                    ):
                        try:
                            return int(retry_delay_str[:-1])
                        except ValueError:
                            return None  # Cannot convert delay string to int
    return None


def extract_quota_info_from_error_details_json(error_message_str: str) -> dict | None:
    """
    Attempts to parse a JSON string (often found within a larger error message)
    to find quota failure details like quotaMetric, quotaId, and quotaValue.

    Args:
        error_message_str: The string potentially containing the JSON with error details.

    Returns:
        A dictionary with quota information if found, otherwise None.
    """
    json_candidate_str = None
    py_dict_start_index = error_message_str.find("{'error':")
    json_dict_start_index = error_message_str.find('{"error":')
    dict_start_index = -1

    if py_dict_start_index != -1 and (
        json_dict_start_index == -1 or py_dict_start_index < json_dict_start_index
    ):
        dict_start_index = py_dict_start_index
    elif json_dict_start_index != -1:
        dict_start_index = json_dict_start_index

    if dict_start_index != -1:
        balance = 0
        for i in range(dict_start_index, len(error_message_str)):
            if error_message_str[i] == "{":
                balance += 1
            elif error_message_str[i] == "}":
                balance -= 1
                if balance == 0:
                    json_candidate_str = error_message_str[dict_start_index : i + 1]
                    break

    if not json_candidate_str:
        return None

    error_data = None
    try:
        error_data = ast.literal_eval(json_candidate_str)
    except (SyntaxError, ValueError):
        try:
            error_data = json.loads(json_candidate_str)
        except json.JSONDecodeError:
            return None

    if isinstance(error_data, dict):
        error_details = error_data.get("error", {}).get("details", [])
        if isinstance(error_details, list):
            for detail_item in error_details:
                if (
                    isinstance(detail_item, dict)
                    and detail_item.get("@type")
                    == "type.googleapis.com/google.rpc.QuotaFailure"
                ):
                    violations = detail_item.get("violations", [])
                    if isinstance(violations, list) and violations:
                        # Return info from the first violation found
                        first_violation = violations[0]
                        if isinstance(first_violation, dict):
                            return {
                                "quotaMetric": first_violation.get("quotaMetric"),
                                "quotaId": first_violation.get("quotaId"),
                                "quotaValue": first_violation.get("quotaValue"),
                            }
    return None


def log_prompt_text_to_file(
    prompt_text: str, save_dir: str, filename: str, logger
) -> str | None:
    """Saves the given prompt text to a file in the specified directory."""
    prompt_log_path = os.path.join(save_dir, filename)
    try:
        with open(prompt_log_path, "w") as f_prompt_log:
            f_prompt_log.write(prompt_text)
        logger.debug(f"Saved prompt to {prompt_log_path}")
        return prompt_log_path
    except Exception as e_save_prompt_log:
        logger.warning(
            f"Failed to save prompt to {filename}: {_format_exception_for_logging(e_save_prompt_log)}"
        )
        return None


def log_code_to_file(code_str: str, save_dir: str, filename: str, logger) -> str | None:
    """Saves the given code string to a file in the specified directory."""
    code_path = os.path.join(save_dir, filename)
    try:
        with open(code_path, "w") as f_py_code:
            f_py_code.write(code_str)
        logger.debug(f"Saved code to {code_path}")
        return code_path
    except Exception as e_save_py_code:
        logger.warning(
            f"Failed to save code to {filename}: {_format_exception_for_logging(e_save_py_code)}"
        )
        return None


def convert_scores_to_refinement_json(
    analysis_snippets_list, scores_array, n_total_samples, logger
) -> str:
    """
    Prepares a JSON string of analysis snippet values and their corresponding scores
    from the scores_array, for use in the refinement prompt.
    """
    value_dict = {}
    score_dict = {}
    if not analysis_snippets_list:
        logger.warning("No analysis snippets provided for refinement JSON generation.")
        return json.dumps({"value": {}, "score": {}})

    for snippet in analysis_snippets_list:
        idx = snippet.get("index")
        val = snippet.get("value")
        if idx is not None and val is not None:
            if 0 <= idx < n_total_samples and 0 <= idx < len(scores_array):
                value_dict[str(idx)] = float(f"{val:.3g}")
                score_dict[str(idx)] = float(f"{scores_array[idx]:.4g}")
            else:
                logger.warning(
                    f"Index {idx} out of bounds for scores_array (len {len(scores_array)}) or n_total_samples ({n_total_samples}). Skipping."
                )
        else:
            logger.warning(
                f"Snippet missing index or value: {snippet}. Skipping for refinement JSON."
            )

    return json.dumps({"value": value_dict, "score": score_dict}, indent=2)


def prepare_training_snippets_for_interest_id(
    feature_data_train,
    feature_labels_train,
    i_feat,
    n_train_samples,
    target_range_size_for_prompt,
    client,
    input_token_limit_override: int | None,
    logger,
):
    all_snippets_for_id_step_list_of_dicts = []
    if feature_labels_train is not None:
        for idx in range(n_train_samples):
            all_snippets_for_id_step_list_of_dicts.append(
                {
                    "index": idx,
                    "value": float(f"{feature_data_train[idx]:.3g}"),
                    "label": int(feature_labels_train[idx]),
                }
            )
    else:
        for idx in range(n_train_samples):
            all_snippets_for_id_step_list_of_dicts.append(
                {
                    "index": idx,
                    "value": float(f"{feature_data_train[idx]:.3g}"),
                    "label": 0,
                }
            )

    base_prompt_token_count_id_step = 250
    model_input_token_limit_id = gemini_api_utils.get_gemini_model_info(
        client,
        65536,
        input_token_limit_override=input_token_limit_override,
        logger=logger,
    ).get("input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT)
    output_token_limit_for_budget_calculation = DEFAULT_OUTPUT_TOKENS_INTEREST_ID
    error_detail = None

    effective_target_range_size_for_llm_prompt = (
        int(target_range_size_for_prompt / 2)
        if target_range_size_for_prompt > 1
        else target_range_size_for_prompt
    )
    if effective_target_range_size_for_llm_prompt == 0 and n_train_samples > 0:
        effective_target_range_size_for_llm_prompt = 1

    try:
        if not client:
            logger.error(f"Ft {i_feat} Interest ID SnippetPrep: Client not provided.")
            error_detail = "Client not provided for Interest ID snippet preparation."
            return (
                [],
                json.dumps(
                    convert_snippet_list_to_final_json(
                        [], include_labels=True, logger=logger
                    ),
                    indent=2,
                ),
                error_detail,
            )

        model_info_dict = gemini_api_utils.get_gemini_model_info(
            client=client,
            default_output_token_limit=2048,
            input_token_limit_override=input_token_limit_override,
            logger=logger,
        )
        model_input_token_limit_id = model_info_dict.get(
            "input_token_limit", DEFAULT_FALLBACK_TOKEN_LIMIT
        )
        fetched_model_output_limit = model_info_dict.get("output_token_limit")
        if fetched_model_output_limit is not None:
            output_token_limit_for_budget_calculation = fetched_model_output_limit

        empty_snippets_map_for_estimation = convert_snippet_list_to_final_json(
            [], include_labels=True, logger=logger
        )
        empty_snippets_json_for_estimation = json.dumps(
            empty_snippets_map_for_estimation
        )

        _temp_prompt_for_token_est = prompt_utils.construct_interest_id_prompt(
            i_feat=i_feat,
            num_train_samples=n_train_samples,
            training_snippets_json_str_for_id_step=empty_snippets_json_for_estimation,
            target_range_size=effective_target_range_size_for_llm_prompt,
        )

        count_response_base_prompt = gemini_api_utils.count_gemini_tokens(
            client=client,
            model_name=FIXED_MODEL_NAME_GEMINI_FLASH,
            contents=[_temp_prompt_for_token_est],
            logger=logger,
        )
        base_prompt_token_count_id_step = count_response_base_prompt.total_tokens

    except Exception as e_budget_setup:
        error_detail = f"Ft {i_feat} Interest ID: Snippet budget setup error: {_format_exception_for_logging(e_budget_setup)}. Using fallback values for limits."
        logger.warning(error_detail)

    anomalous_snippets_id = [
        s for s in all_snippets_for_id_step_list_of_dicts if s.get("label") == 1
    ]
    normal_snippets_id = [
        s for s in all_snippets_for_id_step_list_of_dicts if s.get("label") != 1
    ]
    centered_normal_snippets_id = prepare_centered_list(
        normal_snippets_id, logger=logger
    )
    prioritized_total_snippets_list_of_dicts = (
        anomalous_snippets_id + centered_normal_snippets_id
    )

    # Apply safety factor to the model input token limit before other subtractions
    safe_model_input_token_limit_id = (
        model_input_token_limit_id * TOKEN_LIMIT_SAFETY_FACTOR
    )

    available_tokens_for_snippets_content = int(
        safe_model_input_token_limit_id  # Use the factored limit
        - base_prompt_token_count_id_step
        - output_token_limit_for_budget_calculation
        - THINKING_BUDGET_INTEREST_ID
        - TOKEN_BUDGET_SAFETY_MARGIN_INTEREST_ID  # This is a small fixed buffer
    )

    selected_snippets_list = []
    if (
        available_tokens_for_snippets_content > 10
        and prioritized_total_snippets_list_of_dicts
    ):
        json_wrapper_for_fill = lambda snips_list: json.dumps(
            convert_snippet_list_to_final_json(
                snips_list, include_labels=True, logger=logger
            )
        )

        qualified_model_name_for_fill = FIXED_MODEL_NAME_GEMINI_FLASH

        selected_snippets_list = fill_snippets_by_token_budget(
            client=client,
            model_name_for_counting=qualified_model_name_for_fill,
            prioritized_snippets_list=prioritized_total_snippets_list_of_dicts,
            available_tokens_for_snippets_content=available_tokens_for_snippets_content,
            json_wrapper_template_func=json_wrapper_for_fill,
            logger=logger,
            context_log_prefix=f"Ft {i_feat} Interest ID SnippetFill: ",
        )
    else:
        log_msg = f"Ft {i_feat} Interest ID: Token budget for snippets ({available_tokens_for_snippets_content}) too small or no snippets to prioritize. Skipping token-based fill. "
        logger.warning(log_msg)
        if not error_detail:
            error_detail = log_msg

    final_snippets_dict_for_prompt = convert_snippet_list_to_final_json(
        selected_snippets_list, include_labels=True, logger=logger
    )
    current_snippets_json_for_prompt = json.dumps(
        final_snippets_dict_for_prompt, indent=2
    )

    return selected_snippets_list, current_snippets_json_for_prompt, error_detail


def prepare_training_snippets_for_main_step(
    X_train_fit_data,
    y_train_fit_labels,
    i_feat,
    identified_train_interest_range,
    logger,
):
    source_for_training_snippets_list = []

    if (
        X_train_fit_data is None
        or y_train_fit_labels is None
        or i_feat >= X_train_fit_data.shape[1]
    ):
        if logger:
            logger.debug(
                f"Feature {i_feat}: No training data/labels available or invalid i_feat for main step snippet prep."
            )
        return source_for_training_snippets_list

    train_feature_data_for_snippets_full = X_train_fit_data[:, i_feat]
    train_labels_for_snippets_full = y_train_fit_labels

    temp_full_training_snippets_unprioritized = []
    for idx, (val, lbl) in enumerate(
        zip(
            train_feature_data_for_snippets_full,
            train_labels_for_snippets_full,
        )
    ):
        temp_full_training_snippets_unprioritized.append(
            {
                "index": idx,
                "value": float(f"{val:.3g}"),
                "label": int(lbl),
            }
        )

    focused_training_snippets_by_range = temp_full_training_snippets_unprioritized
    if identified_train_interest_range:
        start_idx_focus, end_idx_focus = identified_train_interest_range
        if (
            isinstance(start_idx_focus, int)
            and isinstance(end_idx_focus, int)
            and start_idx_focus <= end_idx_focus
        ):
            if logger:
                logger.debug(
                    f"Feature {i_feat}: Filtering training snippets to identified range [{start_idx_focus}-{end_idx_focus}] for main step."
                )
            focused_training_snippets_by_range = [
                s
                for s in temp_full_training_snippets_unprioritized
                if start_idx_focus <= s["index"] <= end_idx_focus
            ]
        else:
            if logger:
                logger.warning(
                    f"Feature {i_feat}: Invalid identified_train_interest_range {identified_train_interest_range}. Using all training snippets."
                )

    if not focused_training_snippets_by_range:
        if logger:
            logger.debug(
                f"Feature {i_feat}: No training snippets found in the focus range {identified_train_interest_range if identified_train_interest_range else '(No range provided or invalid)'}. Returning empty list for training examples."
            )
        return source_for_training_snippets_list

    anomalous_train_snippets = [
        s for s in focused_training_snippets_by_range if s.get("label") == 1
    ]
    normal_train_snippets = [
        s for s in focused_training_snippets_by_range if s.get("label") != 1
    ]

    centered_normal_train_snippets = prepare_centered_list(
        normal_train_snippets, logger=logger
    )

    source_for_training_snippets_list = (
        anomalous_train_snippets + centered_normal_train_snippets
    )

    if not source_for_training_snippets_list and focused_training_snippets_by_range:
        if logger:
            logger.debug(
                f"Feature {i_feat}: Prioritization of focused snippets resulted in empty list; using centered list of all focused snippets as fallback."
            )
        source_for_training_snippets_list = prepare_centered_list(
            focused_training_snippets_by_range, logger=logger
        )

    if logger:
        logger.debug(
            f"Feature {i_feat}: Prepared {len(source_for_training_snippets_list)} prioritized training snippets (from range {identified_train_interest_range if identified_train_interest_range and isinstance(identified_train_interest_range, tuple) and len(identified_train_interest_range)==2 else 'N/A or All'}) as source for main step batches."
        )

    return source_for_training_snippets_list


def prepare_analysis_snippets_for_batch(
    candidate_snippets_list,
    client,
    model_name_for_counting,
    tokens_for_analysis_budget_this_batch,
    logger,
):
    """
    Prepares a batch of analysis snippets from a candidate list based on token budget.
    Uses fill_snippets_by_token_budget (binary search) to select the maximum
    number of candidates that fit the token budget.
    Returns the selected snippets.
    """
    collected_snippets_for_batch = []

    if not client:
        if logger:
            logger.error(
                "Analysis snippet prep: Client not provided. Cannot use token budget."
            )
        return []  # Return empty list

    if not candidate_snippets_list:
        if logger:
            logger.debug("No candidate analysis snippets provided for this batch call.")
        return []  # No candidates to process

    # Use fill_snippets_by_token_budget on the provided candidate_snippets_list
    json_wrapper_analysis_for_fill = lambda snips_list: json.dumps(
        convert_snippet_list_to_final_json(
            snips_list, include_labels=False, logger=logger
        )
    )

    if logger:
        logger.debug(
            f"Calling fill_snippets_by_token_budget with {len(candidate_snippets_list)} candidates for analysis batch."
        )

    collected_snippets_for_batch = fill_snippets_by_token_budget(
        client=client,
        model_name_for_counting=model_name_for_counting,
        prioritized_snippets_list=candidate_snippets_list,  # This list is already prioritized or represents all remaining
        available_tokens_for_snippets_content=tokens_for_analysis_budget_this_batch,
        json_wrapper_template_func=json_wrapper_analysis_for_fill,
        logger=logger,
        context_log_prefix="AnalysisBatchFill: ",
    )

    if logger:
        logger.debug(
            f"prepare_analysis_snippets_for_batch: Selected {len(collected_snippets_for_batch)} snippets from {len(candidate_snippets_list)} candidates."
        )

    return collected_snippets_for_batch
