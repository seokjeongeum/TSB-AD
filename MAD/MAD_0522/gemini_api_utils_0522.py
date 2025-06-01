from google import genai

from MAD.core.constants import (DEFAULT_FALLBACK_TOKEN_LIMIT,
                                FIXED_MODEL_NAME_GEMINI_FLASH)

from .mad_utils_0522 import _format_exception_for_logging


def initialize_gemini_client(api_key_value: str, logger):
    if not genai:
        if logger:
            logger.error("Google GenAI library (genai) not available.")
        raise RuntimeError("Google GenAI library (genai) not available.")
    if not api_key_value:
        if logger:
            logger.error("API key value not provided for client initialization.")
        raise ValueError("API key value must be provided.")

    try:
        client = genai.Client(api_key=api_key_value)
        if logger:
            logger.info(
                f"Successfully configured GenAI with API key ending ...{api_key_value[-4:] if len(api_key_value) > 4 else '****'}."
            )
        return client
    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to configure GenAI/create client (key ending ...{api_key_value[-4:] if len(api_key_value) > 4 else '****'}): {_format_exception_for_logging(e)}"
            )
        return None


def execute_gemini_api_call(api_call_func, logger, *args, **kwargs):
    """
    Executes a given API call function.
    `api_call_func` should be a callable that performs the Gemini API operation.
    The `client` argument is largely for historical reasons or specific client-based calls;
    """
    try:
        result = api_call_func(*args, **kwargs)
        return result
    except Exception as e_api:
        log_prompt_on_error = kwargs.pop("log_prompt_on_error", False)
        prompt_to_log = kwargs.pop("prompt_to_log", None)
        if logger:
            log_message = (
                f"Google GenAI API call failed: {_format_exception_for_logging(e_api)}"
            )
            if log_prompt_on_error and prompt_to_log:
                log_message += f"\n--- Failing Prompt Text ---\n{prompt_to_log}\n-------------------------"
            logger.error(log_message)
        raise


def get_gemini_model_info(
    client,
    default_output_token_limit: int,
    input_token_limit_override: int | None,
    logger,
):
    qualified_model_name = (
        FIXED_MODEL_NAME_GEMINI_FLASH
        if FIXED_MODEL_NAME_GEMINI_FLASH.startswith("")
        else f"{FIXED_MODEL_NAME_GEMINI_FLASH}"
    )
    input_limit = input_token_limit_override if input_token_limit_override is not None else DEFAULT_FALLBACK_TOKEN_LIMIT
    using_override_for_input = input_token_limit_override is not None
    output_limit = default_output_token_limit

    try:
        api_call_to_execute = client.models.get

        model_details = execute_gemini_api_call(
            api_call_to_execute,
            logger,
            model=qualified_model_name,
        )

        fetched_input_limit = getattr(model_details, "input_token_limit", None)
        fetched_output_limit = getattr(model_details, "output_token_limit", None)

        if not using_override_for_input and fetched_input_limit is not None:
            input_limit = fetched_input_limit
        
        if fetched_output_limit is not None:
            output_limit = fetched_output_limit

        if logger:
            log_input_limit_source = "override" if using_override_for_input else ("fetched" if fetched_input_limit is not None else "default")
            logger.debug(
                f"Model {qualified_model_name}: Input Limit={input_limit} (source: {log_input_limit_source}), Output Limit={output_limit} (fetched: {fetched_output_limit is not None})."
            )

    except Exception as e_fetch:
        if logger:
            logger.warning(
                f"Could not fetch token limits for {qualified_model_name} using client.models.get: {_format_exception_for_logging(e_fetch)}. Using defaults: In={input_limit}, Out={output_limit}."
            )

    return {
        "input_token_limit": input_limit,
        "output_token_limit": output_limit,
        "qualified_model_name": qualified_model_name,
    }


def upload_file_to_gemini(client, file_path: str, logger):
    if not file_path or not isinstance(file_path, str):
        if logger:
            logger.error(f"Invalid file_path for upload: {file_path}")
        raise ValueError("Valid file_path must be provided for upload.")

    try:
        uploaded_file = execute_gemini_api_call(
            client.files.upload,
            logger,
            file=file_path,
        )
        if logger:
            logger.debug(
                f"Successfully uploaded file {file_path}. Name: {getattr(uploaded_file, 'name', 'N/A')}, URI: {getattr(uploaded_file, 'uri', 'N/A')}"
            )
        return uploaded_file
    except Exception as e_upload:
        if logger:
            logger.error(
                f"Failed to upload file {file_path}: {_format_exception_for_logging(e_upload)}"
            )
        raise


def count_gemini_tokens(client, model_name: str, contents, logger):
    if not model_name:
        if logger:
            logger.error("Model name not provided for token counting.")
        raise ValueError("Model name must be provided for token counting.")

    try:
        qualified_model_name = (
            model_name if model_name.startswith("") else f"{model_name}"
        )
        api_call_to_execute = client.models.count_tokens

        count_response = execute_gemini_api_call(
            api_call_to_execute,  # The function to call (client.models.count_tokens)
            logger,  # Third argument to execute_call_func_ref
            # Keyword arguments for client.models.count_tokens:
            model=qualified_model_name,
            contents=contents,
        )
        if logger:
            logger.debug(
                f"Token count for model {qualified_model_name}: {getattr(count_response, 'total_tokens', 'N/A')}"
            )
        return count_response
    except Exception as e_count:
        if logger:
            logger.error(
                f"Failed to count tokens for model {qualified_model_name}: {_format_exception_for_logging(e_count)}"
            )
        raise
