import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Configuration ---

# Define the log files to be analyzed with user-friendly names
LOG_FILES_TO_ANALYZE = {
    "Dimensionality Reduction Ablated": "eval/score/multi/TSPulse2_dimensionality_reduction_ablated/000_run_TSPulse2_dimensionality_reduction_ablated.log",
    "Full Model": "eval/score/multi/TSPulse2/000_run_TSPulse2.log",
}

# Pricing for Gemini 2.5 Pro, per 1 million tokens, from the provided image.
# The pricing tier depends on the size of the input prompt.
GEMINI_2_5_PRO_PRICING = {
    "standard_prompt": {  # Prompts <= 200k tokens
        "threshold": 200000,
        "input_per_1m": 1.25,
        "output_per_1m": 10.00,
    },
    "large_prompt": {  # Prompts > 200k tokens
        "threshold": float("inf"),
        "input_per_1m": 2.50,
        "output_per_1m": 15.00,
    },
}

# --- Core Logic ---


def parse_log_file(filepath: str) -> Optional[Tuple[List[Dict[str, float]], float]]:
    """
    Parses a log file to extract details for each LLM API call and total time cost.

    Args:
        filepath: The path to the log file.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dictionary represents one API call
          and contains its inference time and token counts.
        - The total time cost from "Success" logs.
        Returns None if the file cannot be found or if there's a mismatch in logs.
    """
    try:
        with open(filepath, "r") as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at '{filepath}'")
        return None

    # Regex to find all occurrences of LLM call metrics
    time_pattern = re.compile(r"LLM: Selected '.*?' for channel '.*?' in ([\d.]+)s")
    token_pattern = re.compile(
        r"LLM token count for .*?:.*?"
        r"candidates_token_count=(\d+).*"
        r"prompt_token_count=(\d+).*"
        r"thoughts_token_count=(\d+).*"
        r"total_token_count=(\d+)"
    )
    total_time_pattern = re.compile(r"Time cost: ([\d.]+)s")

    # Extract all matches
    time_matches = [float(t) for t in time_pattern.findall(log_content)]
    token_matches = token_pattern.findall(log_content)
    total_time_matches = [float(t) for t in total_time_pattern.findall(log_content)]

    total_time = sum(total_time_matches)

    if len(time_matches) != len(token_matches):
        print(
            f"Warning: Mismatch in captured time ({len(time_matches)}) and token ({len(token_matches)}) logs for {filepath}. Analysis may be incomplete."
        )
        return None

    # Combine matches into a structured list
    api_calls = []
    for i, tokens in enumerate(token_matches):
        candidate_tokens = int(tokens[0])
        prompt_tokens = int(tokens[1])
        thinking_tokens = int(tokens[2])
        total_tokens = int(tokens[3])

        api_calls.append(
            {
                "inference_time_s": time_matches[i],
                "prompt_tokens": prompt_tokens,
                "output_tokens": candidate_tokens
                + thinking_tokens,  # Output is the sum of thinking and the final answer
                "total_tokens": total_tokens,
            }
        )
    return api_calls, total_time


def analyze_calls(
    api_calls: List[Dict[str, float]], total_time: Optional[float]
) -> Dict[str, Any]:
    """
    Aggregates metrics and calculates total cost from a list of parsed API calls.

    Args:
        api_calls: A list of dictionaries, each representing one API call.
        total_time: The total execution time from the log file.

    Returns:
        A dictionary containing aggregated statistics and the total estimated cost.
    """
    if not api_calls:
        return {
            "Total LLM Calls": 0,
            "Total Inference Time (s)": 0,
            "Total Input Tokens": 0,
            "Total Output Tokens": 0,
            "Grand Total Tokens": 0,
            "Estimated Cost (USD)": 0,
            "Total Time (s)": total_time or 0,
            "LLM Contribution (%)": 0,
        }

    total_cost = 0.0
    total_inference_time = sum(call["inference_time_s"] for call in api_calls)
    total_input_tokens = sum(call["prompt_tokens"] for call in api_calls)
    total_output_tokens = sum(call["output_tokens"] for call in api_calls)
    grand_total_tokens = sum(call["total_tokens"] for call in api_calls)

    # Calculate cost on a per-call basis because the rate depends on each prompt's size
    for call in api_calls:
        prompt_size = call["prompt_tokens"]

        # Determine which pricing tier to use
        if prompt_size <= GEMINI_2_5_PRO_PRICING["standard_prompt"]["threshold"]:
            pricing = GEMINI_2_5_PRO_PRICING["standard_prompt"]
        else:
            pricing = GEMINI_2_5_PRO_PRICING["large_prompt"]

        # Calculate cost for this specific call
        input_cost = (call["prompt_tokens"] / 1_000_000) * pricing["input_per_1m"]
        output_cost = (call["output_tokens"] / 1_000_000) * pricing["output_per_1m"]
        total_cost += input_cost + output_cost

    llm_contribution = (
        (total_inference_time / total_time) * 100 if total_time else 0
    )

    return {
        "Total LLM Calls": len(api_calls),
        "Total Inference Time (s)": total_inference_time,
        "Total Input Tokens": total_input_tokens,
        "Total Output Tokens": total_output_tokens,
        "Grand Total Tokens": grand_total_tokens,
        "Estimated Cost (USD)": total_cost,
        "Total Time (s)": total_time,
        "LLM Contribution (%)": llm_contribution,
    }


def main():
    """
    Main function to run the analysis and print the comparison table.
    """
    print("--- LLM Inference and Cost Analysis ---")

    results = {}
    for name, path in LOG_FILES_TO_ANALYZE.items():
        print(f"\nAnalyzing log for: '{name}'...")
        parsed_data = parse_log_file(path)
        if parsed_data:
            api_calls, total_time = parsed_data
            results[name] = analyze_calls(api_calls, total_time)
        else:
            # Create an empty result set if file was not found or logs are inconsistent
            results[name] = analyze_calls([], None)

    # Use pandas to create and display a formatted comparison table
    df = pd.DataFrame(results).T  # Transpose to have log names as rows

    # Formatting for better readability
    for col in [
        "Total Inference Time (s)",
        "Total Time (s)",
    ]:
        if col in df.columns:
            df[col] = df[col].map("{:,.4f}".format)

    if "Estimated Cost (USD)" in df.columns:
        df["Estimated Cost (USD)"] = df["Estimated Cost (USD)"].map("${:,.10f}".format)

    if "LLM Contribution (%)" in df.columns:
        df["LLM Contribution (%)"] = df["LLM Contribution (%)"].map("{:,.2f}%".format)

    for col in [
        "Total LLM Calls",
        "Total Input Tokens",
        "Total Output Tokens",
        "Grand Total Tokens",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(int).map("{:,.0f}".format)

    print("\n--- Comparison Summary ---")
    if df.empty:
        print("No data to display. Please check log file paths and content.")
    else:
        # Set pandas display options to show full float precision
        with pd.option_context("display.float_format", "{:.15f}".format):
            print(df.to_string())
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
