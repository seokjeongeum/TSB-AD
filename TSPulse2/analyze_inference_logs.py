import os
import re
from typing import Any, Dict, List, Optional, Set, cast

import pandas as pd

# --- Configuration ---

# Define the log files to be analyzed with user-friendly names
LOG_FILES_TO_ANALYZE = {
    "Dimensionality Reduction Ablated": "eval/score/multi/TSPulse2_dimensionality_reduction_ablated/000_run_TSPulse2_dimensionality_reduction_ablated.log",
    "Full Model (with PCA/Selection)": "eval/score/multi/TSPulse2/000_run_TSPulse2.log",
}

# Pricing for Gemini 2.5 Pro, per 1 million tokens, from the provided pricing page.
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

VALID_FILENAMES_PATH_PRIMARY = "Datasets/File_List/TSPulse2-M.csv"
VALID_FILENAMES_PATH_SECONDARY = "Datasets/File_List/TSB-AD-M-Eva.csv"

# --- Core Logic ---


def load_simple_filenames(path: str) -> Optional[Set[str]]:
    """Loads a set of valid filenames from a single CSV file."""
    try:
        df = pd.read_csv(path)
        return set(df["file_name"])
    except FileNotFoundError:
        print(f"ERROR: Valid filenames file not found at '{path}'.")
        return None
    except KeyError:
        print(f"ERROR: 'file_name' column not found in '{path}'.")
        return None


def get_full_model_filenames(override_path: str, base_path: str) -> Optional[Set[str]]:
    """
    Loads filenames from a base list, but replaces them with files from an
    override list if a corresponding file is found. A corresponding override
    file is one that starts with the base filename (without extension) followed
    by a hyphen.
    """
    try:
        df_override = pd.read_csv(override_path)
        override_files = set(df_override["file_name"])

        df_base = pd.read_csv(base_path)
        base_files = set(df_base["file_name"])

        valid_filenames = set()
        for base_file in base_files:
            base_name = base_file.rsplit(".", 1)[0]

            found_override = None
            for override_file in override_files:
                if override_file.startswith(base_name + "-"):
                    found_override = override_file
                    break

            if found_override:
                valid_filenames.add(found_override)
            else:
                valid_filenames.add(base_file)

        return valid_filenames

    except FileNotFoundError:
        print(
            f"ERROR: A filenames file was not found. Check paths: '{override_path}', '{base_path}'."
        )
        return None
    except KeyError:
        print(f"ERROR: 'file_name' column not found in one of the CSV files.")
        return None


def parse_log_file(
    filepath: str, valid_filenames: Set[str]
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Parses a log file to extract details for each LLM API call and total time cost,
    grouping all results by the source filename.

    Args:
        filepath: The path to the log file.

    Returns:
        A dictionary where keys are source filenames and values are another dictionary
        containing the list of API calls and the time cost for that file.
        Returns None if the file cannot be found.
    """
    try:
        with open(filepath, "r") as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"ERROR: Log file not found at '{filepath}'. Please check the path.")
        return None

    per_file_results: Dict[str, Dict[str, Any]] = {}

    # A robust pattern to split the log file by success messages, but keep them.
    # This creates a list like: [content_before_1st_success, 1st_success_msg, content_before_2nd, 2nd_success_msg, ...]
    split_pattern = r"(Success at .*? using .*? \| Time cost: [\d.]+s.*)"
    log_chunks = re.split(split_pattern, log_content)

    # We iterate through the list, taking a content chunk and its corresponding success message.
    for i in range(0, len(log_chunks) - 1, 2):
        chunk_content = log_chunks[i]
        success_line = log_chunks[i + 1]

        # Extract filename and time cost from the success message itself.
        filename_match = re.search(r"Success at (.*?) using", success_line)
        time_match = re.search(r"Time cost: ([\d.]+)s", success_line)

        if not filename_match or not time_match:
            continue

        filename = filename_match.group(1).strip()
        time_cost = float(time_match.group(1))

        if filename not in valid_filenames:
            continue

        time_pattern = re.compile(r"LLM: Selected '.*?' for channel '.*?' in ([\d.]+)s")
        token_pattern = re.compile(
            r"LLM token count for .*?:.*?"
            r"candidates_token_count=(\d+).*"
            r"prompt_token_count=(\d+).*"
            r"thoughts_token_count=(\d+).*"
            r"total_token_count=(\d+)"
        )
        plot_pattern = re.compile(r"LLM debug plot saved to: (.*)")
        thoughts_pattern = re.compile(r"LLM thoughts saved to: (.*)")

        time_matches = time_pattern.findall(chunk_content)
        token_matches = token_pattern.findall(chunk_content)
        plot_matches = plot_pattern.findall(chunk_content)
        thoughts_matches = thoughts_pattern.findall(chunk_content)

        if len(time_matches) != len(token_matches):
            print(f"Warning: Mismatch for {filename}. Skipping.")
            continue

        api_calls = []
        for j, tokens in enumerate(token_matches):
            candidate_tokens = int(tokens[0])
            prompt_tokens = int(tokens[1])
            thoughts_tokens = int(tokens[2])
            api_calls.append(
                {
                    "inference_time_s": float(time_matches[j]),
                    "candidate_tokens": candidate_tokens,
                    "prompt_tokens": prompt_tokens,
                    "thoughts_token_count": thoughts_tokens,
                    "output_tokens": candidate_tokens + thoughts_tokens,
                    "total_tokens": int(tokens[3]),
                }
            )

        per_file_results[filename] = {
            "api_calls": api_calls,
            "time_cost": time_cost,
            "plot_files": [p.strip() for p in plot_matches],
            "thoughts_files": [t.strip() for t in thoughts_matches],
        }

    return per_file_results


def analyze_calls(
    api_calls: List[Dict[str, float]], total_time: float
) -> Dict[str, Any]:
    """
    Aggregates metrics and calculates total cost from a list of parsed API calls for a single file.

    Args:
        api_calls: A list of dictionaries, each representing one API call.
        total_time: The total execution time from the log file.

    Returns:
        A dictionary containing aggregated statistics and the total estimated cost.
    """
    if not api_calls:
        return {
            "Total LLM Calls": 0,
            "Total LLM Inference Time (s)": 0,
            "Total Input Tokens": 0,
            "Total Output Tokens": 0,
            "Grand Total Tokens": 0,
            "Estimated Cost (USD)": 0,
            "File Time Cost (s)": total_time,
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

        # Determine which pricing tier to use based on the prompt token count
        if prompt_size <= GEMINI_2_5_PRO_PRICING["standard_prompt"]["threshold"]:
            pricing = GEMINI_2_5_PRO_PRICING["standard_prompt"]
        else:
            pricing = GEMINI_2_5_PRO_PRICING["large_prompt"]

        # Calculate cost for this specific call
        input_cost = (call["prompt_tokens"] / 1_000_000) * pricing["input_per_1m"]
        output_cost = (call["output_tokens"] / 1_000_000) * pricing["output_per_1m"]
        total_cost += input_cost + output_cost

    llm_contribution = (
        (total_inference_time / total_time) * 100 if total_time > 0 else 0
    )

    return {
        "Total LLM Calls": len(api_calls),
        "Total LLM Inference Time (s)": total_inference_time,
        "Total Input Tokens": total_input_tokens,
        "Total Output Tokens": total_output_tokens,
        "Grand Total Tokens": grand_total_tokens,
        "Estimated Cost (USD)": total_cost,
        "File Time Cost (s)": total_time,
        "LLM Contribution (%)": llm_contribution,
    }


def main():
    """
    Main function to run the analysis and print the comparison table.
    """
    print("--- LLM Inference and Cost Analysis ---")

    all_file_results = []
    for name, path in LOG_FILES_TO_ANALYZE.items():
        print(f"\nAnalyzing log for: '{name}'...")

        valid_filenames: Optional[Set[str]] = None
        if name == "Full Model (with PCA/Selection)":
            valid_filenames = get_full_model_filenames(
                override_path=VALID_FILENAMES_PATH_PRIMARY,
                base_path=VALID_FILENAMES_PATH_SECONDARY,
            )
        else:
            # For all other models, use a simple file list.
            valid_filenames = load_simple_filenames(VALID_FILENAMES_PATH_SECONDARY)

        if valid_filenames is None:
            continue

        per_file_data = parse_log_file(path, valid_filenames)

        if per_file_data:
            for filename, data in per_file_data.items():
                analysis = analyze_calls(data["api_calls"], data["time_cost"])
                analysis["Model"] = name
                analysis["File"] = filename
                analysis["plot_files"] = data.get("plot_files", [])
                analysis["thoughts_files"] = data.get("thoughts_files", [])
                all_file_results.append(analysis)

    if not all_file_results:
        print("\nNo data to display. Please check log file paths and content.")
        return

    # --- Create and print the detailed per-file DataFrame ---
    df_detailed = pd.DataFrame(all_file_results)
    # Reorder columns for clarity
    cols_order = [
        "Model",
        "File",
        "Total LLM Calls",
        "Total LLM Inference Time (s)",
        "File Time Cost (s)",
        "LLM Contribution (%)",
        "Total Input Tokens",
        "Total Output Tokens",
        "Grand Total Tokens",
        "Estimated Cost (USD)",
    ]
    df_detailed = df_detailed[cols_order]

    # --- Create the overall summary DataFrame ---
    df_summary = df_detailed.copy()
    # Convert necessary columns back to numeric for summation
    numeric_cols = [
        "Total LLM Calls",
        "Total LLM Inference Time (s)",
        "Total Input Tokens",
        "Total Output Tokens",
        "Grand Total Tokens",
        "Estimated Cost (USD)",
        "File Time Cost (s)",
    ]
    for col in numeric_cols:
        # Remove formatting and convert to numeric
        # Ensure the column is treated as a Series to use .str accessor
        series = pd.Series(df_summary[col])
        df_summary[col] = series.astype(str).str.replace(r"[$,%]", "", regex=True)
        df_summary[col] = pd.to_numeric(df_summary[col], errors="coerce")

    # Now group by model and sum to get the aggregate summary
    df_agg = df_summary.groupby("Model")[numeric_cols].sum().reset_index()

    # Get file count per model to calculate average cost.
    model_series = cast(pd.Series, df_detailed["Model"])
    file_counts = model_series.value_counts()
    # Use the model name to map the file count and calculate the average cost.
    model_in_agg_series = cast(pd.Series, df_agg["Model"])
    df_agg["Avg. Cost per Series"] = df_agg[
        "Estimated Cost (USD)"
    ] / model_in_agg_series.map(file_counts)

    # Recalculate the contribution percentage for the aggregate
    df_agg["LLM Contribution (%)"] = 0.0
    if not df_agg.empty:
        non_zero_mask = df_agg["File Time Cost (s)"] > 0
        df_agg.loc[non_zero_mask, "LLM Contribution (%)"] = (
            df_agg.loc[non_zero_mask, "Total LLM Inference Time (s)"]
            / df_agg.loc[non_zero_mask, "File Time Cost (s)"]
        ) * 100

    # --- Formatting for better readability ---
    for col in ["Total LLM Inference Time (s)", "File Time Cost (s)"]:
        df_agg[col] = pd.to_numeric(df_agg[col], errors="coerce").apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")  # type: ignore
    df_agg["Estimated Cost (USD)"] = pd.to_numeric(df_agg["Estimated Cost (USD)"], errors="coerce").apply(  # type: ignore
        lambda x: f"${x:,.4f}" if pd.notna(x) else ""
    )
    df_agg["Avg. Cost per Series"] = pd.to_numeric(df_agg["Avg. Cost per Series"], errors="coerce").apply(  # type: ignore
        lambda x: f"${x:,.4f}" if pd.notna(x) else ""
    )
    df_agg["LLM Contribution (%)"] = pd.to_numeric(df_agg["LLM Contribution (%)"], errors="coerce").apply(  # type: ignore
        lambda x: f"{x:,.2f}%" if pd.notna(x) else ""
    )
    for col in [
        "Total LLM Calls",
        "Total Input Tokens",
        "Total Output Tokens",
        "Grand Total Tokens",
    ]:
        df_agg[col] = pd.to_numeric(df_agg[col], errors="coerce").astype("Int64").apply(lambda x: f"{int(x):,.0f}" if pd.notna(x) else "")  # type: ignore

    # --- Write output to a file ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "analysis_report.txt")
    with open(output_filename, "w") as f:
        f.write("--- LLM Inference and Cost Analysis ---\n\n")

        f.write("--- Row Count Per Model ---\n")
        f.write("=" * 80 + "\n")
        f.write(df_detailed["Model"].value_counts().to_string())  # type: ignore
        f.write("\n\n")

        f.write("--- Overall Comparison Summary ---\n")
        f.write("=" * 80 + "\n")
        f.write(df_agg.to_string(index=False))
        f.write("\n\n")

        f.write("--- LLM Artifacts Map ---\n")
        f.write("=" * 80 + "\n")
        for result in all_file_results:
            filename = result["File"]
            model = result["Model"]
            plot_files = [p for p in result.get("plot_files", []) if p]
            thoughts_files = [t for t in result.get("thoughts_files", []) if t]

            if not plot_files and not thoughts_files:
                continue

            f.write(f"File: {filename} (Model: {model})\n")
            if plot_files:
                f.write("  Plot Files:\n")
                for p in plot_files:
                    f.write(f"    - {p}\n")
            if thoughts_files:
                f.write("  Thoughts Files:\n")
                for t in thoughts_files:
                    f.write(f"    - {t}\n")
            f.write("-" * 40 + "\n")

    print(f"\nAnalysis complete. Report saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
