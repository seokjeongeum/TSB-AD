import os
import re
from typing import Any, Dict, List, Optional, Set, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --- Configuration ---

# Define the log files to be analyzed with user-friendly names
LOG_FILES_TO_ANALYZE = {
    "Ablated": "eval/score/multi/MAD_no_dim_redux/000_run_MAD_no_dim_redux.log",
    "Full": "eval/score/multi/MAD/000_run_MAD.log",
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

VALID_FILENAMES_PATH_PRIMARY = "Datasets/File_List/MAD-M.csv"
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


def save_bar_chart(
    df: pd.DataFrame, title: str, xlabel: str, ylabel: str, output_path_no_ext: str
):
    """Generates and saves a bar chart from a DataFrame to both PDF and PNG."""
    if df.empty:
        print(f"Skipping plot '{title}' because the dataframe is empty.")
        return

    # Sort by value for better visualization
    df_sorted = df.sort_values(by=ylabel, ascending=False).copy()

    plt.figure(figsize=(10, 8))
    bars = plt.bar(
        df_sorted[xlabel],
        df_sorted[ylabel],
        color=plt.cm.viridis(np.linspace(0.4, 0.8, len(df_sorted[xlabel]))),
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha="right")

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.01,
            f"{yval:,.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path_no_ext)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save as PDF
    pdf_path = f"{output_path_no_ext}.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {pdf_path}")

    # Save as PNG
    png_path = f"{output_path_no_ext}.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved plot to: {png_path}")

    plt.close()


def plot_inference_summary(df: pd.DataFrame, output_dir: str):
    """
    Generates a single figure with key inference metrics: time and cost.
    """
    if df.empty:
        print("Skipping summary plot because the dataframe is empty.")
        return

    # --- 1. Data Preparation ---
    df_plot = df.copy()

    # Fix for NaN issue: only convert non-Model object columns to numeric
    for col in df_plot.columns:
        if df_plot[col].dtype == "object" and col != "Model":
            df_plot[col] = (
                df_plot[col]
                .astype(str)
                .str.replace(r"[$,%]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )

    # Use a more descriptive name for File Time Cost
    df_plot.rename(
        columns={"File Time Cost (s)": "Total Execution Time (s)"}, inplace=True
    )
    df_plot.set_index("Model", inplace=True)

    # Calculate the non-LMM time for the stacked bar chart
    df_plot["Other Execution Time (s)"] = (
        df_plot["Total Execution Time (s)"]
        - df_plot["Total LMM Inference Time (s)"]
    )

    # --- 2. Create Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # --- 3. Time Comparison Subplot (Stacked Bar) ---
    ax = axes[0]
    time_cols = ["Total LMM Inference Time (s)", "Other Execution Time (s)"]
    df_plot[time_cols].plot(
        kind="bar", stacked=True, ax=ax, colormap="viridis", rot=15
    )

    ax.set_title("")
    ax.set_ylabel("Time (seconds)", fontsize=36)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=32)
    ax.tick_params(axis="y", labelsize=32)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:,.0f}k"))
    ax.legend(
        ["LMM Inference Time", "Other Execution Time"], fontsize=28, loc="upper left"
    )

    # Add combined total and LMM time labels using bar_label for better placement.
    labels = [
        f"Total: {total:,.0f}s\n(LMM: {lmm:,.0f}s)"
        for total, lmm in zip(
            df_plot["Total Execution Time (s)"],
            df_plot["Total LMM Inference Time (s)"],
        )
    ]

    # Label the top-most container of the stacked bar chart.
    if ax.containers:
        ax.bar_label(
            ax.containers[-1],
            labels=labels,
            fontsize=28,
            weight="bold",
            padding=10,  # Increased padding
            label_type="edge",
        )
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)  # Make space for labels

    # --- 4. Total Cost Subplot ---
    ax = axes[1]
    cost_col = "Estimated Cost (USD)"
    df_plot[cost_col].plot(
        kind="bar", ax=ax, color=plt.cm.viridis(np.linspace(0.4, 0.8, 2)), rot=15
    )
    ax.set_title("")
    ax.set_ylabel("Cost (USD)", fontsize=36)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=32)
    ax.tick_params(axis="y", labelsize=32)
    ax.bar_label(ax.containers[0], fmt="$%.2f", fontsize=28, padding=10)
    ax.margins(y=0.15)

    plt.tight_layout()  # Adjust layout

    # --- 5. Save Figure ---
    output_path_no_ext = os.path.join(output_dir, "inference_summary_time_cost")
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = f"{output_path_no_ext}.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved summary plot to: {pdf_path}")

    png_path = f"{output_path_no_ext}.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved summary plot to: {png_path}")

    plt.close()


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
    Parses a log file to extract details for each LMM API call and total time cost,
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
            r"candidates_token_count=(\d+|None).*"
            r"prompt_token_count=(\d+|None).*"
            r"thoughts_token_count=(\d+|None).*"
            r"total_token_count=(\d+|None)"
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
            (
                candidate_tokens_str,
                prompt_tokens_str,
                thoughts_tokens_str,
                total_tokens_str,
            ) = tokens

            candidate_tokens = (
                int(candidate_tokens_str) if candidate_tokens_str != "None" else 0
            )
            prompt_tokens = int(prompt_tokens_str) if prompt_tokens_str != "None" else 0
            thoughts_tokens = (
                int(thoughts_tokens_str) if thoughts_tokens_str != "None" else 0
            )
            total_tokens = int(total_tokens_str) if total_tokens_str != "None" else 0

            api_calls.append(
                {
                    "inference_time_s": float(time_matches[j]),
                    "candidate_tokens": candidate_tokens,
                    "prompt_tokens": prompt_tokens,
                    "thoughts_token_count": thoughts_tokens,
                    "output_tokens": candidate_tokens + thoughts_tokens,
                    "total_tokens": total_tokens,
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
            "Total LMM Calls": 0,
            "Total LMM Inference Time (s)": 0,
            "Total Input Tokens": 0,
            "Total Output Tokens": 0,
            "Grand Total Tokens": 0,
            "Estimated Cost (USD)": 0,
            "File Time Cost (s)": total_time,
            "LMM Contribution (%)": 0,
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

    lmm_contribution = (
        (total_inference_time / total_time) * 100 if total_time > 0 else 0
    )

    return {
        "Total LMM Calls": len(api_calls),
        "Total LMM Inference Time (s)": total_inference_time,
        "Total Input Tokens": total_input_tokens,
        "Total Output Tokens": total_output_tokens,
        "Grand Total Tokens": grand_total_tokens,
        "Estimated Cost (USD)": total_cost,
        "File Time Cost (s)": total_time,
        "LMM Contribution (%)": lmm_contribution,
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
        if name == "Full":
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
        "Total LMM Calls",
        "Total LMM Inference Time (s)",
        "File Time Cost (s)",
        "LMM Contribution (%)",
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
        "Total LMM Calls",
        "Total LMM Inference Time (s)",
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
    df_agg["LMM Contribution (%)"] = 0.0
    if not df_agg.empty:
        non_zero_mask = df_agg["File Time Cost (s)"] > 0
        df_agg.loc[non_zero_mask, "LMM Contribution (%)"] = (
            df_agg.loc[non_zero_mask, "Total LMM Inference Time (s)"]
            / df_agg.loc[non_zero_mask, "File Time Cost (s)"]
        ) * 100

    # --- Formatting for better readability ---
    for col in ["Total LMM Inference Time (s)", "File Time Cost (s)"]:
        df_agg[col] = pd.to_numeric(df_agg[col], errors="coerce").apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")  # type: ignore
    df_agg["Estimated Cost (USD)"] = pd.to_numeric(df_agg["Estimated Cost (USD)"], errors="coerce").apply(  # type: ignore
        lambda x: f"${x:,.4f}" if pd.notna(x) else ""
    )
    df_agg["Avg. Cost per Series"] = pd.to_numeric(df_agg["Avg. Cost per Series"], errors="coerce").apply(  # type: ignore
        lambda x: f"${x:,.4f}" if pd.notna(x) else ""
    )
    df_agg["LMM Contribution (%)"] = pd.to_numeric(df_agg["LMM Contribution (%)"], errors="coerce").apply(  # type: ignore
        lambda x: f"{x:,.2f}%" if pd.notna(x) else ""
    )
    for col in [
        "Total LMM Calls",
        "Total Input Tokens",
        "Total Output Tokens",
        "Grand Total Tokens",
    ]:
        df_agg[col] = pd.to_numeric(df_agg[col], errors="coerce").astype("Int64").apply(lambda x: f"{int(x):,.0f}" if pd.notna(x) else "")  # type: ignore

    # --- Write output to a file ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "analysis_report.txt")
    with open(output_filename, "w") as f:
        f.write("--- LMM Inference and Cost Analysis ---\n\n")

        f.write("--- Row Count Per Model ---\n")
        f.write("=" * 80 + "\n")
        f.write(df_detailed["Model"].value_counts().to_string())  # type: ignore
        f.write("\n\n")

        f.write("--- Overall Comparison Summary ---\n")
        f.write("=" * 80 + "\n")
        f.write(df_agg.to_string(index=False))
        f.write("\n\n")

        f.write("--- LMM Artifacts Map ---\n")
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

    # --- Generate and Save Visualizations ---
    charts_output_dir = os.path.join(script_dir, "inference_analysis_charts")
    plot_inference_summary(df_agg, charts_output_dir)


if __name__ == "__main__":
    main()
