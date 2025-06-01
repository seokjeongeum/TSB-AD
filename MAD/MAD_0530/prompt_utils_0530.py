import json

import numpy as np


def get_llm_training_context_message(
    X_train_fit_data_param,
    y_train_fit_labels_param,
):
    """Generates a context message about how to use training examples."""
    if X_train_fit_data_param is not None and y_train_fit_labels_param is not None:
        has_anomalies_in_train_examples = np.any(y_train_fit_labels_param == 1)
        
        context_msg = "You are provided with labeled 'training_example_snippets' (JSON format below the plot description). These snippets are a selection from the original training data. Use them to understand patterns of normal (label: 0) and anomalous (label: 1) behavior.\n"
        context_msg += "- **Authoritative Labels:** If a data point in your current 'analysis_data_snippets' has an index that EXACTLY matches an index present in the 'training_example_snippets', you MUST assign an anomaly score strictly based on its label in 'training_example_snippets': score 1.0 if label is 1, and score 0.0 if label is 0. This rule is paramount for those specific overlapping points.\n"
        context_msg += "- **Interpreting Omitted Training Points:** Assume that any data points from the original training period that are NOT included in the provided 'training_example_snippets' were considered normal during the snippet selection process, UNLESS they visually and strongly resemble a clear anomalous pattern shown in the labeled 'training_example_snippets'.\n"
        
        if has_anomalies_in_train_examples:
            context_msg += "- **Learning from Anomalies:** For 'analysis_data_snippets' points NOT directly covered by 'training_example_snippets', apply the patterns of anomalies learned from the 'training_example_snippets' (label: 1) to identify and score similar anomalies.\n"
        else: # No anomalies were in the original training labels, so snippets will also be all normal
            context_msg += "- **Learning from Normality:** The 'training_example_snippets' provided (all labeled 0) represent normal behavior. For 'analysis_data_snippets' points NOT directly covered by 'training_example_snippets', identify and score deviations from this learned normality. These deviations can be sharp spikes, sudden uncharacteristic dips, or other significant changes that break the established local pattern or are hard to predict/reconstruct  considering broader contextual patterns.\n"
        return context_msg
    else:
        return "No labeled training data snippets are provided. Identify anomalies in 'analysis_data_snippets' based on inherent patterns in an unsupervised manner, guided by the plot if available."


def construct_unpredictability_code_gen_prompt(
    i_feat,
    n_samples,
    current_batch_training_snippets_map,
    current_batch_analysis_snippets_map,
    training_data_indices_info_str,
    analysis_data_indices_info_str,
    image_uri_for_llm=None,
    generate_func_with_data_param: bool = False,
    training_feature_col_for_baseline_guidance: np.ndarray | None = None,
    has_train_anomaly_for_unpred_prompt: bool = False,
):
    """Constructs the prompt for the LLM to IDENTIFY hard-to-reconstruct indices
       and generate a SIMPLE Python function assigning 1.0 to them."""

    training_snippets_json_str = (
        "No training example snippets in this batch (or none available)."
    )
    if current_batch_training_snippets_map and current_batch_training_snippets_map.get("value"):
        sorted_training_map = {}
        if "value" in current_batch_training_snippets_map:
            sorted_value_items = sorted(
                current_batch_training_snippets_map["value"].items(),
                key=lambda item: int(item[0])
            )
            sorted_training_map["value"] = dict(sorted_value_items)
        if "label" in current_batch_training_snippets_map:
            sorted_label_items = sorted(
                current_batch_training_snippets_map["label"].items(),
                key=lambda item: int(item[0])
            )
            sorted_training_map["label"] = dict(sorted_label_items)
        training_snippets_json_str = json.dumps(sorted_training_map, indent=2)

    analysis_snippets_json_str = "No analysis data snippets in this batch."
    if current_batch_analysis_snippets_map and current_batch_analysis_snippets_map.get("value"):
        sorted_analysis_map_for_values_only = {}
        if "value" in current_batch_analysis_snippets_map:
            sorted_value_items_analysis = sorted(
                current_batch_analysis_snippets_map["value"].items(),
                key=lambda item: int(item[0])
            )
            sorted_analysis_map_for_values_only["value"] = dict(sorted_value_items_analysis)
        analysis_snippets_json_str = json.dumps(sorted_analysis_map_for_values_only, indent=2)

    training_context_guidance = ""
    if training_feature_col_for_baseline_guidance is not None:
        if has_train_anomaly_for_unpred_prompt:
            training_context_guidance = "The 'Training Snippets' (and green plot section) include labeled anomalies. These define various rates of change. Your focus for unpredictability is on rates of change in analysis data that are *even more extreme or of a different character* than any seen in the entire training set (normal or anomalous examples)."
        else:
            training_context_guidance = "The 'Training Snippets' (and green plot section) represent **NORMAL BEHAVIOR**. Pay close attention to the typical rates of change (single-step differences) observed in this normal training data. This is your baseline for 'expected' rates of change."
    else:
        training_context_guidance = "No specific training data is provided as a baseline. You must infer typical rates of change from the overall visual patterns in the plot. Focus on identifying points in the analysis data that show exceptionally abrupt changes relative to their local context."

    prompt_parts = []
    prompt_parts.append(f"You are an expert in time series analysis. Your primary task is to identify data points (indices) from the 'analysis_data_snippets' that exhibit an **unusually aggressive rate of change (single-step difference)** compared to the patterns observed in the 'Training Snippets' (and the green section of the plot, which typically represents normal behavior if `has_train_anomaly_for_unpred_prompt` is False). After identifying these specific indices based on your analysis of the provided data, you will generate a simple Python function `calculate_unpredictability_scores`. This function will assign a score of 1.0 to your identified indices and 0.0 to all others within the full feature series of length {n_samples}.\n")
    
    prompt_parts.append("**Key Information Provided:**\n")
    prompt_parts.append(f"1.  **Plot of Feature Data (Image URI: {image_uri_for_llm is not None}):**")
    prompt_parts.append(f"    Visualizes Feature {i_feat} (length {n_samples}).")
    prompt_parts.append("    - The **GREEN SECTION** (TRAINING DATA) helps define typical/normal rates of change if `has_train_anomaly_for_unpred_prompt` is False.")
    prompt_parts.append("    - The **ANALYSIS DATA** segment (often highlighted) contains points to assess.")
    prompt_parts.append(f"    - {training_context_guidance}")

    prompt_parts.append(f"2.  **Training Snippets (JSON for context: {training_data_indices_info_str}):**")
    prompt_parts.append("    If `has_train_anomaly_for_unpred_prompt` is False, these exclusively show normal behavior. Analyze typical single-step differences (value[i] - value[i-1]) within these normal examples to understand expected rates of change.")
    prompt_parts.append("    ```json")
    prompt_parts.append(f"    {training_snippets_json_str}")
    prompt_parts.append("    ```")

    prompt_parts.append(f"3.  **Analysis Data Snippets (JSON, values only: {analysis_data_indices_info_str}):**")
    prompt_parts.append("    Examine each point. For `analysis_data_snippets[idx]`, calculate its single-step difference from `analysis_data_snippets[idx-1]`. Compare this to typical differences in normal training data (if available and applicable).")
    prompt_parts.append("    ```json")
    prompt_parts.append(f"    {analysis_snippets_json_str}")
    prompt_parts.append("    ```")

    prompt_parts.append("**Your Task:**")
    prompt_parts.append("1.  **Identify Indices with Unusually Aggressive Rate of Change FROM THE PROVIDED ANALYSIS SNIPPETS:**")
    prompt_parts.append("    Your SOLE FOCUS is on the **rate of change**. For each point in the `analysis_data_snippets`, consider the difference `delta_analysis = current_value - previous_value`.")
    prompt_parts.append("    -   Compare this `delta_analysis` to the range and magnitude of typical single-step deltas observed in the 'Training Snippets' (and green plot section, if it represents normal data).")
    prompt_parts.append("    -   An index from the `analysis_data_snippets` is 'unpredictable' if its `delta_analysis` is **significantly more aggressive** than any delta seen in the normal training data. 'More aggressive' could mean:")
    prompt_parts.append("        - Its absolute magnitude `|delta_analysis|` is much larger than typical `|delta_train_normal|`.")
    prompt_parts.append("        - It's a large positive delta (strong rise) from a certain baseline value, when normal training data only showed much smaller positive deltas from similar baselines.")
    prompt_parts.append("        - It's a large negative delta (strong fall) from a certain baseline value, when normal training data only showed much smaller negative deltas from similar baselines.")
    prompt_parts.append("    -   **Conceptual Example of Your Internal Reasoning Process (Apply this to the actual data you receive):**")
    prompt_parts.append("        'First, I examine the normal training data (green plot section / training snippets) to understand the typical characteristics of single-step changes (`delta_train_normal`). I note the general range, maximum observed magnitudes for rises and falls, and how these changes relate to the baseline values from which they occur.'")
    prompt_parts.append("        'Then, for an analysis point at `some_analysis_index` with value `V_curr` (preceded by `V_prev`), I calculate `delta_analysis = V_curr - V_prev`.'")
    prompt_parts.append("        'I then ask: Is this `delta_analysis` (considering its sign, magnitude, and the baseline `V_prev`) qualitatively and quantitatively different and more extreme than the `delta_train_normal` characteristics I learned? For instance, if `delta_analysis` represents a rise from `V_prev` that is dramatically larger than any rise observed from a similar `V_prev` in the normal training data, then `some_analysis_index` is unpredictable due to this aggressive rate of change.'")
    prompt_parts.append("    -   Your final judgment for each point in `analysis_data_snippets` should be based on this comparison. Do NOT consider the absolute value of the point itself, only the magnitude and character of its change from the *immediately preceding point* relative to changes observed in normal training data.")

    prompt_parts.append("2.  **Generate a Python Function `calculate_unpredictability_scores`:**")
    prompt_parts.append("    -   Name: `calculate_unpredictability_scores`")
    prompt_parts.append(f"    -   Signature: {'`def calculate_unpredictability_scores(X_data_col):`' if generate_func_with_data_param else '`def calculate_unpredictability_scores():`'}")
    prompt_parts.append(f"    -   Body: Initialize `unpredictability_scores = np.zeros({n_samples}, dtype=float)`. For each 0-based index `idx_identified` (from the overall series up to {n_samples-1}) that YOU identified from the `analysis_data_snippets` as having an unusually aggressive rate of change: `unpredictability_scores[idx_identified] = 1.0`")
    prompt_parts.append("    -   Return: The `unpredictability_scores` array.")
    prompt_parts.append("    -   No `import` statements. NumPy (`np`) is pre-available.")
    prompt_parts.append("    -   The function body should be simple: initialize the array, then directly assign 1.0 to the specific indices YOU have identified through your analysis of the provided data.")

    prompt_parts.append(f"**Example Structure (Conceptual - replace placeholders with actual indices YOU identify from the data provided in THIS call):**")
    prompt_parts.append("```python")
    prompt_parts.append("# import numpy as np # This is pre-available, DO NOT include in your output.")
    prompt_parts.append("")
    prompt_parts.append("def calculate_unpredictability_scores():")
    prompt_parts.append(f"    unpredictability_scores = np.zeros({n_samples}, dtype=float)")
    prompt_parts.append("    # Indices identified based on unusually aggressive rate of change ")
    prompt_parts.append("    # compared to normal training data patterns observed in the data for this call:")
    prompt_parts.append("    # E.g.: unpredictability_scores[ACTUAL_INDEX_YOU_FOUND_A] = 1.0 # Justification: e.g., Rate of rise much larger than seen in training from similar baseline.")
    prompt_parts.append("    # E.g.: unpredictability_scores[ACTUAL_INDEX_YOU_FOUND_B] = 1.0 # Justification: e.g., Rate of fall much steeper than seen in training.")
    prompt_parts.append("    return unpredictability_scores")
    prompt_parts.append("```")
    prompt_parts.append(f"(Your actual generated code must use the correct `n_samples` value of {n_samples}. The indices assigned 1.0 must be those YOU determine from your analysis of the provided `analysis_data_snippets` and plot in *this specific call*, based *only* on the rate-of-change criteria.)")
    prompt_parts.append("Output ONLY the Python code string, optionally wrapped in ```python ... ```.")

    prompt_text = "\n".join(prompt_parts)
    return prompt_text


def construct_llm_batch_prompt(
    i_feat,
    n_samples,
    current_batch_training_snippets_map,
    current_batch_analysis_snippets_map,
    training_data_indices_info_str,
    analysis_data_indices_info_str,
    X_train_fit_data_param,
    y_train_fit_labels_param,
    image_uri_for_llm=None,
    is_refinement_attempt: bool = False,
    previous_code_str: str | None = None,
    previous_execution_error_details: str | None = None,
    generate_func_with_data_param: bool = False,
    unpredictability_plot_uri: str | None = None,
    k: int | None = None,
):
    training_snippets_json_str = (
        "No training example snippets in this batch (or none available)."
    )
    if current_batch_training_snippets_map and current_batch_training_snippets_map.get(
        "value"
    ):
        # Sort the inner dictionaries by integer value of index keys
        sorted_training_map = {}
        if "value" in current_batch_training_snippets_map:
            sorted_value_items = sorted(
                current_batch_training_snippets_map["value"].items(),
                key=lambda item: int(item[0])
            )
            sorted_training_map["value"] = dict(sorted_value_items)
        if "label" in current_batch_training_snippets_map:
            sorted_label_items = sorted(
                current_batch_training_snippets_map["label"].items(),
                key=lambda item: int(item[0])
            )
            sorted_training_map["label"] = dict(sorted_label_items)
        
        training_snippets_json_str = json.dumps(
            sorted_training_map, indent=2
        )

    analysis_snippets_json_str = "No analysis data snippets in this batch."
    if current_batch_analysis_snippets_map: # Check if the map itself exists
        # Expects current_batch_analysis_snippets_map to be like: 
        # { "value": {"idx1": val1, ...}, "unpredictability_score": {"idx1": score1, ...} }
        # Sort inner value dict by key
        if "value" in current_batch_analysis_snippets_map and current_batch_analysis_snippets_map["value"]:
            sorted_value_items_analysis = sorted(
                current_batch_analysis_snippets_map["value"].items(),
                key=lambda item: int(item[0])
            )
            # Create a new dict for sorted analysis snippets to preserve original structure if other keys exist
            sorted_analysis_map_for_json = {"value": dict(sorted_value_items_analysis)}
            
            # Include other keys from current_batch_analysis_snippets_map if they exist, like 'unpredictability_score'
            # However, per new requirements, 'unpredictability_score' is no longer provided.
            # if "unpredictability_score" in current_batch_analysis_snippets_map:
            # sorted_analysis_map_for_json["unpredictability_score"] = dict(sorted(current_batch_analysis_snippets_map["unpredictability_score"].items(), key=lambda item: int(item[0]))

            analysis_snippets_json_str = json.dumps(
                sorted_analysis_map_for_json, indent=2
            )

    # Determine the training context message
    training_context_msg = get_llm_training_context_message(
        X_train_fit_data_param,
        y_train_fit_labels_param,
    )

    # Determine the correct function signature and parameter passing for the generated code
    func_signature = "def calculate_anomaly_scores():"
    func_call_example = "scores = calculate_anomaly_scores()"
    if generate_func_with_data_param:
        func_signature = "def calculate_anomaly_scores(X_data_col):"
        func_call_example = "scores = calculate_anomaly_scores(X_data_col)" # X_data_col is conceptual for example

    initial_prompt_header = f"""You are an expert in time series anomaly detection. Your task is to generate a Python function that calculates anomaly scores for a given feature time series.
The function will be named `calculate_anomaly_scores`.
It will take {'`X_data_col` (a NumPy array of the full feature data)' if generate_func_with_data_param else 'no parameters'}.
It must return a NumPy array of anomaly scores, one for each of the {n_samples} data points.
Scores should range from 0.0 (definitely normal) to 1.0 (definitely anomalous)."""

    refinement_instructions = ""
    if is_refinement_attempt:
        initial_prompt_header = f"""You are an expert in time series anomaly detection. Your previous attempt to generate a Python function `calculate_anomaly_scores` failed.
You need to refine the Python code based on the error from the previous execution.
The function will take {'`X_data_col` (a NumPy array of the full feature data)' if generate_func_with_data_param else 'no parameters'}.
It must return a NumPy array of anomaly scores (0.0 to 1.0) for all {n_samples} data points."""
        refinement_instructions = f"""
**Previous Code (that resulted in an error):**
```python
{previous_code_str if previous_code_str else "No previous code available."}
```

**Execution Error from Previous Code:**
{previous_execution_error_details if previous_execution_error_details else "No specific error details from previous execution."}

**Refinement Task:**
Modify the PREVIOUS CODE to fix the error and ensure it correctly calculates anomaly scores according to the original task.
Consider the error message carefully. If it was a `KeyError` or `IndexError` in your anomaly scoring logic, ensure your loops or index accesses are within the bounds of `n_samples` ({n_samples}).
If the error was related to NumPy array shapes, ensure the output `anomaly_scores` array is always of shape `({n_samples},)`.
If the error was "name 'X_data_col' is not defined", it means your function was defined as `def calculate_anomaly_scores():` but tried to access `X_data_col`. You MUST redefine it as `def calculate_anomaly_scores(X_data_col):` and use the passed `X_data_col`.
The refined function must still be named `calculate_anomaly_scores` and have the signature `{func_signature}`.
Focus on robust anomaly identification based on the principles described below.
Output ONLY the MODIFIED Python code string.
"""

    core_scoring_logic_header = "**Core Anomaly Scoring Logic - Reconstructibility and Pattern Adherence:**"

    prompt_text = f"""{initial_prompt_header}
{core_scoring_logic_header}
Your primary goal is to identify points that are "hard to reconstruct" or "unpredictable" based on established patterns and trends in the data.
-   **Low Anomaly Score (e.g., 0.0-0.2):** Assign to points that are easily reconstructible and conform to established local patterns, trends, or seasonality.
    -   **Even if a point's value is outside the range seen in `training_example_snippets`, if it clearly follows an established trend or pattern (e.g., a continuing linear increase, a predictable point in a cycle), it might still be normal and receive a low score.**
-   **High Anomaly Score (e.g., 0.8-1.0):** Assign to points that significantly break from established patterns, are difficult to reconstruct from their neighbors, or represent an uncharacteristic change in behavior.
    -   This includes sharp spikes/dips that are not part of a known volatile pattern, sudden level shifts, or changes in the series' local statistical properties (e.g., variance, periodicity) that are not explained by `training_example_snippets`.

**Characteristic Length `k` for Anomaly Influence:**
A characteristic length `k` = {k if k is not None else 'Not explicitly provided, determine visually/contextually'} will be used by you to define the window of influence for identified anomalous points.
-   If `k` is explicitly provided, use that value. The justification for this `k` is: {f"Provided value k={k}." if k is not None else "You will need to determine `k` based on visual inspection of the anomaly characteristics on the plot(s)."}
-   For sharp, isolated spikes/dips, a very small `k` (e.g., 0, 1, or 2) is appropriate.
-   For broader events (e.g., a sustained level shift, a period of high volatility), a larger `k` reflecting the event's duration is appropriate.
-   When assigning scores, if a point `idx_int` is identified as the core of an anomaly:
    -   `anomaly_scores[idx_int]` should be high (e.g., 1.0).
    -   Its influence can extend to `idx_int - k` and `idx_int + k`. You might apply a decay function (e.g., linear, triangular) to the scores in this window, decreasing from 1.0 at `idx_int` to 0.0 at `idx_int +/- (k+1)`.
    -   If `k=0`, only the point `idx_int` itself gets the high score.

**Information Provided to You:**
1.  **Plot of Feature Data (Image URI: {image_uri_for_llm is not None}):**
    This plot shows the time series for feature {i_feat}.
    Use the plot to visually identify anomalies. Pay attention to:
    -   `Full Data Series for Feature` (`dimgray`): The complete data you are analyzing.
    -   `Train Focus Range` (e.g., `magenta`): If shown, this is a segment of training data identified as particularly relevant for context.
    -   `Overall Training Anomaly Examples` (e.g., `darkred`): If shown, these are known anomalies from the training phase.
    -   `Batch Train Snips Sent` (e.g., `darkgreen`): Specific training examples sent with this batch.
    -   `Batch Analysis Snips` (`darkgoldenrod`): The current segment of data under primary analysis.
    Anomalies are often points or segments that deviate significantly from any established "normal" pattern visible in training data (if available and shown) or from the local surrounding data if no explicit training context is highlighted for "normal".

2.  **Training Example Snippets (JSON, {training_data_indices_info_str}):**
    {training_context_msg}
    ```json
    {training_snippets_json_str}
    ```

3.  **Analysis Data Snippets (JSON, {analysis_data_indices_info_str}):**
    These are the specific data points currently under analysis. Your function should score all {n_samples} points, but these snippets give context to the current batch.
    ```json
    {analysis_snippets_json_str}
    ```
{refinement_instructions}
**Python Function Generation Details:**
-   The function MUST be named `calculate_anomaly_scores`.
-   The function signature MUST be `{func_signature}`.
-   Initialize `anomaly_scores = np.zeros({n_samples}, dtype=float)`.
-   The core of your function should iterate through the `X_data_col` (if `generate_func_with_data_param` is true) or conceptual full data (if not).
-   For each point or identified anomalous segment, apply your logic based on reconstructibility and pattern adherence, considering the characteristic length `k`.
-   Do NOT include `import` statements in the generated code string. NumPy (`np`) is already available in the execution environment.
-   Ensure the final `anomaly_scores` array is returned.

**Example Structure of Generated Code (Conceptual):**
```python
import numpy as np # This import is NOT needed in your output string. np is pre-loaded.

{func_signature}
    anomaly_scores = np.zeros({n_samples}, dtype=float)
    # {'Access X_data_col directly if generate_func_with_data_param is True.' if generate_func_with_data_param else 'Reason about the conceptual full data series.'}

    # --- Your anomaly identification and scoring logic here ---
    # Example for an isolated anomaly at index `idx_anom` with influence `k_val`:
    # idx_anom = 123 # Identified anomalous point
    # k_val = {k if k is not None else 'your_determined_k_for_idx_anom'} # Use the provided k or determine it.
    # anomaly_scores[idx_anom] = 1.0
    # # Apply decay if k_val > 0
    # for j_offset in range(1, k_val + 1):
    #     # Calculate decayed score (e.g., linear decay)
    #     decayed_score = 1.0 - (j_offset / (k_val + 1.0))
    #     if idx_anom - j_offset >= 0:
    #         anomaly_scores[idx_anom - j_offset] = max(anomaly_scores[idx_anom - j_offset], decayed_score)
    #     if idx_an_idx_anom + j_offset < {n_samples}:
    #         anomaly_scores[idx_anom + j_offset] = max(anomaly_scores[idx_anom + j_offset], decayed_score)
    # --- End example ---

    return np.clip(anomaly_scores, 0.0, 1.0) # Ensure scores are clipped
```
Output ONLY the Python code string, optionally wrapped in ```python ... ```. Do NOT output any other text or JSON.
"""
    return prompt_text


def construct_interest_id_prompt(
    i_feat,
    num_train_samples,
    training_snippets_json_str_for_id_step,  # JSON string of training snippets
    target_range_size,  # This is the *effective* target size for the LLM's output range (e.g., halved value)
    image_uri_for_llm=None,
):
    """Constructs the prompt for the LLM to identify a focused range of interest in training data."""
    # Base prompt text
    prompt_text = f"""You are an expert in time series analysis.
For Feature {i_feat}, you are given training data snippets and a target range size.
Your task is to identify a contiguous sub-range (start_index, end_index) within the training data that is most 'interesting' or representative for understanding anomalous behavior for this feature, given the provided snippets and the target size.

**Inputs:**
1.  **Plot of Training Feature Data (if image URI provided: {image_uri_for_llm is not None}):**
    This plot shows the entire training data for feature {i_feat}. It may visually highlight labeled anomalies if they exist.
    The plot may highlight:
    - 'Full Context Ft ... (Non-Training Part)': Data outside the training span, in `dimgray`.
    - 'Training Data Span': The actual training data segment, in `darkblue`.
    - 'Labeled Training Anomalies': Anomalous segments within training data, in `darkred`.
    Use this plot to visually guide your decision.

2.  **Training Snippets (JSON):**
    Structure: {{ ""value"": {{ ""index_str"": float_value }}, ""label"": {{ ""index_str"": int_label }} }}. Label 1=anomaly, 0=normal.
    These are selected snippets from the training data (length {num_train_samples}).
    ```json
    {training_snippets_json_str_for_id_step}
    ```

3.  **Target Range Size (approximate):** {target_range_size}
    The identified range [start_index, end_index] should ideally be around this length. It can be smaller if the data is short, or slightly larger if a coherent anomalous event spans more.

**Your Task: Identify and Output the Most Interesting Range**
Output a JSON object with two keys: "start_index" and "end_index".
-   `start_index`: The 0-based starting index of the most interesting sub-range in the training data.
-   `end_index`: The 0-based ending index of the most interesting sub-range (inclusive).

**Criteria for "Interesting Range":**
1.  **Presence of Anomalies:** Prioritize ranges that include or are very near points labeled as anomalies (label: 1) in the `training_snippets_json_str_for_id_step` and visually confirmed on the plot.
2.  **Transitions:** Ranges that show clear transitions between normal and anomalous behavior, or varied anomalous patterns, are highly valuable.
3.  **Representativeness:** If multiple anomalous regions exist, pick one that seems most representative or challenging, or a dense cluster of anomalies if available.
4.  **Context:** Include some normal data points around the anomalies to provide context, if the target_range_size allows.
5.  **Adherence to Target Size:** Try to make the range `(end_index - start_index + 1)` close to `target_range_size`, but prioritize capturing a complete anomalous event or a significant pattern over strict adherence to the size. If no anomalies are present, select a range that best represents typical patterns or subtle variations if any.
6.  **Validity:** `0 <= start_index <= end_index < num_train_samples ({num_train_samples})`.

**Example Output:**
If `num_train_samples` is 5000 and `target_range_size` is 200, and you identify the most interesting segment to be from index 1000 to 1199 (inclusive):
```json
{{
  "start_index": 1000,
  "end_index": 1199
}}
```
If no anomalies are present and the data is short, e.g., `num_train_samples` is 50, `target_range_size` 200:
```json
{{
  "start_index": 0,
  "end_index": 49
}}
```Output ONLY the JSON object. Do NOT include any other text or markdown.
"""
    return prompt_text
