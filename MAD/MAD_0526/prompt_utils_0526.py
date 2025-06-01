import json

import numpy as np


def get_llm_training_context_message(
    is_analyzing_training_data_itself: bool,
    X_train_fit_data_param,
    y_train_fit_labels_param,
):
    """Generates a context message about how to use training examples."""
    if X_train_fit_data_param is not None and y_train_fit_labels_param is not None:
        has_anomalies_in_train_examples = np.any(y_train_fit_labels_param == 1)
        context_msg = "You are provided with labeled 'training_example_snippets'. Use these to understand patterns of normal (label: 0) and anomalous (label: 1) behavior. "
        if is_analyzing_training_data_itself:
            context_msg += "You are currently analyzing this training data itself. "
            if has_anomalies_in_train_examples:
                context_msg += "Your primary goal is to correctly identify the anomalous points shown in 'training_example_snippets' (label: 1) and mark other points as normal (0)."
            else:
                context_msg += "The provided 'training_example_snippets' (all labeled 0) exemplify normal behavior for this feature. When analyzing this training data itself, use these examples to establish a baseline of normality. Your primary goal is to score points consistent with this learned normality as low (e.g., 0.0-0.1). However, if you observe significant visual or patterned deviations from this established normality within this training set, you may assign higher scores to those points."
        else:
            context_msg += (
                "You are currently analyzing new/unseen 'analysis_data_snippets'. "
            )
            if has_anomalies_in_train_examples:
                context_msg += "Apply the patterns of anomalies learned from 'training_example_snippets' (label: 1) to identify similar anomalies in 'analysis_data_snippets'."
            else:
                context_msg += "The 'training_example_snippets' represent normal behavior (all labels are 0). Identify deviations from this learned normality in 'analysis_data_snippets'."
        return context_msg
    else:
        return "No labeled training data snippets are provided. Identify anomalies in 'analysis_data_snippets' based on inherent patterns in an unsupervised manner, guided by the plot if available."


def construct_llm_batch_prompt(
    i_feat,
    n_samples,
    current_batch_training_snippets_map,
    current_batch_analysis_snippets_map,
    training_metadata_for_prompt,
    analysis_data_indices_info_str,
    is_analyzing_training_data_itself: bool,
    X_train_fit_data_param,
    y_train_fit_labels_param,
    image_uri_for_llm=None,
    is_refinement_attempt: bool = False,
    previous_code_str: str | None = None,
    previous_execution_error_details: str | None = None,
    generate_func_with_data_param: bool = False,
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
    if current_batch_analysis_snippets_map and current_batch_analysis_snippets_map.get(
        "value"
    ):
        # Sort the inner 'value' dictionary by integer value of index keys
        sorted_analysis_map = {}
        if "value" in current_batch_analysis_snippets_map:
            sorted_value_items_analysis = sorted(
                current_batch_analysis_snippets_map["value"].items(),
                key=lambda item: int(item[0])
            )
            sorted_analysis_map["value"] = dict(sorted_value_items_analysis)
        # Analysis snippets don't have labels in their map structure typically

        analysis_snippets_json_str = json.dumps(
            sorted_analysis_map, indent=2
        )

    training_context_msg = get_llm_training_context_message(
        is_analyzing_training_data_itself,
        X_train_fit_data_param,
        y_train_fit_labels_param,
    )

    training_metadata_json_str = "No training data metadata available."
    if training_metadata_for_prompt:
        training_metadata_json_str = json.dumps(training_metadata_for_prompt, indent=2)

    prompt_text = f"""You are an expert anomaly detection model.
For Feature {i_feat} (analysis data length {n_samples}), you are given:

1.  **Plot (if image URI provided: {image_uri_for_llm is not None}):**
    The plot shows the full time series for feature {i_feat}. It may highlight:
    - 'Full Data Series for Feature': The complete data, with non-highlighted parts shown in `dimgray`.
    - 'Overall Training Anomaly Examples': Examples of training anomalies, where the data line appears in `darkred`.
    - 'Batch Train Snips Sent': Training snippets provided in JSON, their corresponding segment on the data line appears in `darkgreen`.
    - 'Batch Analysis Snips Sent': The current data segment being analyzed, its corresponding segment on the data line appears in `darkgoldenrod`.
    Read the plot carefully to understand how these different contexts are visually represented by the data line's appearance and specified colors.

2.  **Training Context Message:** {training_context_msg}

3.  **Training Data Metadata (JSON):**
    Provides summary statistics of the *entire* training dataset for this feature.
    Structure: {{"min_val": float, "max_val": float, "anomalies": [{{"index": int, "value": float}}, ...]}}
    Anomalies list is empty if no anomalies are labeled in training data.
    ```json
    {training_metadata_json_str}
    ```

4.  **Training Snippets (JSON, if provided and fit token budget):**
    These are selected examples from the training data to illustrate patterns.
    Structure: {{ "value": {{ "index_str": float_value }}, "label": {{ "index_str": int_label }} }}. Label 1=anomaly, 0=normal.
    ```json
    {training_snippets_json_str}
    ```

5.  **Analysis Snippets (JSON: {analysis_data_indices_info_str}):**
    Structure: {{ "value": {{ "index_str": float_value }} }}. Your task is to determine their anomaly status and assign appropriate scores.
    ```json
    {analysis_snippets_json_str}
    ```

"""

    if is_refinement_attempt:
        prompt_text += f"""
--- REFINEMENT ATTEMPT ---
You are now in a REFINEMENT ATTEMPT. A previous attempt was made to generate anomaly scores for this batch.
Your task is to ANALYZE THE PREVIOUS ATTEMPT and GENERATE REVISED PYTHON CODE for `calculate_anomaly_scores` that improves upon it.

**Previous Attempt's Artifacts for Your Review:**
A.  **Previous Python Code String (`calculate_anomaly_scores`):**
    This is the Python code that generated the scores and plot mentioned above.
    ```python
    {previous_code_str if previous_code_str else "Not available."}
    ```

**Refinement Instructions:**
- Review the previous attempt's code, its output scores (JSON), and the visualization (plot).
"""

        if previous_execution_error_details:
            prompt_text += f"""
B.  **Previous Execution Error Details:**
    The previous execution of the Python code resulted in the following error:
    ```
    {previous_execution_error_details}
    ```
    """

    prompt_text += f"""
- Identify areas for improvement:
    - Did the previous code correctly apply the scoring logic (e.g., broad influence rule, k >= 50, linear decay)?
    - Were anomalies missed? Were normal points incorrectly scored as anomalous?
    - Was the extent of influence (`k`) for core anomalies appropriate based on visual cues?
    - Are the justification comments in the previous code accurate and sufficient?
- Generate a *new* `calculate_anomaly_scores` Python function. This function should:
    - Incorporate improvements based on your analysis of the previous attempt.
    - Strictly adhere to all original task requirements (output format, scoring logic, justification comments, etc.).
    - If the previous attempt was largely correct, your new code might be very similar but with minor tweaks or improved justifications. If it was significantly flawed, the new code might be substantially different.
- Your justification comments in the new code should not only explain the scoring but also briefly mention how it improves upon or corrects the previous attempt if significant changes were made.
--- END REFINEMENT ATTEMPT ---
"""

    prompt_text += f"""
Your Task (applies to both initial and refinement attempts):
Generate ONLY the Python code string for a function `calculate_anomaly_scores`.
This function MUST:"""

    if generate_func_with_data_param:
        prompt_text += f"""
1.  Accept one argument: `X_data_col` (a 1D NumPy array of length {n_samples} representing the current feature's data).
2.  Return a 1D NumPy array of the length {n_samples}, containing **FLOAT anomaly scores between 0.0 (definitely normal) and 1.0 (definitely anomalous)**."""
    else:
        prompt_text += f"""
1.  Accept NO arguments. It will use the 'analysis_data_snippets' and 'training_example_snippets' provided in the prompt's context.
2.  Return a 1D NumPy array of the length {n_samples}, containing **FLOAT anomaly scores between 0.0 (definitely normal) and 1.0 (definitely anomalous)**."""

    prompt_text += f""" This will be referred to as `anomaly_scores` in the template. Scores can be continuous values within this range.
3.  **CRITICAL SCORING LOGIC (Plot-Focused with Score Influence):**
    *   Initialize `anomaly_scores = np.zeros({{n_samples}}, dtype=float)`. Note `dtype=float`.
    *   **Primary Analysis Source: Plot.** Your main analysis should be driven by visually inspecting the provided plot (either the main input plot or the `previous_anomaly_plot_uri` if in refinement).
    *   **Role of Snippet JSONs:**
        *   The `analysis_data_snippets` JSON tells you the *specific indices* within the plot that are the focus of THIS BATCH. These are your primary points of interest.
        *   The `training_example_snippets` JSON (and plot highlights) helps you understand what normal and anomalous patterns look like *visually*.
        *   The `training_metadata_for_prompt` provides overall context about the training data distribution and any known anomalies.
    *   **Identifying and Scoring Anomalies:** For each index `idx_int` corresponding to an entry in `analysis_data_snippets` for THIS BATCH:
        1.  Locate `idx_int` on the plot within the 'Batch Analysis Snips Sent' segment.
        2.  Visually compare the data line pattern around `idx_int` to learned normal and anomalous patterns (from `training_example_snippets` if provided, and `training_metadata_for_prompt`).
        3.  **Consider the following when assessing `idx_int`:**
            *   **Magnitude:** Does the value significantly deviate from global training min/max or local norms?
            *   **Morphology (Shape):** Is it a sharp spike/dip (e.g., "V-shape"), a sudden level shift, an unusual oscillation, or another distinct pattern that contrasts with typical behavior seen in training examples or the surrounding local data? Consider the "sharpness" or "pointedness" of peaks/troughs.
            *   **Rate of Change:** Is there an unusually rapid ascent or descent to/from `idx_int` compared to normal fluctuations?
            *   **Local Context:** How does the pattern at `idx_int` compare to its immediate preceding and succeeding data points and trends? A deviation might be anomalous primarily due to its local abruptness.
            *   **Follow-up Instability:** Does a primary deviation at `idx_int` lead to subsequent unusual behavior, like smaller "hiccups," oscillations, or a failure to return to a stable baseline quickly?
        4.  If the visual pattern at `idx_int`, considering the factors above, strongly indicates an anomaly, this `idx_int` becomes a candidate for a core anomaly. Proceed to the "Scoring Surrounding Points" rule below. If it appears clearly normal, assign a low score (e.g., 0.0 to 0.1) to `anomaly_scores[idx_int]`. If ambiguous, assign an intermediate score reflecting your confidence, but typically this would not trigger the broad influence rule unless confidence becomes high.
    *   **Scoring Surrounding Points (Mandatory Broad Influence for Core Anomalies):** If you identify `idx_int` (from `analysis_data_snippets`) as the center of a visually distinct anomalous event (e.g., a clear spike, dip, or shift inconsistent with normal training patterns/metadata, considering shape, rate of change, and local context):
        1.  **Set `anomaly_scores[idx_int] = 1.0`.**
        2.  **Determine an influence half-width `k`:**
            *   The choice of `k` (e.g., typical values could range from small integers for very localized effects, or more if strongly supported by visual evidence) should be primarily guided by the visual extent of the anomaly's "area of effect" on the plot. For example, if the plot shows a disturbance lasting for roughly 20 points around `idx_int`, then a `k` value like 10 would be appropriate. If it's a broader effect over ~100 points, k=50 might be suitable.
            *   `k` represents the number of points to *each side* of `idx_int` that are influenced. The total influenced segment is `2*k + 1`.
            *   If visual cues for the full extent of `k` are ambiguous but `idx_int` is clearly anomalous, choose a reasonable default `k` based on the observed characteristics (e.g., k=5 or k=10 for sharp, localized events; k=20 or k=30 for moderately extended events) and state your reasoning. There is no strict minimum for `k`; base it on visual evidence.
        3.  **Apply symmetrical linear decay to surrounding points:**
            *   Scores should decrease linearly from `1.0` at `idx_int` to `0.01` at `idx_int + k` on the right side.
            *   Scores should decrease linearly from `1.0` at `idx_int` to `0.01` at `idx_int - k` on the left side.
            *   Points `idx_int + j` and `idx_int - j` (for `1 <= j <= k`) should have the same score, derived from this linear decay.
            *   Use `np.linspace` for generating these decaying scores and apply them efficiently using NumPy slicing. Ensure array bounds are respected (scores should not be applied outside the `0` to `n_samples-1` range).
            *   The example code in the template demonstrates a robust way to implement this, including handling array boundaries.
        4.  When applying these scores, use `np.maximum(current_scores_in_slice, new_decay_scores)` to ensure that if a point is influenced by multiple considerations (e.g., overlapping anomalies or widespread rules), it retains the highest applicable score.
    *   **Widespread Anomalies in Batch (Plot-Driven):** If a large contiguous portion of the *current 'Batch Analysis Snips Sent' segment on the plot* appears to have a sustained, visually distinct anomalous pattern:
        *   Assign high scores (e.g., 0.8-1.0) consistently across the indices of this visually confirmed widespread anomalous segment.
        *   Consider gentle decay at boundaries if visually appropriate.
        *   This rule can act in conjunction with the point-based decay. Use `np.maximum` to merge scores.
    *   The `anomaly_scores` array should reflect a nuanced scoring based on visual evidence from the plot, with scores ranging from 0.0 to 1.0.
4.  **Adhere to Training Context:** Follow the guidance provided in point 2.
5.  **Justification Comments:** For EACH distinct anomalous event (originating from an `analysis_data_snippets` index) where you apply the "Mandatory Broad Influence" rule (i.e., score set to 1.0 at `idx_int` and decayed broadly):
    a.  State the primary index `idx_int` from `analysis_data_snippets` that triggered this observation.
    b.  Explain the visual patterns on the plot justifying why `idx_int` is considered the center of an anomaly. **Refer to magnitude, morphology (e.g., V-dip, spike, shift), rate of change, local context, and any follow-up instability observed.**
    c.  Describe your reasoning for the scores assigned to surrounding points: **Confirm that `anomaly_scores[idx_int]` is 1.0. State the chosen half-width `k`. Explicitly state the start (`idx_int-k`) and end (`idx_int+k`) indices of the ideal influenced range. Explain how visual cues on the plot (e.g., "the visual disturbance is localized and extends for roughly 20 points, so k=10 was chosen," or "visual extent was ambiguous for this sharp spike, so a default k=5 was used," or "the anomaly shows a broader impact for ~100 points, so k=50 was chosen") justify this choice of `k`. Confirm that scores decay linearly to 0.01 at these `idx_int +/- k` boundaries (or at array edges if `k` extends beyond them).**
    d.  How you've adhered to the Training Data Context based on visual cues.
    e.  (If in refinement mode and changes were made due to previous attempt): Briefly note what was corrected or improved regarding this specific event.
6.  **Efficiency for Scoring Ranges:** Use NumPy slicing (e.g., `anomaly_scores[start:end] = desired_scores_array` or `anomaly_scores[start:end] = constant_score`).
7.  **Constraints:**
    *   All logic must be self-contained within `calculate_anomaly_scores`.
    *   Do NOT include `import` statements. NumPy (`np`) is available.
    *   Return ONLY the `anomaly_scores` 1D NumPy array of FLOATS.

Output ONLY the Python code string, optionally wrapped in ```python ... ```. Do NOT output any other text or JSON.
Example of direct Python code output:
Python function template:
```python"""

    if generate_func_with_data_param:
        prompt_text += f"""
import numpy as np # np is available.
def calculate_anomaly_scores(X_data_col):
    # X_data_col is the 1D NumPy array for the current feature, length {n_samples}.
    # Use X_data_col directly in your logic instead of relying on conceptual analysis_data_snippets for values.
    # The JSON `analysis_data_snippets` is still provided above to indicate WHICH indices are of current interest for THIS BATCH.
    anomaly_scores = np.zeros({n_samples}, dtype=float) # Initialize with float scores
    n_samples_for_code = {n_samples} # Make n_samples available inside the function"""
    else:
        prompt_text += f"""
import numpy as np # np is available.
def calculate_anomaly_scores():
    # The 'analysis_data_snippets' and 'training_example_snippets' are conceptually available.
    # If in a refinement attempt, previous_code_str, previous_anomaly_scores_json_str, and previous_anomaly_plot_uri would also be conceptually available.
    anomaly_scores = np.zeros({n_samples}, dtype=float) # Initialize with float scores
    n_samples_for_code = {n_samples} # Make n_samples available inside the function"""

    prompt_text += f"""

    # --- BEGIN SCORING LOGIC (Analyze THIS BATCH's relevant plot segment via 'analysis_data_snippets') ---
    # Example: analysis_snippets_map = {{ ""value"": {{ ""100"": 0.5, ""101"": 2.3, ""102"": 0.4 }} }}

    # Example for "Mandatory Broad Influence for Core Anomalies":
    # Plot shows a sharp V-dip at index 500, inconsistent with normal patterns and local trend.
    # Visual inspection suggests the anomalous effect is quite localized, say affecting about 30 points in total.
    # Thus, a half-width k=15 is chosen.
    # Primary anomaly point: core_anomaly_idx = 500 (from analysis_data_snippets).
    # Justification: Sharp V-dip at 500. Characterized by rapid descent and ascent. Local context shows stable behavior before and after. Training data shows no such V-dips.
    # Scoring:
    #   - anomaly_scores[500] = 1.0.
    #   - k_half_width = 15 (since 2*15+1 = 31, and visual effect is ~30 points).
    #   - Scores decay linearly from 1.0 at index 500 to 0.01 at indices 485 (500-15) and 515 (500+15).
    
    # Example implementation for the above scenario:
    # core_anomaly_idx = 500 # Assume this came from an analysis snippet
    # k_half_width = 15    # Chosen based on visual extent.
    #
    # if 0 <= core_anomaly_idx < n_samples_for_code:
    #     # Center point
    #     # anomaly_scores[core_anomaly_idx] = 1.0 # This will be handled by linspace if done carefully
    #
    #     # Right side decay (from core_anomaly_idx to core_anomaly_idx + k_half_width)
    #     # This slice includes the core point itself.
    #     start_idx_r = core_anomaly_idx
    #     # Ideal end of this segment of decay
    #     end_idx_r_ideal = core_anomaly_idx + k_half_width 
    #     # Actual end, clipped by array boundary
    #     end_idx_r_actual = min(end_idx_r_ideal, n_samples_for_code - 1) 
    #     
    #     if start_idx_r <= end_idx_r_actual: # Proceed if the slice is valid
    #         # Linspace for the full ideal decay from 1.0 to 0.01 over k_half_width steps (+1 for num points)
    #         full_decay_values_r = np.linspace(1.0, 0.01, num=k_half_width + 1)
    #         # Determine how many points of this decay are actually needed for the current slice
    #         num_actual_points_in_slice_r = end_idx_r_actual - start_idx_r + 1
    #         scores_to_apply_r = full_decay_values_r[:num_actual_points_in_slice_r]
    #
    #         current_slice_r = slice(start_idx_r, end_idx_r_actual + 1)
    #         anomaly_scores[current_slice_r] = np.maximum(
    #             anomaly_scores[current_slice_r],
    #             scores_to_apply_r
    #         )
    #
    #     # Left side decay (from core_anomaly_idx - k_half_width to core_anomaly_idx)
    #     # This slice also includes the core point.
    #     end_idx_l = core_anomaly_idx 
    #     # Ideal start of this segment of decay
    #     start_idx_l_ideal = core_anomaly_idx - k_half_width
    #     # Actual start, clipped by array boundary
    #     start_idx_l_actual = max(0, start_idx_l_ideal)
    #
    #     if start_idx_l_actual <= end_idx_l: # Proceed if the slice is valid
    #         # Linspace for the full ideal decay from 0.01 to 1.0 over k_half_width steps (+1 for num points)
    #         full_decay_values_l = np.linspace(0.01, 1.0, num=k_half_width + 1)
    #         # Determine how many points of this decay are actually needed for the current slice
    #         num_actual_points_in_slice_l = end_idx_l - start_idx_l_actual + 1
    #         # We need the tail end of full_decay_values_l
    #         scores_to_apply_l = full_decay_values_l[(k_half_width + 1 - num_actual_points_in_slice_l):]
    #         
    #         current_slice_l = slice(start_idx_l_actual, end_idx_l + 1)
    #         anomaly_scores[current_slice_l] = np.maximum(
    #             anomaly_scores[current_slice_l],
    #             scores_to_apply_l
    #         )
    #
    #     # After these operations, anomaly_scores[core_anomaly_idx] will effectively be 1.0,
    #     # and scores at core_anomaly_idx +/- k_half_width (if within bounds) will be 0.01.

    # --- END SCORING LOGIC ---
    return anomaly_scores
```
"""
    return prompt_text


def construct_interest_id_prompt(
    i_feat,
    num_train_samples,
    training_snippets_json_str_for_id_step,  # JSON string of training snippets
    target_range_size,  # This is the *effective* target size for the LLM's output range (e.g., halved value)
    image_uri_for_llm=None,
):
    """Constructs the prompt for the interest identification step."""
    # THIS FUNCTION IS NO LONGER USED AND WILL BE REMOVED OR COMMENTED OUT
    # AS THE INTEREST IDENTIFICATION STEP IS BEING REMOVED.
    pass # Placeholder, will be removed or commented

    # target_range_size is now pre-adjusted by the caller before being passed here.
    # No need to halve it again in this function.
    effective_target_range_size = target_range_size

    if num_train_samples > 0 and effective_target_range_size > num_train_samples:
        effective_target_range_size = num_train_samples
    elif num_train_samples == 0:
        effective_target_range_size = 0

    prompt_text = f"""You are an expert anomaly detection model.
For Feature {i_feat} (training data length {num_train_samples}), you are given:
1. A plot of the data for Feature {i_feat}. (Image URI provided: {image_uri_for_llm is not None}). Key elements potentially highlighted on the plot (colors mentioned are typical):
    - Full data series: Typically `dimgray`.
    - Identified training interest/focus range: If applicable, often highlighted (e.g., with `magenta` or `deepskyblue`).
    - Overall training anomaly examples: If shown, often with `darkred` or `lightcoral`.
    - Snippets for this batch (training context): If shown, often with `darkgreen` or `limegreen`.
2. Snippets from this training data (JSON format below). Structure: {{"value": {{"index_str": float_value}}, "label": {{"index_str": int_label}} }}. Label 1=anomaly, 0=normal.
```json
{training_snippets_json_str_for_id_step}
```

Your Task:
Analyze the provided training data (plot and snippets) for feature {i_feat}.
Identify and return a single contiguous range (start_index, end_index) within the training data (indices 0 to {max(0, num_train_samples-1)}) that you are most interested in for detailed examination or believe is most crucial for understanding anomaly patterns.
Prioritize ranges with interesting/complex behavior, volatility changes, or clear anomalies.

The identified range should ideally contain approximately {effective_target_range_size} data points. The actual number of points can vary based on your expert judgment of what forms a cohesive "interesting" segment, but aim for this size.
Ensure start_index <= end_index.
If total samples ({num_train_samples}) is less than {effective_target_range_size}, or if no specific range stands out, or data is very short, you may return a range covering all available samples (e.g., 0 to {max(0, num_train_samples-1)}).

Output ONLY a JSON object with keys "start_index" and "end_index".
Example: {{"start_index": 100, "end_index": 250}}
Constraints: 0 <= start_index <= end_index < {num_train_samples if num_train_samples > 0 else 1}.
If num_train_samples is 0, output: {{"start_index": 0, "end_index": 0}}.
Fallback for no specific range or very short data ({num_train_samples} samples): {{"start_index": 0, "end_index": {max(0, num_train_samples-1)} }}.
"""
    return prompt_text
