import datetime
import io
import logging
import os
import sys
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.preprocessing import MinMaxScaler

from TSPulse2.TSPulse2ADUtility import TSPulse2ADUtility

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.models.tspulse.modeling_tspulse import \
    TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    AggregationFunction, TimeSeriesAnomalyDetectionPipeline, score_smoothing)

load_dotenv()


class TSPulse2Pipeline(TimeSeriesAnomalyDetectionPipeline):
    def __init__(
        self,
        model: TSPulseForReconstruction,
        *args,
        prediction_mode: List[str],
        aggr_function: str = AggregationFunction.MAX.value,
        aggregation_length: int = 32,
        smoothing_length: int = 8,
        probabilistic_processor: Optional[PostHocProbabilisticProcessor] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            prediction_mode=prediction_mode,
            aggr_function=aggr_function,
            aggregation_length=aggregation_length,
            smoothing_length=smoothing_length,
            probabilistic_processor=probabilistic_processor,
            **kwargs,
        )
        self._model_processor = TSPulse2ADUtility(
            model,
            mode=prediction_mode,
            aggregation_length=aggregation_length,
            **kwargs,
        )
        self.width = 160
        self.fontsize = 48

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = (
            super()._sanitize_parameters(**kwargs)
        )
        if "use_llm_selection" in kwargs:
            postprocess_kwargs["use_llm_selection"] = kwargs["use_llm_selection"]
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def _get_single_channel_llm_selection(
        self, client, model, scores_dict, target_channel_name, raw_data
    ):
        """Helper function to perform plotting and LLM call for a single channel."""
        num_scores = len(scores_dict)
        num_plots = num_scores + 1  # Add one for the raw data plot
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(self.width, 6 * num_plots),
            sharex=False,
        )
        locator = mticker.MaxNLocator(nbins=40, prune="both")

        # Plot raw data for the channel
        ax_raw = axes[0]
        ax_raw.set_ylabel("Raw Data Value", color="tab:blue")
        ax_raw.plot(
            raw_data.index, raw_data.values, color="tab:blue", alpha=0.7, linewidth=3
        )
        ax_raw.tick_params(axis="y", labelcolor="tab:blue", labelsize=self.fontsize)
        ax_raw.set_title(f"Raw Data: {target_channel_name}", fontsize=self.fontsize)
        data_min, data_max = raw_data.values.min(), raw_data.values.max()
        padding = (data_max - data_min) * 0.1
        ax_raw.set_ylim(data_min - padding, data_max + padding)
        ax_raw.set_xlim(raw_data.index.min(), raw_data.index.max())
        ax_raw.xaxis.set_major_locator(locator)
        ax_raw.tick_params(axis="x", labelsize=self.fontsize * 0.5)

        # Plot each anomaly score for the channel
        for i, (key, score) in enumerate(scores_dict.items()):
            ax = axes[i + 1]
            ax.set_ylabel("Anomaly Score", color="tab:orange")
            ax.plot(
                raw_data.index,
                score,
                color="tab:orange",
                linestyle="-",
                linewidth=3,
            )
            ax.tick_params(axis="y", labelcolor="tab:orange", labelsize=self.fontsize)
            ax.set_title(
                f"Anomaly Score: {key} (for {target_channel_name})",
                fontsize=self.fontsize,
            )
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(raw_data.index.min(), raw_data.index.max())
            ax.xaxis.set_major_locator(locator)
            ax.tick_params(axis="x", labelsize=self.fontsize * 0.5)

        axes[-1].set_xlabel("Time", fontsize=self.fontsize)
        plt.suptitle(
            f"Analysis for Channel: {target_channel_name}",
            y=0.98,
            fontsize=self.fontsize,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        # --- START: DEBUGGING CODE ---
        # Create a directory to store the debug plots and thoughts
        artifacts_dir = "llm_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_plot_path = os.path.join(artifacts_dir, f"{timestamp}_plot.png")

        # Save the figure to a file
        plt.savefig(debug_plot_path, bbox_inches="tight", pad_inches=0)
        # --- END: DEBUGGING CODE ---
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        image_bytes = buf.read()
        plt.close(fig)

        plot_order = list(scores_dict.keys())
        prompt = f"""
The image displays several plots for a single data channel named '{target_channel_name}'.
The first plot shows the raw data for this channel.
The subsequent plots show different anomaly scores for this same channel. The order is: {plot_order}.
Analyze the raw data in the top plot, then select the score below that most effectively highlights the unusual patterns in the raw data.
The available score methods are: {', '.join(plot_order)}.
Based on your analysis, which anomaly score is the best for channel '{target_channel_name}'?
"""
        contents: types.ContentListUnion = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/png",
                        data=image_bytes,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        score_keys = list(scores_dict.keys())

        response = None
        max_retries = 5
        delay = 2  # Start with a 2-second delay

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=32768, include_thoughts=True
                        ),
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.STRING, enum=score_keys
                        ),
                    ),
                )
                # If successful, break the loop
                break
            except Exception as e:
                logging.warning(
                    f"LLM call failed for '{target_channel_name}' on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        selected_key = None
        if response and response.candidates:
            if hasattr(response, "usage_metadata"):
                logging.info(
                    f"LLM token count for '{target_channel_name}': {response.usage_metadata}"
                )
            first_candidate = response.candidates[0]
            if first_candidate.content and first_candidate.content.parts:
                thought_summary = ""
                for part in first_candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thought_summary += f"Thought: {part.thought}\n"
                    elif hasattr(part, "text") and part.text:
                        cleaned_text = part.text.strip()
                        if cleaned_text:
                            thought_summary += f"Output Part: {cleaned_text}\n"
                            import json

                            try:
                                json_response = json.loads(cleaned_text)
                                if isinstance(json_response, dict):
                                    selected_key = next(iter(json_response.values()))
                                else:
                                    selected_key = json_response
                            except (json.JSONDecodeError, TypeError):
                                selected_key = cleaned_text.strip('"')
                if thought_summary:
                    thought_filename = f"{timestamp}_thoughts.txt"
                    thought_save_path = os.path.join(artifacts_dir, thought_filename)
                    with open(thought_save_path, "w") as f:
                        f.write(thought_summary)
        return selected_key

    def _select_head_with_llm(self, scores_dict, target_columns, raw_data):
        if not scores_dict or all(
            not isinstance(v, np.ndarray) or v.size == 0 for v in scores_dict.values()
        ):
            return {}
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = "gemini-2.5-pro"

        channel_selections = {}
        for i, channel_name in enumerate(target_columns):
            start_time = time.time()
            logging.info(
                f"LLM: Analyzing channel {i+1}/{len(target_columns)}: {channel_name}"
            )
            raw_channel_data = raw_data[[channel_name]]
            single_channel_scores = {
                key: score_array[:, i] for key, score_array in scores_dict.items()
            }

            selected_key = self._get_single_channel_llm_selection(
                client, model, single_channel_scores, channel_name, raw_channel_data
            )
            logging.info(
                f"LLM: Selected '{selected_key}' for channel '{channel_name}' in {time.time() - start_time}s"
            )
            channel_selections[channel_name] = selected_key
        return channel_selections

    def postprocess(self, model_outputs, **postprocess_parameters):
        mangled_name = "_TimeSeriesAnomalyDetectionPipeline__context_memory"
        result = getattr(self, mangled_name)["data"].copy()
        expand_score = postprocess_parameters.get("expand_score", False)
        smoothing_window_size = postprocess_parameters.get("smoothing_length", 1)
        target_columns = postprocess_parameters.get("target_columns", [])
        use_llm_selection = postprocess_parameters.get("use_llm_selection", True)

        report_mode = postprocess_parameters.get("report_mode", False)
        predictive_score_smoothing = postprocess_parameters.get(
            "predictive_score_smoothing", False
        )
        if not isinstance(smoothing_window_size, int):
            try:
                smoothing_window_size = int(smoothing_window_size)
            except ValueError:
                smoothing_window_size = 1

        # adjust scoring and smooth
        extra_kwargs = {}
        if "reference" in getattr(self, mangled_name):
            data = getattr(self, mangled_name)["reference"]
            if len(target_columns) > 0:
                data = data[target_columns]
            extra_kwargs["reference"] = data.values

        model_outputs_ = {}
        for k in model_outputs:
            score = model_outputs[k]
            score = self._model_processor.adjust_boundary(k, score, **extra_kwargs)
            if not predictive_score_smoothing and (
                k == AnomalyScoreMethods.PREDICTIVE.value
            ):  # Skip Smoothing For 1 Lookahead forecast
                model_outputs_[k] = score
            elif k == AnomalyScoreMethods.PROBABILISTIC.value:
                model_outputs_[k] = score_smoothing(
                    score, smoothing_window_size=1
                )  # no smoothing of p-value scores across time
            else:
                model_outputs_[k] = score_smoothing(
                    score, smoothing_window_size=smoothing_window_size
                )
            model_outputs_[k] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                model_outputs_[k]
            )

        # aggregate scores and expand
        score = np.stack([score_ for _, score_ in model_outputs_.items()], axis=0)
        mode_selected = None
        if report_mode and (self.select_function is not None):
            keys = [key for key, _ in model_outputs_.items()]
            sel_index = self.select_function(score, axis=0)
            mode_selected = np.asarray([keys[z] for z in sel_index.ravel()]).reshape(
                sel_index.shape
            )
        ensemble_score = self.aggr_function(score, axis=0)

        if use_llm_selection:
            all_scores = model_outputs_.copy()
            all_scores["ensemble"] = ensemble_score
            raw_data_to_plot = result[target_columns]

            # This will now always return a dictionary, e.g., {'x1': 'time', 'x2': 'ensemble'}
            selections = self._select_head_with_llm(
                all_scores, target_columns, raw_data=raw_data_to_plot
            )

            # Construct the final score array from per-channel selections
            score = np.zeros_like(ensemble_score)
            for i, col_name in enumerate(target_columns):
                selected_key = selections.get(
                    col_name, "ensemble"
                )  # Default to ensemble if a channel is missed
                score[:, i] = all_scores[selected_key][:, i]
        else:
            score = ensemble_score

        expand_score = (len(target_columns) > 1) and expand_score
        model_outputs = {}
        if expand_score:
            if len(target_columns) != score.shape[-1]:
                raise RuntimeError(
                    f"Error: inconsistent state, with target columns {target_columns}"
                )
            for i, col_name in enumerate(target_columns):
                model_outputs[f"{col_name}_anomaly_score"] = score[..., i]
            model_outputs["anomaly_score"] = score.mean(axis=-1).ravel()

            if mode_selected is not None:
                for i, col_name in enumerate(target_columns):
                    model_outputs[f"{col_name}_selected_mode"] = mode_selected[..., i]

        else:
            model_outputs.update(anomaly_score=score.ravel())
            if mode_selected is not None:
                model_outputs.update(selected_mode=mode_selected.ravel())

        # populate dataframe
        for k in model_outputs:
            result[k] = model_outputs[k]
        setattr(self, mangled_name, {})
        return result
