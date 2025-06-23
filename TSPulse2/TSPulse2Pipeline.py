import base64
import io
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.genai import types

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline, score_smoothing)


class TSPulse2Pipeline(TimeSeriesAnomalyDetectionPipeline):
    """Time Series Anomaly Detection using HF time series models. This pipeline consumes a `pandas.DataFrame`
    containing the time series data and produces a new `pandas.DataFrame` with anomaly scores.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # 1. Pop the NEW parameter specific to this subclass.
        #    This removes it from kwargs so the parent doesn't get an unknown argument.
        head_min_max_scale = kwargs.pop("head_min_max_scale", True)
        self.llm_selection = kwargs.pop("llm_selection", False)

        # 2. Call the parent's __init__ with all remaining args and kwargs.
        #    The parent class will handle its own arguments (model, prediction_mode, etc.).
        #    This guarantees the parent's __init__ runs completely and correctly,
        #    creating self.__context_memory on the current instance.
        super().__init__(*args, **kwargs)

        # 3. Now that the parent is initialized, set the child's specific attribute.
        self.head_min_max_scale = head_min_max_scale

    def _create_llm_plot(
        self, series_data, scores_dict, title="", safe_title="", timestamp=0
    ):
        """
        Generates a multi-panel plot with one subplot per anomaly score type.
        This is much clearer for the LLM to analyze than a single overlaid plot.
        """
        valid_scores_dict = {k: v for k, v in scores_dict.items() if v is not None}
        num_scores = len(valid_scores_dict)

        if num_scores == 0:
            return ""

        fig, axes = plt.subplots(
            nrows=num_scores,
            ncols=1,
            figsize=(20, 4 * num_scores),
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16)

        score_colors = {
            "time": "dodgerblue",
            "fft": "forestgreen",
            "forecast": "darkviolet",
            "ensemble": "darkgoldenrod",
            "scaled_ensemble": "orangered",
        }

        for i, (name, scores) in enumerate(valid_scores_dict.items()):
            ax = axes[i]
            ax.plot(series_data, color="gray", linewidth=1.0, label="Time Series Data")
            ax.set_ylabel("Data Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_title(f"Score Type: {name}")

            ax2 = ax.twinx()
            color = score_colors.get(name, "black")
            ax2.plot(scores, color=color, linewidth=1.8, label=f"Score ({name})")
            ax2.set_ylabel("Anomaly Score", color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            if name != "ensemble":
                ax2.set_ylim(-0.05, 1.05)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout(rect=(0, 0.03, 1, 0.96))

        plot_dir = "plots/llm_selection"
        os.makedirs(plot_dir, exist_ok=True)
        # Use passed-in safe_title and timestamp for consistent naming
        filename = f"{timestamp}_{safe_title}.jpeg"
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _get_llm_selection(
        self, b64_image: str, safe_title: str, timestamp: int
    ) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"

        prompt = """Analyze the provided plot, which shows a time series and several anomaly score types. 
Identify which score type ('time', 'fft', 'forecast', 'ensemble', or 'scaled_ensemble') is most effective at detecting the anomalies. 
The best score should be high during anomalous periods and low otherwise.
Respond with only the single string identifier for the best score type."""

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/jpeg", data=base64.b64decode(b64_image)
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        # NEW: Enable thought summaries to capture the model's reasoning process.
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=24576, include_thoughts=True  # Enable thought summaries
            ),
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.STRING,
                enum=["time", "fft", "forecast", "ensemble", "scaled_ensemble"],
            ),
        )
        start_time = time.time()
        response = client.models.generate_content(
            model=model,
            contents=contents,  # type: ignore
            config=generate_content_config,
        )
        end_time = time.time()
        print(
            f"Gemini API call for '{safe_title}' took {end_time - start_time:.2f} seconds."
        )

        selected_key = "scaled_ensemble"  # Default fallback
        if response and response.candidates:
            first_candidate = response.candidates[0]
            if first_candidate.content and first_candidate.content.parts:
                # NEW: Extract and save the thought summary
                thought_summary = ""
                for part in first_candidate.content.parts:
                    if hasattr(part, "thought") and part.thought and part.text:
                        thought_summary += part.text + "\n"
                    elif hasattr(part, "text") and part.text:
                        llm_result = part.text
                        selected_key = llm_result.strip('"')

                if thought_summary:
                    thoughts_dir = "plots/llm_selection/thoughts"
                    os.makedirs(thoughts_dir, exist_ok=True)
                    thought_filename = f"{timestamp}_{safe_title}_thoughts.txt"
                    thought_save_path = os.path.join(thoughts_dir, thought_filename)
                    with open(thought_save_path, "w") as f:
                        f.write(thought_summary)
                    print(f"Saved LLM thoughts to {thought_save_path}")

        return selected_key

    def postprocess(self, model_outputs, **postprocess_parameters):
        mangled_name = "_TimeSeriesAnomalyDetectionPipeline__context_memory"
        result = getattr(self, mangled_name)["data"].copy()

        smoothing_window_size = postprocess_parameters.get("smoothing_length", 1)
        target_columns = postprocess_parameters.get("target_columns")

        if target_columns is None:
            raise ValueError("target_columns is required")

        report_mode = postprocess_parameters.get("report_mode", False)
        predictive_score_smoothing = postprocess_parameters.get(
            "predictive_score_smoothing", False
        )
        if not isinstance(smoothing_window_size, int):
            try:
                smoothing_window_size = int(smoothing_window_size)
            except ValueError:
                smoothing_window_size = 1

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

        original_scores = model_outputs_.copy()

        if self.head_min_max_scale:
            epsilon = 1e-8
            scaled_scores = {}
            for k, v in model_outputs_.items():
                min_v = np.min(v, axis=0, keepdims=True)
                max_v = np.max(v, axis=0, keepdims=True)
                scaled_scores[k] = (v - min_v) / (max_v - min_v + epsilon)

            # BUG FIX: Calculate scaled_ensemble from the newly scaled component scores
            ensemble_components = []
            keys_for_ensemble = ["time", "fft", "forecast"]
            for key in keys_for_ensemble:
                if key in scaled_scores:
                    ensemble_components.append(scaled_scores[key])

            if ensemble_components:
                stacked_ensemble = np.stack(ensemble_components, axis=0)
                model_outputs_["scaled_ensemble"] = self.aggr_function(
                    stacked_ensemble, axis=0
                )

        unscaled_scores_stack = np.stack(
            [s for k, s in original_scores.items() if k in ["time", "fft", "forecast"]],
            axis=0,
        )
        if unscaled_scores_stack.size > 0:
            model_outputs_["ensemble"] = self.aggr_function(
                unscaled_scores_stack, axis=0
            )

        mode_selected = None
        if self.llm_selection:
            num_timesteps = result.shape[0]
            num_targets = len(target_columns)
            score = np.zeros((num_timesteps, num_targets))
            llm_modes = np.full((num_timesteps, num_targets), "", dtype=object)

            for i, col_name in enumerate(target_columns):
                # NEW: Create consistent identifiers for saving plots and thoughts
                safe_title = "".join(
                    c for c in col_name if c.isalnum() or c in (" ", "_", "-")
                ).rstrip()
                timestamp = int(time.time())

                series_data = result[col_name].values
                scores_for_col = {
                    k: v[:, i]
                    for k, v in model_outputs_.items()
                    if k in ["time", "fft", "forecast", "scaled_ensemble", "ensemble"]
                }
                b64_image = self._create_llm_plot(
                    series_data,
                    scores_for_col,
                    title=f"Select best score for {col_name}",
                    safe_title=safe_title,
                    timestamp=timestamp,
                )
                selected_key = self._get_llm_selection(
                    b64_image, safe_title=safe_title, timestamp=timestamp
                )
                print(f"Column '{col_name}': LLM selected '{selected_key}'")
                score[:, i] = scores_for_col[selected_key]
                llm_modes[:, i] = selected_key

            if report_mode:
                mode_selected = llm_modes
        else:
            score = model_outputs_["scaled_ensemble"]
            if report_mode and (self.select_function is not None):
                keys = [key for key, _ in model_outputs_.items()]
                sel_index = self.select_function(score, axis=0)
                mode_selected = np.asarray(
                    [keys[z] for z in sel_index.ravel()]
                ).reshape(sel_index.shape)

        # Final score formatting and population
        expand_score = (len(target_columns) > 1) and postprocess_parameters.get(
            "expand_score", False
        )
        model_outputs = {}
        if expand_score:
            if len(target_columns) != score.shape[-1]:
                raise RuntimeError(
                    f"Error: inconsistent state, with target columns {target_columns}"
                )
            for i, col_name in enumerate(target_columns):
                model_outputs[f"{col_name}_anomaly_score"] = score[..., i]

            if mode_selected is not None:
                for i, col_name in enumerate(target_columns):
                    model_outputs[f"{col_name}_selected_mode"] = mode_selected[..., i]
            model_outputs.update(anomaly_score=score.mean(axis=1))
        else:
            model_outputs.update(anomaly_score=score.ravel())
            if mode_selected is not None:
                model_outputs.update(selected_mode=mode_selected.ravel())

        # populate dataframe
        for k in model_outputs:
            result[k] = model_outputs[k]

        setattr(self, mangled_name, {})
        return result
