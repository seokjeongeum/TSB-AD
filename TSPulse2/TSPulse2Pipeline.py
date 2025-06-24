import base64
import io
import os
import sys
import time

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.genai import types

# Use relative paths for local imports
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from embedding_pipeline import EmbeddingExtractorPipeline
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline, score_smoothing)


class TSPulse2Pipeline(TimeSeriesAnomalyDetectionPipeline):
    """
    Time Series Anomaly Detection Pipeline with enhanced head selection capabilities.
    """

    def __init__(self, *args, **kwargs):
        # 1. Pop NEW, specific parameters for this pipeline subclass.
        self.head_min_max_scale = kwargs.pop("head_min_max_scale", True)
        self.llm_selection = kwargs.pop("llm_selection", False)
        self.head_selector = kwargs.pop(
            "head_selector", None
        )  # NEW: dedicated argument

        # 2. Call the parent's __init__ with all remaining args and kwargs.
        super().__init__(*args, **kwargs)

        # 3. Load the selector model if the mode is set to "model"
        if self.head_selector == "model":
            self._load_selector_model()

    def _load_selector_model(self, model_dir="../trained_selectors/"):
        """Loads the trained head selector model and label encoder."""
        # Construct path relative to this script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir, model_dir, "embedding_selector_model.joblib"
        )
        encoder_path = os.path.join(
            base_dir, model_dir, "embedding_selector_encoder.joblib"
        )

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Selector model not found at {model_path}. Please run train_selector_with_embeddings.py first."
            )

        self.selector_model = joblib.load(model_path)
        self.selector_encoder = joblib.load(encoder_path)
        self.selector_inverse_encoder = {
            i: label for label, i in self.selector_encoder.items()
        }
        print("Successfully loaded embedding-based head selector model.")

    def _create_llm_plot(
        self, series_data, scores_dict, title="", safe_title="", timestamp=0
    ):
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

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=24576, include_thoughts=True
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
        selected_key = "scaled_ensemble"
        if response and response.candidates:
            first_candidate = response.candidates[0]
            if first_candidate.content and first_candidate.content.parts:
                thought_summary = ""
                for part in first_candidate.content.parts:
                    if hasattr(part, "thought") and part.thought and part.text:
                        thought_summary += part.text + "\n"
                    elif hasattr(part, "text") and part.text:
                        selected_key = part.text.strip('"')
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

        target_columns = postprocess_parameters.get("target_columns")
        if target_columns is None:
            raise ValueError("target_columns is required for postprocessing.")

        # --- 1. Calculate and Scale Base Head Scores ---
        smoothing_window_size = postprocess_parameters.get("smoothing_length", 1)
        extra_kwargs = {}
        if "reference" in getattr(self, mangled_name):
            extra_kwargs["reference"] = getattr(self, mangled_name)["reference"][
                target_columns
            ].values

        model_outputs_ = {}
        for k, v in model_outputs.items():
            score = self._model_processor.adjust_boundary(v, **extra_kwargs)
            model_outputs_[k] = score_smoothing(
                score, smoothing_window_size=smoothing_window_size
            )

        original_scores = model_outputs_.copy()

        if self.head_min_max_scale:
            epsilon = 1e-8
            scaled_scores = {}
            for k, v in model_outputs_.items():
                min_v, max_v = np.min(v, axis=0, keepdims=True), np.max(
                    v, axis=0, keepdims=True
                )
                scaled_scores[k] = (v - min_v) / (max_v - min_v + epsilon)
            model_outputs_ = scaled_scores

        ensemble_keys = ["time", "fft", "forecast"]
        if all(k in original_scores for k in ensemble_keys):
            unscaled_stack = np.stack(
                [original_scores[k] for k in ensemble_keys], axis=0
            )
            model_outputs_["ensemble"] = self.aggr_function(unscaled_stack, axis=0)

        if self.head_min_max_scale and all(k in model_outputs_ for k in ensemble_keys):
            scaled_stack = np.stack([model_outputs_[k] for k in ensemble_keys], axis=0)
            model_outputs_["scaled_ensemble"] = self.aggr_function(scaled_stack, axis=0)

        # --- 2. Head Selection Logic ---
        score = None
        mode_selected = None

        if self.head_selector == "model":
            print("Using trained model for head selection...")
            data_df = getattr(self, mangled_name)["data"]
            # FIX: Get batch_size from parameters passed to the pipeline call
            batch_size = postprocess_parameters.get(
                "batch_size", 512
            )  # Default to 512 if not provided
            embedding_pipeline = EmbeddingExtractorPipeline(
                model=self.model, batch_size=batch_size
            )
            # The pipeline now returns embeddings directly from its __call__
            embeddings = embedding_pipeline(data_df, **postprocess_parameters)

            if embeddings is None:
                raise ValueError("Embedding extraction failed and returned None.")

            # FIX: Convert generator to list to allow indexing
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(list(embeddings))

            num_channels = len(target_columns)
            score = np.zeros((len(result), num_channels))
            selected_modes_list = []

            for i in range(num_channels):
                channel_embedding = embeddings[i].reshape(1, -1)
                pred_idx = self.selector_model.predict(channel_embedding)[0]
                selected_head_name = self.selector_inverse_encoder[pred_idx]

                if selected_head_name == "TSPulse2":
                    selected_key = "scaled_ensemble"
                else:
                    selected_key = selected_head_name.split("_")[-1]

                print(f"Channel '{target_columns[i]}': Selector chose '{selected_key}'")
                if selected_key not in model_outputs_:
                    print(
                        f"Warning: Selected key '{selected_key}' not found. Defaulting to scaled_ensemble."
                    )
                    selected_key = "scaled_ensemble"
                score[:, i] = model_outputs_[selected_key][:, i]
                selected_modes_list.append(selected_key)

            if postprocess_parameters.get("report_mode", False):
                mode_selected = np.array(selected_modes_list).reshape(1, -1)

        elif self.head_selector == "llm" or self.llm_selection is True:
            print("Using LLM for head selection...")
            num_timesteps, num_targets = result.shape[0], len(target_columns)
            score = np.zeros((num_timesteps, num_targets))
            llm_modes = np.full((num_timesteps, num_targets), "", dtype=object)
            for i, col_name in enumerate(target_columns):
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
                    f"Select best score for {col_name}",
                    safe_title,
                    timestamp,
                )
                selected_key = self._get_llm_selection(b64_image, safe_title, timestamp)
                print(f"Column '{col_name}': LLM selected '{selected_key}'")
                score[:, i] = scores_for_col[selected_key]
                llm_modes[:, i] = selected_key
            if postprocess_parameters.get("report_mode", False):
                mode_selected = llm_modes

        else:  # Default behavior
            print("Using default 'scaled_ensemble' scores.")
            score = model_outputs_.get("scaled_ensemble")
            if score is None:
                score = model_outputs_.get("ensemble")
            if score is None:
                raise ValueError(
                    "Default 'scaled_ensemble' or 'ensemble' score not found."
                )

        # --- 3. Finalize Output DataFrame ---
        expand_score = (len(target_columns) > 1) and postprocess_parameters.get(
            "expand_score", False
        )
        model_outputs = {}
        if expand_score:
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
        for k, v in model_outputs.items():
            result[k] = v

        setattr(self, mangled_name, {})
        return result
