import os
import sys

import joblib
import numpy as np
import pandas as pd

# Use relative paths for local imports
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.models.tspulse import TSPulseForClassification
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline, score_smoothing)
from tsfm_public.toolkit.time_series_classification_pipeline import \
    TimeSeriesClassificationPipeline


class TSPulse2Pipeline(TimeSeriesAnomalyDetectionPipeline):
    """
    Time Series Anomaly Detection Pipeline with enhanced head selection capabilities.
    """

    def __init__(self, *args, **kwargs):
        # 1. Pop NEW, specific parameters for this pipeline subclass.
        self.head_min_max_scale = kwargs.pop("head_min_max_scale", True)
        self.head_selector = kwargs.pop(
            "head_selector", False
        )  # Default to False, expects a boolean

        # 2. Call the parent's __init__ with all remaining args and kwargs.
        super().__init__(*args, **kwargs)

        # 3. Initialize selector attributes
        self.selector_model = None
        self.selector_preprocessor = None

        # 4. Load the selector model if the mode is set to True
        if self.head_selector:
            self._load_selector_model()

    def _load_selector_model(self, model_dir="classification_output/final_model/"):
        """Loads the trained TSPulseForClassification model and its preprocessor."""
        # Construct path relative to this script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_dir)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Selector model not found at {model_path}. Please run train_classifier.py first."
            )

        self.selector_model = TSPulseForClassification.from_pretrained(model_path)

        preprocessor_path = os.path.join(model_path, "preprocessor.joblib")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Selector preprocessor not found at {preprocessor_path}. "
                "Please ensure 'preprocessor.joblib' is in the model directory."
            )
        self.selector_preprocessor = joblib.load(preprocessor_path)

        print(
            "Successfully loaded TSPulseForClassification head selector model and preprocessor."
        )

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

        # FIX 1: Concatenate list of score arrays from accumulator before processing
        concatenated_outputs = {}
        for k, v_list in model_outputs.items():
            if v_list:
                concatenated_outputs[k] = np.concatenate(v_list, axis=0)

        model_outputs_ = {}
        for k, v in concatenated_outputs.items():
            score = self._model_processor.adjust_boundary(k, v, **extra_kwargs)
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

        if self.head_selector:
            print("Using trained classification model for head selection...")
            data_df = getattr(self, mangled_name)["data"]

            num_channels = len(target_columns)
            score = np.zeros((len(result), num_channels))
            selected_modes_list = []

            device = self.selector_model.device
            classification_pipeline = TimeSeriesClassificationPipeline(
                model=self.selector_model,
                feature_extractor=self.selector_preprocessor,
                device=device,
            )

            for i, col_name in enumerate(target_columns):
                channel_df = pd.DataFrame(
                    {
                        "past_values": [pd.Series(data_df[col_name].values)],
                        "labels": ["ensemble"],
                    }
                )

                prediction_result = classification_pipeline(channel_df)
                selected_head_name = prediction_result["labels_prediction"][0]

                if selected_head_name == "future":
                    selected_key = "forecast"
                else:
                    selected_key = selected_head_name

                print(
                    f"Channel '{target_columns[i]}': Selector chose '{selected_key}' (from '{selected_head_name}')"
                )

                if selected_key not in model_outputs_:
                    print(
                        f"Warning: Selected key '{selected_key}' not found in available scores. Defaulting to scaled_ensemble."
                    )
                    selected_key = "scaled_ensemble"

                score[:, i] = model_outputs_[selected_key][:, i]
                selected_modes_list.append(selected_key)

            if postprocess_parameters.get("report_mode", False):
                mode_selected = np.array(selected_modes_list).reshape(1, -1)

        else:
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
        final_model_outputs = {}
        if expand_score:
            for i, col_name in enumerate(target_columns):
                final_model_outputs[f"{col_name}_anomaly_score"] = score[..., i]
            if mode_selected is not None:
                for i, col_name in enumerate(target_columns):
                    # FIX 2: Assign the scalar string, which pandas will broadcast.
                    final_model_outputs[f"{col_name}_selected_mode"] = mode_selected[
                        0, i
                    ]
            final_model_outputs.update(anomaly_score=score.mean(axis=1))
        else:
            final_model_outputs.update(anomaly_score=score.ravel())
            if mode_selected is not None:
                # FIX 2: Assign the scalar string, which pandas will broadcast.
                final_model_outputs.update(selected_mode=mode_selected.ravel()[0])

        for k, v in final_model_outputs.items():
            result[k] = v

        setattr(self, mangled_name, {})
        return result
