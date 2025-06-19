import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)

from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
    score_smoothing,
)


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

        # 2. Call the parent's __init__ with all remaining args and kwargs.
        #    The parent class will handle its own arguments (model, prediction_mode, etc.).
        #    This guarantees the parent's __init__ runs completely and correctly,
        #    creating self.__context_memory on the current instance.
        super().__init__(*args, **kwargs)

        # 3. Now that the parent is initialized, set the child's specific attribute.
        self.head_min_max_scale = head_min_max_scale

    def postprocess(self, model_outputs, **postprocess_parameters):
        """Overrides the postprocess of the base class. Applies post-processing logic on the model outputs.

        Args:
            model_outputs (dict): dictionary containing model outputs.

        Raises:
            RuntimeError: Returned if there is an inconsistency in the target columns and the resulting scores.

        Returns:
            pd.DataFrame: pandas dataframe with anomaly score attached
        """
        # --- FIX: Access the name-mangled attribute directly ---
        # The parent class creates an attribute named __context_memory. Due to name mangling,
        # it exists on our object as '_TimeSeriesAnomalyDetectionPipeline__context_memory'.
        # We must use this full, "ugly" name to access it from the child class.
        mangled_name = "_TimeSeriesAnomalyDetectionPipeline__context_memory"

        result = getattr(self, mangled_name)["data"].copy()

        expand_score = postprocess_parameters.get("expand_score", False)
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

        # --- FIX: Also access it here ---
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

        # aggregate scores and expand
        score = np.stack([score_ for _, score_ in model_outputs_.items()], axis=0)

        if self.head_min_max_scale:
            min_val = np.min(score, axis=0)
            max_val = np.max(score, axis=0)
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-8
            score = (score - min_val) / (max_val - min_val + epsilon)

        mode_selected = None
        if report_mode and (self.select_function is not None):
            keys = [key for key, _ in model_outputs_.items()]
            sel_index = self.select_function(score, axis=0)
            mode_selected = np.asarray([keys[z] for z in sel_index.ravel()]).reshape(
                sel_index.shape
            )

        score = self.aggr_function(score, axis=0)

        expand_score = (len(target_columns) > 1) and expand_score
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

        # --- FIX: Clear the context memory using the mangled name ---
        setattr(self, mangled_name, {})

        return result
