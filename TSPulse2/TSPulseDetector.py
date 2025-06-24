import os
import sys

import pandas as pd
from accelerate import Accelerator
from tsfm_public.models.tspulse.modeling_tspulse import \
    TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import \
    TimeSeriesAnomalyDetectionPipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TSB_AD.models.base import BaseDetector


def _prepare_df_for_tspulse(data_np):
    """
    Adapter function to convert a numpy array into the DataFrame format
    required by the TSPulse pipeline. This is a necessary shim because the
    tsfm-public library's high-level API is DataFrame-based.
    """
    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)

    num_channels = data_np.shape[1]
    target_columns = [f"value_{i}" for i in range(num_channels)]

    data_df = pd.DataFrame(data_np, columns=target_columns)
    data_df.insert(0, "timestamp", pd.to_datetime(range(len(data_df)), unit="s"))
    return data_df, target_columns


class TSPulseDetector(BaseDetector):
    def __init__(self, **kwargs):
        # Store all kwargs to pass them down to the pipeline
        self.kwargs = kwargs
        self.model_class = self.kwargs.get("model_class", TSPulseForReconstruction)
        self.pipeline_class = self.kwargs.get(
            "pipeline_class", TimeSeriesAnomalyDetectionPipeline
        )
        self.aggregation_length = self.kwargs.get("aggregation_length", 96)
        self.aggr_function = self.kwargs.get("aggr_function", "max")
        self.smoothing_length = self.kwargs.get("smoothing_length", 16)
        self.least_significant_scale = self.kwargs.get("least_significant_scale", 0.01)
        self.least_significant_score = self.kwargs.get("least_significant_score", 0.1)
        self.head_min_max_scale = self.kwargs.get("head_min_max_scale", False)
        self.llm_selection = self.kwargs.get("llm_selection", False)
        self.prediction_mode = self.kwargs.get(
            "prediction_mode",
            [
                AnomalyScoreMethods.PREDICTIVE.value,
                AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ],
        )

    def fit(self, X):
        _, target_columns = _prepare_df_for_tspulse(X)
        num_channels = len(target_columns)

        model = self.model_class.from_pretrained(
            "ibm-granite/granite-timeseries-tspulse-r1",
            num_input_channels=num_channels,
            revision="main",
            mask_type="user",
            ignore_mismatched_sizes=True,
        )

        accelerator = Accelerator()
        model = accelerator.prepare_model(model, device_placement=True)

        # Pass all kwargs down to the pipeline.
        # The pipeline will pick what it needs.
        pipeline_kwargs = self.kwargs.copy()
        pipeline_kwargs.update(
            {
                "model": model,
                "timestamp_column": "timestamp",
                "target_columns": target_columns,
                "prediction_mode": self.prediction_mode,
                "aggregation_length": self.aggregation_length,
                "aggr_function": self.aggr_function,
                "smoothing_length": self.smoothing_length,
                "least_significant_scale": self.least_significant_scale,
                "least_significant_score": self.least_significant_score,
                "head_min_max_scale": self.head_min_max_scale,
                "llm_selection": self.llm_selection,
            }
        )

        self.pipeline = self.pipeline_class(**pipeline_kwargs)

    def decision_function(self, X):
        df, _ = _prepare_df_for_tspulse(X)
        result = self.pipeline(
            df,
            batch_size=2048,
            report_mode=True,
            predictive_score_smoothing=True,
            expand_score=True,
        )
        return result["anomaly_score"].values
