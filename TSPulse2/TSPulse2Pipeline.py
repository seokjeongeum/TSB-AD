import os
import sys
from typing import List, Optional

# Use relative paths for local imports
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.models.tspulse.modeling_tspulse import \
    TSPulseForReconstruction
from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    AggregationFunction, TimeSeriesAnomalyDetectionPipeline)

from TSPulse2.TSPulse2ADUtility import TSPulse2ADUtility


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
        if kwargs.get("fuse_reconstruction", True):
            self._model_processor = TSPulse2ADUtility(
                model,
                mode=prediction_mode,
                aggregation_length=aggregation_length,
                **kwargs,
            )
