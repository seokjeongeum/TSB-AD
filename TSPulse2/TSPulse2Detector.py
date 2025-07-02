import os
import sys

from TSPulse2.TSPulse2Pipeline import TSPulse2Pipeline

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "granite-tsfm",
    ),
)
from notebooks.hfdemo.tspulse.anomaly_detection.utility.model import \
    TSAD_Pipeline


class TSPulse2Detector(TSAD_Pipeline):
    def __init__(
        self,
        batch_size: int,
        aggr_win_size: int,
        num_input_channels: int,
        smoothing_window: int,
        prediction_mode: str,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            aggr_win_size=aggr_win_size,
            num_input_channels=num_input_channels,
            smoothing_window=smoothing_window,
            prediction_mode=prediction_mode,
            **kwargs,
        )
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        self._scorer = TSPulse2Pipeline(
            self._model,
            timestamp_column="timestamp",
            target_columns=self._headers,
            prediction_mode=prediction_mode_array,
            aggregation_length=aggr_win_size,
            smoothing_length=smoothing_window,
            least_significant_scale=0.0,
            least_significant_score=1.0,
            **kwargs,
        )

    def fit(self, X, y=None):
        pass
