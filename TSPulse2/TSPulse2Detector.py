from notebooks.hfdemo.tspulse.anomaly_detection.utility.model import \
    TSAD_Pipeline

from TSPulse2.TSPulse2Pipeline import TSPulse2Pipeline


class TSPulse2Detector(TSAD_Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prediction_mode_array = [
            s_.strip()
            for s_ in str(kwargs.get("prediction_mode", "forecast+time+fft")).split("+")
        ]
        aggregation_length = kwargs.get("aggr_win_size", 96)
        smoothing_length = kwargs.get("smoothing_length", 16)
        self._scorer = TSPulse2Pipeline(
            self._model,
            timestamp_column="timestamp",
            target_columns=self._headers,
            prediction_mode=prediction_mode_array,
            aggregation_length=aggregation_length,
            smoothing_length=smoothing_length,
            least_significant_scale=0.0,
            least_significant_score=1.0,
            head_min_max_scale=kwargs.get("head_min_max_scale", True),
            head_selector=kwargs.get("head_selector", True),
        )
