import pandas as pd
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

from TSB_AD.models.base import BaseDetector
from TSPulse2.TSPulse2Pipeline import TSPulse2Pipeline
from accelerate import Accelerator


def _prepare_df_for_tspulse(data_np):
    """
    Adapter function to convert a numpy array into the DataFrame format
    required by the TSPulse pipeline. This is a necessary shim because the
    tsfm-public library's high-level API is DataFrame-based.
    """
    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)

    num_channels = data_np.shape[1]
    # Create generic column names as required by the pipeline
    target_columns = [f"x_{i}" for i in range(num_channels)]
    df = pd.DataFrame(data_np, columns=target_columns)
    # Add a dummy timestamp column, also required by the pipeline
    df["timestamp"] = pd.to_datetime(
        pd.date_range(start="2000-01-01", periods=len(df), freq="s")
    )
    return df, target_columns


class TSPulseDetector(BaseDetector):
    def __init__(self, **kwargs):
        self.model_class = kwargs.get("model_class", TSPulseForReconstruction)
        self.pipeline_class = kwargs.get("pipeline_class", TSPulse2Pipeline)
        self.aggregation_length = kwargs.get("aggregation_length", 96)
        self.aggr_function = kwargs.get("aggr_function", "max")
        self.smoothing_length = kwargs.get("smoothing_length", 16)
        self.least_significant_scale = kwargs.get("least_significant_scale", 0.01)
        self.least_significant_score = kwargs.get("least_significant_score", 0.1)
        self.head_min_max_scale = kwargs.get("head_min_max_scale", True)
        self.prediction_mode = kwargs.get(
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
        )
        accelerator = Accelerator()
        model = accelerator.prepare_model(model, device_placement=True)
        self.pipeline = self.pipeline_class(
            model=model,
            timestamp_column="timestamp",
            target_columns=target_columns,
            prediction_mode=self.prediction_mode,
            aggregation_length=self.aggregation_length,
            aggr_function=self.aggr_function,
            smoothing_length=self.smoothing_length,
            least_significant_scale=self.least_significant_scale,
            least_significant_score=self.least_significant_score,
            head_min_max_scale=self.head_min_max_scale,
        )

    def decision_function(self, X):
        df, _ = _prepare_df_for_tspulse(X)
        result = self.pipeline(
            df,
            batch_size=512,
            report_mode=True,
            predictive_score_smoothing=True,
            expand_score=True,
        )
        return result["anomaly_score"].values
