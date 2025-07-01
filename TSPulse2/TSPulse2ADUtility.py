import logging
import os
import sys
from collections import OrderedDict
from typing import List

import torch
from torch import nn
from transformers.utils.generic import ModelOutput
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility
from tsfm_public.models.tspulse.utils.helpers import patchwise_stitched_reconstruction


class TSPulse2ADUtility(TSPulseADUtility):
    def __init__(
        self,
        model: TSPulseForReconstruction,
        mode: List[str],
        aggregation_length: int,
        score_exponent: float = 1.0,
        least_significant_scale: float = 1e-2,
        least_significant_score: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            model=model,
            mode=mode,
            aggregation_length=aggregation_length,
            score_exponent=score_exponent,
            least_significant_scale=least_significant_scale,
            least_significant_score=least_significant_score,
            **kwargs,
        )

    def compute_score(
        self,
        payload: dict,
        expand_score: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """Produces required model output for anomaly scoring

        Args:
            payload (dict): data batch.
            expand_score (bool): compute score for each stream for multivariate data. Defaults to False.

        Returns:
            ModelOutput: model output
        """
        mode = kwargs.get("mode", self._mode)
        use_forecast = AnomalyScoreMethods.PREDICTIVE.value in mode
        use_fft = AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value in mode
        use_ts = AnomalyScoreMethods.TIME_RECONSTRUCTION.value in mode
        aggr_win_size = self._aggr_win_size
        anomaly_criterion = nn.MSELoss(reduction="none")

        reconstruct_start = self._model.config.context_length - aggr_win_size
        reconstruct_end = self._model.config.context_length

        batch_x = payload["past_values"]

        # Get TSPulse zeroshot output with stitched masked reconstruction
        keys_to_stitch = ["reconstruction_outputs", "reconstructed_ts_from_fft"]

        model_forward_output = {}
        if use_forecast:
            # model_forward_output = self._model(**payload)
            model_forward_output = self._model(batch_x)

        stitched_dict = {}
        if use_ts or use_fft:
            stitched_dict = patchwise_stitched_reconstruction(
                model=self._model,
                past_values=batch_x,
                patch_size=self._model.config.patch_length,
                keys_to_stitch=keys_to_stitch,
                keys_to_aggregate=[],
                reconstruct_start=reconstruct_start,
                reconstruct_end=reconstruct_end,
                debug=False,
            )
            if isinstance(stitched_dict, tuple):
                stitched_dict = stitched_dict[0]

        # Get desired output from TSPulse outputs
        # output shape: [batch_size, window_size, n_channels]
        scores = OrderedDict()

        reduction_axis = [1] if expand_score else [1, 2]
        if use_ts:
            # time reconstruction
            output = stitched_dict["reconstruction_outputs"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            scores[AnomalyScoreMethods.TIME_RECONSTRUCTION.value] = torch.mean(
                pointwise_score, dim=reduction_axis
            )

        if use_fft:
            # time reconstruction from fft
            output = stitched_dict["reconstructed_ts_from_fft"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            scores[AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value] = torch.mean(
                pointwise_score, dim=reduction_axis
            )

        if use_forecast:
            # forecast output
            batch_future_values = payload["future_values"]
            output = model_forward_output["forecast_output"]
            pointwise_score = anomaly_criterion(
                batch_future_values[:, 0, :], output[:, 0, :]
            ).unsqueeze(1)
            scores[AnomalyScoreMethods.PREDICTIVE.value] = torch.mean(
                pointwise_score, dim=reduction_axis
            )

        print(
            torch.stack(
                [
                    stitched_dict["reconstruction_outputs"][
                        :, reconstruct_start:reconstruct_end, :
                    ].mean(reduction_axis),
                    stitched_dict["reconstructed_ts_from_fft"][
                        :, reconstruct_start:reconstruct_end, :
                    ].mean(reduction_axis),
                    model_forward_output["forecast_output"][:, 0, :]
                    .unsqueeze(1)
                    .mean(reduction_axis),
                ],
                dim=1,
            ).shape
        )

        return ModelOutput(scores)
