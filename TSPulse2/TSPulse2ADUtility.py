import os
import sys
from typing import List

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.models.tspulse.modeling_tspulse import \
    TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility


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
