import datetime
import io
import logging
import os
import sys
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.preprocessing import MinMaxScaler

from TSPulse2.TSPulse2ADUtility import TSPulse2ADUtility

sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "granite-tsfm",
    ),
)
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    AggregationFunction,
    TimeSeriesAnomalyDetectionPipeline,
    score_smoothing,
)


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
        self._model_processor = TSPulse2ADUtility(
            model,
            mode=prediction_mode,
            aggregation_length=aggregation_length,
            **kwargs,
        )
        self.width = 160
        self.fontsize = 48

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = (
            super()._sanitize_parameters(**kwargs)
        )
        if "use_llm_selection" in kwargs:
            postprocess_kwargs["use_llm_selection"] = kwargs["use_llm_selection"]
        if "llm_few_shot_config" in kwargs:
            postprocess_kwargs["llm_few_shot_config"] = kwargs["llm_few_shot_config"]
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def _create_llm_plot(
        self,
        raw_data,  # pd.Series
        scores_dict,
        title,
        labels=None,  # np.array
        save_to_path=None,
    ):
        num_scores = len(scores_dict)
        num_plots = num_scores + 1  # Add one for the raw data plot
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(self.width, 6 * num_plots),
            sharex=False,
        )
        locator = mticker.MaxNLocator(nbins=40, prune="both")

        # Plot raw data for the channel
        ax_raw = axes[0]
        ax_raw.set_ylabel("Raw Data Value", color="tab:blue")
        ax_raw.plot(
            raw_data.index, raw_data.values, color="tab:blue", alpha=0.7, linewidth=3
        )
        ax_raw.tick_params(axis="y", labelcolor="tab:blue", labelsize=self.fontsize)
        ax_raw.set_title(f"Raw Data: {title}", fontsize=self.fontsize)
        data_min, data_max = raw_data.values.min(), raw_data.values.max()
        padding = (data_max - data_min) * 0.1
        ax_raw.set_ylim(data_min - padding, data_max + padding)
        ax_raw.set_xlim(raw_data.index.min(), raw_data.index.max())
        ax_raw.xaxis.set_major_locator(locator)
        ax_raw.tick_params(axis="x", labelsize=self.fontsize * 0.5)

        # Plot each anomaly score for the channel
        for i, (key, score) in enumerate(scores_dict.items()):
            ax = axes[i + 1]
            ax.set_ylabel("Anomaly Score", color="tab:orange")
            ax.plot(
                raw_data.index,
                score,
                color="tab:orange",
                linestyle="-",
                linewidth=3,
            )
            # Highlight anomaly scores if labels are provided
            if labels is not None:
                anomaly_indices = np.where(labels == 1)[0]
                if len(anomaly_indices) > 0:
                    # Ensure indices are within score length
                    anomaly_indices = anomaly_indices[anomaly_indices < len(score)]
                    ax.scatter(
                        raw_data.index[anomaly_indices],
                        score[anomaly_indices],
                        color="red",
                        s=150,
                        zorder=10,
                        marker="x",
                        linewidths=3,
                    )
            ax.tick_params(axis="y", labelcolor="tab:orange", labelsize=self.fontsize)
            ax.set_title(
                f"Anomaly Score: {key} (for {title})",
                fontsize=self.fontsize,
            )
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(raw_data.index.min(), raw_data.index.max())
            ax.xaxis.set_major_locator(locator)
            ax.tick_params(axis="x", labelsize=self.fontsize * 0.5)

        axes[-1].set_xlabel("Time", fontsize=self.fontsize)
        plt.suptitle(
            f"Analysis for: {title}",
            y=0.98,
            fontsize=self.fontsize,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))

        if save_to_path:
            # Ensure the directory exists and save both PNG and PDF
            base_path, _ = os.path.splitext(save_to_path)
            png_path = base_path + ".png"
            pdf_path = base_path + ".pdf"
            os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
            plt.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=150)
            logging.info(f"Saved plot to: {png_path}")
            plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0, format="pdf")
            logging.info(f"Saved plot to: {pdf_path}")

        buf = io.BytesIO()
        plt.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        image_bytes = buf.read()
        plt.close(fig)
        return image_bytes

    def _get_single_channel_llm_selection(
        self,
        client,
        model,
        scores_dict,
        target_channel_name,
        raw_data,
        llm_few_shot_config="default",
    ):
        """Helper function to perform plotting and LLM call for a single channel."""
        artifacts_dir = "llm_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # --- Generate and save the plot for the current channel ---
        current_plot_bytes = self._create_llm_plot(
            raw_data[target_channel_name],
            scores_dict,
            target_channel_name,
            labels=None,
            save_to_path=os.path.join(artifacts_dir, f"{timestamp}_plot.png"),
        )

        # --- Dynamically generate example plots ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        dataset_dir = os.path.join(project_root, "Datasets", "TSB-AD-M")
        score_dir = os.path.join(project_root, "eval", "score", "multi_as_uni")
        score_name_map = {
            "ensemble": "TSPulse_ZS_ensemble",
            "time": "TSPulse_ZS_time",
            "fft": "TSPulse_ZS_fft",
            "forecast": "TSPulse_ZS_forecast",
        }

        example_csvs = {}
        prompt = ""

        if llm_few_shot_config == "forecast_biased":
            example_csvs = {
                "forecast": [
                    "120_TAO_id_5_Environment_tr_500_1st_3-col-0.csv",
                    "149_SMAP_id_6_Sensor_tr_2128_1st_5000-0.csv",
                ]
            }
            prompt = f"""
The final PDF shows plots for the channel to be analyzed: '{target_channel_name}'.
The first plot is the raw data, followed by anomaly scores in the order: {list(scores_dict.keys())}.
You have been provided with an example of when the 'forecast' score performs well. 'Performing well' means the anomaly score is high for ground-truth anomalies (highlighted with red 'x' markers in the examples) and low for normal data points, leading to a high VUS-PR score.
Analyze the final PDF. If its characteristics, including the raw data shape, align with the provided example in EVERY aspect, select 'forecast'.
Otherwise, select the most appropriate score from the other available methods.
The available score methods are: {', '.join(scores_dict.keys())}.
Which anomaly score is the best for channel '{target_channel_name}'?
"""
        elif llm_few_shot_config == "non_forecast_biased":
            example_csvs = {
                "ensemble": [
                    "062_SMD_id_6_Facility_tr_7180_1st_15131-11.csv",
                    "028_MITDB_id_10_Medical_tr_37500_1st_39948-V1.csv",
                    "164_SMAP_id_21_Sensor_tr_1976_1st_4200-0.csv",
                ],
                "fft": [
                    "113_SVDB_id_30_Medical_tr_4552_1st_4652-ECG1.csv",
                    "049_GHL_id_18_Sensor_tr_50000_1st_109001-dL-rand.csv",
                    "131_OPPORTUNITY_id_3_HumanActivity_tr_7016_1st_26691-AccelerometerLAZYCHAIRaccY.csv",
                    "011_MSL_id_10_Sensor_tr_1525_1st_4590-0.csv",
                    "082_LTDB_id_4_Medical_tr_4456_1st_4556-ECG1.csv",
                    "023_MITDB_id_5_Medical_tr_25000_1st_36913-V1.csv",
                    "072_SMD_id_16_Facility_tr_7119_1st_15849-11.csv",
                    "195_Exathlon_id_22_Facility_tr_10766_1st_12590-1-executor-threadpool-activeTasks-value.csv",
                ],
                "time": [
                    "090_SVDB_id_7_Medical_tr_12157_1st_12257-ECG1.csv",
                    "004_MSL_id_3_Sensor_tr_530_1st_630-0.csv",
                    "140_CATSv2_id_3_Sensor_tr_28307_1st_28407-bso2.csv",
                    "040_GHL_id_9_Sensor_tr_50000_1st_92001-dL-rand.csv",
                ],
            }
            prompt = f"""
The final PDF shows plots for the channel to be analyzed: '{target_channel_name}'.
The first plot is the raw data, followed by anomaly scores in the order: {list(scores_dict.keys())}.
You have been provided with examples of when 'ensemble', 'fft', and 'time' scores perform well. 'Performing well' means the anomaly score is high for ground-truth anomalies (highlighted with red 'x' markers in the examples) and low for normal data points, leading to a high VUS-PR score.
Analyze the final PDF. If EVEN ONE aspect of its characteristics aligns with ANY of the provided examples, select the corresponding score.
Otherwise, select 'forecast'.
The available score methods are: {', '.join(scores_dict.keys())}.
Which anomaly score is the best for channel '{target_channel_name}'?
"""
        else:  # Default behavior for TSPulse2
            example_csvs = {
                "forecast": [
                    "120_TAO_id_5_Environment_tr_500_1st_3-col-0.csv",
                    "149_SMAP_id_6_Sensor_tr_2128_1st_5000-0.csv",
                ],
                "ensemble": [
                    "062_SMD_id_6_Facility_tr_7180_1st_15131-11.csv",
                    "028_MITDB_id_10_Medical_tr_37500_1st_39948-V1.csv",
                    "164_SMAP_id_21_Sensor_tr_1976_1st_4200-0.csv",
                ],
                "fft": [
                    "113_SVDB_id_30_Medical_tr_4552_1st_4652-ECG1.csv",
                    "049_GHL_id_18_Sensor_tr_50000_1st_109001-dL-rand.csv",
                    "131_OPPORTUNITY_id_3_HumanActivity_tr_7016_1st_26691-AccelerometerLAZYCHAIRaccY.csv",
                    "011_MSL_id_10_Sensor_tr_1525_1st_4590-0.csv",
                    "082_LTDB_id_4_Medical_tr_4456_1st_4556-ECG1.csv",
                    "023_MITDB_id_5_Medical_tr_25000_1st_36913-V1.csv",
                    "072_SMD_id_16_Facility_tr_7119_1st_15849-11.csv",
                    "195_Exathlon_id_22_Facility_tr_10766_1st_12590-1-executor-threadpool-activeTasks-value.csv",
                ],
                "time": [
                    "090_SVDB_id_7_Medical_tr_12157_1st_12257-ECG1.csv",
                    "004_MSL_id_3_Sensor_tr_530_1st_630-0.csv",
                    "140_CATSv2_id_3_Sensor_tr_28307_1st_28407-bso2.csv",
                    "040_GHL_id_9_Sensor_tr_50000_1st_92001-dL-rand.csv",
                ],
            }
            prompt = f"""
The final PDF shows plots for the channel to be analyzed: '{target_channel_name}'.
The first plot is the raw data, followed by anomaly scores in the order: {list(scores_dict.keys())}.
You have been provided with examples of when each score type ('ensemble', 'fft', 'time', 'forecast') performs well. 'Performing well' means the anomaly score is high for ground-truth anomalies (highlighted with red 'x' markers in the examples) and low for normal data points, leading to a high VUS-PR score.
Analyze the final PDF. Compare its characteristics, including the raw data shape, to the provided examples. Select the score ('ensemble', 'fft', 'time', or 'forecast') corresponding to the example that most closely aligns with the plots for '{target_channel_name}'.
The available score methods are: {', '.join(scores_dict.keys())}.
Which anomaly score is the best for channel '{target_channel_name}'?
"""

        example_parts = []
        all_examples = [
            (method, filename)
            for method, filenames in example_csvs.items()
            for filename in filenames
        ]

        for i, (method, csv_filename) in enumerate(all_examples):
            try:
                example_basename = os.path.splitext(csv_filename)[0]
                example_plot_path = os.path.join(
                    artifacts_dir, f"{example_basename}_example.pdf"
                )

                if os.path.exists(example_plot_path):
                    logging.info(f"Reusing existing plot: {example_plot_path}")
                    with open(example_plot_path, "rb") as f:
                        example_plot_bytes = f.read()
                else:
                    logging.info(f"Generating new plot for: {csv_filename}")
                    example_data_path = os.path.join(dataset_dir, csv_filename)
                    example_df = pd.read_csv(example_data_path)
                    example_raw_data = example_df.iloc[:, 0]
                    example_labels = example_df.iloc[:, -1].values

                    example_scores = {}
                    for key, algo_name in score_name_map.items():
                        score_path = os.path.join(
                            score_dir, algo_name, f"{example_basename}.npy"
                        )
                        if os.path.exists(score_path):
                            score_values = np.load(score_path)
                            if len(score_values) != len(example_raw_data):
                                score_values = np.resize(
                                    score_values, len(example_raw_data)
                                )
                            example_scores[key] = score_values

                    if not example_scores:
                        logging.warning(
                            f"No scores found for example {csv_filename}. Skipping."
                        )
                        continue

                    anonymous_title = f"Example Plot {i + 1} (Best Method: {method})"
                    example_plot_bytes = self._create_llm_plot(
                        example_raw_data,
                        example_scores,
                        anonymous_title,
                        labels=example_labels,
                        save_to_path=example_plot_path,
                    )

                example_parts.append(
                    types.Part.from_bytes(
                        mime_type="application/pdf", data=example_plot_bytes
                    )
                )

            except Exception as e:
                logging.warning(
                    f"Failed to process example file {csv_filename}: {e}",
                    exc_info=True,
                )
            example_parts.append(
                types.Part.from_text(
                    text=f"This is an example where the '{method}' score performed well."
                )
            )

        current_plot_part = types.Part.from_bytes(
            mime_type="application/pdf",
            data=current_plot_bytes,
        )

        all_parts = example_parts + [
            current_plot_part,
            types.Part.from_text(text=prompt),
        ]

        contents: types.ContentListUnion = [
            types.Content(
                role="user",
                parts=all_parts,
            ),
        ]
        score_keys = list(scores_dict.keys())

        response = None
        max_retries = 10
        delay = 2  # Start with a 2-second delay

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=32768, include_thoughts=True
                        ),
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.OBJECT,
                            required=["head"],
                            properties={
                                "head": types.Schema(
                                    type=types.Type.STRING, enum=score_keys
                                )
                            },
                        ),
                    ),
                )
                # If successful, break the loop
                break
            except Exception as e:
                logging.warning(
                    f"LLM call failed for '{target_channel_name}' on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        selected_key = None
        if response and response.candidates:
            if hasattr(response, "usage_metadata"):
                logging.info(
                    f"LLM token count for '{target_channel_name}': {response.usage_metadata}"
                )
            first_candidate = response.candidates[0]
            if first_candidate.content and first_candidate.content.parts:
                thought_summary = ""
                for part in first_candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thought_summary += f"Thought: {part.text}\n"
                    elif hasattr(part, "text") and part.text:
                        cleaned_text = part.text.strip()
                        if cleaned_text:
                            thought_summary += f"Output Part: {cleaned_text}\n"
                            import json

                            try:
                                json_response = json.loads(cleaned_text)
                                if (
                                    isinstance(json_response, dict)
                                    and "head" in json_response
                                ):
                                    selected_key = json_response["head"]
                                else:
                                    # Fallback for safety
                                    selected_key = cleaned_text.strip().strip('"')
                            except json.JSONDecodeError:
                                selected_key = cleaned_text.strip().strip('"')
                if thought_summary:
                    thought_filename = f"{timestamp}_thoughts.txt"
                    thought_save_path = os.path.join(artifacts_dir, thought_filename)
                    with open(thought_save_path, "w") as f:
                        f.write(thought_summary)
                    logging.info(f"LLM thoughts saved to: {thought_save_path}")
        return selected_key

    def _select_head_with_llm(self, scores_dict, target_columns, raw_data, **kwargs):
        if not scores_dict or all(
            not isinstance(v, np.ndarray) or v.size == 0 for v in scores_dict.values()
        ):
            return {}
        load_dotenv()
        # Use a separate API key for this pipeline, with a fallback to the general key
        api_key = os.getenv("TSPulse_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        logging.info(
            f"Attempting to use separate Gemini API key for TSPulse... {api_key}"
        )
        client = genai.Client(
            api_key=api_key,
        )

        model = "gemini-2.5-pro"

        channel_selections = {}
        for i, channel_name in enumerate(target_columns):
            start_time = time.time()
            logging.info(
                f"LLM: Analyzing channel {i+1}/{len(target_columns)}: {channel_name}"
            )
            raw_channel_data = raw_data[[channel_name]]
            single_channel_scores = {
                key: score_array[:, i] for key, score_array in scores_dict.items()
            }

            selected_key = self._get_single_channel_llm_selection(
                client,
                model,
                single_channel_scores,
                channel_name,
                raw_channel_data,
                llm_few_shot_config=kwargs.get("llm_few_shot_config", "default"),
            )
            logging.info(
                f"LLM: Selected '{selected_key}' for channel '{channel_name}' in {time.time() - start_time}s"
            )
            channel_selections[channel_name] = selected_key
        return channel_selections

    def postprocess(self, model_outputs, **postprocess_parameters):
        mangled_name = "_TimeSeriesAnomalyDetectionPipeline__context_memory"
        result = getattr(self, mangled_name)["data"].copy()
        expand_score = postprocess_parameters.get("expand_score", False)
        smoothing_window_size = postprocess_parameters.get("smoothing_length", 1)
        target_columns = postprocess_parameters.get("target_columns", [])
        use_llm_selection = postprocess_parameters.get("use_llm_selection", True)
        llm_few_shot_config = postprocess_parameters.get(
            "llm_few_shot_config", "default"
        )

        report_mode = postprocess_parameters.get("report_mode", False)
        predictive_score_smoothing = postprocess_parameters.get(
            "predictive_score_smoothing", False
        )
        if not isinstance(smoothing_window_size, int):
            try:
                smoothing_window_size = int(smoothing_window_size)
            except ValueError:
                smoothing_window_size = 1

        # adjust scoring and smooth
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
            model_outputs_[k] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                model_outputs_[k]
            )

        # aggregate scores and expand
        score = np.stack([score_ for _, score_ in model_outputs_.items()], axis=0)
        mode_selected = None
        if report_mode and (self.select_function is not None):
            keys = [key for key, _ in model_outputs_.items()]
            sel_index = self.select_function(score, axis=0)
            mode_selected = np.asarray([keys[z] for z in sel_index.ravel()]).reshape(
                sel_index.shape
            )
        ensemble_score = self.aggr_function(score, axis=0)

        if use_llm_selection:
            all_scores = model_outputs_.copy()
            all_scores["ensemble"] = ensemble_score
            raw_data_to_plot = result[target_columns]

            # This will now always return a dictionary, e.g., {'x1': 'time', 'x2': 'ensemble'}
            selections = self._select_head_with_llm(
                all_scores,
                target_columns,
                raw_data=raw_data_to_plot,
                llm_few_shot_config=llm_few_shot_config,
            )

            # Construct the final score array from per-channel selections
            score = np.zeros_like(ensemble_score)
            for i, col_name in enumerate(target_columns):
                selected_key = selections.get(
                    col_name, "ensemble"
                )  # Default to ensemble if a channel is missed
                score[:, i] = all_scores[selected_key][:, i]
        else:
            score = ensemble_score

        expand_score = (len(target_columns) > 1) and expand_score
        model_outputs = {}
        if expand_score:
            if len(target_columns) != score.shape[-1]:
                raise RuntimeError(
                    f"Error: inconsistent state, with target columns {target_columns}"
                )
            for i, col_name in enumerate(target_columns):
                model_outputs[f"{col_name}_anomaly_score"] = score[..., i]
            model_outputs["anomaly_score"] = score.mean(axis=-1).ravel()

            if mode_selected is not None:
                for i, col_name in enumerate(target_columns):
                    model_outputs[f"{col_name}_selected_mode"] = mode_selected[..., i]

        else:
            model_outputs.update(anomaly_score=score.ravel())
            if mode_selected is not None:
                model_outputs.update(selected_mode=mode_selected.ravel())

        # populate dataframe
        for k in model_outputs:
            result[k] = model_outputs[k]
        setattr(self, mangled_name, {})
        return result
