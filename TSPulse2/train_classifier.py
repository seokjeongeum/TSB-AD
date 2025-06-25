"""
This script trains a TSPulseForClassification model on the TSB-AD dataset.
It adapts the logic from the tspulse_classification.ipynb notebook to:
1. Split multivariate series into multiple univariate series.
2. Load data according to a specific train/test split strategy:
   - Test Set: Full series from 'Eva' files.
   - Train/Val Pool: Full series from 'Tuning' files + train splits from 'Eva' files.
3. Dynamically determine the classification label for each series by finding the
   "best performing head" based on the highest VUS-PR score, using different
   metric file configurations for Eva and Tuning sets.
4. Preprocess, fine-tune the univariate TSPulse model, and save the result.
"""

import argparse
import logging
import math
import os
import re
import sys
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "granite-tsfm")),
)
from tsfm_public.models.tspulse import TSPulseForClassification
from tsfm_public.toolkit.dataset import ClassificationDFDataset
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_classification_preprocessor import \
    TimeSeriesClassificationPreprocessor

# --- Configuration ---
SEED = 2024
set_seed(SEED)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# --- Path and Data Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "Datasets")
METRICS_BASE_PATH = os.path.join(PROJECT_ROOT, "eval", "metrics")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "TSPulse2", "classification_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define metric file mappings. One for 'Eva' sets, one for 'Tuning' sets.
EVA_METRIC_FILES = {
    "ensemble": "TSPulse_ZS_ensemble.csv",
    "fft": "TSPulse_ZS_fft.csv",
    "future": "TSPulse_ZS_future.csv",
    "time": "TSPulse_ZS_time.csv",
    "scaled_ensemble": "TSPulse2.csv",
}

TUNING_METRIC_FILES = {
    "scaled_ensemble": "TSPulse2.csv",
}


DATASET_CONFIG = {
    "M-Eva": {
        "list_file": f"{BASE_DATA_PATH}/File_List/TSB-AD-M-Eva.csv",
        "metrics_dir_name": "multi",
        "data_dir_name": "TSB-AD-M",
    },
    "M-Tuning": {
        "list_file": f"{BASE_DATA_PATH}/File_List/TSB-AD-M-Tuning.csv",
        "metrics_dir_name": "multi-tuning",
        "data_dir_name": "TSB-AD-M",
    },
    "U-Eva": {
        "list_file": f"{BASE_DATA_PATH}/File_List/TSB-AD-U-Eva.csv",
        "metrics_dir_name": "uni",
        "data_dir_name": "TSB-AD-U",
    },
    "U-Tuning": {
        "list_file": f"{BASE_DATA_PATH}/File_List/TSB-AD-U-Tuning.csv",
        "metrics_dir_name": "uni-tuning",
        "data_dir_name": "TSB-AD-U",
    },
}

# --- Helper Functions ---


def parse_train_index(filename: str) -> int:
    """Extracts the training index from the TSB-AD filename."""
    match = re.search(r"_tr_(\d+)_", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse train index from filename: {filename}")


@lru_cache(maxsize=8)
def load_metrics_for_dir(metrics_dir: str, files_to_load: tuple) -> dict:
    """Loads specified metric files from a directory into a dictionary of DataFrames."""
    files_to_load_dict = dict(files_to_load)

    logging.info(f"Loading metrics from: {metrics_dir}")
    metrics_data = {}
    for head, fname in files_to_load_dict.items():
        metric_file_path = os.path.join(metrics_dir, fname)
        if not os.path.exists(metric_file_path):
            logging.warning(
                f"Metric file not found: {metric_file_path}. Skipping head '{head}'."
            )
            continue
        df = pd.read_csv(metric_file_path)
        df["file_sanitized"] = df["file"].str.replace(".csv", "", regex=False)
        df = df.set_index("file_sanitized")
        metrics_data[head] = df
    return metrics_data


def get_best_head(filename: str, metrics_data: dict) -> str:
    """Determines the best performing head for a given file based on VUS-PR."""
    sanitized_fname = filename.replace(".csv", "")
    scores = {}
    for head, df in metrics_data.items():
        if sanitized_fname in df.index:
            vus_pr_score = df.loc[sanitized_fname, "VUS-PR"]
            if isinstance(vus_pr_score, pd.Series):
                vus_pr_score = vus_pr_score.iloc[0]
            scores[head] = float(vus_pr_score)
        else:
            scores[head] = -1.0

    if not any(s > -1.0 for s in scores.values()):
        logging.warning(f"File '{filename}' not found in any metric files. Skipping.")
        return None

    if not scores:
        return "scaled_ensemble"

    best_head = max(scores, key=scores.get)
    return best_head


def _process_single_file(
    filename: str,
    group_name: str,
    data_dir: str,
    metrics_cache: dict,
    use_train_split_for_eva: bool,
) -> List[Dict]:
    """Helper function to process a single file for parallel execution."""
    samples = []
    label = get_best_head(filename, metrics_cache)
    if label is None:
        return samples

    try:
        data_path = os.path.join(data_dir, filename)
        df_raw = pd.read_csv(data_path)

        is_eva_group = "Eva" in group_name
        if is_eva_group:
            train_index = parse_train_index(filename)
            if use_train_split_for_eva:
                # Use training part of Eva files for the train/val set
                df_processed = df_raw.iloc[:train_index].copy()
            else:
                # Use test part of Eva files for the test set
                df_processed = df_raw.iloc[train_index:].copy()
        else:
            # For non-eva files (Tuning files), use the whole series
            df_processed = df_raw.copy()

        # Re-identify value columns after potential slicing
        value_cols = [
            c
            for c in df_processed.columns
            if c not in ["is_anomaly", "anomaly", "timestamp"]
        ]
        if not value_cols:
            value_cols = [
                c
                for c in df_processed.columns
                if np.issubdtype(df_processed[c].dtype, np.number)
            ]
            if not value_cols:
                value_cols = ["value"]

        for i, col_name in enumerate(value_cols):
            univariate_series = df_processed[col_name].values
            sample_id = f"{filename.replace('.csv', '')}_dim_{i}"
            samples.append(
                {"id": sample_id, "values": univariate_series, "label": label}
            )
    except Exception as e:
        logging.error(f"Failed to process file {filename}: {e}", exc_info=True)

    return samples


def load_data_from_config(
    groups_to_load: List[str], use_train_split_for_eva: bool
) -> List[Dict]:
    """Loads data, parallelized with joblib."""
    all_univariate_samples = []

    for group_name in groups_to_load:
        config = DATASET_CONFIG[group_name]
        logging.info(f"--- Processing data for: {group_name} ---")

        is_tuning_group = "Tuning" in group_name
        metric_files_to_use = (
            TUNING_METRIC_FILES if is_tuning_group else EVA_METRIC_FILES
        )

        metrics_dir = os.path.join(METRICS_BASE_PATH, config["metrics_dir_name"])
        data_dir = os.path.join(BASE_DATA_PATH, config["data_dir_name"])

        metrics_cache = load_metrics_for_dir(
            metrics_dir, tuple(metric_files_to_use.items())
        )

        if not metrics_cache:
            logging.error(f"No metrics loaded for {metrics_dir}. Cannot proceed.")
            continue

        file_list_df = pd.read_csv(config["list_file"])

        processed_samples_lists = Parallel(n_jobs=-1)(
            delayed(_process_single_file)(
                filename, group_name, data_dir, metrics_cache, use_train_split_for_eva
            )
            for filename in tqdm(
                file_list_df["file_name"], desc=f"Processing {group_name}"
            )
        )

        all_univariate_samples.extend(
            [sample for sublist in processed_samples_lists for sample in sublist]
        )

    return all_univariate_samples


def create_dataframe_for_preprocessor(data_samples: List[Dict]) -> pd.DataFrame:
    """Converts samples into a DataFrame for the preprocessor."""
    data_for_df = {"past_values": [], "labels": []}
    for sample in tqdm(data_samples, desc="Creating preprocessor DataFrame"):
        data_for_df["past_values"].append(pd.Series(sample["values"]))
        data_for_df["labels"].append(sample["label"])
    return pd.DataFrame(data_for_df)


def compute_metrics(p):
    """Compute accuracy for classification task."""
    preds = np.argmax(p.predictions[0], axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


def evaluate_and_log(predictions, dataset_name, tsp, output_dir):
    """Calculates and logs evaluation metrics."""
    preds = np.argmax(predictions.predictions[0], axis=1)
    true_labels = predictions.label_ids
    pred_labels = tsp.label_encoder.inverse_transform(preds)
    true_labels_text = tsp.label_encoder.inverse_transform(true_labels)
    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(
        true_labels_text,
        pred_labels,
        labels=tsp.label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    logging.info(f"\n\n--- {dataset_name} Test Set Evaluation ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + report)

    results_file = os.path.join(output_dir, f"test_results_{dataset_name.lower()}.txt")
    with open(results_file, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)


class TrainerWithTrainAccuracy(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to handle the label key mismatch and log training accuracy.
        """
        # The Trainer moves the column listed in `label_names` to 'labels'.
        # We need to rename it back to 'target_values' for the model's forward pass.
        if "labels" in inputs:
            inputs["target_values"] = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)

        # Loss is part of the model's output
        loss = outputs.loss

        # --- Custom Logic to Compute and Log Training Accuracy ---
        if self.is_in_train:
            logits = outputs.prediction_outputs
            # The labels are now in 'target_values'
            labels = inputs.get("target_values")

            if logits is not None and labels is not None:
                preds = torch.argmax(logits.detach(), dim=-1)
                accuracy = (preds == labels).float().mean()
                self.log({"accuracy": accuracy.item()})

        return (loss, outputs) if return_outputs else loss


def main():
    """Main function to run the full training and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train TSPulseForClassification with custom hyperparameters."
    )
    # --- Add arguments for hyperparameters ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save checkpoints and final model. Defaults to TSPulse2/classification_output.",
    )
    parser.add_argument("--head_reduce_d_model", type=int, default=1)
    parser.add_argument(
        "--decoder_mode",
        type=str,
        default="mix_channel",
        choices=["mix_channel", "common_channel"],
    )
    parser.add_argument(
        "--head_gated_attention_activation",
        type=str,
        default="softmax",
        choices=["softmax", "sigmoid"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--channel_virtual_expand_scale", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("STEP 1: Loading and Splitting Data...")
    # Load the training part of Eva datasets to be used for training and validation
    logging.info("Loading Eva data for training/validation set...")
    train_val_pool_samples = load_data_from_config(
        ["M-Eva", "U-Eva"], use_train_split_for_eva=True
    )

    # Load the Tuning datasets for testing
    logging.info("Loading Tuning data for test sets...")
    uni_test_samples = load_data_from_config(
        ["U-Tuning"], use_train_split_for_eva=False
    )
    multi_test_samples = load_data_from_config(
        ["M-Tuning"], use_train_split_for_eva=False
    )

    if not train_val_pool_samples or not uni_test_samples or not multi_test_samples:
        logging.error("Data loading resulted in empty datasets. Exiting.")
        return

    logging.info(
        f"Loaded {len(train_val_pool_samples)} samples for training/validation pool, "
        f"{len(uni_test_samples)} for univariate testing, and "
        f"{len(multi_test_samples)} for multivariate testing."
    )

    logging.info("STEP 2: Preparing DataFrames...")
    df_train_val_pool = create_dataframe_for_preprocessor(train_val_pool_samples)
    df_uni_test = create_dataframe_for_preprocessor(uni_test_samples)
    df_multi_test = create_dataframe_for_preprocessor(multi_test_samples)

    logging.info("STEP 3: Preprocessing Data...")
    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=["past_values"],
        label_column="labels",
        scaling=True,
        encode_labels=True,
    )

    # Fit preprocessor on all available data to learn all labels.
    logging.info("Fitting preprocessor on all available data to learn all labels...")
    df_full_for_fitting = pd.concat(
        [df_train_val_pool, df_uni_test, df_multi_test], ignore_index=True
    )
    tsp.train(df_full_for_fitting)

    # Split the Eva data pool into training and validation sets.
    train_df = df_train_val_pool.sample(frac=0.9, random_state=SEED).reset_index(
        drop=True
    )
    eval_df = df_train_val_pool.drop(train_df.index).reset_index(drop=True)

    if train_df.empty or eval_df.empty:
        logging.error("Dataset splitting resulted in empty datasets. Exiting.")
        sys.exit(1)

    logging.info("Transforming datasets with the fitted preprocessor...")
    train_df_prep = tsp.preprocess(train_df)
    eval_df_prep = tsp.preprocess(eval_df)
    uni_test_df_prep = tsp.preprocess(df_uni_test)
    multi_test_df_prep = tsp.preprocess(df_multi_test)

    # Create the datasets
    train_dataset = ClassificationDFDataset(
        train_df_prep,
        input_columns=["past_values"],
        label_column="labels",
        context_length=512,
        full_series=True,
    )
    eval_dataset = ClassificationDFDataset(
        eval_df_prep,
        input_columns=["past_values"],
        label_column="labels",
        context_length=512,
        full_series=True,
    )
    uni_test_dataset = ClassificationDFDataset(
        uni_test_df_prep,
        input_columns=["past_values"],
        label_column="labels",
        context_length=512,
        full_series=True,
    )
    multi_test_dataset = ClassificationDFDataset(
        multi_test_df_prep,
        input_columns=["past_values"],
        label_column="labels",
        context_length=512,
        full_series=True,
    )

    # 4. Initialize Model
    logging.info("STEP 4: Initializing Model...")
    num_targets = len(tsp.label_encoder.classes_)
    logging.info(f"Found {num_targets} unique labels: {tsp.label_encoder.classes_}")

    config_dict = {
        "head_gated_attention_activation": args.head_gated_attention_activation,
        "channel_virtual_expand_scale": args.channel_virtual_expand_scale,
        "mask_ratio": args.mask_ratio,
        "head_reduce_d_model": args.head_reduce_d_model,
        "decoder_mode": args.decoder_mode,
        "disable_mask_in_classification_eval": True,
        "fft_time_consistent_masking": True,
        "head_aggregation_dim": "patch",
        "head_aggregation": None,
        "loss": "cross_entropy",
        "ignore_mismatched_sizes": True,
    }

    config_dict["num_input_channels"] = 1  # Each series is treated as univariate
    config_dict["num_targets"] = num_targets

    model = TSPulseForClassification.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1",
        revision="tspulse-block-dualhead-512-p16-r1",
        **config_dict,
    )

    # # Compile the model if using PyTorch 2.0+
    # if hasattr(torch, "compile"):
    #     logging.info("Compiling the model with torch.compile...")
    #     model = torch.compile(model)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.time_encoding.parameters():
        param.requires_grad = True
    for param in model.backbone.fft_encoding.parameters():
        param.requires_grad = True

    # 5. Train Model
    logging.info("STEP 5: Training Model...")

    batch_size = 2048  # Use a constant batch size
    logging.info("Finding optimal learning rate...")
    lr, model = optimal_lr_finder(
        model,
        train_dataset,
        batch_size=batch_size,
    )
    logging.info(f"Using learning rate found by LR finder: {lr}")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        overwrite_output_dir=True,
        learning_rate=lr,
        num_train_epochs=200,
        do_eval=True,
        # Set both strategies to 'epoch' for clean, aggregated logging
        eval_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # --- SPEEDUP CONFIGS ---
        gradient_accumulation_steps=4,  # Simulate effective batch size
        dataloader_num_workers=os.cpu_count(),
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        # --- END SPEEDUP CONFIGS ---
        report_to="tensorboard",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["target_values"],  # Inform Trainer of the correct label key
    )

    trainer = TrainerWithTrainAccuracy(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    logging.info("STEP 6: Evaluating on Test Sets...")

    # Evaluate Univariate
    logging.info("--- Evaluating on Univariate Test Set ---")
    uni_predictions = trainer.predict(uni_test_dataset)
    evaluate_and_log(uni_predictions, "Univariate", tsp, args.output_dir)

    # Evaluate Multivariate
    logging.info("--- Evaluating on Multivariate Test Set ---")
    multi_predictions = trainer.predict(multi_test_dataset)
    evaluate_and_log(multi_predictions, "Multivariate", tsp, args.output_dir)

    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
