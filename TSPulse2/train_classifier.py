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
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report
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

# Define metric file mappings. This is the single source of truth for heads.
EVA_METRIC_FILES = {
    "ensemble": "TSPulse_ZS_ensemble.csv",
    "fft": "TSPulse_ZS_fft.csv",
    "future": "TSPulse_ZS_future.csv",
    "time": "TSPulse_ZS_time.csv",
    "scaled_ensemble": "TSPulse2.csv",
}

# The TUNING_METRIC_FILES dictionary is no longer needed as we will use a unified approach.

# The DATASET_CONFIG dictionary is no longer needed, as paths will be handled
# directly in the simplified data loading function.


# --- Helper Functions ---


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
        # This case is now more important as we only process common files.
        # If a file is in the common list but somehow has no score, log a warning.
        logging.warning(
            f"File '{filename}' was expected in metric files but no score was found. Skipping."
        )
        return None

    # The default return is less likely to be hit, but kept as a fallback.
    if not scores:
        return "scaled_ensemble"

    best_head = max(scores, key=scores.get)
    return best_head


def _process_single_file(
    filename: str, data_dir: str, metrics_cache: dict
) -> List[Dict]:
    """
    Helper function to process a single file for parallel execution.
    This version is simplified and does not handle Eva splits.
    """
    samples = []
    # The label (best head) is now determined from a pre-filtered set of common files
    label = get_best_head(filename, metrics_cache)
    if label is None:
        return samples

    try:
        data_path = os.path.join(data_dir, filename)
        df_processed = pd.read_csv(data_path)

        # Re-identify value columns
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


def load_and_process_data(dataset_type: str) -> List[Dict]:
    """
    Loads and processes data from a given dataset type ('uni' or 'multi'),
    aligning with the logic from train_selector_with_embeddings.py.
    It uses only files common to all metric definitions.
    """
    logging.info(f"\n--- Processing {dataset_type.upper()} tuning data ---")

    if dataset_type == "uni":
        data_dir_name = "TSB-AD-U"
        metrics_dir_name = "uni-tuning"
        file_list_name = "TSB-AD-U-Tuning.csv"
    else:  # multi
        data_dir_name = "TSB-AD-M"
        metrics_dir_name = "multi-tuning"
        file_list_name = "TSB-AD-M-Tuning.csv"

    data_dir = os.path.join(BASE_DATA_PATH, data_dir_name)
    metrics_dir = os.path.join(METRICS_BASE_PATH, metrics_dir_name)
    file_list_path = os.path.join(BASE_DATA_PATH, "File_List", file_list_name)

    # 1. Load all metric files for the given dataset type
    metric_dfs = {}
    for head_name, file_name in EVA_METRIC_FILES.items():
        file_path = os.path.join(metrics_dir, file_name)
        if not os.path.exists(file_path):
            logging.warning(
                f"Metric file not found, skipping head '{head_name}': {file_path}"
            )
            continue
        df = pd.read_csv(file_path)
        # Sanitize filenames to match across different files
        df["file"] = df["file"].apply(
            lambda x: os.path.splitext(x)[0] if isinstance(x, str) else x
        )
        metric_dfs[head_name] = df.set_index("file")

    if not metric_dfs:
        logging.error(f"No metric files found for {dataset_type}. Skipping.")
        return []

    active_heads = list(metric_dfs.keys())
    if len(active_heads) < len(EVA_METRIC_FILES):
        logging.warning(
            f"Not all heads have metric files. Using available heads: {active_heads}"
        )

    # 2. Find common files across all loaded metric dataframes
    common_files = set(metric_dfs[active_heads[0]].index)
    for head_name in active_heads[1:]:
        common_files.intersection_update(metric_dfs[head_name].index)

    logging.info(f"Found {len(common_files)} common files for {dataset_type} data.")

    # 3. Filter the main file list to only include common files
    try:
        file_list_df = pd.read_csv(file_list_path)
        all_tuning_files_with_ext = file_list_df["file_name"].tolist()
    except FileNotFoundError:
        logging.error(f"Tuning file list not found at {file_list_path}.")
        return []

    # Match common_files (without extension) to the full filenames
    files_to_process = [
        f
        for f in all_tuning_files_with_ext
        if os.path.splitext(f)[0] in common_files
    ]
    logging.info(f"Processing {len(files_to_process)} files from the file list.")

    # 4. Create metrics_cache for get_best_head. This is slightly different from
    # the selector's approach but reuses the existing `get_best_head` function.
    metrics_cache = {}
    for head, df in metric_dfs.items():
        # Ensure the index name is what get_best_head expects
        df.index.name = "file_sanitized"
        metrics_cache[head] = df

    # 5. Process the filtered files in parallel
    processed_samples_lists = Parallel(n_jobs=-1)(
        delayed(_process_single_file)(filename, data_dir, metrics_cache)
        for filename in tqdm(files_to_process, desc=f"Processing {dataset_type} files")
    )

    all_univariate_samples = [
        sample for sublist in processed_samples_lists for sample in sublist
    ]
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
    label_distribution = pd.Series(true_labels_text).value_counts()

    logging.info(f"\n\n--- {dataset_name} Test Set Evaluation ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Ground Truth Label Distribution:\n{label_distribution.to_string()}")
    logging.info("Classification Report:\n" + report)

    results_file = os.path.join(output_dir, f"test_results_{dataset_name.lower()}.txt")
    with open(results_file, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Ground Truth Label Distribution:\n")
        f.write(label_distribution.to_string() + "\n\n")
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

    logging.info("STEP 1: Loading and Processing Data...")
    # Load all available "Tuning" data using the new common-file logic.
    uni_samples = load_and_process_data("uni")
    multi_samples = load_and_process_data("multi")
    all_samples = uni_samples + multi_samples

    if not all_samples:
        logging.error("Data loading resulted in an empty dataset. Exiting.")
        return

    logging.info(f"Loaded a total of {len(all_samples)} samples from tuning data.")

    logging.info("STEP 2: Preparing DataFrame and Splitting Data...")
    df_full = create_dataframe_for_preprocessor(all_samples)

    # Shuffle the DataFrame
    df_full = df_full.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 80/10/10 Split
    train_size = int(0.8 * len(df_full))
    val_size = int(0.1 * len(df_full))

    train_df = df_full[:train_size].reset_index(drop=True)
    eval_df = df_full[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df_full[train_size + val_size :].reset_index(drop=True)

    if train_df.empty or eval_df.empty or test_df.empty:
        logging.error("Dataset splitting resulted in empty datasets. Exiting.")
        sys.exit(1)

    logging.info("STEP 3: Preprocessing Data...")
    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=["past_values"],
        label_column="labels",
        scaling=True,
        encode_labels=True,
    )

    # Fit preprocessor on the full dataset to learn all labels and scaling params
    logging.info("Fitting preprocessor on all available data...")
    tsp.train(df_full)

    logging.info("Transforming datasets with the fitted preprocessor...")
    train_df_prep = tsp.preprocess(train_df)
    eval_df_prep = tsp.preprocess(eval_df)
    test_df_prep = tsp.preprocess(test_df)

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
    test_dataset = ClassificationDFDataset(
        test_df_prep,
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

    batch_size = 1  # Use a constant batch size
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

    logging.info("STEP 6: Evaluating on Test Set...")
    predictions = trainer.predict(test_dataset)
    evaluate_and_log(predictions, "Combined Test", tsp, args.output_dir)

    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
