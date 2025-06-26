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
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
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


def get_base_multivariate_file(univariate_filename: str, multi_base_names: list) -> str:
    """
    Finds the original multivariate base filename for a derived univariate filename.
    It iterates through a pre-sorted list of multivariate names to find the longest match.

    Args:
        univariate_filename (str): The filename of the derived univariate series.
        multi_base_names (list): A list of original multivariate base names, sorted by length descending.

    Returns:
        str: The matching multivariate base name, or None if not found.
    """
    # The univariate filename is without extension here
    for base_name in multi_base_names:
        if univariate_filename.startswith(base_name + "_"):
            return base_name
    logging.debug(f"Could not find base name for {univariate_filename}")
    return None


def get_base_multivariate_file_map(univariate_files: list, multi_base_names: list) -> dict:
    """
    Efficiently finds the original multivariate base filename for a list of derived univariate filenames
    using a single compiled regular expression.

    Args:
        univariate_files (list): A list of derived univariate series filenames (without extension).
        multi_base_names (list): A list of original multivariate base names, sorted by length descending.

    Returns:
        dict: A mapping from each univariate filename to its corresponding multivariate base name.
    """
    # The list of base names is pre-sorted by length, so the regex will prioritize the longest match.
    base_name_pattern = "|".join(re.escape(b) for b in multi_base_names)
    # This regex looks for one of the base names at the start of the string, followed by an underscore.
    base_name_regex = re.compile(f"^({base_name_pattern})_")

    mapping = {}
    for f in tqdm(univariate_files, desc="Mapping univariate files to base names"):
        match = base_name_regex.match(f)
        if match:
            # group(1) captures the matched base name from the pattern.
            mapping[f] = match.group(1)
        else:
            logging.debug(f"Could not determine base name for {f}")
            mapping[f] = None
    return mapping


def load_split_and_process_data(
    dataset_type: str, multi_file_lists: dict = None
) -> (List[Dict], List[Dict]):
    """
    Loads data, splits it into train/val and test sets based on file lists,
    and processes them into univariate samples.
    """
    logging.info(f"\n--- Processing {dataset_type.upper()} data ---")

    # 1. Define paths based on dataset type
    full_list_name, eva_list_name = None, None  # Default to None
    if dataset_type == "uni":
        data_dir_name = "TSB-AD-U"
        metrics_dir_name = "uni"
        full_list_name = "TSB-AD-U.csv"
        eva_list_name = "TSB-AD-U-Eva.csv"
    elif dataset_type == "multi":
        data_dir_name = "TSB-AD-M"
        metrics_dir_name = "multi"
        full_list_name = "TSB-AD-M.csv"
        eva_list_name = "TSB-AD-M-Eva.csv"
    elif dataset_type == "multi_as_uni":
        data_dir_name = "TSB-AD-M-univariate"
        metrics_dir_name = "multi_as_uni"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    data_dir = os.path.join(BASE_DATA_PATH, data_dir_name)
    metrics_dir = os.path.join(METRICS_BASE_PATH, metrics_dir_name)

    # 2. Load all metric files to find common files
    metric_dfs = {}
    for head_name, file_name in EVA_METRIC_FILES.items():
        file_path = os.path.join(metrics_dir, file_name)
        if not os.path.exists(file_path):
            logging.warning(
                f"Metric file not found, skipping head '{head_name}': {file_path}"
            )
            continue
        df = pd.read_csv(file_path)
        df["file"] = df["file"].apply(
            lambda x: os.path.splitext(x)[0] if isinstance(x, str) else x
        )
        metric_dfs[head_name] = df.set_index("file")

    if not metric_dfs:
        logging.error(f"No metric files found for {dataset_type}. Skipping.")
        return [], []

    active_heads = list(metric_dfs.keys())
    common_files = set(metric_dfs[active_heads[0]].index)
    for head_name in active_heads[1:]:
        common_files.intersection_update(metric_dfs[head_name].index)
    logging.info(
        f"Found {len(common_files)} common files with metrics for {dataset_type} data."
    )

    # 3. Load file lists to determine train/val and test splits
    if dataset_type == "multi_as_uni":
        train_val_files_set, test_files_set = set(), set()
        multi_eva_files_set = multi_file_lists["eva"]
        multi_base_names = multi_file_lists["base"]

        # --- OPTIMIZATION ---
        # The original method iterated through all base names for each univariate file,
        # which was very slow (O(N*M)). This new method builds a single regex
        # to find all mappings in one pass (much faster).
        uni_to_multi_map = get_base_multivariate_file_map(
            list(common_files), multi_base_names
        )

        for f, base_multi_name in uni_to_multi_map.items():
            if base_multi_name and (base_multi_name + ".csv") in multi_eva_files_set:
                test_files_set.add(f + ".csv")
            else:
                train_val_files_set.add(f + ".csv")
    else:
        full_file_list_path = os.path.join(BASE_DATA_PATH, "File_List", full_list_name)
        eva_file_list_path = os.path.join(BASE_DATA_PATH, "File_List", eva_list_name)
        try:
            full_files_set = set(pd.read_csv(full_file_list_path)["file_name"])
            eva_files_set = set(pd.read_csv(eva_file_list_path)["file_name"])
        except FileNotFoundError as e:
            logging.error(f"File list not found: {e}. Cannot create splits.")
            return [], []

        train_val_files_set = full_files_set - eva_files_set
        test_files_set = eva_files_set

    # 4. Filter file lists to only those with valid metrics
    files_to_process_train_val = [
        f for f in train_val_files_set if os.path.splitext(f)[0] in common_files
    ]
    files_to_process_test = [
        f for f in test_files_set if os.path.splitext(f)[0] in common_files
    ]

    logging.info(
        f"Found {len(files_to_process_train_val)} files for the training/validation pool."
    )
    logging.info(f"Found {len(files_to_process_test)} files for the test set.")

    # 5. Create metrics_cache for get_best_head
    metrics_cache = {head: df for head, df in metric_dfs.items()}

    # 6. Process files in parallel
    logging.info("Processing train/validation files...")
    train_val_samples_lists = Parallel(n_jobs=-1)(
        delayed(_process_single_file)(filename, data_dir, metrics_cache)
        for filename in tqdm(
            files_to_process_train_val,
            desc=f"Processing {dataset_type} train/val files",
        )
    )
    train_val_samples = [
        sample for sublist in train_val_samples_lists for sample in sublist
    ]

    logging.info("Processing test files...")
    test_samples_lists = Parallel(n_jobs=-1)(
        delayed(_process_single_file)(filename, data_dir, metrics_cache)
        for filename in tqdm(
            files_to_process_test, desc=f"Processing {dataset_type} test files"
        )
    )
    test_samples = [sample for sublist in test_samples_lists for sample in sublist]

    return train_val_samples, test_samples


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


class TrainerWithWeightedLoss(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "labels" in inputs:
            inputs["target_values"] = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.prediction_outputs
        labels = inputs.get("target_values")

        # Use a new loss function with the calculated weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        # --- Custom Logic to Compute and Log Training Accuracy ---
        if self.is_in_train and labels is not None:
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
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Start training from scratch, overwriting existing checkpoints. Default is to resume.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("STEP 1: Loading and Processing Data...")

    # Load file lists for multi-variate cases first, as they are needed for multi_as_uni
    multi_full_list_path = os.path.join(BASE_DATA_PATH, "File_List", "TSB-AD-M.csv")
    multi_eva_list_path = os.path.join(BASE_DATA_PATH, "File_List", "TSB-AD-M-Eva.csv")
    try:
        multi_full_df = pd.read_csv(multi_full_list_path)
        multi_eva_files_set = set(pd.read_csv(multi_eva_list_path)["file_name"])
        # Create a list of base names (without .csv), sorted by length descending
        multi_base_names = sorted(
            [os.path.splitext(f)[0] for f in multi_full_df["file_name"]],
            key=len,
            reverse=True,
        )
        multi_file_lists = {"eva": multi_eva_files_set, "base": multi_base_names}
    except FileNotFoundError as e:
        logging.error(f"Multivariate file list not found: {e}. Cannot proceed.")
        return

    # Load and split data using the new logic
    uni_train_val_samples, uni_test_samples = load_split_and_process_data("uni")
    multi_train_val_samples, multi_test_samples = load_split_and_process_data("multi")
    (
        multi_as_uni_train_val_samples,
        multi_as_uni_test_samples,
    ) = load_split_and_process_data("multi_as_uni", multi_file_lists=multi_file_lists)

    all_train_val_samples = (
        uni_train_val_samples + multi_train_val_samples + multi_as_uni_train_val_samples
    )
    all_test_samples = uni_test_samples + multi_test_samples + multi_as_uni_test_samples

    if not all_train_val_samples:
        logging.error(
            "Training/validation data loading resulted in an empty dataset. Exiting."
        )
        return
    if not all_test_samples:
        logging.warning(
            "Test data loading resulted in an empty dataset. Evaluation will be skipped."
        )

    logging.info(
        f"Loaded {len(all_train_val_samples)} samples for training/validation pool."
    )
    logging.info(f"Loaded {len(all_test_samples)} samples for test set.")

    logging.info("STEP 2: Preparing DataFrame and Splitting Data...")
    df_train_val = create_dataframe_for_preprocessor(all_train_val_samples)
    df_test = (
        create_dataframe_for_preprocessor(all_test_samples)
        if all_test_samples
        else pd.DataFrame()
    )

    # Shuffle the training/validation pool
    df_train_val = df_train_val.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Split the pool into 90% training and 10% validation
    train_size = int(0.9 * len(df_train_val))
    train_df = df_train_val[:train_size].reset_index(drop=True)
    eval_df = df_train_val[train_size:].reset_index(drop=True)
    test_df = df_test.reset_index(drop=True)  # Already separate

    logging.info("--- Training Set Label Distribution ---")
    logging.info(train_df["labels"].value_counts(normalize=True).to_string())

    # Combine all data to fit the preprocessor
    df_full = pd.concat([train_df, eval_df, test_df], ignore_index=True)

    if train_df.empty or eval_df.empty:
        logging.error(
            "Dataset splitting resulted in empty train or validation sets. Exiting."
        )
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

    # Calculate class weights for handling imbalance
    class_labels = tsp.label_encoder.classes_
    # We need the original string labels from the training set to calculate weights correctly.
    # The 'train_df' still has the original string labels.
    train_labels_text = train_df["labels"]

    class_weights = compute_class_weight(
        class_weight="balanced", classes=class_labels, y=train_labels_text
    )
    # The trainer expects a tensor on the correct device.
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda")

    logging.info(f"Using class weights: {class_weights_tensor}")

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

    batch_size = 2**14  # Use a constant batch size
    logging.info("Finding optimal learning rate...")
    lr, model = optimal_lr_finder(
        model,
        train_dataset,
        batch_size=batch_size,
    )
    logging.info(f"Using learning rate found by LR finder: {lr}")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        overwrite_output_dir=args.fresh_start,
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

    trainer = TrainerWithWeightedLoss(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        class_weights=class_weights_tensor,  # Pass the weights
    )

    trainer.train(resume_from_checkpoint=not args.fresh_start)

    logging.info("STEP 6: Evaluating on Test Set...")
    predictions = trainer.predict(test_dataset)
    evaluate_and_log(predictions, "Combined Test", tsp, args.output_dir)

    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
