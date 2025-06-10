import sys
import os
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..",)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from TSPulse.tspulse_args import get_tspulse_args


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "granite-tsfm")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import math
import os
import tempfile

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from sktime.datasets import load_tsf_to_dataframe
import datasets
import numpy as np

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tspulse.configuration_tspulse import TSPulseConfig
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.util import count_parameters
from tsfm_public.toolkit.visualization import plot_predictions


logger = logging.getLogger(__file__)


def downsample_dataframe(
    df: pd.DataFrame, k: int, method: str = "average"
) -> pd.DataFrame:
    """
    Downsamples a time series DataFrame by factor k.

    The DataFrame is expected to have columns: 'series_name', 'timestamps', 'value'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        k (int): The downsampling factor. Every k samples will be reduced to 1.
        method (str): 'average' to average k points, 'subsample' to take every k-th point.

    Returns:
        pd.DataFrame: The downsampled DataFrame.
    """
    if k <= 1:
        return df.copy()

    # Group by each individual time series
    grouped = df.groupby("series_name")

    downsampled_list = []

    for name, group in grouped:
        # Skip if group is not large enough for one window
        if len(group) < k:
            continue

        if method == "average":
            # Use rolling average and then pick every k-th point.
            # The new timestamp will be the end of the rolling window.
            group_res = group.reset_index(drop=True)
            downsampled_group = group_res.rolling(window=k).mean(numeric_only=True)
            downsampled_group = downsampled_group.iloc[k - 1 :: k].reset_index(drop=True)

            # Keep original timestamps and series name
            downsampled_group["timestamps"] = group_res["timestamps"].iloc[
                k - 1 :: k
            ].reset_index(drop=True)
            downsampled_group["series_name"] = name

        elif method == "subsample":
            # Simply take every k-th row
            downsampled_group = group.iloc[::k].reset_index(drop=True)
        else:
            raise ValueError("Method must be 'average' or 'subsample'")

        downsampled_list.append(downsampled_group)

    if not downsampled_list:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(downsampled_list, ignore_index=True)


def get_base_model(args):
    """
    This function defines the TSPulse model architecture based on provided arguments,
    aligning with the pre-training methodology described in the TSPulse paper.
    """
    config = TSPulseConfig(
        # --- Core Architecture ---
        context_length=args.context_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_layers=args.num_layers,
        decoder_d_model=args.decoder_d_model,
        decoder_num_layers=args.decoder_num_layers,
        patch_register_tokens=args.patch_register_tokens,
        # --- Self-Supervised Task: Masking ---
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
        loss_apply_mode="mask_and_full",
        # --- Dual-Space (FFT) Learning ---
        fuse_fft=args.fuse_fft,
        fft_weight=args.fft_weight,
        fft_original_signal_loss_weight=args.fft_original_signal_loss_weight,
        enable_fft_prob_loss=args.enable_fft_prob_loss,
        fft_prob_weight=args.fft_prob_weight,
        # --- Auxiliary Forecasting Task ---
        fft_time_add_forecasting_pt_loss=args.fft_time_add_forecasting_pt_loss,
        prediction_length=args.prediction_length,
        fft_time_add_forecasting_pt_loss_weight=args.fft_time_add_forecasting_pt_loss_weight,
        # --- Other Critical Settings ---
        scaling=args.scaling,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        mode="common_channel",
        decoder_mode="common_channel",
    )

    model = TSPulseForReconstruction(config)
    return model


def pretrain(args, model, dset_train, dset_val):
    learning_rate = args.learning_rate

    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoint"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        seed=args.random_seed,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(args.save_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Correctly calculate steps_per_epoch for multi-GPU training
    steps_per_epoch = math.ceil(
        len(dset_train) / (args.batch_size * trainer_args.world_size)
    )
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )

    # Set trainer
    if args.early_stopping:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
        )

    # Train
    trainer.train()

    # Save the pretrained model

    model_save_path = os.path.join(args.save_dir, "tspulse_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def inference(args, model_path, dset_test):
    model = get_model(model_path=model_path)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
        ),
    )
    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE output:", "+" * 20)
    output = trainer.evaluate(dset_test)
    print(output)

    # get predictions

    predictions_dict = trainer.predict(dset_test)

    predictions_np = predictions_dict.predictions[0]

    print(predictions_np.shape)

    # get backbone embeddings (if needed for further analysis)

    backbone_embedding = predictions_dict.predictions[1]

    print(backbone_embedding.shape)

    plot_path = os.path.join(args.save_dir, "plots")
    # plot
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="test_inference",
        channel=0,
    )
    print("Plots saved in location:", plot_path)


if __name__ == "__main__":
    # Arguments
    args = get_tspulse_args()

    # Set seed
    set_seed(args.random_seed)

    logger.info(
        f"{'*' * 20} Pre-training a TSPulse for context len = {args.context_length}, forecast len = {args.prediction_length} {'*' * 20}"
    )

    # Based on Table 4 from TSPulse paper. Key is a substring in dataset directory name.
    DRS_CONFIG = {
        "wind_4_sec": [150, 225, 450, 900],  # 4s to 10m, 15m, 30m, 1h
        "wind_farms_minutely": [10, 15, 30, 60],  # 1m to 10m, 15m, 30m, 1h
        "solar_10_minutes": [3, 6],  # 10m to 30m, 1h
        "australian_electricity_demand": [2, 48],  # 30m to 1h, 1d
        "solar_4_sec": [150, 225, 450, 900],
        "london_smart_meters": [2, 48],  # 30m to 1h, 1d
        "pems": [2, 3, 6, 12],  # 5m to 10m, 15m, 30m, 1h
        "PEMS": [2, 3, 6, 12],  # Case-sensitive check
    }

    # Data prep
    datasets_dir = os.path.join(os.path.dirname(__file__), "..", "TSPulse", "datasets")
    dataset_names = [
        d
        for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ]

    all_data_list = []

    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_dir, dataset_name)
        tsf_path = os.path.join(
            dataset_path,
            f"{dataset_name}.tsf",
        )
        arrow_dir_path = dataset_path

        current_df = None

        # TSF file loading
        if os.path.exists(tsf_path):
            logger.info(f"Loading TSF dataset: {dataset_name}")
            try:
                data, metadata = load_tsf_to_dataframe(
                    tsf_path, value_column_name=None
                )
                data = data.reset_index()

                # Standardize timestamp column name
                time_col_names = ["timestamps", "timestamp", "timepoints", "index"]
                timestamp_col = next(
                    (c for c in time_col_names if c in data.columns), None
                )
                if timestamp_col:
                    data.rename(columns={timestamp_col: "timestamps"}, inplace=True)
                else:
                    logger.error(
                        f"No timestamp column found in {dataset_name}. Skipping."
                    )
                    continue

                # Identify value and ID columns
                id_cols = metadata.get("dimension_names", [])
                if "series_name" in data.columns and "series_name" not in id_cols:
                    id_cols.append("series_name")

                if not id_cols:
                    id_cols = [
                        c
                        for c in data.columns
                        if data[c].dtype == "object" and c != "timestamps"
                    ]

                value_cols = [
                    c for c in data.columns if c not in id_cols and c != "timestamps"
                ]

                if not value_cols:
                    logger.error(
                        f"No value columns found for {dataset_name}. Skipping."
                    )
                    continue

                # Melt multivariate data to create independent univariate series
                if len(value_cols) > 1:
                    logger.info(
                        f"{dataset_name} is multivariate. Unnesting into independent univariate series."
                    )
                    if not id_cols:  # Create a dummy id if none exist
                        data["series_name"] = dataset_name
                        id_cols = ["series_name"]

                    df_long = data.melt(
                        id_vars=id_cols + ["timestamps"],
                        value_vars=value_cols,
                        var_name="channel",
                        value_name="value",
                    )
                    df_long["series_name"] = (
                        df_long[id_cols].astype(str).agg("_".join, axis=1)
                        + "_"
                        + df_long["channel"]
                    )
                    final_df = df_long[["series_name", "timestamps", "value"]]
                # Handle data already in long format (or univariate)
                else:
                    df_long = data.rename(columns={value_cols[0]: "value"})
                    if not id_cols:
                        df_long["series_name"] = dataset_name
                    else:
                        df_long["series_name"] = (
                            df_long[id_cols].astype(str).agg("_".join, axis=1)
                        )
                    current_df = df_long[["series_name", "timestamps", "value"]]

            except Exception as e:
                logger.error(
                    f"Failed to process TSF dataset {dataset_name}. Error: {e}",
                    exc_info=True,
                )

        # Arrow file loading
        elif os.path.exists(os.path.join(arrow_dir_path, "dataset_info.json")):
            logger.info(f"Loading arrow dataset: {dataset_name}")
            try:
                hf_dataset = datasets.load_from_disk(arrow_dir_path)

                if isinstance(hf_dataset, datasets.DatasetDict):
                    df = hf_dataset.get("train") or hf_dataset.get(
                        "test"
                    )  # take first available split
                    if df is None:
                        logger.error(
                            f"Arrow dataset {dataset_name} has no 'train' or 'test' split. Skipping."
                        )
                        continue
                    df = df.to_pandas()
                else:
                    df = hf_dataset.to_pandas()

                processed_dfs = []
                for _, row in df.iterrows():
                    target = row.get("target")
                    if target is None:
                        continue

                    num_periods = (
                        len(target)
                        if isinstance(target, (list, np.ndarray, pd.Series))
                        else 1
                    )
                    target_values = (
                        target
                        if isinstance(target, (list, np.ndarray, pd.Series))
                        else [target]
                    )

                    freq = row.get("freq", "D")
                    if freq == "T":
                        freq = "min"  # Fix for pandas FutureWarning

                    timestamps = pd.date_range(
                        start=row["start"], periods=num_periods, freq=freq
                    )
                    temp_df = pd.DataFrame(
                        {
                            "timestamps": timestamps,
                            "value": target_values,
                            "series_name": row.get("item_id", dataset_name),
                        }
                    )
                    processed_dfs.append(temp_df)

                if processed_dfs:
                    current_df = pd.concat(processed_dfs, ignore_index=True)
            except Exception as e:
                logger.error(
                    f"Failed to load arrow dataset {dataset_name}. Error: {e}",
                    exc_info=True,
                )

        if current_df is not None and not current_df.empty:
            # Add original high-resolution dataset
            all_data_list.append(current_df)
            logger.info(
                f"Loaded {dataset_name}. Original size: {len(current_df)} rows."
            )

            # Apply Diverse Resolution Sampling (DRS)
            for name_key, k_factors in DRS_CONFIG.items():
                if name_key in dataset_name:
                    logger.info(
                        f"Applying DRS to {dataset_name} with k-factors: {k_factors}"
                    )
                    for k in k_factors:
                        # Use averaging method as it is more robust
                        df_downsampled = downsample_dataframe(
                            current_df, k, method="average"
                        )
                        if not df_downsampled.empty:
                            all_data_list.append(df_downsampled)
                            logger.info(
                                f"  > Added downsampled version (k={k}) with {len(df_downsampled)} rows."
                            )
                    break  # Stop after first match
        elif os.path.exists(tsf_path) or os.path.exists(
            os.path.join(arrow_dir_path, "dataset_info.json")
        ):
            # A file existed but was not loaded, previous logs have details.
            pass
        else:
            logger.warning(f"No loadable data found for {dataset_name}, skipping.")
            continue

    if not all_data_list:
        logger.error("No datasets were loaded. Exiting.")
        sys.exit(1)

    # Concatenate all datasets
    full_data = pd.concat(all_data_list, ignore_index=True)


    timestamp_column = "timestamps"
    id_columns = ["series_name"]  # mention the ids that uniquely identify a time-series.

    target_columns = ["value"]

    # mention the train, valid and split config.
    split_config = {
        "train": 0.7,
        "valid": 0.15,
        "test": 0.15,
    }

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, dset_valid, dset_test = get_datasets(tsp, full_data, split_config)

    # Get model
    model = get_base_model(args)
    
    logger.info(f"Model has {count_parameters(model)/1e6:.2f}M parameters")


    # Pretrain
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    inference(args=args, model_path=model_save_path, dset_test=dset_test)

    print("inference completed..")
