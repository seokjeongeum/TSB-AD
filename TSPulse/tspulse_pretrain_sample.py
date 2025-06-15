import logging

from rich.logging import RichHandler
from transformers import logging as hf_logging

# try:
#     import cudf
#     IS_CUDA_AVAILABLE = True
# except ImportError:
#     IS_CUDA_AVAILABLE = False
hf_logging.disable_default_handler()
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
rich_handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_path=False,
)
log_formatter = logging.Formatter(
    fmt="%(message)s    [dim](%(pathname)s:%(lineno)d)[/dim]"
)
rich_handler.setFormatter(log_formatter)

root_logger.addHandler(rich_handler)
logger = logging.getLogger("TSPulsePretrain")
import argparse
import math
import os
import pprint
import sys
import tempfile
import time

import datasets
import pandas as pd
import polars as pl
from rich import print as rprint
from rich.panel import Panel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)

hf_logging.set_verbosity_info()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TSPulse.tspulse_args import get_tspulse_args

tsfm_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "granite-tsfm")
)
if tsfm_project_root not in sys.path:
    sys.path.insert(0, tsfm_project_root)

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tspulse.configuration_tspulse import TSPulseConfig
from tsfm_public.models.tspulse.modeling_tspulse import \
    TSPulseForReconstruction
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.util import count_parameters
from tsfm_public.toolkit.visualization import plot_predictions


def _parse_freq_str_for_polars_duration(freq_str: str) -> dict:
    """
    Parses a pandas-style frequency string (e.g., '5T', '1H', 'D') and returns a
    kwargs dictionary suitable for the `polars.duration()` expression.
    """
    if not freq_str or not isinstance(freq_str, str):
        return {"days": 1}  # Default case

    freq_str = freq_str.upper()

    # Isolate the numeric part and the unit part
    numeric_part = "".join(filter(str.isdigit, freq_str))
    unit_part = "".join(filter(str.isalpha, freq_str))

    value = int(numeric_part) if numeric_part else 1

    if unit_part in ("T", "MIN"):
        return {"minutes": value}
    elif unit_part == "H":
        return {"hours": value}
    elif unit_part == "D":
        return {"days": value}
    elif unit_part == "S":
        return {"seconds": value}
    elif unit_part == "W":
        return {"weeks": value}
    else:
        logger.warning(f"Unsupported frequency unit '{unit_part}', defaulting to days.")
        return {"days": value}


def convert_arrow_to_long_polars(
    arrow_dataset: datasets.Dataset, dataset_name: str
) -> datasets.Dataset:
    """
    Converts a Hugging Face Arrow Dataset to a standardized long format using
    a fast, vectorized Polars operation. This version robustly handles datasets
    that need their timestamp range to be generated from 'start' and 'freq' columns,
    and also correctly unnests multivariate series.
    """
    logger.info(
        f"  - Vectorized conversion of Arrow data for '[bold]{dataset_name}[/bold]' using Polars..."
    )

    pl_df = pl.from_arrow(arrow_dataset.data.table)

    # Path for datasets that are "wide" and need timestamp generation
    if "target" in pl_df.columns and isinstance(pl_df["target"][0], (list, pl.Series)):
        logger.info(
            "    -> Detected nested 'target' column. Performing high-speed 'explode' operation."
        )

        if "item_id" not in pl_df.columns:
            pl_df = pl_df.with_columns(pl.lit(dataset_name).alias("item_id"))

        freq_str = "1d"
        if "freq" in pl_df.columns and len(pl_df) > 0 and pl_df["freq"][0] is not None:
            freq_str = pl_df["freq"][0]

        duration_kwargs = _parse_freq_str_for_polars_duration(freq_str)
        duration_expr = pl.duration(**duration_kwargs)

        long_pl_df = (
            pl_df.lazy()
            .explode("target")
            .with_columns(pl.int_range(0, pl.len()).over("item_id").alias("time_idx"))
            .with_columns(
                (pl.col("start") + pl.col("time_idx") * duration_expr).alias(
                    "timestamps"
                )
            )
            .rename({"item_id": "series_name", "target": "value"})
            .select(["series_name", "timestamps", "value"])
            .collect(engine="streaming")
        )
    # Path for datasets that are already somewhat "long"
    else:
        logger.info(
            "    -> Data appears to be in long format. Standardizing column names."
        )
        rename_map = {}
        if "timestamp" in pl_df.columns and "timestamps" not in pl_df.columns:
            rename_map["timestamp"] = "timestamps"
        if "item_id" in pl_df.columns and "series_name" not in pl_df.columns:
            rename_map["item_id"] = "series_name"
        if "target" in pl_df.columns and "value" not in pl_df.columns:
            rename_map["target"] = "value"
        long_pl_df = pl_df.rename(rename_map)

        required_cols = ["series_name", "timestamps", "value"]
        missing_cols = [col for col in required_cols if col not in long_pl_df.columns]
        if missing_cols:
            from polars.exceptions import ColumnNotFoundError

            raise ColumnNotFoundError(
                f"The following required columns are missing after processing: {missing_cols}"
            )

        long_pl_df = long_pl_df.select(required_cols)

    if isinstance(long_pl_df["value"].dtype, pl.List):
        logger.info(
            "    -> Detected multivariate series (List type). Exploding to long format."
        )
        long_pl_df = (
            long_pl_df.with_columns(
                dim_index=pl.col("value").list.eval(pl.int_range(0, pl.len()))
            )
            .explode(["value", "dim_index"])
            .with_columns(
                series_name=(
                    pl.col("series_name").cast(pl.Utf8)
                    + "_dim_"
                    + pl.col("dim_index").cast(pl.Utf8)
                )
            )
            .drop("dim_index")
        )

    logger.info(
        f"    -> Conversion complete. Resulting in [bold magenta]{len(long_pl_df):,}[/bold magenta] rows."
    )

    return datasets.Dataset.from_dict(long_pl_df.to_dict(as_series=False))


def _get_common_args_parser():
    """Gets a parser with arguments common to all backends."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--drs_backend",
        type=str,
        default="polars",
        choices=["pandas", "polars", "cudf"],
        help="The backend to use for DRS processing. 'pandas' is the slow baseline, "
        "'polars' is a fast multi-core CPU backend, and 'cudf' is a GPU backend.",
    )
    return parser


def apply_drs_pandas(
    hf_dataset: datasets.Dataset, drs_factors: list, dataset_name: str
):
    """Original pandas-based DRS implementation with detailed logging."""
    logger.info(
        f"Applying DRS to '[bold]{dataset_name}[/bold]' with factors: {drs_factors} using [red]pandas[/red]."
    )
    df = hf_dataset.to_pandas().sort_values(["series_name", "timestamps"])
    all_resampled_chunks = []

    for k in drs_factors:
        logger.info(
            f"  - Processing DRS factor k=[bold red]{k}[/bold red]..."
        )
        if len(df) < k:
            logger.warning(f"    Dataset is shorter than k={k}, skipping.")
            continue

        resampled_chunks_for_k = []
        for name, group in tqdm(
            df.groupby("series_name"), desc=f"Pandas DRS (k={k})", leave=False
        ):
            if len(group) < k:
                continue
            resampled_group = group.copy()
            resampled_group["value"] = (
                resampled_group["value"].rolling(window=k, min_periods=k).mean()
            )
            resampled_group = resampled_group.iloc[k - 1 :: k].copy()
            resampled_group["series_name"] = f"{name}_drs_{k}"
            resampled_chunks_for_k.append(resampled_group)

        if resampled_chunks_for_k:
            k_df = pd.concat(resampled_chunks_for_k)
            logger.info(
                f"    -> Generated [bold magenta]{len(k_df):,}[/bold magenta] rows for k={k}."
            )
            all_resampled_chunks.append(k_df)

    if not all_resampled_chunks:
        logger.warning(
            f"    No DRS data generated for '[yellow]{dataset_name}[/yellow]'"
        )
        return None

    final_drs_df = pd.concat(all_resampled_chunks, ignore_index=True)
    logger.info(
        f"  -> Generated a total of [bold magenta]{len(final_drs_df):,}[/bold magenta] DRS rows."
    )
    return datasets.Dataset.from_pandas(final_drs_df)


def apply_drs_polars(
    hf_dataset: datasets.Dataset, drs_factors: list, dataset_name: str
):
    """
    High-performance Polars (multi-core CPU) DRS implementation with intelligent handling of short series.

    - For series with length >= k: Applies a rolling mean and keeps all valid windows.
    - For series with length < k: Computes the mean of the entire series, resulting in a single data point.
    """
    logger.info(f"Applying DRS to '[bold]{dataset_name}[/bold]' with factors: {drs_factors} using [cyan]Polars (Advanced Logic)[/cyan].")
    try:
        pl_df = pl.from_arrow(hf_dataset.data.table)
    except Exception:
        pl_df = pl.from_pandas(hf_dataset.to_pandas())
        
    pl_df = pl_df.with_columns(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    
    pl_df = pl_df.sort("series_name", "timestamps")
    all_resampled_dfs = []

    for k in drs_factors:
        logger.info(f"  - Processing DRS factor k=[bold cyan]{k}[/bold cyan]...")

        series_lengths = pl_df.group_by("series_name", maintain_order=True).len()
        short_series_names = series_lengths.filter(pl.col("len") < k)["series_name"]
        long_series_names = series_lengths.filter(pl.col("len") >= k)["series_name"]

        num_long_series = len(long_series_names)
        num_short_series = len(short_series_names)
        logger.info(f"    -> Partitioning: [bold cyan]{num_long_series:,}[/bold cyan] series (L>={k}) | [bold yellow]{num_short_series:,}[/bold yellow] series (L<{k}).")

        k_results = []

        if num_long_series > 0:
            df_long = pl_df.filter(pl.col("series_name").is_in(long_series_names.to_list()))
            resampled_long = (
                df_long.lazy()
                .with_columns(
                    pl.col("value")
                    .rolling_mean(window_size=k, min_samples=k)
                    .over("series_name")
                    .alias("value_resampled")
                )
                .filter(pl.col("value_resampled").is_not_null())
                .select(["series_name", "timestamps", pl.col("value_resampled").alias("value")])
                .collect(engine='streaming')
            )
            if len(resampled_long) > 0:
                logger.info(f"    -> Generated [bold magenta]{len(resampled_long):,}[/bold magenta] rows from the 'long' partition.")
                k_results.append(resampled_long)

        if num_short_series > 0:
            df_short = pl_df.filter(pl.col("series_name").is_in(short_series_names.to_list()))
            resampled_short = (
                df_short.lazy()
                .group_by("series_name")
                .agg(
                    pl.mean("value").alias("value"),
                    pl.last("timestamps").alias("timestamps")
                )
                .select(["series_name", "timestamps", "value"])
                .collect(engine='streaming')
            )
            if len(resampled_short) > 0:
                logger.info(f"    -> Generated [bold magenta]{len(resampled_short):,}[/bold magenta] rows from the 'short' partition (as single-point averages).")
                k_results.append(resampled_short)

        if k_results:
            combined_k_df = pl.concat(k_results)
            final_k_df = combined_k_df.with_columns(
                pl.col("series_name") + f"_drs_{k}"
            ).select(["series_name", "timestamps", "value"])
            all_resampled_dfs.append(final_k_df)
        else:
             logger.warning(f"    -> Generated 0 rows for k={k}. This can happen if the input dataset for this factor was empty.")

    if not all_resampled_dfs:
        logger.warning(f"    No DRS data generated for '[yellow]{dataset_name}[/yellow]'")
        return None

    final_drs_pl_df = pl.concat(all_resampled_dfs)
    logger.info(f"  -> Generated a total of [bold magenta]{len(final_drs_pl_df):,}[/bold magenta] DRS rows across all factors.")

    return datasets.Dataset.from_dict(final_drs_pl_df.to_dict(as_series=False))


# def apply_drs_cudf(hf_dataset: datasets.Dataset, drs_factors: list, dataset_name: str):
#     """GPU-accelerated cuDF DRS implementation with detailed logging."""
#     logger.info(f"Applying DRS to '[bold]{dataset_name}[/bold]' with factors: {drs_factors} using [green]cuDF (GPU)[/green].")
#     gdf = cudf.from_pandas(hf_dataset.to_pandas()).sort_values(["series_name", "timestamps"])
#     all_resampled_gdfs = []

#     for k in drs_factors:
#         logger.info(f"  - Processing DRS factor k=[bold green]{k}[/bold green] on GPU...") # Announce the start
#         if len(gdf) < k:
#             logger.warning(f"    Dataset is shorter than k={k}, skipping.")
#             continue

#         # Correctly perform rolling mean and downsampling in cuDF
#         rolling_mean_col = gdf.groupby('series_name')['value'].rolling(k, min_periods=k).mean().reset_index(drop=True)
#         gdf['value_resampled'] = rolling_mean_col
#         gdf['group_idx'] = gdf.groupby('series_name').cumcount()

#         resampled_gdf = gdf[gdf['group_idx'] % k == (k - 1)].copy()
#         resampled_gdf = resampled_gdf.dropna(subset=['value_resampled'])

#         if len(resampled_gdf) > 0:
#             resampled_gdf['series_name'] = resampled_gdf['series_name'] + f"_drs_{k}"
#             resampled_gdf['value'] = resampled_gdf['value_resampled']

#             logger.info(f"    -> Generated [bold magenta]{len(resampled_gdf):,}[/bold magenta] rows for k={k}.")
#             all_resampled_gdfs.append(resampled_gdf[['series_name', 'timestamps', 'value']])

#     if 'value_resampled' in gdf.columns:
#         del gdf['value_resampled']
#     if 'group_idx' in gdf.columns:
#         del gdf['group_idx']

#     if not all_resampled_gdfs:
#         logger.warning(f"    No DRS data generated for '[yellow]{dataset_name}[/yellow]'")
#         return None

#     final_drs_gdf = cudf.concat(all_resampled_gdfs, ignore_index=True)
#     logger.info(f"  -> Generated a total of [bold magenta]{len(final_drs_gdf):,}[/bold magenta] DRS rows.")
#     return datasets.Dataset.from_pandas(final_drs_gdf.to_pandas())


def get_base_model(args):
    """This function defines the TSPulse model architecture based on provided arguments."""
    config = TSPulseConfig(
        context_length=args.context_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_layers=args.num_layers,
        decoder_d_model=args.decoder_d_model,
        decoder_num_layers=args.decoder_num_layers,
        patch_register_tokens=args.patch_register_tokens,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
        loss_apply_mode="mask_and_full",
        fuse_fft=args.fuse_fft,
        fft_weight=args.fft_weight,
        fft_original_signal_loss_weight=args.fft_original_signal_loss_weight,
        enable_fft_prob_loss=args.enable_fft_prob_loss,
        fft_prob_weight=args.fft_prob_weight,
        fft_time_add_forecasting_pt_loss=args.fft_time_add_forecasting_pt_loss,
        prediction_length=args.prediction_length,
        fft_time_add_forecasting_pt_loss_weight=args.fft_time_add_forecasting_pt_loss_weight,
        scaling=args.scaling,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        mode="common_channel",
        decoder_mode="common_channel",
    )
    return TSPulseForReconstruction(config)


def pretrain(args, model, dset_train, dset_val):
    """Handles the training process."""
    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoint"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        seed=args.random_seed,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(args.save_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(
        len(dset_train) / (args.batch_size * trainer_args.world_size)
    )
    scheduler = OneCycleLR(
        optimizer,
        args.learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    callbacks = (
        [EarlyStoppingCallback(early_stopping_patience=10)]
        if args.early_stopping
        else []
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
    )
    trainer.train()
    model_save_path = os.path.join(args.save_dir, "tspulse_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def inference(args, model_path, dset_test):
    """Handles the inference and evaluation process."""
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
    logger.info("[bold green]Test MSE output:[/bold green]")
    output = trainer.evaluate(dset_test)
    logger.info(output)
    predictions_dict = trainer.predict(dset_test)
    logger.info(
        f"Predictions shape: [bold cyan]{predictions_dict.predictions[0].shape}[/bold cyan]"
    )
    logger.info(
        f"Backbone embeddings shape: [bold cyan]{predictions_dict.predictions[1].shape}[/bold cyan]"
    )
    plot_path = os.path.join(args.save_dir, "plots")
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="test_inference",
        channel=0,
    )
    logger.info(f"Plots saved in location: [underline]{plot_path}[/underline]")


def load_tsf_to_hf_dataset_fast(tsf_path: str, dataset_name: str) -> datasets.Dataset:
    """
    A high-performance parser that reads a .tsf file and directly outputs
    a clean, long-format Hugging Face Dataset using Polars. Replaces sktime.
    This version is robust to complex TSF headers.
    """
    logger.info(f"  - Fast-parsing TSF file for '[bold]{dataset_name}[/bold]' with Polars...")
    header_info = {}
    
    rows_to_skip = 0
    with open(tsf_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().lower().startswith("@data"):
                break
            else:
                if line.strip().startswith("@"):
                    parts = line.strip().split(" ", 1)
                    key = parts[0][1:]
                    value = parts[1] if len(parts) > 1 else ""
                    header_info[key] = value
                rows_to_skip += 1
    
    is_multivariate = header_info.get("univariate", "false").lower() != "true"
    
    df = (
        pl.read_csv(
            tsf_path,
            has_header=False,
            separator=",",
            skip_rows=rows_to_skip + 1,
            infer_schema_length=None,
        )
        .with_row_index("index")
    )

    series_data_cols = df.columns[1:-1]

    long_df = df.unpivot(
        index="index",
        on=series_data_cols,
        variable_name="time_step_str",
        value_name="value"
    ).filter(pl.col("value").is_not_null())

    final_df = long_df.with_columns(
        timestamps=pl.col("time_step_str").str.extract(r"(\d+)").cast(pl.Int64)
    )

    if is_multivariate:
        final_df = (
            final_df
            .with_columns(pl.col("value").str.split(":").alias("value"))
            .with_columns(
                dim_index=pl.col("value").list.eval(pl.int_range(0, pl.len()))
            )
            .explode(["value", "dim_index"])
            .with_columns(
                series_name=pl.lit(f"{dataset_name}_") + pl.col("index").cast(pl.Utf8) + "_dim_" + pl.col("dim_index").cast(pl.Utf8),
                value=pl.col("value").cast(pl.Float64, strict=False)
            )
            .drop("dim_index")
        )
    else:
        final_df = final_df.with_columns(
            series_name=pl.lit(f"{dataset_name}_") + pl.col("index").cast(pl.Utf8),
            value=pl.col("value").cast(pl.Float64, strict=False)
        )

    final_df = final_df.select("series_name", "timestamps", "value").filter(pl.col("value").is_not_null())

    logger.info(f"    -> Parsing complete. Resulting in [bold magenta]{len(final_df):,}[/bold magenta] rows.")
    return datasets.Dataset.from_dict(final_df.to_dict(as_series=False))


if __name__ == "__main__":
    parent_parser = _get_common_args_parser()
    args = get_tspulse_args(parent_parser)

    args_dict = vars(args)
    pretty_args_str = pprint.pformat(args_dict, indent=2)
    args_panel = Panel(
        pretty_args_str,
        title="[bold cyan]Run Configuration[/bold cyan]",
        subtitle="[dim]Script Arguments[/dim]",
        border_style="green",
    )
    rprint(args_panel)

    set_seed(args.random_seed)
    logger.info(
        f"[bold] {'*' * 20} Pre-training TSPulse (Efficient Loader) {'*' * 20} [/bold]"
    )

    DRS_CONFIG = {
        "wind_4_seconds": [150, 225, 450, 900],
        "wind_farms_minutely": [10, 15, 30, 60],
        "solar_10_minutes": [3, 6],
        "australian_electricity_demand": [2, 48],
        "solar_4_seconds": [150, 225, 450, 900],
        "london_smart_meters": [2, 48],
        "pems": [2, 3, 6, 12],
        "PEMS": [2, 3, 6, 12],
        "LOS_LOOP": [2, 3, 6, 12],
    }

    DRS_BACKEND_MAP = {
        "pandas": apply_drs_pandas,
        "polars": apply_drs_polars,
        # "cudf": apply_drs_cudf,
    }
    selected_backend_name = args.drs_backend
    # if selected_backend_name == "cudf" and not IS_CUDA_AVAILABLE:
    #     logger.warning("[yellow]cuDF backend was selected, but cuDF is not available. Falling back to Polars.[/yellow]")
    #     selected_backend_name = "polars"
    apply_drs_func = DRS_BACKEND_MAP[selected_backend_name]
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    cache_dir = os.path.join(datasets_dir, "incremental_cache")
    if args.use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using memory-mapped cache directory: [cyan]{cache_dir}[/cyan]")

    logger.info("Starting data loading and processing...")
    dataset_names = [
        d
        for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d)) and d != "incremental_cache"
    ]
    all_processed_datasets = []

    for dataset_name in dataset_names:
        individual_cache_path = os.path.join(cache_dir, dataset_name)
        cache_exists_check_path = os.path.join(
            individual_cache_path, "dataset_info.json"
        )
        if args.use_cache and os.path.exists(cache_exists_check_path):
            logger.info(
                f"Cache hit for '[bold]{dataset_name}[/bold]'. Loading from [cyan]{individual_cache_path}[/cyan]"
            )
            try:
                loaded_dataset = datasets.load_from_disk(individual_cache_path)
                all_processed_datasets.append(loaded_dataset)
                continue
            except Exception as e:
                logger.warning(
                    f"Failed to load cache for [yellow]{dataset_name}[/yellow]. Reprocessing. Error: {e}"
                )
        logger.info(
            f"Cache miss for '[bold]{dataset_name}[/bold]'. Starting full processing."
        )

        datasets_for_current_source = []
        dataset_path = os.path.join(datasets_dir, dataset_name)
        tsf_path = os.path.join(dataset_path, f"{dataset_name}.tsf")
        current_hf_dataset = None

        if os.path.exists(tsf_path):
            try:
                current_hf_dataset = load_tsf_to_hf_dataset_fast(tsf_path, dataset_name)
            except Exception as e:
                logger.error(
                    f"Failed to process TSF dataset [red]{dataset_name}[/red].",
                    exc_info=True,
                )
        elif os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            try:
                logger.info(
                    f"  -> Found original HF dataset for '[bold]{dataset_name}[/bold]'. Loading and converting..."
                )
                hf_dataset_from_disk = datasets.load_from_disk(dataset_path)
                dataset_to_process = (
                    hf_dataset_from_disk.get("train")
                    or hf_dataset_from_disk.get("test")
                    or next(iter(hf_dataset_from_disk.values()), None)
                    if isinstance(hf_dataset_from_disk, datasets.DatasetDict)
                    else hf_dataset_from_disk
                )
                if dataset_to_process:
                    current_hf_dataset = convert_arrow_to_long_polars(
                        arrow_dataset=dataset_to_process, dataset_name=dataset_name
                    )
                else:
                    logger.error(
                        f"Could not find a valid split in Arrow dataset [red]{dataset_name}[/red]."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to load original Arrow dataset [red]{dataset_name}[/red].",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"No loadable data found for [yellow]{dataset_name}[/yellow]."
            )
            continue

        if current_hf_dataset and len(current_hf_dataset) > 0:
            datasets_for_current_source.append(current_hf_dataset)
            drs_factors = next(
                (factors for key, factors in DRS_CONFIG.items() if key in dataset_name),
                None,
            )
            if drs_factors:
                start_time = time.perf_counter()
                drs_dataset = apply_drs_func(
                    hf_dataset=current_hf_dataset,
                    drs_factors=drs_factors,
                    dataset_name=dataset_name,
                )
                end_time = time.perf_counter()
                duration = end_time - start_time
                logger.info(
                    f"DRS processing with [bold]{selected_backend_name}[/bold] for '{dataset_name}' took [bold yellow]{duration:.4f}s[/bold yellow]"
                )
                if drs_dataset:
                    datasets_for_current_source.append(drs_dataset)
            consolidated_hf_dataset = datasets.concatenate_datasets(
                datasets_for_current_source
            )
            all_processed_datasets.append(consolidated_hf_dataset)
            if args.use_cache:
                logger.info(
                    f"Saving processed data for '[bold]{dataset_name}[/bold]' to cache: [cyan]{individual_cache_path}[/cyan]"
                )
                consolidated_hf_dataset.save_to_disk(individual_cache_path)
        else:
            logger.warning(
                f"No loadable data found or processed for [yellow]{dataset_name}[/yellow], skipping."
            )
    if not all_processed_datasets:
        logger.error(
            "[bold red]No datasets were loaded or processed. Exiting.[/bold red]"
        )
        sys.exit(1)

    logger.info(
        f"Concatenating data from [bold cyan]{len(all_processed_datasets)}[/bold cyan] processed sources..."
    )
    full_hf_dataset = datasets.concatenate_datasets(all_processed_datasets)
    logger.info(
        f"Final dataset has [bold magenta]{len(full_hf_dataset)}[/bold magenta] total rows."
    )

    timestamp_column = "timestamps"
    id_columns = ["series_name"]
    target_columns = ["value"]
    split_config = {"train": [0, 0.7], "valid": [0.7, 0.85], "test": [0.85, 1.0]}
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
    logger.info("Preparing datasets for training...")
    dset_train, dset_valid, dset_test = get_datasets(tsp, full_hf_dataset, split_config)
    logger.info(
        f"Train dataset size: [bold cyan]{len(dset_train)}[/bold cyan], Valid: [bold cyan]{len(dset_valid)}[/bold cyan], Test: [bold cyan]{len(dset_test)}[/bold cyan]"
    )
    model = get_base_model(args)
    logger.info(
        f"Model has [bold yellow]{count_parameters(model)/1e6:.2f}M[/bold yellow] parameters"
    )
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    logger.info("[bold green on black] Pretraining Completed! [/bold green on black]")
    logger.info(f"Model saved in location: [underline]{model_save_path}[/underline]")
    inference(args=args, model_path=model_save_path, dset_test=dset_test)
    logger.info("[bold green on black] Inference completed. [/bold green on black]")
