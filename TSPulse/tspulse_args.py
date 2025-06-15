# Copyright contributors to the TSFM project
#
"""Utilities for TSPULSE notebooks"""

import argparse
import logging
import os
import tempfile

import torch


logger = logging.getLogger(__name__)


def get_tspulse_args(parser):  # pragma: no cover
    # parser = argparse.ArgumentParser(description="TSPulse pretrain arguments.")

    # Core Architecture
    parser.add_argument(
        "--context_length", "-cl", type=int, default=512, help="History context length"
    )
    parser.add_argument(
        "--prediction_length",
        "-fl",
        type=int,
        default=24,
        help="Forecast length for aux task",
    )
    parser.add_argument("--patch_length", "-pl", type=int, default=8, help="Patch length")
    parser.add_argument(
        "--patch_stride", "-ps", type=int, default=8, help="Patch stride"
    )
    parser.add_argument(
        "--d_model_scale",
        "-dms",
        type=int,
        default=3,
        help="Scale for d_model (d_model = patch_length * d_model_scale)",
    )
    parser.add_argument(
        "--num_layers", "-nl", type=int, default=8, help="Number of encoder layers"
    )
    parser.add_argument(
        "--decoder_d_model_scale",
        "-ddms",
        type=int,
        default=3,
        help="Scale for decoder d_model",
    )
    parser.add_argument(
        "--decoder_num_layers", "-dnl", type=int, default=2, help="Number of decoder layers"
    )
    parser.add_argument(
        "--patch_register_tokens",
        "-prt",
        type=int,
        default=8,
        help="Number of patch register tokens",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--head_dropout", type=float, default=0.2, help="Head dropout rate"
    )
    parser.add_argument(
        "--scaling", type=str, default="revin", help="Scaling method (e.g., 'revin', 'standard')"
    )

    # Self-Supervised Task: Masking
    parser.add_argument(
        "--mask_type", type=str, default="var_hybrid", help="Masking type"
    )
    parser.add_argument("--mask_ratio", type=float, default=0.4, help="Masking ratio")

    # Dual-Space (FFT) Learning
    parser.add_argument(
        "--fuse_fft",
        type=int,
        default=1,
        help="Enable FFT fusion (0 for False, 1 for True)",
    )
    parser.add_argument(
        "--fft_weight", type=float, default=1.0, help="Weight for FFT reconstruction loss"
    )
    parser.add_argument(
        "--fft_original_signal_loss_weight",
        type=float,
        default=1.0,
        help="Weight for original signal loss in FFT path",
    )
    parser.add_argument(
        "--enable_fft_prob_loss",
        type=int,
        default=1,
        help="Enable FFT probability loss (0 for False, 1 for True)",
    )
    parser.add_argument(
        "--fft_prob_weight",
        type=float,
        default=1.0,
        help="Weight for FFT probability loss",
    )

    # Auxiliary Forecasting Task
    parser.add_argument(
        "--fft_time_add_forecasting_pt_loss",
        type=int,
        default=1,
        help="Enable auxiliary forecasting task (0 for False, 1 for True)",
    )
    parser.add_argument(
        "--fft_time_add_forecasting_pt_loss_weight",
        type=float,
        default=1.0,
        help="Weight for auxiliary forecasting loss",
    )

    # Training
    parser.add_argument(
        "--num_gpus", "-ng", type=int, default=None, help="Number of GPUs"
    )
    parser.add_argument(
        "--random_seed", "-rs", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=256, help="Batch size per device"
    )
    parser.add_argument(
        "--num_epochs", "-ne", type=int, default=25, help="Number of epochs"
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--early_stopping",
        "-es",
        type=int,
        default=1,
        help="Whether to use early stopping (0 or 1)",
    )
    parser.add_argument(
        "--save_dir",
        "-sd",
        type=str,
        default=tempfile.gettempdir(),
        help="Directory to save model",
    )

    # Data
    parser.add_argument(
        "--data_root_path",
        "-drp",
        type=str,
        default="datasets/",
        help="Root path for datasets",
    )

    parser.add_argument(
        "--use_cache",
        action="store_false",
        help="If set, caches the pre-processed data to avoid re-loading from source files on subsequent runs.",
    )

    args = parser.parse_args()

    # Post-process arguments
    args.fuse_fft = int_to_bool(args.fuse_fft)
    args.enable_fft_prob_loss = int_to_bool(args.enable_fft_prob_loss)
    args.fft_time_add_forecasting_pt_loss = int_to_bool(
        args.fft_time_add_forecasting_pt_loss
    )
    args.early_stopping = int_to_bool(args.early_stopping)

    args.d_model = args.patch_length * args.d_model_scale
    args.decoder_d_model = args.patch_length * args.decoder_d_model_scale

    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
        logger.info(f"Automatically determined number of GPUs: {args.num_gpus}")

    args.save_dir = os.path.join(
        args.save_dir,
        f"TSPulse_cl-{args.context_length}_pl-{args.patch_length}_ne-{args.num_epochs}",
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # For compatibility with pretrain script
    args.forecast_length = args.prediction_length

    return args


def int_to_bool(value):  # pragma: no cover
    if value == 0:
        return False
    elif value == 1:
        return True
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (0 or 1)")

