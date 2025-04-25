# coding: utf-8

from __future__ import division, print_function

import math
import os
import pickle
import random
import time
import traceback  # Added for padding
import warnings  # Added for filtering warnings

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
# ****** Added MinMaxScaler import ******
from sklearn.preprocessing import MinMaxScaler
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes # No longer needed for inset plot
from torch import nn
# **************************************
from torch.utils.data import DataLoader, Dataset  # Added Dataset

# Assuming these are in the same project structure or copied
# Need to ensure these imports work in the execution environment
try:
    from TSB_AD.evaluation.basic_metrics import basic_metricor
except ImportError:
    print("Warning: TSB_AD.evaluation.basic_metrics not found. Extended label analysis might be affected.")
    basic_metricor = None
try:
    from TSB_AD.evaluation.metrics import get_metrics
except ImportError:
    print("Warning: TSB_AD.evaluation.metrics not found. Metrics calculation will be skipped.")
    get_metrics = None
try:
    from TSB_AD.models.base import BaseDetector
except ImportError:
    print("Warning: TSB_AD.models.base not found. Custom_AD class might not inherit correctly if BaseDetector is expected.")
    # Define a dummy BaseDetector if needed

    class BaseDetector:
        def __init__(self, *args, **kwargs): pass
        def fit(self, data): pass
        def decision_function(self, data): pass
try:
    from TSB_AD.utils.slidingWindows import find_length_rank
except ImportError:
    print("Warning: TSB_AD.utils.slidingWindows not found. find_length_rank will not be available.")
    # Define a dummy function if needed
    def find_length_rank(data, rank=1): return 100  # Default fallback
try:
    from TSB_AD.utils.torch_utility import EarlyStoppingTorch, get_gpu
except ImportError:
    print("Warning: TSB_AD.utils.torch_utility not found. GPU/EarlyStopping support might be affected.")
    # Define dummy functions/classes if needed

    def get_gpu(use_cuda=True): return torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    class EarlyStoppingTorch:
        def __init__(self, patience=5, verbose=False):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.best_state_dict = None  # Store best model state

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score:
                self.counter += 1
                if self.verbose:
                    print(
                        f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            # Use deepcopy to ensure state_dict is independent
            self.best_state_dict = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
            self.val_loss_min = val_loss


# Assuming ReconstructDataset is defined correctly elsewhere,
# if not, define a basic one here.
try:
    from TSB_AD.utils.dataset import ReconstructDataset
except ImportError:
    print("Warning: TSB_AD.utils.dataset not found. Using basic fallback ReconstructDataset.")

    class ReconstructDataset(Dataset):
        def __init__(self, data, window_size):
            # Convert data to float32 tensor
            self.data = torch.from_numpy(data.astype(np.float32)).float()
            self.window_size = window_size

        def __len__(self):
            # Calculate the number of possible windows
            return max(0, len(self.data) - self.window_size + 1)

        def __getitem__(self, index):
            # Extract the window
            window = self.data[index:index + self.window_size]
            # Return window and itself (target for reconstruction)
            return window.float(), window.float()


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# --- Helper: _pad_array (Copied from Custom_AD context for use here) ---
def _pad_array(data_unpadded, n_samples_original, sliding_window):
    # Input validation
    if not isinstance(data_unpadded, np.ndarray):
        try:
            data_unpadded = np.array(data_unpadded)
        except:
            # Return shape based on target length and inferred dimensions
            ndim = data_unpadded.ndim if hasattr(data_unpadded, 'ndim') else 1
            # Adjust shape calculation for potentially missing second dimension info
            feat_dim = 1
            if ndim > 1 and hasattr(data_unpadded, 'shape') and len(data_unpadded.shape) > 1:
                feat_dim = data_unpadded.shape[1] if data_unpadded.shape[1] > 0 else 1
            shape = (n_samples_original,) if ndim == 1 else (
                n_samples_original, feat_dim)

            print(
                f"Warning (_pad_array): Input conversion failed. Returning zeros with shape {shape}.")
            return np.zeros(shape)

    if data_unpadded.ndim == 0:  # Handle scalar input gracefully
        # Convert scalar to 1-element array
        data_unpadded = data_unpadded.reshape(1)

    # Determine target shape based on original samples and feature dimension (if any)
    target_shape = (n_samples_original,)
    feat_dim = 1
    if data_unpadded.ndim == 2:
        feat_dim = data_unpadded.shape[1]
        if feat_dim == 0:  # Handle case where input might be (N, 0)
            # print("Warning (_pad_array): Input array has 0 features. Padding as 1D.")
            target_shape = (n_samples_original,)
        else:
            target_shape = (n_samples_original, feat_dim)
    elif data_unpadded.ndim > 2:
        print(
            f"Warning (_pad_array): Input array has >2 dimensions ({data_unpadded.ndim}). Padding only first axis.")
        target_shape = (n_samples_original,) + data_unpadded.shape[1:]

    if n_samples_original <= 0:
        # Return empty with correct shape
        return np.array([]).reshape(target_shape)

    n_unpadded = data_unpadded.shape[0]

    if n_unpadded == 0:
        # Return zeros with the correct target shape
        return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)

    # Ensure sliding window is a positive integer
    sliding_window = max(1, int(sliding_window)) if isinstance(
        sliding_window, (int, float)) else 1

    # --- Handle length matching ---
    if n_unpadded == n_samples_original:
        # Ensure shape matches exactly, reshape if necessary and possible
        if data_unpadded.shape == target_shape:
            return data_unpadded.copy()
        else:
            try:
                # print(f"Info (_pad_array): Length matches but shape mismatch ({data_unpadded.shape} vs {target_shape}). Reshaping.")
                return data_unpadded.reshape(target_shape).copy()
            except ValueError:
                print(
                    f"Warning (_pad_array): Length matches but shape mismatch ({data_unpadded.shape} vs {target_shape}). Cannot reshape. Returning zeros.")
                return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)

    elif n_unpadded > n_samples_original:
        # print(f"Warning in _pad_array: Unpadded data ({n_unpadded}) longer than original ({n_samples_original}). Truncating.")
        truncated_data = data_unpadded[:n_samples_original]
        # Ensure shape matches after truncation
        if truncated_data.shape == target_shape:
            return truncated_data.copy()
        else:
            try:
                # print(f"Info (_pad_array): Truncated shape mismatch ({truncated_data.shape} vs {target_shape}). Reshaping.")
                return truncated_data.reshape(target_shape).copy()
            except ValueError:
                print(
                    f"Warning (_pad_array): Truncated shape mismatch ({truncated_data.shape} vs {target_shape}). Cannot reshape. Returning zeros.")
                return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)
    # else: n_unpadded < n_samples_original, so padding is needed

    # --- Calculate padding amounts (Center-biased padding) ---
    if sliding_window <= 1:
        pad_before = 0
        pad_after = n_samples_original - n_unpadded
    else:
        # Points before the start of scores
        pad_before = math.ceil((sliding_window - 1) / 2)
        # Ensure pad_before is within valid range
        pad_before = min(pad_before, n_samples_original - 1)
        # Calculate remaining padding needed after the scores end
        pad_after = n_samples_original - n_unpadded - pad_before
        # Ensure pad_after is non-negative
        pad_after = max(0, pad_after)

    # Check if calculated padding matches total required padding
    if pad_before + pad_after != (n_samples_original - n_unpadded):
        # Fallback if calculation seems off (shouldn't happen with logic above)
        # print(f"Warning (_pad_array): Padding calculation mismatch ({pad_before}+{pad_after} != {n_samples_original - n_unpadded}). Using fallback.")
        pad_before = (sliding_window - 1) // 2
        pad_after = n_samples_original - n_unpadded - pad_before
        pad_before = max(0, pad_before)  # Ensure non-negative
        pad_after = max(0, pad_after)   # Ensure non-negative
        # Recalculate if still doesn't match (e.g., if n_samples_original < sliding_window)
        if pad_before + pad_after != (n_samples_original - n_unpadded):
            # print(f"Warning (_pad_array): Fallback padding calculation mismatch. Defaulting padding.")
            pad_before = max(0, (n_samples_original - n_unpadded)) // 2
            pad_after = max(0, n_samples_original - n_unpadded - pad_before)

    # --- Perform padding using np.pad ---
    try:
        # Define padding widths for each dimension.
        # Pad only the first axis (axis 0). No padding for other axes.
        pad_width = [(pad_before, pad_after)] + \
            [(0, 0)] * (data_unpadded.ndim - 1)

        # Use 'edge' mode to repeat the first/last value (row for 2D)
        padded = np.pad(data_unpadded, pad_width, mode='edge')

    except Exception as e:
        print(
            f"Error during np.pad in _pad_array: {e}. Returning zeros with target shape {target_shape}.")
        traceback.print_exc()  # Print traceback for debugging np.pad issues
        return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)

    # --- Final length verification (np.pad should guarantee this) ---
    if padded.shape[0] != n_samples_original:
        print(
            f"CRITICAL Warning in _pad_array: Final padded length {padded.shape[0]} != target {n_samples_original} despite using np.pad. Resizing forcefully.")
        # Force resize if something went extremely wrong
        if padded.shape[0] < n_samples_original:
            # Re-pad to fix size
            extra_pad = n_samples_original - padded.shape[0]
            pad_width_fix = [(0, extra_pad)] + [(0, 0)] * (padded.ndim - 1)
            try:
                padded = np.pad(padded, pad_width_fix, mode='edge')
            except Exception as e_fix:
                print(
                    f"Error during forceful resize padding: {e_fix}. Returning zeros.")
                return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)

        else:  # Too long
            padded = padded[:n_samples_original]

    # Final shape check
    if padded.shape != target_shape:
        print(
            f"CRITICAL Warning in _pad_array: Final padded shape {padded.shape} != target {target_shape}. Attempting reshape.")
        try:
            # Attempt reshape as last resort
            padded = padded.reshape(target_shape)
        except ValueError:
            print(
                f"Error: Cannot reshape padded array to target shape {target_shape}. Returning zeros.")
            return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)

    return padded
# --- End Helper ---


"""
This function is adapted from [OmniAnomaly] by [TsingHuasuya et al.]
Original source: [https://github.com/NetManAIOps/OmniAnomaly]
"""


class OmniAnomalyModel(nn.Module):
    def __init__(self, feats, device):
        super(OmniAnomalyModel, self).__init__()
        self.name = 'OmniAnomaly'
        self.device = device
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            # nn.Flatten(),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        bs = x.shape[0]
        win = x.shape[1]

        # hidden = torch.rand(2, bs, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        hidden = torch.rand(2, bs, self.n_hidden).to(
            self.device) if hidden is not None else hidden

        out, hidden = self.lstm(x.view(-1, bs, self.n_feats), hidden)

        # print('out: ', out.shape)       # (L, bs, n_hidden)
        # print('hidden: ', hidden.shape) # (2, bs, n_hidden)

        # Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        # Reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std
        # Decoder
        x = self.decoder(x)             # (L, bs, n_feats)
        return x.reshape(bs, win*self.n_feats), mu.reshape(bs, win*self.n_latent), logvar.reshape(bs, win*self.n_latent), hidden


class OmniAnomaly(BaseDetector):
    def __init__(self,
                 win_size=5,
                 feats=1,
                 batch_size=128,
                 epochs=50,
                 patience=3,
                 lr=0.002,
                 validation_size=0.2
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size

        self.model = OmniAnomalyModel(
            feats=self.feats, device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss(reduction='none')

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        mses, klds = [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            n = epoch + 1
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:
                d = d.to(self.device)
                # print('d: ', d.shape)

                y_pred, mu, logvar, hidden = self.model(
                    d, hidden if idx else None)
                d = d.view(-1, self.feats*self.win_size)
                MSE = torch.mean(self.criterion(y_pred, d), axis=-1)
                KLD = -0.5 * torch.sum(1 + logvar -
                                       mu.pow(2) - logvar.exp(), dim=-1)
                loss = torch.mean(MSE + self.model.beta * KLD)

                mses.append(torch.mean(MSE).item())
                klds.append(self.model.beta * torch.mean(KLD).item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(),
                                 avg_loss=avg_loss / (idx + 1))

            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                loop = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=True
                )
                with torch.no_grad():
                    for idx, (d, _) in loop:
                        d = d.to(self.device)
                        y_pred, mu, logvar, hidden = self.model(
                            d, hidden if idx else None)
                        d = d.view(-1, self.feats*self.win_size)
                        MSE = torch.mean(self.criterion(y_pred, d), axis=-1)
                        KLD = -0.5 * \
                            torch.sum(1 + logvar - mu.pow(2) -
                                      logvar.exp(), dim=-1)
                        loss = torch.mean(MSE + self.model.beta * KLD)

                        avg_loss_val += loss.cpu().item()
                        loop.set_description(
                            f"Validation Epoch [{epoch}/{self.epochs}]"
                        )
                        loop.set_postfix(loss=loss.item(),
                                         avg_loss_val=avg_loss_val / (idx + 1))

            self.scheduler.step()
            if len(valid_loader) > 0:
                avg_loss = avg_loss_val / len(valid_loader)
            else:
                avg_loss = avg_loss / len(train_loader)
            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        scores = []
        y_preds = []
        loop = tqdm.tqdm(enumerate(test_loader),
                         total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (d, _) in loop:
                d = d.to(self.device)
                # print('d: ', d.shape)

                y_pred, _, _, hidden = self.model(d, hidden if idx else None)
                y_preds.append(y_pred)
                d = d.view(-1, self.feats*self.win_size)

                # print('y_pred: ', y_pred.shape)
                # print('d: ', d.shape)
                loss = torch.mean(self.criterion(y_pred, d), axis=-1)
                # print('loss: ', loss.shape)

                scores.append(loss.cpu())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) +
                                            list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass


def run_Custom_AD_Unsupervised(data, HP):
    """ Runs unsupervised training and scoring """
    clf = OmniAnomaly(feats=data.shape[1], **HP)
    print("Fitting model (Unsupervised)...")
    clf.fit(data)
    print("Calculating scores (Unsupervised)...")
    score = clf.decision_function(data)
    return score, clf


def run_Custom_AD_Semisupervised(data_train, data_test, **HP):
    """Runs fitting on train data and calculates the standard anomaly score on test data."""
    num_features = data_train.shape[1]
    clf = OmniAnomaly(feats=num_features, **HP)
    print("Fitting model (Semi-supervised)...")
    clf.fit(data_train)
    print("Calculating scores (Semi-supervised)...")
    raw_score = clf.decision_function(data_test)
    return raw_score

# --- Helper and Visualization Functions ---

# --- _find_clusters (Unchanged) ---


def _find_clusters(indices):
    """Finds contiguous clusters of indices."""
    if not isinstance(indices, (list, np.ndarray)) or len(indices) == 0:
        return []
    try:
        indices = np.unique(np.asarray(indices, dtype=int))
    except ValueError:
        return []
    if len(indices) == 0:
        return []
    if len(indices) == 1:
        return [(indices[0], indices[0])]
    diffs = np.diff(indices)
    split_points = np.where(diffs > 1)[0]
    start_indices = np.insert(indices[split_points + 1], 0, indices[0])
    end_indices = np.append(indices[split_points], indices[-1])
    return list(zip(start_indices, end_indices))


# --- visualize_errors (Plots only overall score, highlights anomalies in red) ---
def visualize_errors(raw_data, original_label, score,  # Takes original_label for highlighting
                     base_plot_filename, plot_dir, model_prefix="",
                     chunk_size=1500, chunk_indices_to_plot=None,
                     feature_names=None):
    """
    Visualizes data and overall anomaly score.
    Highlights segments of raw data corresponding to true anomalies (label=1) in red.
    Plots only chunks containing true anomalies.
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    n_samples = len(original_label)
    valid_input = True

    # --- Input Validation ---
    if not isinstance(raw_data, np.ndarray) or raw_data.ndim not in [1, 2]:
        valid_input = False
        print("Vis Error: raw_data format")
    elif raw_data.shape[0] != n_samples:
        valid_input = False
        print(
            f"Vis Error: raw_data length ({raw_data.shape[0]}) vs label ({n_samples})")
    if not isinstance(original_label, np.ndarray) or original_label.ndim != 1:
        valid_input = False
        print("Vis Error: original_label format")
    if not isinstance(score, np.ndarray) or score.ndim != 1:
        valid_input = False
        print("Vis Error: score format")
    elif valid_input and score.shape[0] != n_samples:
        valid_input = False
        print(
            f"Vis Error: score length ({score.shape[0]}) vs label ({n_samples})")

    n_dims = raw_data.shape[1] if raw_data.ndim == 2 else 1
    if feature_names and valid_input and len(feature_names) != n_dims:
        print(f"Vis Warning: Feature name count mismatch.")
        feature_names = None

    if not valid_input:
        print(f"Error ({model_prefix}): Invalid input for visualization.")
        return
    if not chunk_indices_to_plot:
        print(
            f"Info ({model_prefix}): No anomaly chunks specified for plotting.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    num_subplots = n_dims
    fig_height = max(8, 2.5 * num_subplots) if num_subplots > 0 else 8
    chunks_to_iterate = sorted(list(set(chunk_indices_to_plot)))
    print(f"Generating plot(s) for anomaly chunk(s): {chunks_to_iterate}...")

    # Loop Through Specified (Anomaly) Chunks
    for i_chunk in chunks_to_iterate:
        chunk_start = i_chunk * chunk_size
        chunk_end = min(n_samples, (i_chunk + 1) * chunk_size)
        time_range_chunk = np.arange(chunk_start, chunk_end)
        if len(time_range_chunk) == 0:
            continue

        # Slice data
        score_chunk = score[chunk_start:chunk_end]
        original_label_chunk = original_label[chunk_start:chunk_end]
        raw_data_chunk = raw_data[chunk_start:chunk_end,
                                  :] if raw_data.ndim == 2 else raw_data[chunk_start:chunk_end]

        if len(score_chunk) != len(time_range_chunk):
            print(f"Vis Error (Chunk {i_chunk}): Score length mismatch.")
            continue

        # Create Figure
        if num_subplots == 0:
            continue
        try:
            fig_ts, axes = plt.subplots(num_subplots, 1, figsize=(
                14, fig_height), sharex=True, squeeze=False)
            axes = axes.ravel()
        except Exception as e:
            print(f"Error creating subplots for chunk {i_chunk}: {e}")
            continue

        # Plot Dimensions
        for d in range(n_dims):
            ax_raw = axes[d]
            lns_plot = []
            feature_label = feature_names[d] if feature_names else f'Dim {d}'
            data_to_plot = raw_data_chunk[:,
                                          d] if raw_data.ndim == 2 else raw_data_chunk
            try:
                # 1. Plot ALL Raw Data
                ln_raw = ax_raw.plot(time_range_chunk, data_to_plot,
                                     label=f'Raw Data ({feature_label})', color='steelblue', linewidth=1.0, zorder=1)
                ax_raw.set_ylabel(
                    f'Raw Data ({feature_label})', color='steelblue')
                ax_raw.tick_params(axis='y', labelcolor='steelblue')
                ax_raw.grid(True, linestyle='--', alpha=0.6)
                lns_plot.extend(ln_raw)

                # 2. Plot Anomalous Segments in Red
                anomaly_indices_local = np.where(original_label_chunk == 1)[0]
                if len(anomaly_indices_local) > 0:
                    anomaly_segments = _find_clusters(anomaly_indices_local)
                    for i_seg, (start_local, end_local) in enumerate(anomaly_segments):
                        start_local = max(0, start_local)
                        end_local = min(len(time_range_chunk) - 1, end_local)
                        if start_local > end_local:
                            continue
                        time_seg = time_range_chunk[start_local: end_local + 1]
                        data_seg = data_to_plot[start_local: end_local + 1]
                        seg_label = 'Anomaly' if d == 0 and i_seg == 0 else "_nolegend_"
                        ln_anom = ax_raw.plot(
                            time_seg, data_seg, color='red', linewidth=1.5, label=seg_label, zorder=2)
                        if seg_label != "_nolegend_":
                            lns_plot.extend(ln_anom)

                # 3. Plot Overall Score
                ax_score = ax_raw.twinx()
                ln_scr_total = ax_score.plot(
                    time_range_chunk, score_chunk, label=f'Overall Score ({model_prefix})', color='orange', linestyle='-', alpha=0.8, linewidth=1.2)
                lns_plot.extend(ln_scr_total)
                ax_score.set_ylabel('Overall Score', color='orange')
                ax_score.tick_params(axis='y', labelcolor='orange')

                # 4. Combined Legend
                plot_labs = [l.get_label() for l in lns_plot]
                unique_handles_labels = {}
                for h, l in zip(lns_plot, plot_labs):
                    if l != "_nolegend_" and l not in unique_handles_labels:
                        unique_handles_labels[l] = h
                if d == 0:
                    ax_raw.legend(unique_handles_labels.values(), unique_handles_labels.keys(
                    ), loc='upper left', fontsize='small', frameon=True, facecolor='white', framealpha=0.8)
            except Exception as e:
                print(f"Error plotting subplot dim {d} chunk {i_chunk}: {e}")
                traceback.print_exc()

        # Final Figure Adjustments
        try:
            axes[-1].set_xlabel('Time Index')
            title = f'{model_prefix} - {base_plot_filename}\nChunk Index {i_chunk} (Indices {chunk_start}-{chunk_end-1}) - Overall Score & Anomalies'
            fig_ts.suptitle(title, fontsize=11)
            fig_ts.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_filename_ts = os.path.join(
                plot_dir, f"{base_plot_filename}_{model_prefix}_ChunkIdx_{i_chunk}_{chunk_start}-{chunk_end-1}_Score_AnomHighlight.png")
            plt.savefig(plot_filename_ts, bbox_inches='tight', dpi=300)
            print(
                f"  Anomaly plot saved: {os.path.basename(plot_filename_ts)}")
        except Exception as e:
            print(f"Error adjusting/saving plot chunk {i_chunk}: {e}")
            traceback.print_exc()
        finally:
            plt.close(fig_ts)
    print(
        f"Anomaly chunk plotting complete for {base_plot_filename} ({model_prefix}).")


# ----------------------------------------------------
# Seeding (Unchanged)
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# BASE_METRIC_ORDER (Unchanged)
BASE_METRIC_ORDER = ["VUS-ROC", "AUC-PR", "AUC-ROC", "Affiliation-F",
                     "R-based-F1", "Event-based-F1", "PA-F1", "Standard-F1"]
# MODEL_NAMES defined in main now
MODEL_NAMES = ['Custom_AD']


def get_ordered_columns(current_columns):
    """ Orders columns for the results CSV. """
    model_prefixes = {name: name for name in MODEL_NAMES}
    final_order = []
    processed_columns = set()
    if 'filename' in current_columns:
        final_order.append('filename')
        processed_columns.add('filename')
    for prefix in model_prefixes.values():
        col = f'{prefix}_VUS-PR'
        if col in current_columns:
            final_order.append(col)
            processed_columns.add(col)
    for prefix in model_prefixes.values():
        other_metrics = [f'{prefix}_{m}' for m in BASE_METRIC_ORDER]
        for col in other_metrics:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)
        other_info = [f'{prefix}_runtime', f'{prefix}_FP_count_ext', f'{prefix}_FN_count_ext', f'{prefix}_Error', f'{prefix}_Scaling_Error',
                      f'{prefix}_Metrics_Threshold_Error', f'{prefix}_Analysis_Error', f'{prefix}_Metrics_Error', f'{prefix}_Metrics_Result']
        for col in other_info:
            if col in current_columns and col not in processed_columns:
                final_order.append(col)
                processed_columns.add(col)
    remaining_columns = sorted(list(set(current_columns) - processed_columns))
    final_order.extend(remaining_columns)
    return final_order


# --- Main Script ---
if __name__ == '__main__':
    overall_start_time = time.time()
    # Configuration
    FILE_LIST_PATH = os.path.join('Datasets', 'File_List', 'TSB-AD-M-Eva.csv')
    DATA_DIR = os.path.join('Datasets', 'TSB-AD-M')
    PROGRESS_FILE_PREFIX = "progress_Custom_AD"
    RESULTS_CSV_PATH = "Custom_AD_detailed_results.csv"
    MODEL_NAMES = ['Custom_AD']
    VISUALIZE_ANOMALIES = True  # Plot anomaly chunks?
    PLOT_DIR_BASE = "Custom_AD_Anomaly_Highlights"
    CHUNK_SIZE_VIS = 1500
    # Default HPs (can be overridden if needed)
    DEFAULT_HP = {'win_size': 100, 'lr': 0.002}

    # Initialize metricor safely
    metricor = basic_metricor() if basic_metricor else None

    # --- Load Progress & Existing Results ---
    processed_files = set()
    all_results = []
    progress_file = f"{PROGRESS_FILE_PREFIX}.pkl"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, set):
                processed_files = loaded_data
            else:
                print(
                    f"Warning: Progress file type {type(loaded_data)}. Reinitializing.")
                processed_files = set()
            print(f"Loaded progress: {len(processed_files)} files processed.")
        except Exception as e:
            print(
                f"Warning: Could not load progress file {progress_file}: {e}. Starting fresh.")
            processed_files = set()

    if os.path.exists(RESULTS_CSV_PATH):
        try:
            print(f"Loading existing results from {RESULTS_CSV_PATH}")
            existing_results_df = pd.read_csv(RESULTS_CSV_PATH)
            if 'filename' in existing_results_df.columns:
                all_results = existing_results_df.where(
                    pd.notnull(existing_results_df), None).to_dict('records')
                valid_filenames = existing_results_df['filename'].dropna(
                ).unique().tolist()
                processed_files.update(valid_filenames)
                print(
                    f"Total unique processed (incl. existing results): {len(processed_files)}")
            else:
                print(f"Warning: Existing results file missing 'filename'. Ignoring.")
                all_results = []
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}.")
            all_results = []

    # --- Load File List ---
    try:
        filenames_df = pd.read_csv(FILE_LIST_PATH)
        if filenames_df.shape[1] > 0:
            filenames = filenames_df.iloc[:, 0].dropna().astype(
                str).unique().tolist()
        else:
            print(f"Warning: File list {FILE_LIST_PATH} seems empty.")
            filenames = []
    except Exception as e:
        print(f"Error reading file list {FILE_LIST_PATH}: {e}")
        filenames = []

    total_files = len(filenames)
    files_to_process = [f for f in filenames if isinstance(
        f, str) and f.strip() and f not in processed_files]
    total_to_process_this_run = len(files_to_process)
    processed_files_count_this_run = 0
    print(
        f"Total files: {total_files}. Processed: {len(processed_files)}. To process now: {total_to_process_this_run}.")

    # --- Main File Loop ---
    for i, filename in enumerate(files_to_process):
        print(
            f"\nProcessing file {i+1}/{total_to_process_this_run} ({processed_files_count_this_run} successful this run): {filename}")
        file_path = os.path.join(DATA_DIR, filename)
        file_success = False
        current_file_results = {'filename': filename}
        model_prefix = MODEL_NAMES[0]  # Only one model
        model_data_for_vis = {}
        clf_instance = None

        try:
            # --- Data Loading and Validation ---
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()

            slidingWindow = find_length_rank(data, rank=1)
            train_index = filename.split('.')[0].split('_')[-3]
            data_train = data[:int(train_index), :]
            Optimal_Det_HP = {'win_size': 100, 'lr': 0.002}

            output = run_Custom_AD_Semisupervised(
                data_train, data, **Optimal_Det_HP)
            output = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                output.reshape(-1, 1)).ravel()
            evaluation_result = get_metrics(
                output, label, slidingWindow=slidingWindow)
            print('Evaluation Result: ', evaluation_result)

            if evaluation_result:
                for key, value in evaluation_result.items():
                    result_key = f'{key}'
                    if isinstance(value, (np.integer, np.floating)):
                        current_file_results[result_key] = value.item()
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        current_file_results[result_key] = value
                    else:
                        current_file_results[result_key] = str(value)
            # --- Visualization Call ---
            # ***** Visualization logic moved inside the main try block *****
            if VISUALIZE_ANOMALIES:
                base_fname_vis = os.path.splitext(
                    os.path.basename(filename))[0]
                model_name_vis = model_prefix  # Use the single model prefix
                print(f"  Generating visualizations for {model_name_vis}...")

                # Prepare data for visualization
                score_for_vis = output  # Use the final scaled score
                original_label_for_vis = label
                raw_data_for_vis = data

                # Basic validation before calling visualize_errors
                vis_data_valid = True
                if score_for_vis is None or original_label_for_vis is None or raw_data_for_vis is None:
                    print(
                        f"    Skipping visualization for {model_name_vis}: Missing essential data (score/label/raw).")
                    vis_data_valid = False
                elif not isinstance(score_for_vis, np.ndarray) or score_for_vis.ndim != 1 or \
                        not isinstance(original_label_for_vis, np.ndarray) or original_label_for_vis.ndim != 1 or \
                        not isinstance(raw_data_for_vis, np.ndarray) or \
                        score_for_vis.shape[0] != original_label_for_vis.shape[0] or \
                        raw_data_for_vis.shape[0] != original_label_for_vis.shape[0]:
                    print(
                        f"    Skipping visualization for {model_name_vis}: Data shape mismatch or type error.")
                    print(f"      Shapes: score={score_for_vis.shape if isinstance(score_for_vis, np.ndarray) else type(score_for_vis)}, "
                          f"label={original_label_for_vis.shape if isinstance(original_label_for_vis, np.ndarray) else type(original_label_for_vis)}, "
                          f"raw_data={raw_data_for_vis.shape if isinstance(raw_data_for_vis, np.ndarray) else type(raw_data_for_vis)}")
                    vis_data_valid = False

                if vis_data_valid:
                    chunk_indices_to_plot = set()
                    if np.any(original_label_for_vis == 1):
                        anomaly_clusters = _find_clusters(
                            np.where(original_label_for_vis == 1)[0])
                        for start, end in anomaly_clusters:
                            # Find all chunks that overlap with the anomaly segment
                            start_chunk = start // CHUNK_SIZE_VIS
                            end_chunk = end // CHUNK_SIZE_VIS
                            chunk_indices_to_plot.update(
                                range(start_chunk, end_chunk + 1))
                    else:
                        print(
                            "    No true anomalies found in label. Skipping anomaly chunk visualization.")

                    chunk_indices_list = sorted(list(chunk_indices_to_plot))

                    if chunk_indices_list:
                        print(
                            f"    Plotting anomaly chunks: {chunk_indices_list}")
                        model_plot_dir = os.path.join(
                            PLOT_DIR_BASE, model_name_vis)
                        try:
                            # Call visualize_errors (note: window_context removed)
                            visualize_errors(raw_data=raw_data_for_vis,
                                             original_label=original_label_for_vis,
                                             score=score_for_vis,
                                             base_plot_filename=base_fname_vis,
                                             plot_dir=model_plot_dir,
                                             model_prefix=model_name_vis,
                                             chunk_size=CHUNK_SIZE_VIS,
                                             chunk_indices_to_plot=chunk_indices_list)
                        except Exception as vis_e:
                            print(
                                f"    Error during visualize_errors call for {model_name_vis}: {vis_e}")
                            # Log error but don't stop file processing
                            current_file_results[f'{model_prefix}_Visualization_Error'] = f"Visualize Error: {vis_e}"
                            traceback.print_exc()
                    elif np.any(original_label_for_vis == 1):
                        # This case shouldn't happen if logic is correct, but log if it does
                        print(
                            "    Anomalies present, but no chunks identified for plotting.")
            # ***** End of Visualization logic *****
            file_success = True  # Mark success if reached here

        # --- Outer Loop Exception Handling ---
        except FileNotFoundError as fnf_e:
            print(f"Error: {fnf_e}")
            current_file_results['Error'] = 'File Not Found (Outer)'
        except (ValueError, RuntimeError, TypeError) as data_load_e:
            print(f"Data Loading/Validation Error: {data_load_e}")
            traceback.print_exc()
            current_file_results['Error'] = f'Data Load/Validation Error: {data_load_e}'
        except pd.errors.EmptyDataError as ede:
            print(f"Pandas EmptyDataError: {ede}. Skipping.")
            current_file_results['Error'] = 'Pandas EmptyDataError'
        except MemoryError as me:
            print(f"CRITICAL MemoryError (Outer Loop): {me}. Stopping file.")
            traceback.print_exc()
            current_file_results['Error'] = 'MemoryError (Outer Loop)'
            file_success = False
        except Exception as e:
            print(f"Unexpected critical error: {e}")
            traceback.print_exc()
            current_file_results['Error'] = f'Critical fail: {e}'
            file_success = False

        # --- Append result AND Save Incrementally ---
        if 'filename' not in current_file_results:
            current_file_results['filename'] = filename
        existing_file_index = next((idx for idx, res in enumerate(
            all_results) if res.get('filename') == filename), -1)
        if existing_file_index != -1:
            all_results[existing_file_index].update(current_file_results)
        else:
            all_results.append(current_file_results)
        if file_success:
            processed_files.add(filename)
            processed_files_count_this_run += 1
            print(f"  File processed successfully.")
        else:
            processed_files.add(filename)
            print(f"  File processing marked as unsuccessful/skipped.")
        try:  # Save progress
            with open(progress_file, 'wb') as f:
                pickle.dump(processed_files, f)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
        try:  # Save intermediate CSV
            if all_results:
                temp_df = pd.DataFrame(all_results)
                temp_df = temp_df.drop_duplicates(
                    subset=['filename'], keep='last')
                if not temp_df.empty:
                    ordered_cols = get_ordered_columns(
                        temp_df.columns.tolist())
                    temp_df = temp_df.reindex(columns=ordered_cols)
                temp_df.to_csv(RESULTS_CSV_PATH, index=False)
        except Exception as e:
            print(f"Warning: Could not save intermediate CSV: {e}")

    # --- End File Loop ---

    # --- Final Summary ---
    print(f"\n--- Run Summary ---")
    print(f"Attempted: {total_to_process_this_run} new files. Succeeded: {processed_files_count_this_run}. Total processed (incl. errors): {len(processed_files)}")

    # --- Final Save & Averaging ---
    if all_results:
        final_results_df = pd.DataFrame(all_results)
        final_results_df = final_results_df.drop_duplicates(
            subset=['filename'], keep='last')
        try:
            if not final_results_df.empty:
                ordered_cols = get_ordered_columns(
                    final_results_df.columns.tolist())
                final_results_df = final_results_df.reindex(
                    columns=ordered_cols)
            final_results_df.to_csv(RESULTS_CSV_PATH, index=False)
            print(f"Final results saved to {RESULTS_CSV_PATH}")
        except Exception as e:
            print(f"Error saving final results CSV: {e}")

        # Calculate Averages
        error_columns = ['Error']
        for m in MODEL_NAMES:
            error_columns.extend(
                [f'{m}_Error', f'{m}_Scaling_Error', f'{m}_Analysis_Error', f'{m}_Metrics_Error'])
        existing_error_columns = [
            col for col in error_columns if col in final_results_df.columns]
        successful_filter = pd.Series(
            [True] * len(final_results_df), index=final_results_df.index)
        for err_col in existing_error_columns:
            successful_filter &= ~(final_results_df[err_col].notna() & (
                final_results_df[err_col] != ''))
        successful_df = final_results_df[successful_filter]
        num_successful = len(successful_df)
        num_failed = len(final_results_df) - num_successful
        if not successful_df.empty:
            print(
                f"\n--- Average Metrics Across {num_successful} Successfully Processed Files --- ({num_failed} failed/error files excluded)")

            def print_average(df, col_key, prefix="Average"):
                if col_key in df.columns:
                    numeric_col = pd.to_numeric(df[col_key], errors='coerce')
                    if numeric_col.notna().any():
                        avg_val = np.nanmean(numeric_col.astype(float))
                        metric_name = col_key.split(
                            '_', 1)[1] if '_' in col_key else col_key
                        print(
                            f'      "{prefix} {metric_name}": {avg_val:.6f},')
                        return True
                return False
            for model_key in MODEL_NAMES:
                print(
                    f"\n  --- {model_key} Averages ({num_successful} files) ---")
                print(f"    -- Key Metrics --")
                if not print_average(successful_df, f'{model_key}_VUS-PR') and not print_average(successful_df, f'{model_key}_VUS-ROC'):
                    print("      VUS-PR/VUS-ROC not available.")
                print(f"\n    -- Other TSB-AD Metrics --")
                metrics_printed = 0
                for metric in BASE_METRIC_ORDER:
                    if metric not in ["VUS-PR", "VUS-ROC"] and print_average(successful_df, f'{model_key}_{metric}'):
                        metrics_printed += 1
                if metrics_printed == 0:
                    print("      No other standard metrics available.")
                print(f"\n    -- Other Info --")
                print_average(
                    successful_df, f"{model_key}_runtime", prefix="Avg Runtime (s)")
                print_average(
                    successful_df, f"{model_key}_FP_count_ext", prefix="Avg FP (ext)")
                print_average(
                    successful_df, f"{model_key}_FN_count_ext", prefix="Avg FN (ext)")
            print("\n  ------------------------------------")
        else:
            print(
                f"\nNo successfully processed files found for averaging metrics ({len(final_results_df)} total files in results).")
    else:
        print("\nNo results generated or loaded.")
    overall_end_time = time.time()
    print(
        f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds")
