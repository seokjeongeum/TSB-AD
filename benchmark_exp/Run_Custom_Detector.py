# coding: utf-8
# Author: Qinghua Liu liu.11085@osu.edu # Modified by AI Assistant & OmniAnomaly Paper Alignment
# License: Apache-2.0 License

from __future__ import division, print_function

import math
import os
import pickle
import random
import time
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
# Use torch distributions for log_prob
from torch.distributions import Normal as TorchNormal
from torch.nn import GaussianNLLLoss, Linear, Module, ModuleList, Parameter
from torch.utils.data import DataLoader

# --- Ensure these imports point to the correct location ---
try:
    from TSB_AD.evaluation.metrics import basic_metricor, get_metrics
    # from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict # HPs passed directly
    from TSB_AD.models.base import BaseDetector
    from TSB_AD.models.feature import Window
    from TSB_AD.utils.dataset import ReconstructDataset
    from TSB_AD.utils.slidingWindows import find_length_rank
    from TSB_AD.utils.torch_utility import EarlyStoppingTorch, get_gpu
    from TSB_AD.utils.utility import check_parameter, standardizer
except ImportError:
    print("Warning: TSB_AD package not found. Trying relative imports...")
    # Add relative imports or error handling as needed
    # ... (previous relative import attempts) ...
    pass
# -----------------------------------------------------------


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# --- Planar Normalizing Flow Layer (Unchanged) ---


class PlanarFlow(Module):
    """Implements a single layer of Planar Normalizing Flow."""

    def __init__(self, dim):
        super().__init__()
        self.weight = Parameter(torch.Tensor(1, dim))  # w
        self.scale = Parameter(torch.Tensor(1, dim))  # u
        self.bias = Parameter(torch.Tensor(1))       # b
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        nn.init.normal_(self.scale, mean=0, std=0.01)
        nn.init.constant_(self.bias, 0)

    def _u_h(self, u, w):
        # Enforce invertibility constraint: w^T * u >= -1
        wu = torch.sum(w * u, dim=1, keepdim=True)
        m_wu = -1 + F.softplus(wu)
        w_norm_sq = torch.sum(w**2, dim=1, keepdim=True)
        u_h = u + (m_wu - wu) * w / (w_norm_sq + 1e-8)
        return u_h

    def forward(self, z):
        u_h = self._u_h(self.scale, self.weight)
        act = F.tanh(torch.matmul(z, self.weight.t()) + self.bias)
        z_next = z + u_h * act
        psi = (1 - act**2) * self.weight
        log_det_jacobian = torch.log(
            torch.abs(1 + torch.sum(psi * u_h, dim=1)) + 1e-8)
        return z_next, log_det_jacobian

# --- Gaussian NLL Helper (Modified) ---


def gaussian_nll(mu, log_var, x, elementwise=False):
    """
    Calculates the negative log-likelihood of x under a diagonal Gaussian N(mu, exp(log_var)).

    Args:
        mu (Tensor): Mean tensor, shape [..., N]
        log_var (Tensor): Log variance tensor, shape [..., N]
        x (Tensor): Target tensor, shape [..., N]
        elementwise (bool): If True, return NLL per element [..., N].
                            If False, return NLL summed across last dim [...].

    Returns:
        Tensor: Negative log-likelihood.
    """
    # Ensure feature dimensions match
    if mu.shape[-1] != x.shape[-1] or log_var.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"Feature mismatch: mu {mu.shape[-1]}, log_var {log_var.shape[-1]}, x {x.shape[-1]}")

    # NLL component for each feature/element
    nll_elementwise = 0.5 * (math.log(2 * math.pi) +
                             log_var + ((x - mu)**2) * torch.exp(-log_var))

    if elementwise:
        return nll_elementwise  # Shape: [..., N]
    else:
        # Sum NLL across the last dimension (features) for each sample/step
        nll_persample = torch.sum(nll_elementwise, dim=-1)
        return nll_persample  # Shape: [...]

# --- Standard Gaussian Log Prob Helper (Unchanged) ---


def log_prob_standard_gaussian(z):
    """Calculates log probability of z under N(0, I)"""
    dim = z.shape[-1]
    log_unnormalized = -0.5 * torch.sum(z**2, dim=-1)
    log_normalization = -0.5 * dim * math.log(2 * math.pi)
    return log_unnormalized + log_normalization

# --- Gaussian Log Prob (Markov Prior) Helper ---


def log_prob_markov_gaussian(z_t, z_t_minus_1):
    """Calculates log prob of z_t under N(z_t | z_{t-1}, I)"""
    # Assumes identity covariance for the transition prior
    dim = z_t.shape[-1]
    log_unnormalized = -0.5 * torch.sum((z_t - z_t_minus_1)**2, dim=-1)
    log_normalization = -0.5 * dim * math.log(2 * math.pi)
    return log_unnormalized + log_normalization


class OmniAnomalyModel(nn.Module):
    # Added use_connected_z_p flag
    def __init__(self, feats, device, win_size, n_hidden=32, n_latent=8, rnn_layers=2, nf_layers=0, use_connected_z_p=True):
        super().__init__()
        self.name = 'OmniAnomaly_Fuller'
        self.device = device
        self.n_feats = feats
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.win_size = win_size
        self.nf_layers = nf_layers
        self.use_connected_z_p = use_connected_z_p  # Store flag

        # GRU (Unchanged)
        self.gru = nn.GRU(feats, self.n_hidden, rnn_layers, batch_first=True)

        # Encoder (Unchanged - already takes z_prev)
        encoder_input_dim = self.n_hidden + self.n_latent
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_input_dim, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )

        # Decoder (Unchanged - predicts params for current x_t)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_feats)
        )

        # Normalizing Flow Layers (Unchanged)
        if self.nf_layers > 0:
            self.flows = ModuleList([PlanarFlow(self.n_latent)
                                    for _ in range(self.nf_layers)])
        else:
            self.flows = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(self.gru.num_layers, batch_size,
                        self.n_hidden).to(self.device)
        z_prev = torch.zeros(batch_size, self.n_latent).to(self.device)

        all_mu_z, all_logvar_z = [], []
        all_z0, all_z_k = [], []
        all_mu_x, all_logvar_x = [], []
        all_log_dets = []

        for t in range(self.win_size):
            xt = x[:, t:t+1, :]
            gru_out, h = self.gru(xt, h)
            ht = gru_out.squeeze(1)

            # --- Encoder (Unchanged) ---
            encoder_input = torch.cat([ht, z_prev], dim=1)
            mu_z0_t, logvar_z0_t = torch.chunk(
                self.encoder_fc(encoder_input), 2, dim=-1)

            # --- Sample z0, Apply Flows (Unchanged) ---
            z0_t = self.reparameterize(mu_z0_t, logvar_z0_t)
            z_k_t = z0_t
            step_log_dets = torch.zeros(batch_size).to(self.device)
            if self.flows is not None:
                zk = z0_t
                for flow in self.flows:
                    zk, log_det_k = flow(zk)
                    step_log_dets += log_det_k
                z_k_t = zk

            # --- Decoder (Unchanged) ---
            mu_xt, logvar_xt = torch.chunk(self.decoder_fc(z_k_t), 2, dim=-1)

            # --- Store results (Unchanged) ---
            all_mu_z.append(mu_z0_t)
            all_logvar_z.append(logvar_z0_t)
            all_z0.append(z0_t)
            all_z_k.append(z_k_t)
            all_mu_x.append(mu_xt)
            all_logvar_x.append(logvar_xt)
            if self.flows is not None:
                all_log_dets.append(step_log_dets)

            z_prev = z_k_t  # Use final sample for next step

        # --- Stack results (Unchanged) ---
        mu_z = torch.stack(all_mu_z, dim=1)
        logvar_z = torch.stack(all_logvar_z, dim=1)
        z0 = torch.stack(all_z0, dim=1)
        z_k = torch.stack(all_z_k, dim=1)
        mu_x = torch.stack(all_mu_x, dim=1)
        logvar_x = torch.stack(all_logvar_x, dim=1)
        log_dets = torch.stack(all_log_dets, dim=1) if self.flows else None

        return mu_x, logvar_x, mu_z, logvar_z, z0, z_k, log_dets


class Custom_AD(BaseDetector):
    # Added use_connected_z_p
    DEFAULT_HP = {
        'win_size': 100, 'batch_size': 128, 'epochs': 50, 'patience': 3,
        'lr': 0.002, 'validation_size': 0.2, 'n_hidden': 32, 'n_latent': 8,
        'rnn_layers': 2, 'beta': 0.01, 'nf_layers': 3,
        # Flag to use the Markov prior N(z_t|z_{t-1}, I)
        'use_connected_z_p': True
    }

    def __init__(self, feats, **HP):
        super().__init__()
        self.__anomaly_score = None
        self.cuda = True
        self.device = get_gpu(self.cuda)
        self.feats = feats

        current_hp = {**self.DEFAULT_HP, **HP}

        self.win_size = int(current_hp['win_size'])
        self.batch_size = int(current_hp['batch_size'])
        self.epochs = int(current_hp['epochs'])
        self.patience = int(current_hp['patience'])
        self.lr = float(current_hp['lr'])
        self.validation_size = float(current_hp['validation_size'])
        self.n_hidden = int(current_hp['n_hidden'])
        self.n_latent = int(current_hp['n_latent'])
        self.rnn_layers = int(current_hp['rnn_layers'])
        self.beta = float(current_hp['beta'])
        self.nf_layers = int(current_hp['nf_layers'])
        self.use_connected_z_p = bool(
            current_hp['use_connected_z_p'])  # Get prior flag

        print(f"  [Custom_AD Init] Using HPs: {current_hp}")

        self.model = OmniAnomalyModel(
            feats=self.feats, device=self.device, win_size=self.win_size,
            n_hidden=self.n_hidden, n_latent=self.n_latent,
            rnn_layers=self.rnn_layers, nf_layers=self.nf_layers,
            # Pass flag to model (though not used internally there)
            use_connected_z_p=self.use_connected_z_p
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 5, 0.9)
        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)

    def fit(self, data):
        # --- Data splitting and Loader setup (keep robust logic) ---
        # ... (Ensure flatten=False) ...
        # ... (Same as previous version) ...
        min_data_needed = self.win_size + 2
        if self.validation_size > 0 and len(data) < min_data_needed / (1-self.validation_size):
            print(
                f"Warning: Data length ({len(data)}) potentially too short for win_size ({self.win_size}) and val_split ({self.validation_size}). Reducing validation_size.")
            max_val_size = 1.0 - float(self.win_size+1) / len(data)
            self.validation_size = max(
                0.0, min(self.validation_size, max_val_size * 0.5))
            print(f"  Adjusted validation_size to {self.validation_size:.3f}")

        split_idx = int((1 - self.validation_size) * len(data))
        if self.validation_size > 0 and split_idx < self.win_size + 1:
            split_idx = self.win_size + 1
        if self.validation_size > 0 and len(data) - split_idx < 1:
            split_idx = len(data) - 1

        tsTrain = data[:split_idx]
        tsValid = data[split_idx:]

        # Use batch_first=True in ReconstructDataset if model expects it
        train_loader = DataLoader(
            # REMOVE flatten=False from here
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        valid_loader = DataLoader(
            # REMOVE flatten=False from here
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        # --- End Loader Setup ---

        train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_epoch_loss, avg_recon_loss, avg_kld_loss = 0, 0, 0
            batch_count = 0
            if not train_loader:
                break
            loop = tqdm.tqdm(enumerate(train_loader),
                             total=len(train_loader), leave=False)

            for idx, batch_data in loop:
                if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 1:
                    continue
                d_window = batch_data[0]
                if d_window.shape[0] == 0:
                    continue
                d_window = d_window.to(self.device)

                # Forward pass
                mu_x, logvar_x, mu_z, logvar_z, z0, z_k, log_dets = self.model(
                    d_window)

                # --- Calculate Loss (ELBO = NLL + beta * KL) ---
                # 1. Reconstruction Loss (NLL)
                nll = gaussian_nll(mu_x.reshape(-1, self.feats), logvar_x.reshape(-1,
                                   self.feats), d_window.reshape(-1, self.feats))
                # [batch, win_size]
                nll = nll.view(d_window.shape[0], self.win_size)
                # Sum NLL over sequence steps (averaged over batch)
                recon_loss = torch.sum(torch.mean(nll, dim=0))

                # 2. KL Divergence Term Calculation
                # log q(z_k | x) = log q0(z0 | x) - sum(log_dets)
                # log q0(z0 | x) = sum_t log N(z0_t | mu_z_t, var_z_t)
                log_q0_z0_seq = TorchNormal(mu_z, torch.exp(
                    0.5 * logvar_z)).log_prob(z0).sum(dim=-1)  # [batch, win_size]
                log_q0_z0_total = log_q0_z0_seq.sum(
                    dim=1)  # Sum over time -> [batch]

                # Sum log determinants over time
                sum_log_dets_total = torch.zeros_like(
                    log_q0_z0_total)  # [batch]
                if log_dets is not None:
                    sum_log_dets_total = log_dets.sum(dim=1)  # [batch]

                # log q(z_k | x) term per sequence
                log_qk_zk_total = log_q0_z0_total - \
                    sum_log_dets_total  # [batch]

                # Prior term: log p(z_k)
                if self.use_connected_z_p:
                    # Markov Prior: p(z) = p(z0) * p(z1|z0) * ...
                    log_p_z0 = log_prob_standard_gaussian(
                        z_k[:, 0, :])  # p(z0)=N(0,I) -> [batch]
                    log_p_zt_cond = torch.zeros(
                        d_window.shape[0], self.win_size - 1).to(self.device)  # [batch, win_size-1]
                    for t in range(1, self.win_size):
                        # p(zt|z_{t-1})=N(zt|z_{t-1},I)
                        log_p_zt_cond[:, t-1] = log_prob_markov_gaussian(
                            z_k[:, t, :], z_k[:, t-1, :])
                    log_p_zk_total = log_p_z0 + \
                        log_p_zt_cond.sum(dim=1)  # [batch]
                else:
                    # Standard Prior: p(z) = Prod_t p(zt) where p(zt)=N(0,I)
                    log_p_standard = log_prob_standard_gaussian(
                        z_k.view(-1, self.n_latent)).view(d_window.shape[0], self.win_size)  # [batch, win_size]
                    log_p_zk_total = log_p_standard.sum(dim=1)  # [batch]

                # KL = E_q[log q(z_k|x) - log p(z_k)]
                # Since we only have one sample z_k from q per x, approximate E_q[] with the value at z_k
                kl_term = log_qk_zk_total - log_p_zk_total  # [batch]
                kld_loss = torch.mean(kl_term)  # Average KL over batch

                # 3. Total ELBO Loss = Recon_NLL + beta * KL
                loss = recon_loss + self.beta * kld_loss

                # --- Backward Pass and Optimization ---
                if not torch.isfinite(loss):
                    print(
                        f"Warning: NaN/Inf loss epoch {epoch}, batch {idx}. Skip update.")
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_epoch_loss += loss.item()
                avg_recon_loss += recon_loss.item()
                avg_kld_loss += kld_loss.item()
                batch_count += 1
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(),
                                 recon=recon_loss.item(), kld=kld_loss.item())

            # --- Validation (similar loss calculation - simplified here for brevity) ---
            val_loss_for_stopping = avg_epoch_loss / \
                batch_count if batch_count > 0 else float('inf')
            if valid_loader and len(valid_loader) > 0:
                self.model.eval()
                avg_val_loss, val_batch_count = 0, 0
                loop_val = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=False)
                with torch.no_grad():
                    for idx_val, batch_data_val in loop_val:
                        # ... (data loading) ...
                        if not isinstance(batch_data_val, (list, tuple)) or len(batch_data_val) < 1:
                            continue
                        d_window_val = batch_data_val[0]
                        if d_window_val.shape[0] == 0:
                            continue
                        d_window_val = d_window_val.to(self.device)

                        # Use the same loss calculation logic as in training
                        mu_x_val, logvar_x_val, mu_z_val, logvar_z_val, z0_val, z_k_val, log_dets_val = self.model(
                            d_window_val)
                        # NLL
                        nll_val = gaussian_nll(
                            mu_x_val.view(-1, self.feats), logvar_x_val.view(-1, self.feats), d_window_val.view(-1, self.feats))
                        nll_val = nll_val.view(
                            d_window_val.shape[0], self.win_size)
                        recon_loss_val = torch.sum(torch.mean(nll_val, dim=0))
                        # KL
                        log_q0_z0_seq_val = TorchNormal(mu_z_val, torch.exp(
                            0.5 * logvar_z_val)).log_prob(z0_val).sum(dim=-1)
                        log_q0_z0_total_val = log_q0_z0_seq_val.sum(dim=1)
                        sum_log_dets_total_val = torch.zeros_like(
                            log_q0_z0_total_val)
                        if log_dets_val is not None:
                            sum_log_dets_total_val = log_dets_val.sum(dim=1)
                        log_qk_zk_total_val = log_q0_z0_total_val - sum_log_dets_total_val
                        if self.use_connected_z_p:
                            log_p_z0_val = log_prob_standard_gaussian(
                                z_k_val[:, 0, :])
                            log_p_zt_cond_val = torch.zeros(
                                d_window_val.shape[0], self.win_size - 1).to(self.device)
                            for t in range(1, self.win_size):
                                log_p_zt_cond_val[:, t-1] = log_prob_markov_gaussian(
                                    z_k_val[:, t, :], z_k_val[:, t-1, :])
                            log_p_zk_total_val = log_p_z0_val + \
                                log_p_zt_cond_val.sum(dim=1)
                        else:
                            log_p_standard_val = log_prob_standard_gaussian(
                                z_k_val.view(-1, self.n_latent)).view(d_window_val.shape[0], self.win_size)
                            log_p_zk_total_val = log_p_standard_val.sum(dim=1)
                        kl_term_val = log_qk_zk_total_val - log_p_zk_total_val
                        kld_loss_val = torch.mean(kl_term_val)
                        # Total Loss
                        loss_val = recon_loss_val + self.beta * kld_loss_val

                        if not torch.isfinite(loss_val):
                            continue

                        avg_val_loss += loss_val.item()
                        val_batch_count += 1
                        loop_val.set_description(
                            f"Validation Epoch [{epoch}/{self.epochs}]")
                        loop_val.set_postfix(val_loss=loss_val.item())

                if val_batch_count > 0:
                    val_loss_for_stopping = avg_val_loss / val_batch_count
                    print(
                        f"  Epoch {epoch}: Avg Val Loss={val_loss_for_stopping:.4f}")
                else:
                    print(f"  Epoch {epoch}: No validation batches processed.")
                    val_loss_for_stopping = float(
                        'inf') if batch_count > 0 else val_loss_for_stopping
            # --- End Validation ---

            self.scheduler.step()
            self.early_stopping(val_loss_for_stopping, self.model)
            if self.early_stopping.early_stop:
                print(f"   Early stopping triggered after epoch {epoch}.")
                break

    # --- decision_function remains the same as the previous refactoring ---
    # --- It uses NLL of the last point as the score ---
    def decision_function(self, data):
        if len(data) < self.win_size:
            print(
                f"Error: Test data length ({len(data)}) < window size ({self.win_size}). Returning zeros.")
            return np.zeros(len(data))

        test_loader = DataLoader(
            # REMOVE flatten=False from here
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        self.model.eval()
        scores_list = []
        loop = tqdm.tqdm(enumerate(test_loader),
                         total=len(test_loader), leave=False)

        with torch.no_grad():
            for idx, batch_data in loop:
                if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 1:
                    continue
                d_window = batch_data[0]
                if d_window.shape[0] == 0:
                    continue
                d_window = d_window.to(self.device)

                mu_x, logvar_x, _, _, _, _, _ = self.model(d_window)

                # Score using NLL of the last time step
                mu_x_last = mu_x[:, -1, :]
                logvar_x_last = logvar_x[:, -1, :]
                d_last = d_window[:, -1, :]

                nll_scores = gaussian_nll(mu_x_last, logvar_x_last, d_last)

                if not torch.all(torch.isfinite(nll_scores)):
                    print(
                        f"Warning: NaN/Inf scores detected test batch {idx}. Replacing.")
                    max_finite_score = torch.finfo(nll_scores.dtype).max / 10
                    nll_scores = torch.nan_to_num(
                        nll_scores, nan=max_finite_score, posinf=max_finite_score, neginf=0.0)

                scores_list.append(nll_scores.cpu())

        if not scores_list:
            return np.zeros(len(data))
        scores_unpadded = torch.cat(scores_list, dim=0).numpy()

        # --- Padding ---
        n_samples_original = len(data)
        # Use helper for padding
        padded_scores = _pad_array(
            scores_unpadded, n_samples_original, self.win_size)

        self.__anomaly_score = padded_scores
        return self.__anomaly_score

    def decision_function_per_feature(self, data):
        """Calculates the PER-FEATURE NLL score for the last point of each window."""
        if len(data) < self.win_size:
            print(
                f"Error: Test data length ({len(data)}) < window size ({self.win_size}). Returning zeros.")
            # Return 2D array of zeros
            return np.zeros((len(data), self.feats))

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        self.model.eval()
        scores_list_per_feature = []  # List to store [batch, feats] tensors
        loop = tqdm.tqdm(enumerate(test_loader),
                         total=len(test_loader), leave=False)

        with torch.no_grad():
            for idx, batch_data in loop:
                if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 1:
                    continue
                d_window = batch_data[0]
                if d_window.shape[0] == 0:
                    continue
                d_window = d_window.to(self.device)

                mu_x, logvar_x, _, _, _, _, _ = self.model(d_window)

                # Score using ELEMENTWISE NLL of the last time step
                mu_x_last = mu_x[:, -1, :]      # Shape: [batch, feats]
                logvar_x_last = logvar_x[:, -1, :]  # Shape: [batch, feats]
                d_last = d_window[:, -1, :]    # Shape: [batch, feats]

                # Calculate NLL per feature using elementwise=True
                nll_scores_feature = gaussian_nll(
                    mu_x_last, logvar_x_last, d_last, elementwise=True)  # Shape: [batch, feats]

                # Handle potential NaNs/Infs per feature
                if not torch.all(torch.isfinite(nll_scores_feature)):
                    print(
                        f"Warning: NaN/Inf per-feature scores detected test batch {idx}. Replacing.")
                    max_finite_score = torch.finfo(
                        nll_scores_feature.dtype).max / 10
                    nll_scores_feature = torch.nan_to_num(
                        nll_scores_feature, nan=max_finite_score, posinf=max_finite_score, neginf=0.0)

                scores_list_per_feature.append(
                    nll_scores_feature.cpu())  # Append [batch, feats]

        if not scores_list_per_feature:
            # Return 2D array of zeros
            return np.zeros((len(data), self.feats))

        # Concatenate along batch dimension -> [total_windows, feats]
        scores_per_feature_unpadded = torch.cat(
            scores_list_per_feature, dim=0).numpy()

        # --- Padding (Now needs to handle 2D) ---
        n_samples_original = len(data)
        # Use helper for padding (will modify _pad_array next)
        padded_scores_per_feature = _pad_array(
            scores_per_feature_unpadded, n_samples_original, self.win_size)

        # Note: We don't store this in self.__anomaly_score, which holds the total score
        return padded_scores_per_feature  # Return 2D padded scores

    def get_interpretation(self, data_window):
        """
        Provides anomaly interpretation for the last point of a given data window.

        Calculates the NLL contribution for each feature of the last point
        and returns them ranked in descending order (highest NLL first).

        Args:
            data_window (np.ndarray): A single window of time series data,
                                      shape [window_size, num_features].

        Returns:
            tuple: A tuple containing:
                - ranked_feature_indices (np.ndarray): Indices of features ranked by NLL contribution (desc).
                - ranked_nll_scores (np.ndarray): Corresponding NLL scores for each feature.
                Returns (None, None) if input is invalid or processing fails.
        """
        if not isinstance(data_window, np.ndarray):
            print("Error: Input data_window must be a numpy array.")
            return None, None
        if data_window.ndim != 2 or data_window.shape[0] != self.win_size or data_window.shape[1] != self.feats:
            print(
                f"Error: Input data_window shape mismatch. Expected [{self.win_size}, {self.feats}], got {data_window.shape}")
            return None, None
        if self.model is None:
            print("Error: Model not trained or loaded.")
            return None, None

        self.model.eval()  # Ensure model is in evaluation mode

        # Prepare input tensor
        window_tensor = torch.from_numpy(data_window).float().unsqueeze(
            0).to(self.device)  # Add batch dim [1, win, feats]

        with torch.no_grad():
            try:
                # Forward pass to get reconstruction parameters
                mu_x, logvar_x, _, _, _, _, _ = self.model(window_tensor)
                # mu_x, logvar_x shape: [1, win_size, n_feats]

                # --- Calculate NLL score for EACH FEATURE of the LAST point ---
                mu_x_last = mu_x[0, -1, :]      # Shape: [n_feats]
                logvar_x_last = logvar_x[0, -1, :]  # Shape: [n_feats]
                d_last = window_tensor[0, -1, :]    # Shape: [n_feats]

                # Calculate NLL per feature using the modified helper
                nll_per_feature = gaussian_nll(
                    mu_x_last, logvar_x_last, d_last, elementwise=True)  # Shape: [n_feats]

                # Handle potential NaNs/Infs
                if not torch.all(torch.isfinite(nll_per_feature)):
                    print(
                        f"Warning: NaN/Inf interpretation scores detected. Replacing.")
                    max_finite_score = torch.finfo(
                        nll_per_feature.dtype).max / 10
                    nll_per_feature = torch.nan_to_num(
                        nll_per_feature, nan=max_finite_score, posinf=max_finite_score, neginf=0.0)

                nll_scores_np = nll_per_feature.cpu().numpy()

                # --- Rank features based on NLL score (descending) ---
                # Indices of features from highest NLL to lowest
                ranked_feature_indices = np.argsort(nll_scores_np)[::-1]
                # Scores sorted accordingly
                ranked_nll_scores = nll_scores_np[ranked_feature_indices]

                return ranked_feature_indices, ranked_nll_scores

            except Exception as e:
                print(f"Error during interpretation forward pass: {e}")
                traceback.print_exc()
                return None, None

    # --- anomaly_score() and param_statistic() remain the same ---

    def anomaly_score(self) -> np.ndarray:
        if self.__anomaly_score is None:
            return None
        return self.__anomaly_score

    def param_statistic(self, save_file): pass

# --- run_... functions remain the same ---


def run_Custom_AD_Unsupervised(data, HP):
    clf = Custom_AD(feats=data.shape[1], **HP)
    clf.fit(data)
    score = clf.decision_function(data)
    return score


def run_Custom_AD_Semisupervised(data_train, data_test, **HP):
    num_features = data_train.shape[1]
    clf = Custom_AD(feats=num_features, **HP)
    clf.fit(data_train)
    raw_score = clf.decision_function(data_test)
    return raw_score, clf

# --- Helper and Visualization Functions ---


def _pad_array(data_unpadded, n_samples_original, sliding_window):
    """Pads a 1D or 2D numpy array (along axis 0) to match the original number of samples."""
    # Input validation
    if not isinstance(data_unpadded, np.ndarray):
        try:
            data_unpadded = np.array(data_unpadded)
        except:
            # Return shape based on target length and inferred dimensions
            ndim = data_unpadded.ndim if hasattr(data_unpadded, 'ndim') else 1
            shape = (n_samples_original,) if ndim == 1 else (n_samples_original,
                                                             data_unpadded.shape[1] if ndim > 1 and hasattr(data_unpadded, 'shape') else 1)
            return np.zeros(shape)

    if data_unpadded.ndim == 0:  # Handle scalar input gracefully
        # Convert scalar to 1-element array
        data_unpadded = data_unpadded.reshape(1)

    # Determine target shape based on original samples and feature dimension (if any)
    target_shape = (n_samples_original,)
    if data_unpadded.ndim == 2:
        target_shape = (n_samples_original, data_unpadded.shape[1])
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
        return data_unpadded.copy()
    elif n_unpadded > n_samples_original:
        # print(f"Warning in _pad_array: Unpadded data ({n_unpadded}) longer than original ({n_samples_original}). Truncating.")
        return data_unpadded[:n_samples_original].copy()
    # else: n_unpadded < n_samples_original, so padding is needed

    # --- Calculate padding amounts (Center-biased padding) ---
    if sliding_window <= 1:
        pad_before = 0
        pad_after = n_samples_original - n_unpadded
    else:
        # Ensure calculation works even if sliding_window is odd
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
        pad_before = (sliding_window - 1) // 2
        pad_after = n_samples_original - n_unpadded - pad_before

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
            f"Error during np.pad in _pad_array: {e}. Returning zeros with target shape.")
        return np.zeros(target_shape, dtype=data_unpadded.dtype if hasattr(data_unpadded, 'dtype') else float)

    # --- Final length verification (np.pad should guarantee this) ---
    if padded.shape[0] != n_samples_original:
        print(
            f"CRITICAL Warning in _pad_array: Final padded length {padded.shape[0]} != target {n_samples_original} despite using np.pad. Resizing forcefully.")
        # Force resize if something went extremely wrong
        if padded.shape[0] < n_samples_original:
            # Re-pad to fix size
            extra_pad = n_samples_original - padded.shape[0]
            pad_width = [(0, extra_pad)] + [(0, 0)] * (padded.ndim - 1)
            padded = np.pad(padded, pad_width, mode='edge')
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


def _find_clusters(indices):
    """Finds contiguous clusters of indices."""
    if not isinstance(indices, (list, np.ndarray)) or len(indices) == 0:
        return []
    # Ensure indices are sorted and unique integers
    try:
        indices = np.unique(np.asarray(indices, dtype=int))
    except ValueError:
        # print("Warning (_find_clusters): Indices could not be converted to integers. Returning empty list.")
        return []

    if len(indices) == 0:
        return []  # Handle empty after unique/conversion

    clusters = []
    if len(indices) == 1:  # Handle single index case
        return [(indices[0], indices[0])]

    # Find differences between consecutive indices
    diffs = np.diff(indices)

    # Points where the difference is greater than 1 mark cluster breaks
    split_points = np.where(diffs > 1)[0]

    # Determine start and end of clusters
    start_indices = np.insert(indices[split_points + 1], 0, indices[0])
    end_indices = np.append(indices[split_points], indices[-1])

    # Create list of (start, end) tuples
    clusters = list(zip(start_indices, end_indices))

    return clusters


# --- Modified visualize_errors ---
# Add inset_axes to imports (should already be there)

def visualize_errors(raw_data, extended_label, score, per_feature_nll_scores, # Added per_feature_nll_scores
                     fp_clusters, # Keep fp_clusters parameter if needed elsewhere, but not used for main plotting line
                     window_context, base_plot_filename, plot_dir, model_prefix="",
                     chunk_size=1500, chunk_indices_to_plot=None,
                     interpretation_map=None, feature_names=None, top_n_interpretation=None):
    """
    Visualizes specific chunks of the time series.
    Plots score on all dimensions AND per-feature NLL contribution.
    If interpretation_map is provided, adds an inset bar plot.
    """
    # Style
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except: plt.style.use('seaborn-whitegrid')

    # --- Input Validation ---
    n_samples = len(extended_label)
    valid_input = True
    if not isinstance(raw_data, np.ndarray) or raw_data.ndim not in [1, 2]:
        valid_input = False
    if not isinstance(extended_label, np.ndarray) or extended_label.ndim != 1:
        valid_input = False
    if not isinstance(score, np.ndarray) or score.ndim != 1:
        valid_input = False
    if valid_input:
        if raw_data.shape[0] != n_samples:
            valid_input = False
            print("Vis Error: raw_data length mismatch")
        if score.shape[0] != n_samples:
            valid_input = False
            print("Vis Error: score length mismatch")
    
    if not isinstance(per_feature_nll_scores, np.ndarray) or per_feature_nll_scores.ndim != 2:
        valid_input = False
        print("Vis Error: per_feature_nll_scores must be a 2D numpy array.")
    elif per_feature_nll_scores.shape[0] != n_samples:
         valid_input = False
         print("Vis Error: per_feature_nll_scores length mismatch.")

    n_dims = raw_data.shape[1] if raw_data.ndim == 2 else 1
    if valid_input and per_feature_nll_scores.shape[1] != n_dims:
        valid_input = False
        print(f"Vis Error: per_feature_nll_scores feature dimension ({per_feature_nll_scores.shape[1]}) mismatch with raw data ({n_dims}).")
    if not valid_input:
        print(f"Error ({model_prefix}): Invalid input for visualize_errors.")
        return

    if not chunk_indices_to_plot:
        print(
            f"Info ({model_prefix}): No specified chunks to plot. Skipping visualization.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    num_subplots = n_dims
    fig_height = max(8, 2.5 * num_subplots) if num_subplots > 0 else 8

    chunks_to_iterate = sorted(list(set(chunk_indices_to_plot)))
    print(f"Generating specified plot chunk(s) with feature NLL: {chunks_to_iterate}...")

    # --- Loop Through Chunks ---
    for i_chunk in chunks_to_iterate:
        chunk_start = i_chunk * chunk_size
        chunk_end = min(n_samples, (i_chunk + 1) * chunk_size)
        time_range_chunk = np.arange(chunk_start, chunk_end)
        if len(time_range_chunk) == 0: continue

        # --- Find first interpretable anomaly index within this chunk ---
        first_interp_idx_in_chunk = None
        interp_data_for_chunk = None
        if interpretation_map:
            indices_in_chunk = set(time_range_chunk)
            interpretable_indices_in_chunk = sorted(
                list(indices_in_chunk.intersection(interpretation_map.keys())))
            if interpretable_indices_in_chunk:
                first_interp_idx_in_chunk = interpretable_indices_in_chunk[0]
                interp_data_for_chunk = interpretation_map.get(
                    first_interp_idx_in_chunk)

        # --- Create Figure for Time Series Chunk ---
        if num_subplots == 0: continue
        try:
            fig_ts, axes = plt.subplots(num_subplots, 1, figsize=(14, fig_height), sharex=True, squeeze=False)
            axes = axes.ravel()
        except Exception as e:
            print(f"Error creating subplots chunk {i_chunk+1}: {e}"); continue

        # --- Plot Dimensions ---
        for d in range(n_dims):
            ax_raw = axes[d]
            lns_plot = [] # Handles and labels for combined legend
            try:
                # --- Plot Raw Data, Label --- (Unchanged)
                data_to_plot = raw_data[chunk_start:chunk_end, d] if raw_data.ndim == 2 else raw_data[chunk_start:chunk_end]
                raw_label_str = f'Raw Data (Dim {d})' if n_dims > 1 else 'Raw Data'
                ln_raw = ax_raw.plot(time_range_chunk, data_to_plot, label=raw_label_str, color='steelblue', linewidth=1.0)
                ax_raw.set_ylabel(raw_label_str, color='steelblue')
                ax_raw.tick_params(axis='y', labelcolor='steelblue')
                ax_raw.grid(True, linestyle='--', alpha=0.6)
                lns_plot.extend(ln_raw)

                ax_raw_label = ax_raw.twinx()
                ln_raw_lbl = ax_raw_label.plot(time_range_chunk, extended_label[chunk_start:chunk_end], label='Extended Label', color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.7)
                ax_raw_label.set_ylabel('Label', color='darkgreen')
                ax_raw_label.tick_params(axis='y', labelcolor='darkgreen')
                ax_raw_label.set_ylim(-0.1, 1.1)
                lns_plot.extend(ln_raw_lbl)

                # --- Plot Scores (Total and Per-Feature) ---
                ax_score = ax_raw.twinx()
                ax_score.spines["right"].set_position(("axes", 1.06)) # Position the score axis

                # Plot TOTAL Score (original orange line)
                ln_scr_total = ax_score.plot(
                    time_range_chunk, score[chunk_start:chunk_end], # Use the 1D score array
                    label=f'Total Score ({model_prefix})',
                    color='orange', linestyle='-', alpha=0.8, linewidth=1.2)
                lns_plot.extend(ln_scr_total)

                # Plot PER-FEATURE NLL Score for this dimension (d)
                # This line shows the "contribution" of this feature's raw data to the anomaly score
                feature_nll = per_feature_nll_scores[chunk_start:chunk_end, d]
                ln_scr_feat = ax_score.plot(
                    time_range_chunk, feature_nll,
                    label=f'NLL (Dim {d})', # Label specific to dimension
                    color='firebrick', # Choose a different color
                    linestyle='--', alpha=0.6, linewidth=1.0) # Different style/alpha
                lns_plot.extend(ln_scr_feat)

                # Set label and ticks for the shared score axis
                ax_score.set_ylabel('Anomaly Score / NLL', color='orange') # Generic label for the axis
                ax_score.tick_params(axis='y', labelcolor='orange')
                # Auto-scaling usually works well here for ylim

                # --- Add Vertical Line for Interpreted Anomaly --- (Unchanged)
                if first_interp_idx_in_chunk is not None:
                    line_label = f'Interpreted Pt ({first_interp_idx_in_chunk})' if d == 0 else "_nolegend_"
                    vl = ax_raw.axvline(x=first_interp_idx_in_chunk, color='red',
                                        linestyle='--', linewidth=1.2, label=line_label, alpha=0.9)
                    if d == 0:  # Add vline to legend only once
                        lns_plot.append(vl)

                # --- Legend (Create Combined Legend) ---
                plot_labs = [l.get_label() for l in lns_plot]
                unique_handles_labels = {}
                for h, l in zip(lns_plot, plot_labs):
                    if l != "_nolegend_" and l not in unique_handles_labels:
                        unique_handles_labels[l] = h
                # Place legend carefully to avoid overlap
                ax_raw.legend(unique_handles_labels.values(), unique_handles_labels.keys(),
                              loc='upper left', fontsize='small', frameon=True, facecolor='white', framealpha=0.8)

            except Exception as e:
                print(f"Error plotting subplot dim {d} chunk {i_chunk+1}: {e}")
                traceback.print_exc() # Print traceback for plotting errors


        # --- Add Inset Interpretation Plot --- (Unchanged)
        if first_interp_idx_in_chunk is not None and interp_data_for_chunk is not None:
            try:
                ranked_indices = interp_data_for_chunk['ranked_indices']
                ranked_scores = interp_data_for_chunk['ranked_scores']
                # Original number before top_n
                num_total_features = len(ranked_indices)

                features_to_plot = ranked_indices
                scores_to_plot = ranked_scores
                plot_title_suffix = ""
                if top_n_interpretation is not None and top_n_interpretation > 0 and top_n_interpretation < len(ranked_indices):
                    features_to_plot = ranked_indices[:top_n_interpretation]
                    scores_to_plot = ranked_scores[:top_n_interpretation]
                    plot_title_suffix = f" - Top {top_n_interpretation}"

                if feature_names is not None and len(feature_names) == num_total_features:
                    x_labels = [feature_names[i] for i in features_to_plot]
                else:
                    x_labels = [f"Feat_{i}" for i in features_to_plot]

                # Create inset axes on the *last* subplot
                ax_inset = inset_axes(axes[-1], width="40%", height="35%", loc='upper right',
                                      # Adjust anchor position slightly below main plot area
                                      bbox_to_anchor=(0.05, -0.15, 1, 1),
                                      bbox_transform=axes[-1].transAxes)

                ax_inset.bar(range(len(scores_to_plot)), scores_to_plot, color='orangered', alpha=0.7)
                ax_inset.set_xlabel("Feature", fontsize=7)
                ax_inset.set_ylabel("NLL Contribution", fontsize=7)
                ax_inset.set_title(
                    f"Interpretation: Idx {first_interp_idx_in_chunk}{plot_title_suffix}", fontsize=8)
                ax_inset.set_xticks(range(len(scores_to_plot)))
                ax_inset.set_xticklabels(
                    x_labels, rotation=90, ha='center', fontsize=6)
                ax_inset.tick_params(axis='y', labelsize=6)
                ax_inset.grid(axis='y', linestyle=':', alpha=0.6)

            except Exception as e:
                print(f"Error creating inset interpretation plot for index {first_interp_idx_in_chunk}: {e}")
        # --- End Inset Plotting ---

        # --- Final Figure Adjustments for Time Series Plot ---
        try:
            axes[-1].set_xlabel('Time Index')
            title = f'{model_prefix} - {base_plot_filename}\nChunk Index {i_chunk} (Indices {chunk_start}-{chunk_end-1}) - Scores & Feat NLL' # Updated title
            if first_interp_idx_in_chunk is not None:
                title += f' / Interpretation Inset: Idx {first_interp_idx_in_chunk}'

            fig_ts.suptitle(title, fontsize=11)
            fig_ts.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly if needed

            # Update filename to reflect content change
            plot_filename_ts = os.path.join(
                plot_dir, f"{base_plot_filename}_{model_prefix}_ChunkIdx_{i_chunk}_{chunk_start}-{chunk_end-1}_ScoresFeatNLL_InterpInset.png" if first_interp_idx_in_chunk is not None else f"{base_plot_filename}_{model_prefix}_ChunkIdx_{i_chunk}_{chunk_start}-{chunk_end-1}_ScoresFeatNLL.png")

            # *** SAVE WITH HIGHER DPI ***
            plt.savefig(plot_filename_ts, bbox_inches='tight', dpi=300) # Added dpi=300

            print(f"  Time series plot saved (High Res): {os.path.basename(plot_filename_ts)}")
        except Exception as e:
            print(f"Error saving plot chunk {i_chunk+1}: {e}")
        finally:
            plt.close(fig_ts)

    print(f"Chunk plotting complete for {base_plot_filename} ({model_prefix}). Check '{plot_dir}'.")


# ----------------------------------------------------
# Set random seed
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Define BASE_METRIC_ORDER (excluding VUS-PR as it will be prioritized)
BASE_METRIC_ORDER = [
    "VUS-ROC", "AUC-PR", "AUC-ROC", "Affiliation-F",
    "R-based-F1", "Event-based-F1", "PA-F1", "Standard-F1"
]

# --- Helper Function for CSV Column Ordering ---
# Unchanged - seems robust enough


def get_ordered_columns(current_columns):
    """
    Orders columns for the results CSV. Adapts based on available models.
    """
    # Adaptable based on MODEL_NAMES defined in main
    # Map name to prefix (can be different if needed)
    model_prefixes = {name: name for name in MODEL_NAMES}

    final_order = []
    processed_columns = set()

    # 1. Filename
    if 'filename' in current_columns:
        final_order.append('filename')
        processed_columns.add('filename')

    # 2. Priority Metrics (VUS-PR for all models)
    for model_prefix in model_prefixes.values():
        col = f'{model_prefix}_VUS-PR'
        if col in current_columns:
            final_order.append(col)
            processed_columns.add(col)

    # 3. Other metrics, then other info for each model
    for model_prefix in model_prefixes.values():
        # Other Metrics
        other_metrics = [f'{model_prefix}_{m}' for m in BASE_METRIC_ORDER]
        for col in other_metrics:
            if col in current_columns:
                final_order.append(col)
                processed_columns.add(col)
        # Other Info (Runtime, Counts, Errors)
        other_info = [
            f'{model_prefix}_runtime',
            f'{model_prefix}_FP_count_ext',
            f'{model_prefix}_FN_count_ext',
            f'{model_prefix}_Error',
            f'{model_prefix}_Scaling_Error',
            f'{model_prefix}_Metrics_Threshold_Error',
            f'{model_prefix}_Analysis_Error',
            f'{model_prefix}_Metrics_Error',
            f'{model_prefix}_Metrics_Result'
        ]
        # Add specific model info if needed (like best window size)
        other_info.extend(
            [f'{model_prefix}_Best_Window_VUS_ROC', f'{model_prefix}_Best_Window_VUS_PR'])

        for col in other_info:
            if col in current_columns and col not in processed_columns:  # Avoid duplicates
                final_order.append(col)
                processed_columns.add(col)

    # 4. Add any remaining columns alphabetically (e.g., general 'Error' column)
    remaining_columns = sorted(list(set(current_columns) - processed_columns))
    final_order.extend(remaining_columns)

    return final_order

# --- End Helper Function ---


if __name__ == '__main__':
    overall_start_time = time.time()
    # --- Configuration ---
    FILE_LIST_PATH = os.path.join('Datasets', 'File_List', 'TSB-AD-M-Eva.csv')
    DATA_DIR = os.path.join('Datasets', 'TSB-AD-M')
    PROGRESS_FILE_PREFIX = "progress_Custom_AD"
    RESULTS_CSV_PATH = "Custom_AD_detailed_results.csv"
    MODEL_NAMES = ['Custom_AD']  # Only running Custom_AD
    VISUALIZE_ERRORS = True
    # Find optimal HPs from dict if available (assumes dict structure)
    # Example: Update win_size from a potentially pre-defined optimal dict
    # filename_base = os.path.basename(FILE_LIST_PATH).split('.')[0] # Need to get filename inside loop
    # if Optimal_Multi_algo_HP_dict and filename_base in Optimal_Multi_algo_HP_dict:
    #      if 'Custom_AD' in Optimal_Multi_algo_HP_dict[filename_base]:
    #           optimal_hp = Optimal_Multi_algo_HP_dict[filename_base]['Custom_AD']
    #           if 'win_size' in optimal_hp: HP_CONFIG['Custom_AD']['win_size'] = optimal_hp['win_size']
    #           if 'lr' in optimal_hp: HP_CONFIG['Custom_AD']['lr'] = optimal_hp['lr']
    #           # Add other HPs as needed
    # NOTE: HP loading needs to be done *inside* the file loop if HPs are file-specific.

    # Ensure metricor is available
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
                        f"Warning: Progress file {progress_file} contained unexpected data type ({type(loaded_data)}). Reinitializing progress.")
                    processed_files = set()
            print(
                f"Loaded progress: {len(processed_files)} files previously processed.")
        except (pickle.UnpicklingError, EOFError, TypeError, AttributeError) as e:
            print(
                f"Warning: Could not load or parse progress file {progress_file}: {e}. Starting fresh.")
            processed_files = set()
        except Exception as e:
            print(
                f"Warning: An unexpected error occurred loading progress file {progress_file}: {e}. Starting fresh.")
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
                    f"Total unique files processed (including existing results): {len(processed_files)}")
            else:
                print(
                    f"Warning: Existing results file {RESULTS_CSV_PATH} is missing 'filename' column. Ignoring existing results.")
                all_results = []
        except pd.errors.EmptyDataError:
            print(
                f"Warning: Existing results file {RESULTS_CSV_PATH} is empty. Starting fresh.")
            all_results = []
        except Exception as e:
            print(
                f"Warning: Could not load or parse existing results from {RESULTS_CSV_PATH}: {e}. Starting without them.")
            all_results = []

    # --- Get File List ---
    try:
        file_list_df = pd.read_csv(FILE_LIST_PATH)
        if file_list_df.empty or file_list_df.shape[1] == 0:
            print(
                f"Warning: File list {FILE_LIST_PATH} is empty or has no columns.")
            filenames = []
        else:
            # Assuming filename is the first column
            filenames = file_list_df.iloc[:, 0].dropna().unique().tolist()
        total_files = len(filenames)
        print(
            f"Found {total_files} unique valid filenames listed in {FILE_LIST_PATH}")
    except FileNotFoundError:
        print(f"Error: File list {FILE_LIST_PATH} not found.")
        filenames = []
        total_files = 0
    except Exception as e:
        print(f"Error reading file list {FILE_LIST_PATH}: {e}")
        filenames = []
        total_files = 0

    # Determine files to process in this run
    files_to_process = [
        f for f in filenames if f not in processed_files and isinstance(f, str) and f.strip()]
    total_to_process_this_run = len(files_to_process)
    processed_files_count_this_run = 0
    print(f"Starting processing for {total_to_process_this_run} new files.")

    # --- Main Loop ---
    for i, filename in enumerate(files_to_process):
        print(
            f"\nProcessing new file {i+1}/{total_to_process_this_run}: {filename} (Overall index: {filenames.index(filename)+1}/{total_files})")
        file_path = os.path.join(DATA_DIR, filename)
        file_success = False
        current_file_results = {'filename': filename}
        model_data_for_vis = {}

        try:
            # --- Data Loading ---
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}. Skipping.")
                current_file_results['Error'] = 'File not found'
                # Do not add to processed_files here
                all_results.append(current_file_results)  # Record the error
                continue  # Skip to next file

            df = pd.read_csv(file_path)
            print(f"  Step 1: Read CSV done")

            # --- Basic Data Validation ---
            if 'Label' not in df.columns:
                print(
                    f"Warning: Skipping {filename} ('Label' column missing).")
                current_file_results['Error'] = 'Label column missing'
                all_results.append(current_file_results)
                continue
            if df.shape[1] < 2:  # Need at least one feature + Label
                print(
                    f"Warning: Skipping {filename} (needs >=1 feature column besides 'Label').")
                current_file_results['Error'] = 'Insufficient columns (<2)'
                all_results.append(current_file_results)
                continue

            try:
                # Convert label first, as it's crucial
                df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
            except Exception as e:
                print(
                    f"Warning: Error converting 'Label' column to numeric for {filename}: {e}. Skipping file.")
                current_file_results['Error'] = f'Label column conversion error: {e}'
                all_results.append(current_file_results)
                continue

            initial_rows = len(df)
            # Drop rows where *any* column has NaN (features or label)
            df = df.dropna()
            rows_after_dropna = len(df)
            if rows_after_dropna < initial_rows:
                print(
                    f"  Info: Dropped {initial_rows - rows_after_dropna} rows with NaN values.")

            if df.empty:
                print(
                    f"Warning: Skipping {filename} (empty after dropping NaNs).")
                current_file_results['Error'] = 'Empty after dropna'
                all_results.append(current_file_results)
                continue

            # Separate features and label *after* dropna
            data = df.iloc[:, :-1].values
            # Label already checked for numeric conversion
            label = df['Label'].values.astype(int)

            try:
                # Convert feature data to float
                data = data.astype(float)
            except ValueError as ve:
                print(
                    f"Warning: Skipping {filename} (non-numeric data in feature columns): {ve}")
                current_file_results['Error'] = f'Non-numeric feature data: {ve}'
                all_results.append(current_file_results)
                continue

            n_samples = data.shape[0]
            n_features = data.shape[1]
            if n_samples < 2:  # Need at least 2 points for some operations
                print(
                    f"Warning: Skipping {filename} (insufficient data points after dropna: {n_samples}).")
                current_file_results['Error'] = f'Insufficient data points (<2): {n_samples}'
                all_results.append(current_file_results)
                continue
            if n_features == 0:
                print(
                    f"Warning: Skipping {filename} (zero feature columns found).")
                current_file_results['Error'] = 'Zero feature columns'
                all_results.append(current_file_results)
                continue
            print(f"  Data Shape: {data.shape}, Label Shape: {label.shape}")
            # --- End Data Validation ---

            # --- Sliding Window Determination (Needed for get_metrics) ---
            # Using rank=1 often gives a reasonable window size estimate
            slidingWindow = find_length_rank(data, rank=1)
            print(f"  Estimated Sliding Window: {slidingWindow}")
            # --- End Sliding Window ---

            # --- Process Each Model (just Custom_AD) ---
            model_processing_failed = False
            for model_name in MODEL_NAMES:  # Loop iterates once
                print(f"    Running Model: {model_name}")
                # --- Load File-Specific HPs (Example) ---
                hp_to_use = {'win_size': 100, 'lr': 0.002,
                             'patience': 40, 'epochs': 200, }
                # Get filename without extension
                filename_base = os.path.splitext(filename)[0]
                print(f"      Using HPs: {hp_to_use}")
                # Ensure win_size is appropriate
                if 'win_size' in hp_to_use:
                    # Basic sanity check on win_size
                    hp_to_use['win_size'] = max(
                        1, min(hp_to_use['win_size'], n_samples // 2))
                    print(
                        f"      Adjusted win_size (if needed): {hp_to_use['win_size']}")

                anomaly_scores_raw = None
                start_time = time.time()

                # --- Run Model ---
                try:
                    # --- !!! MODIFIED TRAINING SPLIT LOGIC !!! ---
                    try:
                        filename_parts = filename_base.split('_')
                        if len(filename_parts) >= 3:  # Ensure there are enough parts
                            # Use the third-to-last part
                            train_end_index_str = filename_parts[-3]
                        else:
                            # Fallback if filename format is unexpected
                            print(
                                f"Warning: Filename '{filename_base}' does not have enough parts for '-3' index split. Using default 30% split.")
                            train_end_index_str = None  # Signal to use default
                            train_end_index = int(0.3 * n_samples)

                        # Attempt conversion if we got a string index
                        if train_end_index_str is not None:
                            try:
                                train_end_index = int(train_end_index_str)
                                # Basic sanity check: ensure index is within bounds and reasonable
                                if not (0 < train_end_index < n_samples):
                                    print(
                                        f"Warning: Parsed train index {train_end_index} (from part '{train_end_index_str}') out of bounds (0, {n_samples}). Using default 30% split.")
                                    train_end_index = int(0.3 * n_samples)
                            except ValueError:
                                print(
                                    f"Warning: Could not parse '{train_end_index_str}' (from part -3) as train index. Using default 30% split.")
                                train_end_index = int(
                                    0.3 * n_samples)  # Default fallback

                        # If using default split (e.g., 30%), ensure it's at least 1
                        train_end_index = max(1, train_end_index)

                        data_train = data[:train_end_index, :]

                        # Ensure data_train has enough samples for at least one window + validation split logic within fit
                        # (Let the Custom_AD class handle the internal validation split)
                        # Need at least window + 2 for potential internal split
                        min_train_samples_needed = hp_to_use.get(
                            'win_size', 10) + 2
                        if data_train.shape[0] < min_train_samples_needed:
                            print(
                                f"Warning: Training data size ({data_train.shape[0]}) based on index {train_end_index} might be too small for window/validation. Trying default 30% split.")
                            train_end_index = max(min_train_samples_needed, int(
                                0.3 * n_samples))  # Use larger of min needed or default
                            # Ensure test data exists
                            train_end_index = min(
                                train_end_index, n_samples - 1)
                            data_train = data[:train_end_index, :]
                            print(
                                f"      Adjusted train split to index {train_end_index}")

                        print(
                            f"      Using train data up to index: {train_end_index} (Shape: {data_train.shape}) <-- Using filename part -3")
                        # --- !!! END MODIFIED TRAINING SPLIT LOGIC !!! ---

                    except Exception as split_e:
                        print(
                            f"Error determining train/test split from filename (using part -3): {split_e}. Using default 30% split.")
                        train_end_index = int(0.3 * n_samples)
                        # Ensure at least 1 sample
                        data_train = data[:max(1, train_end_index), :]

                    # Ensure there's test data left (the full 'data' is passed as test set)
                    if train_end_index >= n_samples:
                        print(
                            "Error: Training split index covers entire dataset. Cannot run semi-supervised.")
                        current_file_results[
                            f'{model_name}_Error'] = 'Invalid train/test split (covers all data)'
                        continue  # Skip model processing

                    # Run the semi-supervised function
                    anomaly_scores_raw, clf = run_Custom_AD_Semisupervised(
                        data_train, data, **hp_to_use)  # Pass full data as test

                    anomaly_scores_per_feature_raw = clf.decision_function_per_feature(
                        data)  # Returns 2D [samples, features]

                    run_time = time.time() - start_time
                    current_file_results[f'{model_name}_runtime'] = run_time
                    print(
                        f"    Step 3 ({model_name}): Model scoring done (runtime: {run_time:.2f}s)")

                    # --- Post-Run Checks ---
                    # Check TOTAL scores (used for metrics)
                    if anomaly_scores_raw is None or not isinstance(anomaly_scores_raw, np.ndarray) or anomaly_scores_raw.ndim != 1:
                        print(
                            f"Error: TOTAL score computation failed or invalid for {model_name}. Setting scores to zero.")
                        current_file_results[f'{model_name}_Error'] = 'Total Score computation failed or invalid'
                        anomaly_scores_raw = np.zeros(n_samples)
                        # If total score failed, per-feature likely did too or is unusable
                        anomaly_scores_per_feature_raw = np.zeros(
                            (n_samples, n_features))
                        continue  # Skip further processing for this model
                    elif len(anomaly_scores_raw) != n_samples:
                        print(
                            f"Error: TOTAL score length mismatch for {model_name} ({len(anomaly_scores_raw)} vs {n_samples}). Critical Issue.")
                        # Log error and attempt fix
                        current_file_results[
                            f'{model_name}_Error'] = f'Total score length mismatch ({len(anomaly_scores_raw)} vs {n_samples})'
                        adjusted_scores = np.zeros(n_samples)
                        copy_len = min(len(anomaly_scores_raw), n_samples)
                        if copy_len > 0:
                            adjusted_scores[:copy_len] = anomaly_scores_raw[:copy_len]
                        anomaly_scores_raw = adjusted_scores
                        # Also fix per-feature shape if total was wrong
                        anomaly_scores_per_feature_raw = np.zeros(
                            (n_samples, n_features))
                        # Don't continue here, let downstream handle potentially bad per-feature data if needed

                    # Check PER-FEATURE scores (used for visualization)
                    if anomaly_scores_per_feature_raw is None or not isinstance(anomaly_scores_per_feature_raw, np.ndarray) or anomaly_scores_per_feature_raw.ndim != 2:
                        print(
                            f"Warning: Per-feature score computation failed or invalid for {model_name}. Visualization will lack detail.")
                        if f'{model_name}_Error' not in current_file_results:
                            current_file_results[f'{model_name}_Error'] = 'Per-feature Score computation failed or invalid'
                        anomaly_scores_per_feature_raw = np.zeros(
                            (n_samples, n_features))  # Create dummy array
                    elif anomaly_scores_per_feature_raw.shape != (n_samples, n_features):
                        print(
                            f"Warning: Per-feature score shape mismatch ({anomaly_scores_per_feature_raw.shape} vs {(n_samples, n_features)}). Visualization might be incorrect.")
                        if f'{model_name}_Error' not in current_file_results:
                            current_file_results[
                                f'{model_name}_Error'] = f'Per-feature score shape mismatch ({anomaly_scores_per_feature_raw.shape})'
                        # Attempt to fix shape - pad/truncate first dim, zero-pad second if needed
                        temp_scores = np.zeros((n_samples, n_features))
                        copy_rows = min(
                            anomaly_scores_per_feature_raw.shape[0], n_samples)
                        copy_cols = min(
                            anomaly_scores_per_feature_raw.shape[1], n_features)
                        if copy_rows > 0 and copy_cols > 0:
                            temp_scores[:copy_rows,
                                        :copy_cols] = anomaly_scores_per_feature_raw[:copy_rows, :copy_cols]
                        anomaly_scores_per_feature_raw = temp_scores

                except MemoryError as mem_e:
                    print(
                        f"CRITICAL MemoryError running model {model_name}: {mem_e}")
                    traceback.print_exc()
                    current_file_results[f'{model_name}_Error'] = f'MemoryError during model run'
                    model_processing_failed = True  # Mark file as failed
                    break  # Stop processing models for this file
                except Exception as model_run_e:
                    print(
                        f"Critical Error running model {model_name}: {model_run_e}")
                    traceback.print_exc()
                    current_file_results[f'{model_name}_Error'] = f'Model run failed: {model_run_e}'
                    continue  # Skip rest of steps for this model

                # --- Score Scaling (MinMax 0-1) ---
                scaled_scores = np.zeros_like(anomaly_scores_raw)
                try:
                    # Handle potential NaNs/Infs in raw scores *before* scaling
                    scores_to_scale = np.nan_to_num(
                        anomaly_scores_raw, nan=0.0, posinf=np.inf, neginf=-np.inf)
                    finite_mask = np.isfinite(scores_to_scale)

                    if np.any(finite_mask):  # Check if there are any finite values
                        min_finite = np.min(scores_to_scale[finite_mask])
                        max_finite = np.max(scores_to_scale[finite_mask])
                        # Replace Infs with finite min/max
                        scores_to_scale[scores_to_scale == np.inf] = max_finite
                        scores_to_scale[scores_to_scale == -
                                        np.inf] = min_finite

                        score_min, score_max = np.min(
                            scores_to_scale), np.max(scores_to_scale)
                        score_range = score_max - score_min

                        if score_range > 1e-9:  # Avoid division by zero
                            scaled_scores = (
                                scores_to_scale - score_min) / score_range
                        elif len(scores_to_scale) > 0:  # All scores are the same finite value
                            scaled_scores = np.full_like(
                                scores_to_scale, 0.5)  # Assign midpoint
                        # else: scores_to_scale is empty, scaled_scores remains zeros

                    else:  # All values were NaN/Inf or array was empty
                        scaled_scores = np.zeros_like(
                            scores_to_scale)  # Assign zeros

                    # Clamp scaled scores to [0, 1] just in case of numerical issues
                    scaled_scores = np.clip(scaled_scores, 0.0, 1.0)

                except Exception as scale_e:
                    print(
                        f"Warning: Scaling failed for {model_name}: {scale_e}. Using raw scores clamped 0-1 as fallback.")
                    traceback.print_exc()
                    current_file_results[f'{model_name}_Scaling_Error'] = str(
                        scale_e)
                    # Fallback: Clamp raw scores (after nan_to_num)
                    try:
                        scores_clamped = np.nan_to_num(
                            anomaly_scores_raw, nan=0.0, posinf=1.0, neginf=0.0)
                        min_s, max_s = np.min(
                            scores_clamped), np.max(scores_clamped)
                        if max_s > min_s:
                            scaled_scores = (
                                scores_clamped - min_s) / (max_s - min_s)
                        else:
                            scaled_scores = np.full_like(scores_clamped, 0.5)
                        scaled_scores = np.clip(scaled_scores, 0.0, 1.0)
                    except Exception:  # Final fallback
                        scaled_scores = np.zeros_like(anomaly_scores_raw)

                print(f"    Step 6 ({model_name}): Scaling done")

                # --- Error Analysis Setup ---
                _FP_count_ext, _FN_count_ext = 0, 0
                _fp_clusters, _fn_clusters = [], []
                _pred_for_analysis = np.zeros_like(label)
                _extended_label = label.copy().astype(float)
                _extension_window = 1  # Will be recalculated

                try:
                    # 1. Extended Labels
                    if metricor and hasattr(metricor, 'range_convers_new') and hasattr(metricor, 'sequencing'):
                        try:
                            _seq = metricor.range_convers_new(label)
                            # Use the general slidingWindow determined earlier for extension consistency
                            _extension_window = max(1, slidingWindow // 2)
                            if _seq is not None and len(_seq) > 0:
                                _extended_label = metricor.sequencing(
                                    label, _seq, window=_extension_window)
                                print(
                                    f"      Extended labels generated with window: {_extension_window}")
                            else:
                                # No anomalies found, no extension needed
                                _extended_label = label.astype(float)
                                print(
                                    "      No anomaly sequences found for label extension.")
                        except Exception as ext_label_e:
                            print(
                                f"Warning: Error during extended label generation: {ext_label_e}. Using original labels.")
                            _extended_label = label.astype(float)
                            # Still estimate window for context
                            _extension_window = max(1, slidingWindow // 2)
                    else:
                        print(
                            "Warning: Metricor unavailable or missing methods for extended labels. Using original labels.")
                        _extended_label = label.astype(float)
                        _extension_window = max(
                            1, slidingWindow // 2)  # Estimate window

                    # 2. Prediction for Analysis (Median threshold on RAW scores)
                    # Use raw scores before scaling for analysis threshold, handle non-finites
                    valid_raw_scores = anomaly_scores_raw[np.isfinite(
                        anomaly_scores_raw)] if anomaly_scores_raw is not None else np.array([])
                    if len(valid_raw_scores) > 0:
                        # Check if there's variation in scores
                        if np.ptp(valid_raw_scores) > 1e-9:
                            _median_threshold = np.median(valid_raw_scores)
                            # Handle cases where median might be non-finite if input had issues (shouldn't happen with check above)
                            if not np.isfinite(_median_threshold):
                                _median_threshold = 0
                            # Apply threshold to original raw scores (handle NaNs by treating them as non-anomalous)
                            _pred_for_analysis = (np.nan_to_num(
                                anomaly_scores_raw, nan=_median_threshold-1) > _median_threshold).astype(int)
                            print(
                                f"      Analysis prediction threshold (median of raw): {_median_threshold:.4f}")
                        else:  # All valid scores are the same
                            print(
                                "      Analysis prediction: All valid raw scores are identical. Predicting all non-anomalous (0).")
                            _pred_for_analysis = np.zeros_like(
                                anomaly_scores_raw)
                    else:  # No valid scores
                        print(
                            "      Analysis prediction: No valid raw scores found. Predicting all non-anomalous (0).")
                        _pred_for_analysis = np.zeros_like(anomaly_scores_raw)

                    # 3. FP/FN Indices and Clusters (using Extended Label vs. Analysis Prediction)
                    if len(_extended_label) == len(_pred_for_analysis):
                        _fp_indices_ext = np.where(
                            (_pred_for_analysis == 1) & (_extended_label < 0.5))[0]
                        _fn_indices_ext = np.where(
                            (_pred_for_analysis == 0) & (_extended_label >= 0.5))[0]
                        _FP_count_ext = len(_fp_indices_ext)
                        _FN_count_ext = len(_fn_indices_ext)
                        _fp_clusters = _find_clusters(_fp_indices_ext)
                        _fn_clusters = _find_clusters(_fn_indices_ext)
                        print(
                            f"      Analysis ({model_name}): FP={_FP_count_ext}, FN={_FN_count_ext} | FP Clusters={len(_fp_clusters)}, FN Clusters={len(_fn_clusters)}")
                    else:
                        print(
                            f"Warning: Length mismatch during error analysis. Extended Label: {len(_extended_label)}, Analysis Pred: {len(_pred_for_analysis)}")
                        _FP_count_ext, _FN_count_ext = -1, -1  # Indicate error

                except Exception as e:
                    print(
                        f"Warning: Could not compute analysis details for {model_name}: {e}")
                    traceback.print_exc()
                    current_file_results[f'{model_name}_Analysis_Error'] = str(
                        e)
                    _FP_count_ext, _FN_count_ext = -1, -1  # Indicate error

                current_file_results[f'{model_name}_FP_count_ext'] = _FP_count_ext
                current_file_results[f'{model_name}_FN_count_ext'] = _FN_count_ext
                print(
                    f"    Step 8 ({model_name}): Error analysis details done")

                # --- Store data for visualization ---
                model_data_for_vis[model_name] = {
                    'anomaly_scores_raw': anomaly_scores_raw, # TOTAL raw score (1D)
                    'anomaly_scores_per_feature_raw': anomaly_scores_per_feature_raw, # PER-FEATURE raw scores (2D)
                    'pred_for_analysis': _pred_for_analysis,
                    'fp_clusters': _fp_clusters,
                    'fn_clusters': _fn_clusters, # Keep for finding chunks, but don't pass to plot func
                    'extended_label': _extended_label,
                    'extension_window': _extension_window
                }
                # ---------------------------------

                # --- Evaluate Metrics (using SCALED scores) ---
                evaluation_result = {}
                best_window_roc, best_window_pr = -1, -1  # Initialize defaults
                if get_metrics is None:
                    evaluation_result = {
                        "Metrics_Error": "get_metrics function unavailable"}
                    print("Warning: get_metrics function is not available.")
                elif scaled_scores is None or len(scaled_scores) != len(label):
                    error_msg = f"Invalid scaled scores for metrics (Length {len(scaled_scores) if scaled_scores is not None else 'None'} vs Label {len(label)})"
                    evaluation_result = {"Metrics_Error": error_msg}
                    print(f"Warning: {error_msg}")
                else:
                    try:
                        # Ensure scores are finite for evaluation
                        scores_eval = np.nan_to_num(
                            scaled_scores, nan=0.0, posinf=1.0, neginf=0.0)
                        label_eval = label  # Use original label
                        print(
                            f"\n    Calling get_metrics with {len(scores_eval)} scaled scores, {len(label_eval)} labels, window {slidingWindow}")

                        # Call get_metrics (assuming it handles scores and labels directly)
                        evaluation_result = get_metrics(
                            scores_eval, label_eval, slidingWindow=slidingWindow)

                        if not isinstance(evaluation_result, dict):
                            print(
                                f"Warning: get_metrics did not return a dictionary. Result: {evaluation_result}")
                            evaluation_result = {"Metrics_Result": str(
                                evaluation_result)}  # Store string representation

                        # Store all returned metrics prefixed with model name
                        for key, value in evaluation_result.items():
                            current_file_results[f'{model_name}_{key}'] = value

                        print(
                            f"      Metrics obtained: {list(evaluation_result.keys())}")

                    except TypeError as te:
                        print(
                            f"TypeError during get_metrics for {model_name}: {te}.")
                        traceback.print_exc()
                        current_file_results[f'{model_name}_Metrics_Error'] = f"TypeError: {te}"
                    except Exception as e:
                        print(
                            f"Error during get_metrics evaluation for {model_name}: {e}")
                        traceback.print_exc()
                        current_file_results[f'{model_name}_Metrics_Error'] = f"Evaluation failed: {e}"

                print(
                    f"    Step 9 ({model_name}): get_metrics evaluation attempted.")
                # --- End Evaluate Metrics ---

            # --- End Model Loop ---
            if model_processing_failed:
                print(
                    f"Stopping processing for {filename} due to critical model failure.")
                # Error already logged in current_file_results
                all_results.append(current_file_results)
                # Do *not* add to processed_files if critical error occurred
                continue  # Skip to next file

            # --- Print Combined Results ---
            print(f"\n      --- Evaluation Result Summary for: {filename} ---")
            printed_keys_summary = set(['filename'])

            def print_result(key, value):
                if isinstance(value, (float, np.floating)) and np.isfinite(value):
                    value_str = f"{value:.6f}"
                elif isinstance(value, (float, np.floating)):
                    value_str = f'"{str(value)}"'  # Quote non-finite floats
                elif isinstance(value, (int, np.integer)):
                    value_str = f"{value}"
                else:
                    value_str = f'"{str(value)}"'  # Quote strings and others
                print(f'          "{key}": {value_str},')
                printed_keys_summary.add(key)

            print(
                f'      "filename": "{current_file_results.get("filename", "N/A")}",')

            # Loop through models (just Custom_AD here)
            for model_key in MODEL_NAMES:
                print(f'      --- {model_key} Results ---')
                # Priority VUS-PR
                key = f'{model_key}_VUS-PR'
                if key in current_file_results:
                    print_result(key, current_file_results[key])

                # Other Metrics
                ordered_metric_keys = [
                    f"{model_key}_{m}" for m in BASE_METRIC_ORDER]
                for key in ordered_metric_keys:
                    if key in current_file_results:
                        print_result(key, current_file_results[key])

                # Other Info
                other_keys = [
                    f"{model_key}_runtime", f"{model_key}_FP_count_ext", f"{model_key}_FN_count_ext",
                    # Add best windows if available
                    f"{model_key}_Best_Window_VUS_ROC", f"{model_key}_Best_Window_VUS_PR",
                    f"{model_key}_Error", f"{model_key}_Scaling_Error", f"{model_key}_Metrics_Threshold_Error",
                    f"{model_key}_Analysis_Error", f"{model_key}_Metrics_Error", f"{model_key}_Metrics_Result"
                ]
                for key in other_keys:
                    if key in current_file_results and key not in printed_keys_summary:
                        print_result(key, current_file_results[key])

            print(f"      -----------------------------------------------------")
            # --- End Combined Results ---

            # --- Find Chunks for Visualization ---
            # Logic to find chunks containing TRUE anomalies and potentially FNs is kept
            # but fn_clusters will not be passed to the plotting function itself.
            chunk_size_vis = 1500
            chunk_indices_to_plot = set()
            if np.any(label):
                true_anomaly_indices = np.where(label == 1)[0]
                anomaly_clusters = _find_clusters(true_anomaly_indices)
            else:
                anomaly_clusters = []
            for start, end in anomaly_clusters:
                start_chunk_idx, end_chunk_idx = start // chunk_size_vis, end // chunk_size_vis
                chunk_indices_to_plot.update(
                    range(start_chunk_idx, end_chunk_idx + 1))
            # Add chunks with FNs (use _fn_clusters calculated earlier in analysis setup)
            # _fn_clusters should be available from the error analysis step
            if '_fn_clusters' in locals() and _fn_clusters:  # Check if _fn_clusters exists
                for start, end in _fn_clusters:
                    start_chunk_idx, end_chunk_idx = start // chunk_size_vis, end // chunk_size_vis
                    chunk_indices_to_plot.update(
                        range(start_chunk_idx, end_chunk_idx + 1))
            chunk_indices_list = sorted(list(chunk_indices_to_plot))
            if chunk_indices_list:
                print(
                    f"    Identified {len(chunk_indices_list)} chunk(s) containing True Anomalies or FNs for visualization: {chunk_indices_list}")
            else:
                print(f"    No chunks identified containing True Anomalies or FNs.")

            # --- Visualization Call ---
            if VISUALIZE_ERRORS and chunk_indices_list:
                plot_dir_base = "Custom_AD_Anomaly_Chunks_AllDims_FeatNLL" # Adjusted directory name
                base_fname = os.path.splitext(os.path.basename(filename))[0]
                print(f"  Step 10: Generating visualizations for {len(chunk_indices_list)} relevant chunk(s)...")
                for model_name_vis, vis_data in model_data_for_vis.items():
                    # Required keys for visualize_errors
                    required_keys = [
                        'anomaly_scores_raw', # Total score for the orange line
                        'anomaly_scores_per_feature_raw', # 2D scores for detailed lines
                        'extended_label',
                        'fp_clusters', # Still used by visualize_errors internally? Check func def. (Original doesn't use it for plotting)
                        'extension_window'
                    ]
                    if not all(k in vis_data and vis_data[k] is not None for k in required_keys):
                        print(f"Skipping visualization {model_name_vis}: Missing data.")
                        continue
                    # Check dimensions of per-feature scores
                    if not isinstance(vis_data['anomaly_scores_per_feature_raw'], np.ndarray) or \
                       vis_data['anomaly_scores_per_feature_raw'].ndim != 2 or \
                       vis_data['anomaly_scores_per_feature_raw'].shape != (n_samples, n_features):
                        print(f"Skipping visualization {model_name_vis}: Invalid per-feature score shape ({vis_data['anomaly_scores_per_feature_raw'].shape if isinstance(vis_data['anomaly_scores_per_feature_raw'], np.ndarray) else 'None'}). Expected ({n_samples}, {n_features}).")
                        continue

                    model_plot_dir = os.path.join(plot_dir_base, model_name_vis)
                    try:
                        # Call visualize_errors, passing BOTH total and per-feature scores
                        visualize_errors(
                            raw_data=data,
                            extended_label=vis_data['extended_label'],
                            score=vis_data['anomaly_scores_raw'], # TOTAL score (1D)
                            per_feature_nll_scores=vis_data['anomaly_scores_per_feature_raw'], # PER-FEATURE scores (2D)
                            fp_clusters=vis_data['fp_clusters'], # Pass FPs (though not plotted by default)
                            # fn_clusters argument removed/ignored by updated visualize_errors
                            window_context=vis_data['extension_window'],
                            base_plot_filename=base_fname,
                            plot_dir=model_plot_dir,
                            model_prefix=model_name_vis,
                            chunk_size=chunk_size_vis,
                            chunk_indices_to_plot=chunk_indices_list,
                            # --- Pass optional interpretation args if needed ---
                            interpretation_map=None, # Or pass actual map if generated
                            feature_names=None, # Or pass list of feature names
                            top_n_interpretation=None # Or specify N
                        )
                    except Exception as vis_e:
                        print(f"Error during visualize_errors call: {vis_e}")
                        traceback.print_exc()

            file_success = True  # Mark as success only if all steps complete without critical errors

        # --- Outer Exception Handling ---
        except FileNotFoundError:
            # Already handled at the start of the loop
            pass
        except pd.errors.EmptyDataError as ede:
            print(
                f"Pandas EmptyDataError processing {filename}: {ede}. Skipping.")
            if 'Error' not in current_file_results:
                current_file_results['Error'] = 'Pandas EmptyDataError'
            file_success = False
        except MemoryError as me:
            print(
                f"CRITICAL MemoryError processing file {filename}: {me}. Skipping.")
            traceback.print_exc()
            if 'Error' not in current_file_results:
                current_file_results['Error'] = 'MemoryError (Outer Loop)'
            file_success = False
        except Exception as e:
            print(f"Unexpected critical error processing {filename}: {e}")
            traceback.print_exc()
            if 'Error' not in current_file_results:
                current_file_results['Error'] = f'Critical processing failed: {e}'
            file_success = False

        # --- Append result AND Save Incrementally ---
        # Check if results for this file already exist (e.g., from previous failed run)
        existing_file_index = -1
        for idx, res in enumerate(all_results):
            if res.get('filename') == filename:
                existing_file_index = idx
                break

        if existing_file_index != -1:
            # Update existing record instead of appending duplicate
            all_results[existing_file_index].update(current_file_results)
            print(f"  Updated results for existing file entry: {filename}")
        else:
            # Append new result
            all_results.append(current_file_results)

        if file_success:
            processed_files.add(filename)  # Add to set *only* if successful
            processed_files_count_this_run += 1
            print(f"  File {filename} processed successfully.")
        else:
            print(
                f"  File {filename} processing marked as unsuccessful or skipped.")
            # Do not add to processed_files set

        # Save progress (set of successfully processed filenames)
        try:
            with open(progress_file, 'wb') as f:
                pickle.dump(processed_files, f)
        except Exception as e:
            print(f"Warning: Could not save progress to {progress_file}: {e}")

        # Save intermediate results CSV
        try:
            # Create DataFrame from potentially updated all_results list
            temp_df = pd.DataFrame(all_results)
            # Ensure filename uniqueness hasn't been violated (shouldn't if update logic works)
            if 'filename' in temp_df.columns:
                temp_df = temp_df.drop_duplicates(
                    subset=['filename'], keep='last')

            if not temp_df.empty:
                # Reorder columns before saving
                ordered_cols = get_ordered_columns(temp_df.columns.tolist())
                temp_df = temp_df.reindex(columns=ordered_cols)
            temp_df.to_csv(RESULTS_CSV_PATH, index=False)
        except Exception as e:
            print(
                f"Warning: Could not save intermediate results to {RESULTS_CSV_PATH}: {e}")
        # --- End Incremental Save ---

    # --- End File Loop ---

    # --- Final Summary ---
    print(f"\n--- Run Summary ---")
    print(
        f"Attempted processing for {total_to_process_this_run} listed new files.")
    print(
        f"Successfully processed {processed_files_count_this_run} new files in this run.")
    print(
        f"Total unique files marked as successfully processed (cumulative): {len(processed_files)}")

    if all_results:
        # Final save of results
        final_results_df = pd.DataFrame(all_results)
        if 'filename' in final_results_df.columns:
            # Ensure final uniqueness just in case
            final_results_df = final_results_df.drop_duplicates(
                subset=['filename'], keep='last')
        try:
            if not final_results_df.empty:
                # Use the ordering function for final save
                ordered_cols = get_ordered_columns(
                    final_results_df.columns.tolist())
                final_results_df = final_results_df.reindex(
                    columns=ordered_cols)
            final_results_df.to_csv(RESULTS_CSV_PATH, index=False)
            print(f"Final detailed results saved to {RESULTS_CSV_PATH}")
        except Exception as e:
            print(f"Error saving final results CSV: {e}")

        # --- Calculate and Print Averages ---
        # Filter based on successful processing (no critical errors logged for the file's main 'Error' or model-specific 'Error' fields)
        error_columns = [col for col in final_results_df.columns if 'Error' in col and col !=
                         'Metrics_Error' and col != 'Scaling_Error']  # Focus on critical errors
        # Include the general file error column if it exists
        error_columns.append('Error')

        successful_filter = pd.Series(
            [True] * len(final_results_df), index=final_results_df.index)
        for err_col in error_columns:
            if err_col in final_results_df.columns:
                # A file is unsuccessful if any critical error column has a non-null/non-empty value
                successful_filter &= final_results_df[err_col].isnull() | (
                    final_results_df[err_col] == '')

        successful_df = final_results_df[successful_filter]
        num_successful_files = len(successful_df)
        num_failed_files = len(final_results_df) - num_successful_files

        if not successful_df.empty:
            print(
                f"\n--- Average Metrics Across {num_successful_files} Successfully Processed Files ---")
            if num_failed_files > 0:
                failed_files_list = final_results_df[~successful_filter]['filename'].tolist(
                )
                print(
                    f"    ({num_failed_files} files excluded due to processing errors: {failed_files_list})")

            def print_average(col_key, prefix="Average"):
                if col_key in successful_df.columns:
                    # Split only on the first underscore
                    metric_name_parts = col_key.split('_', 1)
                    metric_name = metric_name_parts[1] if len(
                        metric_name_parts) > 1 else col_key

                    # Convert to numeric, coercing errors to NaN
                    numeric_col = pd.to_numeric(
                        successful_df[col_key], errors='coerce')

                    if numeric_col.notna().any():  # Check if there are any valid numbers
                        # nanmean ignores NaNs
                        avg_val = np.nanmean(numeric_col.astype(float))
                        value_str = f"{avg_val:.6f}" if np.isfinite(
                            avg_val) else 'NaN'
                        print(f'      "{prefix} {metric_name}": {value_str},')
                        return True
                    else:
                        # print(f'      "{prefix} {metric_name}": "N/A (No valid data)",') # Less verbose option
                        pass  # Don't print if no valid data
                return False

            # Loop through models (just Custom_AD) for averages
            for model_key in MODEL_NAMES:
                print(f"\n  --- {model_key} Averages ---")
                print(f"    -- Priority Metric --")
                print_average(f'{model_key}_VUS-PR')

                print(f"\n    -- Other Metrics --")
                metrics_printed = 0
                for metric in BASE_METRIC_ORDER:
                    if print_average(f'{model_key}_{metric}'):
                        metrics_printed += 1
                if metrics_printed == 0:
                    print("      No average metrics calculated (no valid data).")

                print(f"\n    -- Other Info --")
                print_average(f"{model_key}_runtime")
                print_average(f"{model_key}_FP_count_ext",
                              prefix="Avg FP Count (Analysis)")
                print_average(f"{model_key}_FN_count_ext",
                              prefix="Avg FN Count (Analysis)")
                # Add average best window if desired
                print_average(f"{model_key}_Best_Window_VUS_ROC",
                              prefix="Avg Best Window (VUS-ROC)")
                print_average(f"{model_key}_Best_Window_VUS_PR",
                              prefix="Avg Best Window (VUS-PR)")

            print("\n  ------------------------------------")

        else:
            print("\nNo successfully processed files found for averaging metrics.")
            if num_failed_files > 0:
                print(
                    f"({num_failed_files} files had critical processing errors recorded)")

    else:
        print("\nNo results generated or loaded in this run.")

    print(
        f"\nTotal execution time: {time.time() - overall_start_time:.2f} seconds")
