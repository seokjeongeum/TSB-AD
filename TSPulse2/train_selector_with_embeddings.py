# TSPulse2/train_selector_with_embeddings.py

import argparse
import os
import sys

import joblib
import lightgbm as lgbm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import copy

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "granite-tsfm")),
)
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction


def get_full_decoder_embeddings(model, past_values):
    """
    Extracts the full, flattened decoder embeddings for a batch of time series.
    This includes time, fft, and register token embeddings.
    """
    with torch.no_grad():
        outputs = model(past_values=past_values, return_loss=False)

    # decoder_hidden_state is already flattened along the patch dimension
    # Shape: (batch_size, num_channels, num_patches * d_model)
    # We flatten it further to get one vector per channel.
    embeddings = outputs.decoder_hidden_state.flatten(start_dim=1)
    return embeddings.cpu().numpy()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        # Wider and deeper MLP
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),  # Wider
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),  # a bit less dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),  # a bit less dropout
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    """A pre-activation residual block."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(size)
        self.act1 = nn.ReLU()
        self.l1 = nn.Linear(size, size)
        self.d = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(size)
        self.act2 = nn.ReLU()
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.act1(out)
        out = self.d(out)  # Dropout after activation
        out = self.l1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.l2(out)

        return identity + out


class ResMLP(nn.Module):
    """A simplified ResMLP for a small dataset."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            ResBlock(256, 0.5),
            ResBlock(256, 0.5),
            ResBlock(256, 0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionMLP(nn.Module):
    """An MLP with a self-attention layer, structured like a Transformer block."""

    def __init__(self, input_dim, num_classes, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),  # Feed-forward expansion
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(0.1)

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x_for_attn = x.unsqueeze(1)
        # Self-attention part
        attn_output, _ = self.attention(x_for_attn, x_for_attn, x_for_attn)
        x = x + self.dropout1(attn_output.squeeze(1))
        x = self.layer_norm1(x)

        # Feed-forward part
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.layer_norm2(x)

        return self.classifier(x)


class GRUClassifier(nn.Module):
    """A GRU-based classifier."""

    def __init__(self, input_dim, num_classes, hidden_dim=128, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension for GRU
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1, :, :])  # Get the last hidden state
        return out


class LSTMClassifier(nn.Module):
    """An LSTM-based classifier."""

    def __init__(self, input_dim, num_classes, hidden_dim=128, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1, :, :])
        return out


class TransformerClassifier(nn.Module):
    """A Transformer-based classifier."""

    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = x + self.pos_encoder
        output = self.transformer_encoder(x)
        output = output.squeeze(1)  # Remove sequence dimension
        return self.classifier(output)


class FNetLayer(nn.Module):
    """A simple FNet block that uses Fourier Transforms for token mixing."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_fft = torch.fft.fft(x, dim=-1)
        x_ifft = torch.fft.ifft(x_fft, dim=-1).real
        return self.norm(x + x_ifft)


class FNetClassifier(nn.Module):
    """A classifier that uses FNet layers."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fnet_block = nn.Sequential(
            FNetLayer(input_dim),
            FNetLayer(input_dim),
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fnet_block(x)
        return self.classifier(x)


class SkipMLP(nn.Module):
    """An MLP with a skip connection, updated with LayerNorm."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.initial_layer = nn.Linear(input_dim, 256)
        self.block = nn.Sequential(
            nn.LayerNorm(256),  # Changed from BatchNorm1d
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
        )
        self.final_layers = nn.Sequential(
            nn.LayerNorm(256),  # Changed from BatchNorm1d
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        initial_x = self.initial_layer(x)
        processed_x = self.block(initial_x)
        # Add the skip connection
        skipped_x = initial_x + processed_x
        return self.final_layers(skipped_x)


class BestOfBreedMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class CNN1DClassifier(nn.Module):
    """A 1D CNN-based classifier."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Calculate the flattened size after convolutions
        flattened_size = input_dim
        flattened_size = (flattened_size // 2) // 2
        self.classifier = nn.Linear(32 * flattened_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


class EncoderClassifier(nn.Module):
    """A classifier that uses an encoder to learn a latent representation."""

    def __init__(self, input_dim, num_classes, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        latent_x = self.encoder(x)
        return self.classifier(latent_x)


class HybridConvGRUClassifier(nn.Module):
    """A hybrid model with Convolutional and GRU layers."""

    def __init__(self, input_dim, num_classes, conv_out_channels=16, gru_hidden=32):
        super().__init__()
        # Assume input_dim is divisible by 4
        self.conv1d = nn.Conv1d(1, conv_out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # The input to the GRU will have `conv_out_channels` features
        self.gru = nn.GRU(conv_out_channels, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.relu(self.conv1d(x))  # (batch_size, conv_out_channels, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, conv_out_channels)
        _, h_n = self.gru(x)  # h_n shape: (1, batch_size, gru_hidden)
        out = self.fc(h_n.squeeze(0))  # (batch_size, num_classes)
        return out


def process_dataset(dataset_type, args, model, heads_and_files):
    """Helper function to process datasets and generate full decoder embeddings."""
    print(f"\nProcessing {dataset_type.upper()} tuning data...")

    if dataset_type == "uni":
        dataset_dir = args.uni_dataset_dir
        metrics_dir = args.uni_metrics_dir
        file_list_path = args.uni_tuning_list
    else:  # multi
        dataset_dir = args.multi_dataset_dir
        metrics_dir = args.multi_metrics_dir
        file_list_path = args.multi_tuning_list

    # --- 1. Load all metric files for the given dataset type ---
    metric_dfs = {}
    for head_name, file_name in heads_and_files.items():
        file_path = os.path.join(metrics_dir, file_name)
        if not os.path.exists(file_path):
            print(
                f"Warning: Metric file not found for {dataset_type} data, skipping head '{head_name}': {file_path}"
            )
            continue

        df = pd.read_csv(file_path)
        df["file"] = df["file"].apply(
            lambda x: os.path.splitext(x)[0] if isinstance(x, str) else x
        )
        metric_dfs[head_name] = df.set_index("file")

    if not metric_dfs:
        print(f"No metric files found for {dataset_type}. Skipping.")
        return [], []

    active_heads = list(metric_dfs.keys())
    if len(active_heads) < len(heads_and_files):
        print(
            f"Warning: Not all heads have metric files for {dataset_type}. Training will proceed with available heads: {active_heads}"
        )

    # --- 2. Find common files and generate training instances ---
    common_files = set(metric_dfs[active_heads[0]].index)
    for head_name in active_heads[1:]:
        common_files.intersection_update(metric_dfs[head_name].index)

    print(f"Found {len(common_files)} common {dataset_type} files for training.")

    X_train = []
    y_train = []

    try:
        file_list_df = pd.read_csv(file_list_path)
        all_tuning_files_with_ext = file_list_df["file_name"].tolist()
    except FileNotFoundError:
        print(
            f"Warning: Tuning file list not found at {file_list_path}. Cannot process files."
        )
        return [], []

    tuning_files = [
        f for f in all_tuning_files_with_ext if os.path.splitext(f)[0] in common_files
    ]

    device = next(model.parameters()).device
    context_length = model.config.context_length

    model.config.num_input_channels = 1
    model.config.channel_virtual_expand_scale = 1

    # --- BATCHED Embedding Generation ---
    embedding_batch_size = 1024

    # Collect all data first
    data_to_process = []
    labels_to_process = []

    for filename in sorted(tuning_files):
        data_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(data_path):
            continue
        basename = os.path.splitext(filename)[0]
        df = pd.read_csv(data_path).dropna()
        if df.shape[0] < 10:
            continue
        data_np = df.iloc[:, 0:-1].values.astype(float)
        num_channels = data_np.shape[1] if data_np.ndim > 1 else 1
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        try:
            vus_pr_scores = [
                metric_dfs[h].loc[basename, "VUS-PR"] for h in active_heads
            ]
            best_head_idx_local = np.argmax(vus_pr_scores)
            best_head_name = active_heads[best_head_idx_local]
            global_best_head_idx = list(heads_and_files.keys()).index(best_head_name)

            for i in range(num_channels):
                channel_data = data_np[:, i : i + 1]
                current_length = channel_data.shape[0]
                padded_channel_data = np.zeros((context_length, 1))
                if current_length >= context_length:
                    padded_channel_data = channel_data[-context_length:]
                else:
                    padded_channel_data[-current_length:] = channel_data

                data_to_process.append(padded_channel_data)
                labels_to_process.append(global_best_head_idx)
        except Exception as e:
            print(f"Could not prepare file {filename}. Error: {e}")

    # Process in batches
    for i in tqdm(
        range(0, len(data_to_process), embedding_batch_size),
        desc=f"Generating {dataset_type} Embeddings",
    ):
        batch_data_np = data_to_process[i : i + embedding_batch_size]
        batch_labels = labels_to_process[i : i + embedding_batch_size]
        past_values_batch = torch.tensor(
            np.array(batch_data_np), dtype=torch.float32
        ).to(device)
        batch_embeddings = get_full_decoder_embeddings(model, past_values_batch)
        
        X_train.extend(batch_embeddings)
        y_train.extend(batch_labels)

    return X_train, y_train


def evaluate_performance(
    model, X_data, y_data, num_classes, device, criterion=None
):
    """Helper function to evaluate model performance on a given dataset."""
    if len(X_data) == 0:
        return 0.0, float("inf")

    if isinstance(model, nn.Module):
        model.eval()
        model = model.to(device)  # Ensure model is on the correct device for evaluation
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_data, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            if criterion is not None:
                loss = criterion(outputs, y_tensor).item()
            else:
                loss = float("inf")
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        return accuracy, loss
    else:  # scikit-learn
        if not hasattr(model, "predict_proba"):
            return accuracy_score(y_data, model.predict(X_data)), 0.0

        y_pred_proba = model.predict_proba(X_data)
        accuracy = accuracy_score(y_data, np.argmax(y_pred_proba, axis=1))
        try:
            loss = log_loss(y_data, y_pred_proba, labels=np.arange(num_classes))
        except ValueError:
            loss = 0.0  #
        return accuracy, loss


def perform_cross_validation(
    model_name,
    model_prototype,
    X_data,
    y_data,
    n_splits_in,
    data_type,
    args,
    device,
    num_classes,
):
    """Performs k-fold cross-validation for a given model and dataset."""
    if len(X_data) == 0:
        return 0.0, 0.0

    n_splits = n_splits_in
    unique_labels, counts = np.unique(y_data, return_counts=True)
    min_class_count = np.min(counts) if len(counts) > 0 else 0

    if min_class_count < n_splits:
        print(
            f"Warning: Smallest class in {data_type} data has {min_class_count} samples. "
            f"Reducing K-Fold splits from {n_splits} to {min_class_count} for this validation."
        )
        n_splits = int(min_class_count)

    if n_splits < 2:
        print(
            f"Warning: Cannot perform cross-validation for {model_name} on {data_type} data. "
            f"Need at least 2 samples in each class for a split. Skipping."
        )
        return 0.0, 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_accuracies = []

    pbar = tqdm(
        skf.split(X_data, y_data),
        total=n_splits,
        desc=f"CV for {model_name} on {data_type}",
        leave=False,
    )
    for train_index, val_index in pbar:
        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        if isinstance(model_prototype, nn.Module):
            model_instance = copy.deepcopy(model_prototype).to(device)
            # --- PyTorch Training with optional Mixup Augmentation ---
            # Create a standard dataset. Augmentation will happen in the training loop.
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

            train_dataloader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(X_train)), shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model_instance.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

            pbar_epochs = tqdm(range(args.num_epochs), desc="Training Epochs", leave=False)
            for epoch in pbar_epochs:
                model_instance.train()
                for batch_X, batch_y in train_dataloader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    
                    if args.augment:
                        # Apply mixup to the batch
                        mixed_batch_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=0.4)
                        outputs = model_instance(mixed_batch_X)
                        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    else:
                        outputs = model_instance(batch_X)
                        loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()
                scheduler.step()
            trained_model = model_instance

        else:  # scikit-learn
            # --- Sklearn Training with optional Noise Augmentation ---
            if args.augment:
                noise = np.random.normal(0, 0.02, X_train.shape).astype(np.float32)
                X_train_aug = np.vstack([X_train, X_train + noise])
                y_train_aug = np.concatenate([y_train, y_train])
            else:
                X_train_aug, y_train_aug = X_train, y_train

            model_instance = clone(model_prototype)
            model_instance.fit(X_train_aug, y_train_aug)
            trained_model = model_instance

        acc, _ = evaluate_performance(trained_model, X_val, y_val, num_classes, device, criterion=nn.CrossEntropyLoss())
        fold_accuracies.append(acc)

    return np.mean(fold_accuracies), np.std(fold_accuracies)


# Mixup helper functions
def mixup_data(x, y, alpha=0.4):
    """Applies mixup to a given batch of data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main(args):
    """
    Trains specialist models to select the best TSPulse head for univariate and
    multivariate data separately.
    """
    set_seed(args.seed)

    heads_and_files = {
        "TSPulse_ZS_time": "TSPulse_ZS_time.csv",
        "TSPulse_ZS_fft": "TSPulse_ZS_fft.csv",
        "TSPulse_ZS_future": "TSPulse_ZS_future.csv",
        "TSPulse_ZS_ensemble": "TSPulse_ZS_ensemble.csv",
        "TSPulse2": "TSPulse2.csv",
    }
    all_head_names = list(heads_and_files.keys())
    label_encoder = {head_name: i for i, head_name in enumerate(all_head_names)}

    # Initialize the model for embedding extraction
    model = TSPulseForReconstruction.from_pretrained(
        args.model_name,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Process both datasets
    X_uni, y_uni = process_dataset("uni", args, model, heads_and_files)
    X_multi, y_multi = process_dataset("multi", args, model, heads_and_files)

    if not (X_uni or X_multi):
        print("Fatal Error: No training data could be generated. Exiting.")
        return

    # --- Prepare for Training ---
    X_uni_np, y_uni_np = np.array(X_uni), np.array(y_uni)
    X_multi_np, y_multi_np = np.array(X_multi), np.array(y_multi)

    print(f"\nTotal univariate instances: {len(y_uni_np)}")
    print(f"Total multivariate instances: {len(y_multi_np)}")

    # --- Architectures ---
    input_dim = X_uni_np.shape[1] if len(X_uni_np) > 0 else X_multi_np.shape[1]
    num_classes = len(all_head_names)

    architectures = {
        "BestOfBreedMLP": lambda: BestOfBreedMLP(input_dim=input_dim, num_classes=num_classes),
        "MLP": lambda: MLP(input_dim=input_dim, num_classes=num_classes),
        "ResMLP": lambda: ResMLP(input_dim=input_dim, num_classes=num_classes),
        "SkipMLP": lambda: SkipMLP(input_dim=input_dim, num_classes=num_classes),
        "CNN1D": lambda: CNN1DClassifier(input_dim=input_dim, num_classes=num_classes),
        "Encoder": lambda: EncoderClassifier(input_dim=input_dim, num_classes=num_classes),
        "RandomForest": lambda: RandomForestClassifier(random_state=args.seed, n_jobs=-1, n_estimators=200),
        "XGBoost": lambda: xgb.XGBClassifier(random_state=args.seed, eval_metric="mlogloss"),
        "CatBoost": lambda: CatBoostClassifier(random_state=args.seed, verbose=0, iterations=500, learning_rate=0.05),
        "ExtraTrees": lambda: ExtraTreesClassifier(random_state=args.seed, n_jobs=-1, n_estimators=200),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=args.seed, verbose=1),
        "AdaBoost": lambda: AdaBoostClassifier(random_state=args.seed),
        "SVC": lambda: SVC(random_state=args.seed, probability=True),
        "KNN": lambda: KNeighborsClassifier(n_jobs=-1),
        "LogisticRegression": lambda: LogisticRegression(random_state=args.seed, max_iter=1000),
        "GaussianNB": lambda: GaussianNB(),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=args.seed),
        "ExtraTrees": lambda: ExtraTreesClassifier(random_state=args.seed, n_jobs=-1),
        "QDA": lambda: QuadraticDiscriminantAnalysis(),
        "Bagging": lambda: BaggingClassifier(random_state=args.seed, n_jobs=-1),
        "PassiveAggressive": lambda: PassiveAggressiveClassifier(random_state=args.seed),
    }

    # --- Process each data type separately ---
    for data_type, X_data, y_data in [
        # ("univariate", X_uni_np, y_uni_np),
        ("multivariate", X_multi_np, y_multi_np),
    ]:
        if len(X_data) == 0:
            print(f"\nNo data for {data_type} specialist model. Skipping.")
            continue

        print(f"\n--- Finding Best Specialist for {data_type.upper()} Data ---")
        performances = []
        for model_name, model_builder in architectures.items():
            try:
                model_prototype = model_builder()
                mean_acc, std_dev = perform_cross_validation(
                    model_name,
                    model_prototype,
                    X_data,
                    y_data,
                    3,  # n_splits
                    data_type,
                    args,
                    device,
                    num_classes,
                )
                if mean_acc > 0:  # Only add models that could be trained
                    performances.append(
                        {"model_name": model_name, "accuracy": mean_acc, "std_dev": std_dev}
                    )
            except Exception as e:
                print(f"Failed to train/evaluate {model_name} for {data_type} data. Error: {e}")

        if not performances:
            print(f"No models could be successfully trained for {data_type} data.")
            continue

        performances.sort(key=lambda x: x["accuracy"], reverse=True)
        print(f"\n--- {data_type.upper()} Specialist Performance Summary ---")
        for perf in performances:
            print(
                f"Model: {perf['model_name']:<20} | CV Accuracy: {perf['accuracy']:.4f} \u00B1 {perf['std_dev']:.4f}"
            )

        best_model_name = performances[0]["model_name"]
        print(f"\nRetraining best specialist ({best_model_name}) on all {data_type} data...")

        best_model_builder = architectures[best_model_name]
        final_model = best_model_builder()
        
        if isinstance(final_model, nn.Module):
            # Full training for PyTorch model
            final_model = final_model.to(device).train()
            # Full training loop for PyTorch model
            batch_size = min(args.batch_size, len(X_data))
            full_train_dataset = TensorDataset(
                torch.tensor(X_data, dtype=torch.float32),
                torch.tensor(y_data, dtype=torch.long),
            )
            full_train_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
            optimizer = optim.AdamW(final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
            criterion = nn.CrossEntropyLoss()

            pbar_final = tqdm(range(args.num_epochs), desc=f"Final Training {best_model_name}")
            pbar_epochs = tqdm(range(args.num_epochs), desc="Training Epochs", leave=False)
            for epoch in pbar_epochs:
                for batch_X, batch_y in full_train_dataloader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = final_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            trained_model = final_model

        else:  # sklearn
            final_model.fit(X_data, y_data)
            trained_model = final_model

        # Save the specialist model
        output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.output_model_dir)), data_type)
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(label_encoder, os.path.join(output_dir, "embedding_selector_encoder.joblib"))
        model_path = os.path.join(output_dir, "embedding_selector_model.joblib" if not isinstance(trained_model, nn.Module) else "embedding_selector_model.pt")
        
        if isinstance(trained_model, nn.Module):
            torch.save(trained_model.cpu().state_dict(), model_path)
        else:
            joblib.dump(trained_model, model_path)
        print(f"{data_type.capitalize()} specialist model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train specialist head selectors using TSPulse embeddings.")
    # --- File Paths ---
    parser.add_argument(
        "--model_name", type=str, default="ibm-granite/granite-timeseries-tspulse-r1"
    )
    parser.add_argument("--uni_dataset_dir", type=str, default="Datasets/TSB-AD-U/")
    parser.add_argument("--multi_dataset_dir", type=str, default="Datasets/TSB-AD-M/")
    parser.add_argument(
        "--uni_metrics_dir", type=str, default="eval/metrics/uni-tuning/"
    )
    parser.add_argument(
        "--multi_metrics_dir", type=str, default="eval/metrics/multi-tuning/"
    )
    parser.add_argument(
        "--uni_tuning_list", type=str, default="Datasets/File_List/TSB-AD-U-Tuning.csv"
    )
    parser.add_argument(
        "--multi_tuning_list",
        type=str,
        default="Datasets/File_List/TSB-AD-M-Tuning.csv",
    )
    parser.add_argument("--output_model_dir", type=str, default="trained_selectors/")

    # --- Training Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_false", help="Enable data augmentation for training.")

    args = parser.parse_args()
    main(args)

