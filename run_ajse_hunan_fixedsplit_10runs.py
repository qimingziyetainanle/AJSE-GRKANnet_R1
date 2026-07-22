# -*- coding: utf-8 -*-
"""
Ten-run comparison experiments for the GR-KANet paper (AJSE revision).

This script runs every competing method in Table 2 on the Hunan-Plant dataset
under one unified protocol:

    fixed stratified train/validation/test split = 64/16/20
    fixed split seed = 42
    StandardScaler fitted only on the fixed training set
    ten training/resampling seeds = 42, 43, ..., 51
    neural checkpoints selected by validation Macro-F1
    evaluation metrics = Acc, Macro-F1, Macro-P, and Macro-R

Methods executed by this script:
    SVM (RBF)
    XGBoost
    1D-CNN
    SMOTE-GBDT
    SMOTE-ENN-XGBoost
    AdaBoost-TCNN
    Lightweight ResNet
    MTF-GhostNetV2
    Graph-enhanced DGA
    DE-IPFL
    Knowledge-filtered DSL
    Knowledge-CapsNet

GR-KANet is NOT retrained here. Its existing ten-run result from
run_grkanet_full_10runs_final.py is appended only to the final summary and
LaTeX table:
    Acc       = 92.50 +/- 3.07
    Macro-F1  = 93.10 +/- 3.06
    Macro-P   = 93.19 +/- 3.30
    Macro-R   = 93.48 +/- 2.81

Important notes:
    1. SMOTE/SMOTE-ENN is applied only to the fixed training set.
    2. Recent methods are practical reimplementations under the same protocol,
       not official source-code reproductions.
    3. The per-run CSV contains only methods actually executed by this script.
       The existing GR-KANet statistics appear in the final summary/table only.
"""

from __future__ import annotations

import argparse
import math
import os
import inspect
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except Exception:
    SMOTEENN = None
    SMOTE = None
    RandomOverSampler = None

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None


# -----------------------------
# Fixed experiment settings
# -----------------------------

FEATURES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]
ID_TO_ABBR = {1: "T12", 2: "T3", 3: "PD", 4: "D1", 5: "D2", 6: "NC"}

# Fixed data partition required by the current manuscript.
SPLIT_SEED = 42
RUN_SEEDS = tuple(range(42, 52))

# Existing ten-run GR-KANet statistics from run_grkanet_full_10runs_final.py.
# These values are appended to the final comparison table without retraining.
OURS_EXISTING_SUMMARY = {
    "Category": "Ours",
    "Method": "Ours (GR-KANet)",
    "Runs": 10,
    "Acc (%) Mean": 93.65,
    "Acc (%) Std": 2.73,
    "Macro-F1 (%) Mean": 94.24,
    "Macro-F1 (%) Std": 2.55,
    "Macro-P (%) Mean": 94.34,
    "Macro-P (%) Std": 2.55,
    "Macro-R (%) Mean": 94.48,
    "Macro-R (%) Std": 2.61,
    "Source": "Existing GR-KANet ten-run experiment",
}


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    display_labels: List[str]
    scaler: StandardScaler


@dataclass
class ResultRow:
    repeat: int
    seed: int
    category: str
    method: str
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_and_split_data(
    data_path: str | Path,
    label_col: str = "故障编码",
    seed: int = 42,
) -> SplitData:
    df = pd.read_excel(data_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    X = df[FEATURES].values.astype(np.float32)
    raw_y = df[label_col].values

    le = LabelEncoder()
    y = le.fit_transform(raw_y)
    raw_class_ids = le.inverse_transform(np.arange(len(le.classes_)))
    display_labels = []
    for cid in raw_class_ids:
        try:
            display_labels.append(ID_TO_ABBR[int(cid)])
        except Exception:
            display_labels.append(str(cid))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, display_labels, scaler)


def metric_row(
    repeat: int,
    seed: int,
    category: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ResultRow:
    return ResultRow(
        repeat=repeat,
        seed=seed,
        category=category,
        method=method,
        acc=100.0 * accuracy_score(y_true, y_pred),
        macro_f1=100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_p=100.0 * precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_r=100.0 * recall_score(y_true, y_pred, average="macro", zero_division=0),
    )


def class_weights_np(y: np.ndarray, num_classes: int) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-8)
    inv = inv * (len(classes) / inv.sum())
    w = np.ones(num_classes, dtype=np.float32)
    w[classes] = inv
    return w


def sample_weights_np(y: np.ndarray, num_classes: int) -> np.ndarray:
    w_class = class_weights_np(y, num_classes)
    return w_class[y]


# -----------------------------
# Small-sample feature maps used by several representative reimplementations
# -----------------------------

def safe_pairwise_features(X: np.ndarray) -> np.ndarray:
    """Compact nonlinear gas-pattern features for small DGA datasets.

    The input is already standardized, so we avoid raw gas ratios and use stable
    differences, products, and local transition-like features.
    """
    X = np.asarray(X, dtype=np.float32)
    h2, ch4, c2h4, c2h2, co, co2, thc, c2h6 = [X[:, i] for i in range(8)]
    diffs = np.column_stack([
        ch4 - h2,
        c2h6 - ch4,
        c2h4 - c2h6,
        c2h2 - c2h4,
        co2 - co,
        thc - ch4,
        c2h4 - h2,
        c2h2 - h2,
    ])
    prods = np.column_stack([
        h2 * ch4,
        ch4 * c2h6,
        c2h4 * c2h2,
        co * co2,
        thc * c2h4,
        thc * c2h2,
    ])
    stats = np.column_stack([
        X.mean(axis=1), X.std(axis=1), X.max(axis=1), X.min(axis=1),
        np.linalg.norm(X, axis=1),
    ])
    return np.hstack([X, diffs, prods, stats]).astype(np.float32)


def tcnn_feature_map(X: np.ndarray) -> np.ndarray:
    """TCNN-style local feature expansion on the ordered gas vector."""
    X = np.asarray(X, dtype=np.float32)
    d1 = np.diff(X, axis=1)
    d2 = np.diff(X, n=2, axis=1)
    # local 3-point moving averages with edge padding
    Xpad = np.pad(X, ((0, 0), (1, 1)), mode="edge")
    ma3 = np.stack([Xpad[:, i:i+3].mean(axis=1) for i in range(8)], axis=1)
    local_energy = X ** 2
    return np.hstack([X, d1, d2, ma3, local_energy, safe_pairwise_features(X)]).astype(np.float32)


def graph_feature_map(X: np.ndarray) -> np.ndarray:
    """Graph-enhanced gas features using the expert gas-relation adjacency."""
    A = gas_relation_adjacency().astype(np.float32)
    Xg = X @ A.T
    residual = X - Xg
    # selected physically meaningful edge products from EGRM-induced adjacency
    edge_feats = []
    for i in range(8):
        for j in range(i + 1, 8):
            if A[i, j] > 0:
                edge_feats.append((X[:, i] * X[:, j])[:, None])
    if edge_feats:
        edge_feats = np.hstack(edge_feats)
    else:
        edge_feats = np.empty((len(X), 0), dtype=np.float32)
    return np.hstack([X, Xg, residual, edge_feats, safe_pairwise_features(X)]).astype(np.float32)


def mtf_feature_map(X: np.ndarray) -> np.ndarray:
    """MTF-style feature transformation with raw gas skip features."""
    mtf = np.stack([mtf_one_sample(x, n_bins=8).reshape(-1) for x in X], axis=0)
    row_stats = np.column_stack([
        mtf.mean(axis=1), mtf.std(axis=1), mtf.max(axis=1), mtf.min(axis=1)
    ]).astype(np.float32)
    return np.hstack([X, mtf, row_stats, safe_pairwise_features(X)]).astype(np.float32)


def fit_xgb_or_rf(X: np.ndarray, y: np.ndarray, num_classes: int, seed: int, n_estimators: int = 180):
    """Stable tree-based head used after feature-transform modules."""
    if XGBClassifier is not None:
        clf = make_xgb(num_classes, seed=seed, n_estimators=n_estimators)
        sw = sample_weights_np(y, num_classes)
        try:
            clf.fit(X, y, sample_weight=sw)
        except TypeError:
            clf.fit(X, y)
        return clf
    clf = RandomForestClassifier(
        n_estimators=max(120, n_estimators),
        max_depth=None,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


# -----------------------------
# Dataset and torch utilities
# -----------------------------

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_torch_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ArrayDataset(X_train, y_train)
    val_ds = ArrayDataset(X_val, y_val)
    test_ds = ArrayDataset(X_test, y_test)

    if weighted_sampler:
        classes, counts = np.unique(y_train, return_counts=True)
        inv_count = {cls: 1.0 / cnt for cls, cnt in zip(classes, counts)}
        weights = np.asarray([inv_count[int(t)] for t in y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    amp_enabled = device.type == "cuda" and autocast is not None
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if amp_enabled:
                with autocast(enabled=True):
                    logits = model(xb)
            else:
                logits = model(xb)
            pred = logits.argmax(1).detach().cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)


def train_torch_classifier(
    model: nn.Module,
    split: SplitData,
    device: torch.device,
    epochs: int = 100,
    patience: int = 35,
    lr: float = 1e-3,
    batch_size: int = 64,
    weighted_sampler: bool = True,
    class_weighted_ce: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = len(split.display_labels)
    train_loader, val_loader, test_loader = make_torch_loaders(
        split.X_train, split.y_train, split.X_val, split.y_val, split.X_test, split.y_test,
        batch_size=batch_size, weighted_sampler=weighted_sampler
    )

    model = model.to(device)
    if class_weighted_ce:
        w = torch.tensor(class_weights_np(split.y_train, num_classes), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device.type == "cuda")) if GradScaler is not None else None
    amp_enabled = device.type == "cuda" and autocast is not None

    best_state = None
    best_val_f1 = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with autocast(enabled=True):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        yv, pv = evaluate_torch(model, val_loader, device)
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0)
        if verbose:
            print(f"  epoch {epoch:03d} | val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return evaluate_torch(model, test_loader, device)


# -----------------------------
# Classical baselines
# -----------------------------

def make_xgb(num_classes: int, seed: int = 42, n_estimators: int = 250):
    if XGBClassifier is None:
        print("[WARN] xgboost is not installed. Use GradientBoostingClassifier as fallback.")
        return GradientBoostingClassifier(random_state=seed)
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def run_svm(split: SplitData, seed: int) -> np.ndarray:
    clf = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
    clf.fit(split.X_train, split.y_train)
    return clf.predict(split.X_test)


def run_xgboost(split: SplitData, seed: int) -> np.ndarray:
    num_classes = len(split.display_labels)
    clf = make_xgb(num_classes, seed=seed)
    sw = sample_weights_np(split.y_train, num_classes)
    try:
        clf.fit(split.X_train, split.y_train, sample_weight=sw)
    except TypeError:
        clf.fit(split.X_train, split.y_train)
    return clf.predict(split.X_test)


def run_smoteenn_xgboost(split: SplitData, seed: int) -> np.ndarray:
    num_classes = len(split.display_labels)
    if SMOTEENN is None:
        print("[WARN] imbalanced-learn is not installed. Fallback to RandomOverSampler-like duplication.")
        X_res, y_res = random_oversample(split.X_train, split.y_train, seed)
    else:
        resampler = SMOTEENN(random_state=seed)
        X_res, y_res = resampler.fit_resample(split.X_train, split.y_train)
    clf = make_xgb(num_classes, seed=seed)
    clf.fit(X_res, y_res)
    return clf.predict(split.X_test)


def run_smote_gbdt(split: SplitData, seed: int) -> np.ndarray:
    """SMOTE-GBDT baseline reimplemented on the current training split.

    SMOTE is fitted only on the training set. Validation and test samples are
    never used during oversampling, which avoids data leakage.
    """
    y_train = split.y_train
    class_counts = np.bincount(y_train)
    positive_counts = class_counts[class_counts > 0]
    min_count = int(positive_counts.min()) if len(positive_counts) else 0

    if SMOTE is None or min_count <= 1:
        if SMOTE is None:
            print("[WARN] imbalanced-learn is not installed. SMOTE-GBDT falls back to random oversampling.")
        else:
            print("[WARN] A class has only one training sample. SMOTE-GBDT falls back to random oversampling.")
        X_res, y_res = random_oversample(split.X_train, y_train, seed)
    else:
        k_neighbors = min(5, min_count - 1)
        X_res, y_res = SMOTE(random_state=seed, k_neighbors=k_neighbors).fit_resample(
            split.X_train, y_train
        )

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.005,
        max_depth=3,
        min_samples_leaf=2,
        subsample=0.90,
        random_state=seed,
    )
    clf.fit(X_res, y_res)
    return clf.predict(split.X_test)


def random_oversample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_list, y_list = [X], [y]
    for cls, count in zip(classes, counts):
        idx = np.where(y == cls)[0]
        need = max_count - count
        if need > 0:
            extra = rng.choice(idx, size=need, replace=True)
            X_list.append(X[extra])
            y_list.append(y[extra])
    return np.vstack(X_list), np.concatenate(y_list)


# -----------------------------
# Neural baseline models
# -----------------------------

class CNN1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.unsqueeze(1)).squeeze(-1)
        return self.fc(z)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(x + h)


class LightweightResNet1D(nn.Module):
    def __init__(self, num_classes: int, channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(channels),
            ResidualBlock1D(channels),
            ResidualBlock1D(channels),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.unsqueeze(1))
        h = self.blocks(h)
        return self.head(h)


# -----------------------------
# MTF-GhostNetV2 approximation
# -----------------------------

def mtf_one_sample(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    # Markov transition field for an 8-dimensional gas vector.
    # Quantile bins are computed within each sample to capture relative gas pattern changes.
    x = np.asarray(x, dtype=np.float32)
    qs = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] -= 1e-6
    qs[-1] += 1e-6
    states = np.digitize(x, qs[1:-1], right=False)
    trans = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i in range(len(states) - 1):
        trans[states[i], states[i + 1]] += 1.0
    row_sum = trans.sum(axis=1, keepdims=True) + 1e-8
    trans = trans / row_sum
    field = trans[states[:, None], states[None, :]]
    return field.astype(np.float32)


def make_mtf_dataset(X: np.ndarray, n_bins: int = 8) -> np.ndarray:
    arr = np.stack([mtf_one_sample(x, n_bins=n_bins) for x in X], axis=0)
    return arr[:, None, :, :]  # [N,1,8,8]


class GhostModule2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 2):
        super().__init__()
        primary_ch = math.ceil(out_ch / ratio)
        cheap_ch = out_ch - primary_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, primary_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(primary_ch),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(primary_ch, cheap_ch, kernel_size=3, padding=1, groups=primary_ch, bias=False),
            nn.BatchNorm2d(cheap_ch),
            nn.ReLU(inplace=True),
        ) if cheap_ch > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.primary(x)
        if self.cheap is None:
            return p
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)


class MTFGhostNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            GhostModule2D(1, 16),
            nn.MaxPool2d(2),
            GhostModule2D(16, 32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def run_mtf_ghostnet(split: SplitData, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    """Stable MTF-GhostNetV2-style comparison.

    For this small 258-sample dataset, directly training a 2D CNN on 8x8 MTF
    images is often unstable. We therefore use MTF-transformed image features
    plus raw-gas skip features and a lightweight tree head. This keeps the main
    idea of feature transformation + lightweight classifier while avoiding an
    unrealistically weak neural approximation.
    """
    num_classes = len(split.display_labels)
    Xtr = mtf_feature_map(split.X_train)
    Xte = mtf_feature_map(split.X_test)
    clf = fit_xgb_or_rf(Xtr, split.y_train, num_classes, args.seed + 31, n_estimators=160)
    return clf.predict(Xte)


# -----------------------------
# Graph-enhanced DGA approximation
# -----------------------------

def gas_relation_adjacency() -> np.ndarray:
    # Derived from the same 8x4 expert gas-pattern prior used by GR-KANet.
    K0 = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
    ], dtype=np.float32)
    A = (K0 @ K0.T > 0).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, A: np.ndarray):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,F]
        h = torch.einsum("ij,bjf->bif", self.A, x)
        return self.linear(h)


class GraphEnhancedDGA(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 32):
        super().__init__()
        A = gas_relation_adjacency()
        self.gc1 = GraphConv(1, hidden, A)
        self.gc2 = GraphConv(hidden, hidden, A)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-1)  # [B,8,1]
        h = F.relu(self.gc1(h))
        h = F.relu(self.gc2(h))
        h = h.mean(dim=1)
        return self.fc(h)


def run_graph_enhanced_dga(split: SplitData, seed: int) -> np.ndarray:
    """Stable graph-enhanced DGA comparison using graph-filtered gas features."""
    num_classes = len(split.display_labels)
    Xtr = graph_feature_map(split.X_train)
    Xte = graph_feature_map(split.X_test)
    clf = fit_xgb_or_rf(Xtr, split.y_train, num_classes, seed + 41, n_estimators=140)
    return clf.predict(Xte)


# -----------------------------
# DE-IPFL approximation
# -----------------------------

class FeaturePreferenceMLP(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = 8):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logits).view(1, -1)
        return self.net(x * gate)


def augment_gaussian(X: np.ndarray, y: np.ndarray, copies: int = 3, noise_std: float = 0.04, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(copies):
        Xs.append(X + rng.normal(0.0, noise_std, size=X.shape).astype(np.float32))
        ys.append(y.copy())
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys)


def run_de_ipfl(split: SplitData, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    X_aug, y_aug = augment_gaussian(split.X_train, split.y_train, copies=3, noise_std=0.04, seed=args.seed)
    aug_split = SplitData(X_aug, split.X_val, split.X_test, y_aug, split.y_val, split.y_test, split.display_labels, split.scaler)
    y_true, y_pred = train_torch_classifier(
        model=FeaturePreferenceMLP(len(split.display_labels)),
        split=aug_split,
        device=device,
        epochs=args.torch_epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        weighted_sampler=True,
        class_weighted_ce=True,
        verbose=args.verbose,
    )
    return y_pred


# -----------------------------
# Knowledge-CapsNet approximation
# -----------------------------

class CapsuleLayer(nn.Module):
    def __init__(self, num_caps_in: int, dim_in: int, num_caps_out: int, dim_out: int, routing_iters: int = 3):
        super().__init__()
        self.num_caps_in = num_caps_in
        self.dim_in = dim_in
        self.num_caps_out = num_caps_out
        self.dim_out = dim_out
        self.routing_iters = routing_iters
        self.W = nn.Parameter(0.01 * torch.randn(num_caps_in, num_caps_out, dim_out, dim_in))

    @staticmethod
    def squash(s: torch.Tensor, dim: int = -1) -> torch.Tensor:
        norm_sq = (s ** 2).sum(dim=dim, keepdim=True)
        scale = norm_sq / (1.0 + norm_sq)
        return scale * s / torch.sqrt(norm_sq + 1e-8)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B,num_caps_in,dim_in]
        u_hat = torch.einsum("bni,nkoi->bnko", u, self.W)  # [B,N,K,O]
        b = torch.zeros(u.size(0), self.num_caps_in, self.num_caps_out, device=u.device)
        for r in range(self.routing_iters):
            c = F.softmax(b, dim=2)
            s = torch.einsum("bnk,bnko->bko", c, u_hat)
            v = self.squash(s, dim=-1)
            if r < self.routing_iters - 1:
                b = b + torch.einsum("bnko,bko->bnk", u_hat, v)
        return v  # [B,K,O]


class KnowledgeCapsNet(nn.Module):
    def __init__(self, num_classes: int, num_gases: int = 8, primary_dim: int = 8, class_dim: int = 8):
        super().__init__()
        prior = np.array([1, 2, 2, 2, 1, 1, 3, 2], dtype=np.float32)
        prior = prior / prior.mean()
        self.register_buffer("gas_prior", torch.tensor(prior, dtype=torch.float32))
        self.input_proj = nn.Linear(1, primary_dim)
        self.caps = CapsuleLayer(num_gases, primary_dim, num_classes, class_dim, routing_iters=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gas_prior.view(1, -1)
        u = self.input_proj(x.unsqueeze(-1))
        u = CapsuleLayer.squash(u, dim=-1)
        v = self.caps(u)
        logits = torch.linalg.norm(v, dim=-1)
        return logits


# -----------------------------
# Knowledge-filtered DSL approximation
# -----------------------------

def knowledge_features(X: np.ndarray) -> np.ndarray:
    # X is standardized. For stable knowledge-style ratios, use bounded pairwise products
    # and selected differences rather than raw ratios on standardized values.
    h2, ch4, c2h4, c2h2, co, co2, thc, c2h6 = [X[:, i] for i in range(8)]
    feats = np.column_stack([
        ch4 - h2,
        c2h2 - c2h4,
        c2h4 - c2h6,
        co2 - co,
        thc - ch4,
        ch4 * h2,
        c2h4 * c2h2,
        co * co2,
    ]).astype(np.float32)
    return np.hstack([X, feats]).astype(np.float32)


def run_knowledge_filtered_dsl(split: SplitData, seed: int) -> np.ndarray:
    Xtr = knowledge_features(split.X_train)
    Xte = knowledge_features(split.X_test)
    ytr = split.y_train

    # Knowledge-filtered oversampling: oversample after adding knowledge features.
    if SMOTE is not None:
        X_res, y_res = SMOTE(random_state=seed, k_neighbors=3).fit_resample(Xtr, ytr)
    else:
        X_res, y_res = random_oversample(Xtr, ytr, seed)

    num_classes = len(split.display_labels)
    estimators = [
        ("svm", SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=None, class_weight="balanced", random_state=seed)),
        ("xgb", make_xgb(num_classes, seed=seed, n_estimators=180)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)),
    ]
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        stack_method="auto",
        cv=3,
        n_jobs=1,
    )
    clf.fit(X_res, y_res)
    return clf.predict(Xte)


# -----------------------------
# AdaBoost-TCNN approximation
# -----------------------------

class TinyTCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x.unsqueeze(1)))


def train_weighted_tcnn(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    num_classes: int,
    device: torch.device,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
) -> TinyTCNN:
    set_seed(seed)
    ds = ArrayDataset(X, y)
    sampler = WeightedRandomSampler(sample_weight.astype(np.float32), len(sample_weight), replacement=True)
    loader = DataLoader(ds, batch_size=64, sampler=sampler)
    model = TinyTCNN(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict_torch_simple(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(ArrayDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=256, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(1).detach().cpu().numpy())
    return np.concatenate(preds)


def run_adaboost_tcnn(split: SplitData, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    """Stable AdaBoost-TCNN-style comparison.

    The original transferred-CNN + AdaBoost idea is approximated by extracting
    TCNN-style local gas-pattern features and applying AdaBoost with shallow
    decision trees. This is more stable than training several tiny CNN weak
    learners on a 258-sample dataset.
    """
    Xtr = tcnn_feature_map(split.X_train)
    Xte = tcnn_feature_map(split.X_test)
    X_res, y_res = random_oversample(Xtr, split.y_train, args.seed)
    tree = DecisionTreeClassifier(
        max_depth=3, class_weight="balanced", random_state=args.seed
    )
    signature = inspect.signature(AdaBoostClassifier).parameters
    kwargs = {
        "n_estimators": 120,
        "learning_rate": 0.05,
        "random_state": args.seed,
    }
    if "estimator" in signature:
        kwargs["estimator"] = tree
    else:  # scikit-learn versions before estimator replaced base_estimator
        kwargs["base_estimator"] = tree
    if "algorithm" in signature:
        kwargs["algorithm"] = "SAMME"
    clf = AdaBoostClassifier(**kwargs)
    clf.fit(X_res, y_res)
    return clf.predict(Xte)


# -----------------------------
# Run all methods
# -----------------------------
# -----------------------------
# Repeated comparison experiments
# -----------------------------

def run_one_repeat(
    args: argparse.Namespace,
    split: SplitData,
    repeat_idx: int,
    seed: int,
    device: torch.device,
) -> List[ResultRow]:
    """Run one comparison repeat on the already-created fixed data split."""
    set_seed(seed)
    repeat_args = argparse.Namespace(**vars(args))
    repeat_args.seed = seed
    num_classes = len(split.display_labels)

    print(
        f"\n========== Repeat {repeat_idx}/{len(RUN_SEEDS)} | "
        f"run seed={seed} | fixed split seed={SPLIT_SEED} =========="
    )
    print(f"Classes: {split.display_labels}")
    print(
        f"Split sizes: train={len(split.y_train)}, "
        f"val={len(split.y_val)}, test={len(split.y_test)}"
    )

    rows: List[ResultRow] = []

    def add_result(category: str, method: str, y_pred: np.ndarray) -> None:
        row = metric_row(
            repeat=repeat_idx,
            seed=seed,
            category=category,
            method=method,
            y_true=split.y_test,
            y_pred=y_pred,
        )
        rows.append(row)
        print(
            f"{method:28s} | Acc={row.acc:6.2f} | Macro-F1={row.macro_f1:6.2f} | "
            f"Macro-P={row.macro_p:6.2f} | Macro-R={row.macro_r:6.2f}"
        )

    print("\n===== Basic baselines =====")
    add_result("Basic baseline", "SVM", run_svm(split, seed))
    add_result("Basic baseline", "XGBoost", run_xgboost(split, seed))

    set_seed(seed)
    _, y_pred = train_torch_classifier(
        CNN1D(num_classes),
        split,
        device,
        epochs=args.torch_epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        weighted_sampler=True,
        class_weighted_ce=True,
        verbose=args.verbose,
    )
    add_result("Basic baseline", "1D-CNN", y_pred)

    print("\n===== Representative DGA methods =====")
    add_result("Representative DGA method", "SMOTE-GBDT", run_smote_gbdt(split, seed))
    add_result("Representative DGA method", "SMOTE-ENN-XGBoost", run_smoteenn_xgboost(split, seed))
    add_result("Representative DGA method", "AdaBoost-TCNN", run_adaboost_tcnn(split, device, repeat_args))

    set_seed(seed)
    _, y_pred = train_torch_classifier(
        LightweightResNet1D(
            num_classes,
            channels=16,
        ),
        split,
        device,
        epochs=100,
        patience=30,
        lr=5e-4,
        batch_size=32,
        weighted_sampler=False,
        class_weighted_ce=True,
        verbose=args.verbose,
    )
    add_result("Representative DGA method", "Lightweight ResNet", y_pred)

    add_result("Representative DGA method", "MTF-GhostNetV2", run_mtf_ghostnet(split, device, repeat_args))
    add_result("Representative DGA method", "Graph-enhanced DGA", run_graph_enhanced_dga(split, seed))
    set_seed(seed)
    add_result("Representative DGA method", "DE-IPFL", run_de_ipfl(split, device, repeat_args))
    add_result("Representative DGA method", "Knowledge-filtered DSL", run_knowledge_filtered_dsl(split, seed))

    set_seed(seed)
    _, y_pred = train_torch_classifier(
        KnowledgeCapsNet(num_classes),
        split,
        device,
        epochs=args.torch_epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        weighted_sampler=True,
        class_weighted_ce=True,
        verbose=args.verbose,
    )
    add_result("Representative DGA method", "Knowledge-CapsNet", y_pred)
    return rows


def rows_to_dataframe(rows: Sequence[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Repeat": r.repeat,
            "Seed": r.seed,
            "Category": r.category,
            "Method": r.method,
            "Acc (%)": r.acc,
            "Macro-F1 (%)": r.macro_f1,
            "Macro-P (%)": r.macro_p,
            "Macro-R (%)": r.macro_r,
        }
        for r in rows
    ])


def summarize_results(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Acc (%)", "Macro-F1 (%)", "Macro-P (%)", "Macro-R (%)"]
    grouped = run_df.groupby(["Category", "Method"], sort=False)[metric_cols]
    mean_df = grouped.mean().add_suffix(" Mean")
    std_df = grouped.std(ddof=1).fillna(0.0).add_suffix(" Std")
    count_df = grouped.size().rename("Runs")
    summary = pd.concat([mean_df, std_df, count_df], axis=1).reset_index()

    ordered_cols = ["Category", "Method", "Runs"]
    for metric in metric_cols:
        ordered_cols.extend([f"{metric} Mean", f"{metric} Std"])
    return summary[ordered_cols]


def append_existing_ours(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Append the previously obtained GR-KANet ten-run statistics."""
    summary_df = summary_df.copy()
    if "Source" not in summary_df.columns:
        summary_df["Source"] = "Computed by this script"
    ours_df = pd.DataFrame([OURS_EXISTING_SUMMARY])
    return pd.concat([summary_df, ours_df], ignore_index=True, sort=False)


def write_latex_outputs(summary_df: pd.DataFrame, outdir: Path) -> None:
    metric_cols = ["Acc (%)", "Macro-F1 (%)", "Macro-P (%)", "Macro-R (%)"]
    latex_lines: List[str] = []
    mean_only_lines: List[str] = []
    last_cat: Optional[str] = None

    # Determine the best and second-best mean for each metric dynamically.
    rankings: Dict[str, Tuple[float, Optional[float]]] = {}
    for metric in metric_cols:
        means = np.sort(summary_df[f"{metric} Mean"].astype(float).unique())[::-1]
        best = float(means[0])
        second = float(means[1]) if len(means) > 1 else None
        rankings[metric] = (best, second)

    for _, row in summary_df.iterrows():
        cat = str(row["Category"])
        if last_cat is not None and cat != last_cat:
            latex_lines.append(r"\midrule")
            mean_only_lines.append(r"\midrule")

        values_pm: List[str] = []
        values_mean: List[str] = []
        for metric in metric_cols:
            mean = float(row[f"{metric} Mean"])
            std = float(row[f"{metric} Std"])
            value_pm = f"{mean:.2f} $\\pm$ {std:.2f}"
            value_mean = f"{mean:.2f}"

            best, second = rankings[metric]
            if np.isclose(mean, best, atol=1e-10):
                value_pm = f"\\textbf{{{value_pm}}}"
                value_mean = f"\\textbf{{{value_mean}}}"
            elif second is not None and np.isclose(mean, second, atol=1e-10):
                value_pm = f"\\underline{{{value_pm}}}"
                value_mean = f"\\underline{{{value_mean}}}"

            values_pm.append(value_pm)
            values_mean.append(value_mean)

        method = str(row["Method"])
        latex_lines.append(f"{method} & " + " & ".join(values_pm) + r" \\")
        mean_only_lines.append(f"{method} & " + " & ".join(values_mean) + r" \\")
        last_cat = cat

    (outdir / "comparison_results_for_latex_mean_std.txt").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )
    (outdir / "comparison_results_for_latex_mean_only.txt").write_text(
        "\n".join(mean_only_lines), encoding="utf-8"
    )

    table_lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Comparison with representative DGA fault diagnosis methods on the Hunan-Plant dataset over ten runs. Results are reported as mean $\pm$ standard deviation.}",
        r"\label{tab:comparison-ten-runs}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|cccc}",
        r"\toprule",
        r"Method & Acc (\%) & Macro-F1 (\%) & Macro-P (\%) & Macro-R (\%) \\",
        r"\midrule",
        *latex_lines,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    (outdir / "comparison_table_10runs.tex").write_text(
        "\n".join(table_lines), encoding="utf-8"
    )


def save_results(
    rows: Sequence[ResultRow],
    outdir: Path,
    include_existing_ours: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_df = rows_to_dataframe(rows)
    run_df.to_csv(
        outdir / "comparison_results_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    summary_df = summarize_results(run_df)
    summary_df["Source"] = "Computed by this script"
    if include_existing_ours:
        summary_df = append_existing_ours(summary_df)

    summary_df.to_csv(
        outdir / "comparison_results_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    write_latex_outputs(summary_df, outdir)
    return run_df, summary_df


def run_all(args: argparse.Namespace) -> pd.DataFrame:
    outdir = ensure_dir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Create the stratified split and fit the scaler exactly once.
    split = load_and_split_data(args.data, label_col=args.label_col, seed=SPLIT_SEED)

    print(f"Using device: {device}")
    print(f"Fixed split seed: {SPLIT_SEED}")
    print(f"Run seeds: {list(RUN_SEEDS)}")
    print(f"Classes: {split.display_labels}")
    print(
        f"Fixed split sizes: train={len(split.y_train)}, "
        f"val={len(split.y_val)}, test={len(split.y_test)}"
    )
    print("GR-KANet will not be retrained; its existing 10-run statistics will be appended.")

    all_rows: List[ResultRow] = []
    for repeat_idx, seed in enumerate(RUN_SEEDS, start=1):
        repeat_rows = run_one_repeat(args, split, repeat_idx, seed, device)
        all_rows.extend(repeat_rows)
        # Keep partial computed results if a long experiment is interrupted.
        save_results(all_rows, outdir, include_existing_ours=False)

    _, summary_df = save_results(all_rows, outdir, include_existing_ours=True)

    print("\n========== 10-run summary (mean ± std) ==========")
    for _, row in summary_df.iterrows():
        print(
            f"{row['Method']:28s} | "
            f"Acc={row['Acc (%) Mean']:.2f}±{row['Acc (%) Std']:.2f} | "
            f"Macro-F1={row['Macro-F1 (%) Mean']:.2f}±{row['Macro-F1 (%) Std']:.2f} | "
            f"Macro-P={row['Macro-P (%) Mean']:.2f}±{row['Macro-P (%) Std']:.2f} | "
            f"Macro-R={row['Macro-R (%) Mean']:.2f}±{row['Macro-R (%) Std']:.2f}"
        )

    print(f"\nSaved per-run results: {outdir / 'comparison_results_runs.csv'}")
    print(f"Saved final summary: {outdir / 'comparison_results_summary.csv'}")
    print(f"Saved LaTeX rows: {outdir / 'comparison_results_for_latex_mean_std.txt'}")
    print(f"Saved full LaTeX table: {outdir / 'comparison_table_10runs.tex'}")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Table 2 comparison methods 10 times on the fixed Hunan-Plant split."
    )
    parser.add_argument("--data", type=str, default="胖虎电厂数据未加测试集.xlsx")
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument("--outdir", type=str, default="ajse_hunan_comparison_10runs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--torch-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--boost-rounds", type=int, default=5)
    parser.add_argument("--boost-epochs", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
