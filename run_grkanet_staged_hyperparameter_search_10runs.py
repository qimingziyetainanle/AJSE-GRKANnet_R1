# -*- coding: utf-8 -*-
"""
Stage-wise ten-run hyperparameter search for the full GR-KANet.

New searches performed by this script
-------------------------------------
1) KAN-DGAM EMA coefficient: 0.70, 0.80, 0.90
2) Adam learning rate:       5e-4, 1e-3, 2e-3
3) Adam weight decay:        0, 1e-5, 1e-4
4) Dropout probability:      0, 0.10, 0.20

Search protocol
---------------
- Full GR-KANet only.
- Expert-binary EGRM.
- alpha initialization = 2.0.
- KAN-DGAM mask update interval = 20 global optimizer steps.
- Single-best checkpoint selected by validation Macro-F1.
- Fixed stratified train/validation/test split = 64/16/20.
- Split seed = 42.
- Training seeds = 42, 43, ..., 51.
- StandardScaler fitted only on the training set.
- Weighted sampler + class-weighted cross-entropy.
- Maximum epochs = 150; early-stopping patience = 80.

The four new parameters are searched stage by stage. At each stage, the
candidate with the highest ten-run mean validation Macro-F1 is locked before
moving to the next stage. Validation standard deviation is used as the first
tie-breaker. Test metrics are recorded for reporting but NEVER used by the
selection code.

The script also embeds the previously completed EGRM-prior, alpha, mask-update,
and checkpoint-strategy summaries supplied by the user. These previous results
are merged with the new results into one master CSV and one grouped LaTeX table.
No previous experiment is retrained.

Python compatibility: Python 3.8+.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None


# -----------------------------------------------------------------------------
# Fixed data/model settings
# -----------------------------------------------------------------------------

FEATURES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]
PATTERNS = ["P1", "P2", "P3", "P4"]
ID_TO_ABBR = {1: "T12", 2: "T3", 3: "PD", 4: "D1", 5: "D2", 6: "NC"}

EXPERT_BINARY = np.asarray(
    [
        [1, 0, 0, 0],  # H2
        [1, 1, 0, 0],  # CH4
        [0, 1, 1, 0],  # C2H4
        [0, 1, 1, 0],  # C2H2
        [0, 0, 0, 1],  # CO
        [0, 0, 0, 1],  # CO2
        [1, 1, 1, 0],  # THC
        [0, 1, 1, 0],  # C2H6
    ],
    dtype=np.float32,
)


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    display_labels: List[str]
    scaler: StandardScaler


@dataclass
class EvalResult:
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float


@dataclass
class ModelConfig:
    dgam_ema: float = 0.8
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.0


@dataclass
class SearchRunResult:
    stage: int
    parameter: str
    candidate: float
    run_id: int
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float
    elapsed_seconds: float
    final_alpha: float
    egrm_change_fro: float
    dgam_ema: float
    lr: float
    weight_decay: float
    dropout: float


class DGADataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Reproducibility and data
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_fixed_split(
    data_path: str,
    label_col: str = "故障编码",
    split_seed: int = 42,
    test_size: float = 0.20,
    val_ratio_in_trainval: float = 0.20,
) -> SplitData:
    df = pd.read_excel(data_path)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing feature columns: {}".format(missing))
    if label_col not in df.columns:
        raise ValueError("Missing label column: {}".format(label_col))

    X = df[FEATURES].values.astype(np.float32)
    raw_y = df[label_col].values

    le = LabelEncoder()
    y = le.fit_transform(raw_y)

    raw_class_ids = le.inverse_transform(np.arange(len(le.classes_)))
    display_labels: List[str] = []
    for cid in raw_class_ids:
        try:
            display_labels.append(ID_TO_ABBR[int(cid)])
        except Exception:
            display_labels.append(str(cid))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=split_seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio_in_trainval,
        stratify=y_trainval,
        random_state=split_seed,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=le,
        display_labels=display_labels,
        scaler=scaler,
    )


def compute_class_weights_np(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-8)
    inv = inv * (len(classes) / inv.sum())
    weights = np.ones(num_classes, dtype=np.float32)
    weights[classes] = inv
    return weights


def make_loaders(
    split: SplitData,
    batch_size: int,
    balance_mode: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DGADataset(split.X_train, split.y_train)
    val_ds = DGADataset(split.X_val, split.y_val)
    test_ds = DGADataset(split.X_test, split.y_test)

    if balance_mode not in {"none", "ce_only", "sampler_only", "both"}:
        raise ValueError("Invalid balance_mode: {}".format(balance_mode))

    if balance_mode in {"sampler_only", "both"}:
        classes, counts = np.unique(split.y_train, return_counts=True)
        inv_count = {
            int(cls): 1.0 / float(cnt) for cls, cnt in zip(classes, counts)
        }
        sample_weights = np.asarray(
            [inv_count[int(t)] for t in split.y_train], dtype=np.float32
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# GR-KANet model
# -----------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(in_features)
            ]
        )
        self.linear = nn.Linear(in_features, out_features)

    def transformed_components(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for j, function_j in enumerate(self.functions):
            x_j = x[:, j : j + 1]
            outputs.append(function_j(x_j) + x_j)
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.transformed_components(x))


class KANDGAM(nn.Module):
    def __init__(
        self,
        channels: int,
        num_gases: int,
        kan_layer: KANLayer,
        mode: str = "energy",
        ema: float = 0.8,
        reference_points: int = 200,
        entropy_bins: int = 30,
    ):
        super().__init__()
        if mode not in {"energy", "entropy"}:
            raise ValueError("mode must be 'energy' or 'entropy'")
        if not 0.0 <= ema < 1.0:
            raise ValueError("EMA coefficient must satisfy 0 <= ema < 1")

        self.channels = channels
        self.num_gases = num_gases
        self.kan_layer = kan_layer
        self.mode = mode
        self.ema = ema
        self.reference_points = reference_points
        self.entropy_bins = entropy_bins
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Sigmoid())
        self.register_buffer("kan_mask", torch.ones(channels))
        self.channel_scores: Optional[np.ndarray] = None

    @torch.no_grad()
    def _compute_channel_scores(self) -> torch.Tensor:
        device = self.kan_mask.device
        C = self.channels
        G = self.num_gases
        if self.kan_layer.in_features != C * G:
            raise ValueError("KAN input dimension must equal channels*num_gases")

        grid = torch.linspace(
            -3.0, 3.0, self.reference_points, device=device
        ).unsqueeze(1)
        derivatives = []
        for function_j in self.kan_layer.functions:
            y = function_j(grid).squeeze(1)
            dy = torch.gradient(y)[0]
            derivatives.append(dy.detach().float().cpu().numpy())

        derivatives = np.asarray(derivatives, dtype=np.float32).reshape(
            C, G, self.reference_points
        )

        if self.mode == "energy":
            scores = np.mean(np.square(derivatives), axis=(1, 2)).astype(np.float32)
        else:
            score_list = []
            for channel in range(C):
                curve = np.abs(derivatives[channel].reshape(-1))
                hist, _ = np.histogram(
                    curve, bins=self.entropy_bins, density=False
                )
                probabilities = hist.astype(np.float64)
                probabilities = probabilities / (probabilities.sum() + 1e-12)
                score_list.append(
                    float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
                )
            scores = np.asarray(score_list, dtype=np.float32)

        self.channel_scores = scores.copy()
        normalized = (scores - scores.min()) / (
            scores.max() - scores.min() + 1e-8
        )
        return torch.tensor(normalized, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_mask_from_kan(self) -> None:
        new_mask = self._compute_channel_scores()
        self.kan_mask.mul_(self.ema).add_(new_mask * (1.0 - self.ema))

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        sample_gate = self.gate(x_k)
        mask = self.kan_mask.view(1, -1, 1)
        return x_k * sample_gate * mask


class GRKANet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        egrm_init: np.ndarray,
        num_gases: int = 8,
        channels: int = 32,
        kan_hidden: int = 8,
        kan_latent: int = 64,
        dgam_mode: str = "energy",
        dropout: float = 0.0,
        alpha_init: float = 2.0,
        dgam_ema: float = 0.8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_gases = num_gases
        self.channels = channels

        egrm_init = np.asarray(egrm_init, dtype=np.float32)
        expected_shape = (num_gases, channels // num_gases)
        if egrm_init.shape != expected_shape:
            raise ValueError(
                "EGRM shape must be {}, got {}".format(
                    expected_shape, egrm_init.shape
                )
            )

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual1 = ResidualBlock1D(channels)
        self.residual2 = ResidualBlock1D(channels)
        self.kan = KANLayer(
            channels * num_gases, kan_latent, hidden=kan_hidden
        )
        self.dgam = KANDGAM(
            channels,
            num_gases,
            self.kan,
            mode=dgam_mode,
            ema=dgam_ema,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(kan_latent, num_classes)

        K0 = torch.tensor(egrm_init, dtype=torch.float32)
        self.register_buffer("egrm_init", K0)
        self.egrm_delta = nn.Parameter(torch.zeros_like(K0))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    @property
    def egrm(self) -> torch.Tensor:
        return self.egrm_init + self.egrm_delta

    def egrm_channel_prior(self) -> torch.Tensor:
        return self.egrm.reshape(1, self.channels, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x.unsqueeze(1))
        x1 = self.residual1(x0)
        x_k = x1 + self.alpha * self.egrm_channel_prior().to(x1.device)
        x_a = self.dgam(x_k)
        return self.residual2(x1 + x_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        latent = self.kan(features.flatten(1))
        latent = self.dropout(latent)
        return self.classifier(latent)


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    amp_enabled = device.type == "cuda" and autocast is not None

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if amp_enabled:
                with autocast(enabled=True):
                    logits = model(xb)
                    _ = criterion(logits, yb)
            else:
                logits = model(xb)
                _ = criterion(logits, yb)
            prediction = logits.argmax(dim=1)
            y_true.extend(yb.detach().cpu().numpy().tolist())
            y_pred.extend(prediction.detach().cpu().numpy().tolist())

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return EvalResult(
        acc=accuracy_score(yt, yp),
        macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
        macro_p=precision_score(yt, yp, average="macro", zero_division=0),
        macro_r=recall_score(yt, yp, average="macro", zero_division=0),
    )


def train_one_seed(
    split: SplitData,
    config: ModelConfig,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[EvalResult, int, float, float, float]:
    set_seed(seed)
    num_classes = len(split.display_labels)
    train_loader, val_loader, test_loader = make_loaders(
        split,
        batch_size=args.batch_size,
        balance_mode=args.balance_mode,
    )

    model = GRKANet(
        num_classes=num_classes,
        egrm_init=EXPERT_BINARY,
        channels=args.channels,
        kan_hidden=args.kan_hidden,
        kan_latent=args.kan_latent,
        dgam_mode=args.dgam_mode,
        dropout=config.dropout,
        alpha_init=args.alpha_init,
        dgam_ema=config.dgam_ema,
    ).to(device)

    if args.balance_mode in {"ce_only", "both"}:
        class_weights = torch.tensor(
            compute_class_weights_np(split.y_train, num_classes),
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = (
        GradScaler(enabled=(device.type == "cuda"))
        if GradScaler is not None
        else None
    )
    amp_enabled = device.type == "cuda" and autocast is not None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_macro_f1 = -1.0
    no_improve = 0

    model.dgam.update_mask_from_kan()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with autocast(enabled=True):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                if scaler is None:
                    raise RuntimeError("AMP is enabled but GradScaler is unavailable")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            global_step += 1
            if global_step % args.mask_update_every == 0:
                model.dgam.update_mask_from_kan()

        val_result = evaluate_model(model, val_loader, criterion, device)
        val_macro_f1 = val_result.macro_f1

        if args.verbose:
            print(
                "    epoch={:03d} | val Macro-F1={:.4f}".format(
                    epoch, val_macro_f1
                )
            )

        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was obtained")

    model.load_state_dict(best_state)
    model.to(device)
    test_result = evaluate_model(model, test_loader, criterion, device)

    final_alpha = float(model.alpha.detach().cpu().item())
    learned_egrm = model.egrm.detach().cpu().numpy().astype(np.float32)
    egrm_change_fro = float(
        np.linalg.norm(learned_egrm - EXPERT_BINARY, ord="fro")
    )
    return (
        test_result,
        best_epoch,
        best_val_macro_f1,
        final_alpha,
        egrm_change_fro,
    )


# -----------------------------------------------------------------------------
# Statistics and output
# -----------------------------------------------------------------------------

def parse_seed_list(text: str, n_runs: int) -> List[int]:
    if text.strip():
        seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not seeds:
            raise ValueError("No valid seed was found in --seeds")
        return seeds
    return [42 + i for i in range(n_runs)]


def parse_float_list(text: str, name: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not np.isfinite(value):
            raise ValueError("{} contains a non-finite value".format(name))
        values.append(value)
    if not values:
        raise ValueError("{} must contain at least one numeric value".format(name))
    return list(dict.fromkeys(values))


def sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    if n <= 1:
        return mean, mean, 0.0
    std = float(array.std(ddof=1))
    standard_error = std / math.sqrt(n)
    critical = (
        float(student_t.ppf(0.975, df=n - 1))
        if student_t is not None
        else 1.96
    )
    half_width = critical * standard_error
    return mean - half_width, mean + half_width, half_width


def format_candidate(parameter: str, value: float) -> str:
    if parameter == "Learning rate":
        return "{:.1e}".format(value)
    if parameter == "Weight decay":
        if value == 0:
            return "0"
        return "{:.1e}".format(value)
    return ("{:.6f}".format(value)).rstrip("0").rstrip(".")


def summarize_new_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    metric_columns = [
        "Best val Macro-F1 (%)",
        "Acc (%)",
        "Macro-F1 (%)",
        "Macro-P (%)",
        "Macro-R (%)",
        "Final alpha",
        "EGRM change Frobenius",
        "Elapsed seconds",
    ]

    group_columns = ["Stage", "Parameter", "Candidate"]
    for keys, group in run_df.groupby(group_columns, sort=False):
        stage, parameter, candidate = keys
        row: Dict[str, object] = {
            "Stage": int(stage),
            "Parameter": str(parameter),
            "Candidate": float(candidate),
            "Candidate display": format_candidate(str(parameter), float(candidate)),
            "Runs": int(len(group)),
            "Best epoch Mean": float(group["Best epoch"].mean()),
            "Best epoch Std": sample_std(group["Best epoch"].values),
            "Search context": str(group["Search context"].iloc[0]),
        }
        for metric in metric_columns:
            values = group[metric].values.astype(float)
            mean = float(values.mean())
            std = sample_std(values)
            ci_low, ci_high, ci_half = confidence_interval_95(values)
            row[metric + " Mean"] = mean
            row[metric + " Std"] = std
            row[metric + " 95% CI Low"] = ci_low
            row[metric + " 95% CI High"] = ci_high
            row[metric + " 95% CI Half-width"] = ci_half
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def choose_validation_best(stage_summary: pd.DataFrame) -> pd.Series:
    ranked = stage_summary.sort_values(
        by=[
            "Best val Macro-F1 (%) Mean",
            "Best val Macro-F1 (%) Std",
            "Candidate",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def previous_results_rows() -> List[Dict[str, object]]:
    """Embed all previously supplied summaries without retraining them."""
    rows: List[Dict[str, object]] = []

    def add(
        group: str,
        parameter: str,
        candidate: str,
        val_mean: float,
        val_std: float,
        acc_mean: float,
        acc_std: float,
        f1_mean: float,
        f1_std: float,
        p_mean: float,
        p_std: float,
        r_mean: float,
        r_std: float,
        decision: str,
        context: str,
    ) -> None:
        rows.append(
            {
                "Experiment group": group,
                "Parameter": parameter,
                "Candidate": candidate,
                "Runs": 10,
                "Val Macro-F1 (%) Mean": val_mean,
                "Val Macro-F1 (%) Std": val_std,
                "Acc (%) Mean": acc_mean,
                "Acc (%) Std": acc_std,
                "Macro-F1 (%) Mean": f1_mean,
                "Macro-F1 (%) Std": f1_std,
                "Macro-P (%) Mean": p_mean,
                "Macro-P (%) Std": p_std,
                "Macro-R (%) Mean": r_mean,
                "Macro-R (%) Std": r_std,
                "Decision": decision,
                "Source": "Previously completed experiment",
                "Search context": context,
            }
        )

    # EGRM prior robustness experiment.
    prior_context = "alpha=2.0; mask interval=20; EMA=0.8; lr=1e-3; wd=0; dropout=0"
    add("Previous", "EGRM prior", "expert_binary", 94.60, 1.07, 93.65, 2.73, 94.24, 2.55, 94.34, 2.55, 94.48, 2.61, "Retained physically informed prior", prior_context)
    add("Previous", "EGRM prior", "expert_row_norm", 94.39, 1.27, 93.46, 3.29, 94.15, 3.17, 94.23, 3.23, 94.69, 3.03, "Sensitivity candidate", prior_context)
    add("Previous", "EGRM prior", "expert_col_norm", 94.40, 1.28, 92.31, 2.22, 93.03, 2.14, 93.39, 2.16, 93.22, 2.18, "Sensitivity candidate", prior_context)
    add("Previous", "EGRM prior", "uniform_density", 94.71, 1.22, 93.08, 2.60, 93.77, 2.60, 94.18, 2.54, 93.95, 2.73, "Unstructured control", prior_context)
    add("Previous", "EGRM prior", "random_same_density", 94.56, 1.69, 91.54, 3.76, 92.24, 3.76, 92.67, 3.95, 92.44, 3.38, "Randomized control", prior_context)
    add("Previous", "EGRM prior", "zero_init", 94.20, 1.09, 92.88, 3.01, 93.38, 2.87, 93.66, 2.87, 93.66, 2.63, "No-prior initialization control", prior_context)

    # Alpha sensitivity experiment.
    alpha_context = "expert_binary; mask interval=20; EMA=0.8; lr=1e-3; wd=0; dropout=0"
    add("Previous", "Alpha initialization", "0.10", 94.20, 1.10, 93.46, 2.60, 94.09, 2.66, 94.42, 2.72, 94.39, 2.70, "Candidate", alpha_context)
    add("Previous", "Alpha initialization", "0.25", 94.11, 0.95, 92.31, 3.51, 92.92, 3.48, 93.21, 3.57, 93.27, 3.12, "Candidate", alpha_context)
    add("Previous", "Alpha initialization", "0.50", 94.21, 1.09, 92.88, 2.04, 93.49, 1.89, 93.76, 2.03, 93.77, 1.77, "Candidate", alpha_context)
    add("Previous", "Alpha initialization", "1.00", 94.40, 1.28, 93.65, 2.04, 94.17, 1.94, 94.41, 2.41, 94.35, 1.54, "Candidate", alpha_context)
    add("Previous", "Alpha initialization", "2.00", 94.60, 1.07, 93.65, 2.73, 94.24, 2.55, 94.34, 2.55, 94.48, 2.61, "Selected by validation", alpha_context)

    # Mask-update interval experiment.
    mask_context = "expert_binary; alpha=2.0; EMA=0.8; lr=1e-3; wd=0; dropout=0"
    add("Previous", "Mask update interval", "1", 94.58, 1.40, 92.69, 2.53, 93.34, 2.34, 93.43, 2.29, 93.59, 2.31, "Candidate", mask_context)
    add("Previous", "Mask update interval", "5", 94.60, 1.06, 92.50, 3.07, 93.10, 3.06, 93.19, 3.30, 93.48, 2.81, "Candidate", mask_context)
    add("Previous", "Mask update interval", "10", 94.40, 1.28, 92.50, 2.30, 93.18, 2.36, 93.22, 2.54, 93.60, 2.24, "Candidate", mask_context)
    add("Previous", "Mask update interval", "20", 94.60, 1.07, 93.65, 2.73, 94.24, 2.55, 94.34, 2.55, 94.48, 2.61, "Selected: equivalent-best validation and lower cost", mask_context)

    # Checkpoint-strategy experiment.
    checkpoint_context = "expert_binary; alpha=2.0; mask interval=20; EMA=0.8; lr=1e-3; wd=0; dropout=0"
    add("Previous", "Checkpoint strategy", "Single-best", 94.60, 1.07, 93.65, 2.73, 94.24, 2.55, 94.34, 2.55, 94.48, 2.61, "Selected by validation", checkpoint_context)
    add("Previous", "Checkpoint strategy", "Top-3 weight average", 94.23, 0.61, 95.00, 2.60, 95.47, 2.51, 95.43, 2.52, 95.76, 2.50, "Not selected: lower validation mean", checkpoint_context)
    add("Previous", "Checkpoint strategy", "Top-5 weight average", 94.23, 1.08, 95.00, 2.07, 95.47, 2.13, 95.44, 2.16, 95.76, 2.19, "Not selected: lower validation mean", checkpoint_context)

    return rows


def write_new_outputs(
    results: Sequence[SearchRunResult],
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[Dict[str, object]] = []
    for result in results:
        context = (
            "EMA={:.6g}; lr={:.6g}; wd={:.6g}; dropout={:.6g}".format(
                result.dgam_ema,
                result.lr,
                result.weight_decay,
                result.dropout,
            )
        )
        run_rows.append(
            {
                "Stage": result.stage,
                "Parameter": result.parameter,
                "Candidate": result.candidate,
                "Run": result.run_id,
                "Seed": result.seed,
                "Best epoch": result.best_epoch,
                "Best val Macro-F1 (%)": 100.0 * result.best_val_macro_f1,
                "Acc (%)": 100.0 * result.acc,
                "Macro-F1 (%)": 100.0 * result.macro_f1,
                "Macro-P (%)": 100.0 * result.macro_p,
                "Macro-R (%)": 100.0 * result.macro_r,
                "Final alpha": result.final_alpha,
                "EGRM change Frobenius": result.egrm_change_fro,
                "Elapsed seconds": result.elapsed_seconds,
                "DGAM EMA": result.dgam_ema,
                "Learning rate": result.lr,
                "Weight decay": result.weight_decay,
                "Dropout": result.dropout,
                "Search context": context,
            }
        )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(
        outdir / "new_hyperparameter_search_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    summary_df = summarize_new_runs(run_df)
    summary_df.to_csv(
        outdir / "new_hyperparameter_search_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    ranking_parts = []
    for (stage, parameter), group in summary_df.groupby(
        ["Stage", "Parameter"], sort=False
    ):
        ranked = group.sort_values(
            by=[
                "Best val Macro-F1 (%) Mean",
                "Best val Macro-F1 (%) Std",
                "Candidate",
            ],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        ranked.insert(0, "Validation rank", np.arange(1, len(ranked) + 1))
        ranking_parts.append(ranked)
    ranking_df = pd.concat(ranking_parts, ignore_index=True)
    ranking_df.to_csv(
        outdir / "new_hyperparameter_validation_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return run_df, summary_df


def make_master_table(
    new_summary: pd.DataFrame,
    selected_by_stage: Dict[int, float],
    outdir: Path,
) -> pd.DataFrame:
    master_rows = previous_results_rows()

    for _, row in new_summary.iterrows():
        stage = int(row["Stage"])
        candidate = float(row["Candidate"])
        selected = abs(candidate - float(selected_by_stage[stage])) < 1e-12
        master_rows.append(
            {
                "Experiment group": "New stage {}".format(stage),
                "Parameter": str(row["Parameter"]),
                "Candidate": str(row["Candidate display"]),
                "Runs": int(row["Runs"]),
                "Val Macro-F1 (%) Mean": float(row["Best val Macro-F1 (%) Mean"]),
                "Val Macro-F1 (%) Std": float(row["Best val Macro-F1 (%) Std"]),
                "Acc (%) Mean": float(row["Acc (%) Mean"]),
                "Acc (%) Std": float(row["Acc (%) Std"]),
                "Macro-F1 (%) Mean": float(row["Macro-F1 (%) Mean"]),
                "Macro-F1 (%) Std": float(row["Macro-F1 (%) Std"]),
                "Macro-P (%) Mean": float(row["Macro-P (%) Mean"]),
                "Macro-P (%) Std": float(row["Macro-P (%) Std"]),
                "Macro-R (%) Mean": float(row["Macro-R (%) Mean"]),
                "Macro-R (%) Std": float(row["Macro-R (%) Std"]),
                "Decision": "Selected by validation" if selected else "Candidate",
                "Source": "Computed by this script",
                "Search context": str(row["Search context"]),
            }
        )

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(
        outdir / "combined_all_sensitivity_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    write_master_latex(master_df, outdir)
    write_compact_new_latex(new_summary, selected_by_stage, outdir)
    return master_df


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    output = str(text)
    for old, new in replacements.items():
        output = output.replace(old, new)
    return output


def pm(mean_value: float, std_value: float) -> str:
    return "{:.2f} $\\pm$ {:.2f}".format(mean_value, std_value)


def write_master_latex(master_df: pd.DataFrame, outdir: Path) -> None:
    lines = [
        r"\begin{longtable}{llc|ccccc}",
        r"\caption{Combined robustness and hyperparameter sensitivity results of GR-KANet. All values are mean $\pm$ standard deviation over ten runs. Parameter selection uses validation Macro-F1 only.}\label{tab:combined-sensitivity}\\",
        r"\toprule",
        r"Group & Parameter & Candidate & Val F1 (\%) & Acc (\%) & Macro-F1 (\%) & Macro-P (\%) & Macro-R (\%) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Group & Parameter & Candidate & Val F1 (\%) & Acc (\%) & Macro-F1 (\%) & Macro-P (\%) & Macro-R (\%) \\",
        r"\midrule",
        r"\endhead",
    ]

    previous_parameter: Optional[str] = None
    for _, row in master_df.iterrows():
        parameter = str(row["Parameter"])
        if previous_parameter is not None and parameter != previous_parameter:
            lines.append(r"\midrule")
        decision = str(row["Decision"])
        candidate = latex_escape(str(row["Candidate"]))
        if decision.startswith("Selected") or decision.startswith("Retained"):
            candidate = r"\textbf{" + candidate + "}"
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(str(row["Experiment group"])),
                latex_escape(parameter),
                candidate,
                pm(row["Val Macro-F1 (%) Mean"], row["Val Macro-F1 (%) Std"]),
                pm(row["Acc (%) Mean"], row["Acc (%) Std"]),
                pm(row["Macro-F1 (%) Mean"], row["Macro-F1 (%) Std"]),
                pm(row["Macro-P (%) Mean"], row["Macro-P (%) Std"]),
                pm(row["Macro-R (%) Mean"], row["Macro-R (%) Std"]),
            )
        )
        previous_parameter = parameter

    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (outdir / "combined_all_sensitivity_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_compact_new_latex(
    new_summary: pd.DataFrame,
    selected_by_stage: Dict[int, float],
    outdir: Path,
) -> None:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Stage-wise sensitivity of GR-KANet to DGAM EMA, learning rate, weight decay, and dropout. Results are mean $\pm$ standard deviation over ten runs. Bold values denote the validation-selected setting at each stage.}",
        r"\label{tab:new-hyperparameter-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll|ccccc}",
        r"\toprule",
        r"Parameter & Candidate & Val Macro-F1 (\%) & Acc (\%) & Macro-F1 (\%) & Macro-P (\%) & Macro-R (\%) \\",
        r"\midrule",
    ]

    previous_stage: Optional[int] = None
    for _, row in new_summary.sort_values(["Stage", "Candidate"]).iterrows():
        stage = int(row["Stage"])
        if previous_stage is not None and stage != previous_stage:
            lines.append(r"\midrule")
        selected = abs(
            float(row["Candidate"]) - float(selected_by_stage[stage])
        ) < 1e-12
        cells = [
            latex_escape(str(row["Parameter"])),
            latex_escape(str(row["Candidate display"])),
            pm(row["Best val Macro-F1 (%) Mean"], row["Best val Macro-F1 (%) Std"]),
            pm(row["Acc (%) Mean"], row["Acc (%) Std"]),
            pm(row["Macro-F1 (%) Mean"], row["Macro-F1 (%) Std"]),
            pm(row["Macro-P (%) Mean"], row["Macro-P (%) Std"]),
            pm(row["Macro-R (%) Mean"], row["Macro-R (%) Std"]),
        ]
        if selected:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(*cells))
        previous_stage = stage

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    (outdir / "new_hyperparameter_sensitivity_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Stage-wise search
# -----------------------------------------------------------------------------

def config_with_candidate(
    base_config: ModelConfig,
    parameter: str,
    candidate: float,
) -> ModelConfig:
    values = asdict(base_config)
    mapping = {
        "DGAM EMA": "dgam_ema",
        "Learning rate": "lr",
        "Weight decay": "weight_decay",
        "Dropout": "dropout",
    }
    values[mapping[parameter]] = float(candidate)
    return ModelConfig(**values)


def run_stage(
    stage: int,
    parameter: str,
    candidates: Sequence[float],
    base_config: ModelConfig,
    split: SplitData,
    seeds: Sequence[int],
    device: torch.device,
    args: argparse.Namespace,
    all_results: List[SearchRunResult],
    outdir: Path,
) -> Tuple[ModelConfig, float, pd.DataFrame]:
    print("\n" + "#" * 78)
    print("Stage {}: {}".format(stage, parameter))
    print("Base config entering stage: {}".format(asdict(base_config)))
    print("Candidates: {}".format(list(candidates)))
    print("#" * 78)

    stage_start_index = len(all_results)
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_config = config_with_candidate(base_config, parameter, candidate)
        print(
            "\n========== Stage {} | {} candidate {}/{}: {} ==========".format(
                stage,
                parameter,
                candidate_index,
                len(candidates),
                format_candidate(parameter, candidate),
            )
        )
        print("Candidate config: {}".format(asdict(candidate_config)))

        for run_id, seed in enumerate(seeds, start=1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            (
                test_result,
                best_epoch,
                best_val_macro_f1,
                final_alpha,
                egrm_change_fro,
            ) = train_one_seed(
                split=split,
                config=candidate_config,
                seed=seed,
                device=device,
                args=args,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            result = SearchRunResult(
                stage=stage,
                parameter=parameter,
                candidate=float(candidate),
                run_id=run_id,
                seed=seed,
                best_epoch=best_epoch,
                best_val_macro_f1=best_val_macro_f1,
                acc=test_result.acc,
                macro_f1=test_result.macro_f1,
                macro_p=test_result.macro_p,
                macro_r=test_result.macro_r,
                elapsed_seconds=elapsed,
                final_alpha=final_alpha,
                egrm_change_fro=egrm_change_fro,
                dgam_ema=candidate_config.dgam_ema,
                lr=candidate_config.lr,
                weight_decay=candidate_config.weight_decay,
                dropout=candidate_config.dropout,
            )
            all_results.append(result)

            print(
                "Run {:02d}/{:02d} | seed={} | best_epoch={:03d} | "
                "ValF1={:.2f} | Acc={:.2f} | Macro-F1={:.2f} | "
                "Macro-P={:.2f} | Macro-R={:.2f} | time={:.1f}s".format(
                    run_id,
                    len(seeds),
                    seed,
                    best_epoch,
                    100.0 * best_val_macro_f1,
                    100.0 * test_result.acc,
                    100.0 * test_result.macro_f1,
                    100.0 * test_result.macro_p,
                    100.0 * test_result.macro_r,
                    elapsed,
                )
            )

            # Preserve all completed runs if interrupted.
            write_new_outputs(all_results, outdir)

    _, full_summary = write_new_outputs(all_results, outdir)
    stage_summary = full_summary[full_summary["Stage"] == stage].copy()
    selected_row = choose_validation_best(stage_summary)
    selected_candidate = float(selected_row["Candidate"])
    selected_config = config_with_candidate(
        base_config, parameter, selected_candidate
    )

    print("\n----- Stage {} summary: {} -----".format(stage, parameter))
    for _, row in stage_summary.sort_values("Candidate").iterrows():
        print(
            "{}={} | ValF1={:.2f}±{:.2f} | Acc={:.2f}±{:.2f} | "
            "Macro-F1={:.2f}±{:.2f} | Macro-P={:.2f}±{:.2f} | "
            "Macro-R={:.2f}±{:.2f}".format(
                parameter,
                row["Candidate display"],
                row["Best val Macro-F1 (%) Mean"],
                row["Best val Macro-F1 (%) Std"],
                row["Acc (%) Mean"],
                row["Acc (%) Std"],
                row["Macro-F1 (%) Mean"],
                row["Macro-F1 (%) Std"],
                row["Macro-P (%) Mean"],
                row["Macro-P (%) Std"],
                row["Macro-R (%) Mean"],
                row["Macro-R (%) Std"],
            )
        )

    print(
        "Validation-selected {}: {}".format(
            parameter, format_candidate(parameter, selected_candidate)
        )
    )
    print("Locked config after stage {}: {}".format(stage, asdict(selected_config)))
    print(
        "Selection used validation Macro-F1 only. Test metrics did not enter the ranking."
    )

    # Safety check: the stage must have contributed exactly len(candidates)*len(seeds) runs.
    expected_new = len(candidates) * len(seeds)
    actual_new = len(all_results) - stage_start_index
    if actual_new != expected_new:
        raise RuntimeError(
            "Stage {} produced {} runs, expected {}".format(
                stage, actual_new, expected_new
            )
        )

    return selected_config, selected_candidate, stage_summary


# -----------------------------------------------------------------------------
# Command line and main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a four-stage ten-seed GR-KANet hyperparameter search for "
            "DGAM EMA, learning rate, weight decay, and dropout."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default="胖虎电厂数据未加测试集.xlsx",
    )
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument(
        "--outdir",
        type=str,
        default="grkanet_staged_hyperparameter_search_10runs_outputs",
    )

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated training seeds. Empty uses 42,43,...",
    )

    parser.add_argument(
        "--ema-values",
        type=str,
        default="0.70,0.80,0.90",
    )
    parser.add_argument(
        "--lr-values",
        type=str,
        default="0.0005,0.001,0.002",
    )
    parser.add_argument(
        "--weight-decay-values",
        type=str,
        default="0,0.00001,0.0001",
    )
    parser.add_argument(
        "--dropout-values",
        type=str,
        default="0,0.10,0.20",
    )

    # Fixed full-model protocol.
    parser.add_argument("--alpha-init", type=float, default=2.0)
    parser.add_argument("--mask-update-every", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--balance-mode",
        choices=["none", "ce_only", "sampler_only", "both"],
        default="both",
    )
    parser.add_argument(
        "--dgam-mode",
        choices=["energy", "entropy"],
        default="energy",
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kan-hidden", type=int, default=8)
    parser.add_argument("--kan-latent", type=int, default=64)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive")
    if args.mask_update_every <= 0:
        raise ValueError("--mask-update-every must be positive")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")

    ema_values = parse_float_list(args.ema_values, "--ema-values")
    lr_values = parse_float_list(args.lr_values, "--lr-values")
    weight_decay_values = parse_float_list(
        args.weight_decay_values, "--weight-decay-values"
    )
    dropout_values = parse_float_list(args.dropout_values, "--dropout-values")

    if len(ema_values) != 3:
        raise ValueError("--ema-values must contain exactly three candidates")
    if len(lr_values) != 3:
        raise ValueError("--lr-values must contain exactly three candidates")
    if len(weight_decay_values) != 3:
        raise ValueError(
            "--weight-decay-values must contain exactly three candidates"
        )
    if len(dropout_values) != 3:
        raise ValueError("--dropout-values must contain exactly three candidates")
    if any(not 0.0 <= value < 1.0 for value in ema_values):
        raise ValueError("Every EMA candidate must satisfy 0 <= value < 1")
    if any(value <= 0 for value in lr_values):
        raise ValueError("Every learning-rate candidate must be positive")
    if any(value < 0 for value in weight_decay_values):
        raise ValueError("Every weight-decay candidate must be nonnegative")
    if any(not 0.0 <= value < 1.0 for value in dropout_values):
        raise ValueError("Every dropout candidate must satisfy 0 <= value < 1")

    outdir = ensure_dir(args.outdir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    seeds = parse_seed_list(args.seeds, args.n_runs)
    split = load_fixed_split(
        args.data,
        label_col=args.label_col,
        split_seed=args.split_seed,
    )

    pd.DataFrame(
        EXPERT_BINARY,
        index=FEATURES,
        columns=PATTERNS,
    ).to_csv(
        outdir / "fixed_expert_binary_egrm.csv",
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    print("Using device: {}".format(device))
    print("Data file: {}".format(args.data))
    print("Fixed split seed: {}".format(args.split_seed))
    print("Training seeds: {}".format(seeds))
    print("Classes: {}".format(split.display_labels))
    print(
        "Split sizes: train={}, val={}, test={}".format(
            len(split.y_train), len(split.y_val), len(split.y_test)
        )
    )
    print(
        "Fixed protocol | expert_binary EGRM | alpha={} | mask_every={} | "
        "single-best checkpoint | batch={} | epochs={} | patience={} | balance={}".format(
            args.alpha_init,
            args.mask_update_every,
            args.batch_size,
            args.epochs,
            args.patience,
            args.balance_mode,
        )
    )
    print(
        "Four-stage coordinate search: EMA -> learning rate -> weight decay -> dropout"
    )
    print("Each stage uses validation Macro-F1 only for candidate selection.")
    print("Total new training runs: {}".format(4 * 3 * len(seeds)))

    # Baseline entering Stage 1.
    current_config = ModelConfig(
        dgam_ema=0.8,
        lr=1e-3,
        weight_decay=0.0,
        dropout=0.0,
    )

    all_results: List[SearchRunResult] = []
    selected_by_stage: Dict[int, float] = {}
    stage_history: List[Dict[str, object]] = []

    stages = [
        (1, "DGAM EMA", ema_values),
        (2, "Learning rate", lr_values),
        (3, "Weight decay", weight_decay_values),
        (4, "Dropout", dropout_values),
    ]

    for stage, parameter, candidates in stages:
        entering_config = asdict(current_config)
        current_config, selected_candidate, stage_summary = run_stage(
            stage=stage,
            parameter=parameter,
            candidates=candidates,
            base_config=current_config,
            split=split,
            seeds=seeds,
            device=device,
            args=args,
            all_results=all_results,
            outdir=outdir,
        )
        selected_by_stage[stage] = selected_candidate
        selected_row = choose_validation_best(stage_summary)
        stage_history.append(
            {
                "Stage": stage,
                "Parameter": parameter,
                "Entering config": json.dumps(entering_config, ensure_ascii=False),
                "Selected candidate": selected_candidate,
                "Selected candidate display": format_candidate(
                    parameter, selected_candidate
                ),
                "Selected validation Macro-F1 (%) Mean": float(
                    selected_row["Best val Macro-F1 (%) Mean"]
                ),
                "Selected validation Macro-F1 (%) Std": float(
                    selected_row["Best val Macro-F1 (%) Std"]
                ),
                "Locked config after stage": json.dumps(
                    asdict(current_config), ensure_ascii=False
                ),
            }
        )
        pd.DataFrame(stage_history).to_csv(
            outdir / "selected_stage_trajectory.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.8f",
        )
        with (outdir / "final_selected_configuration.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "selection_basis": "ten-run mean validation Macro-F1; validation std tie-breaker",
                    "fixed": {
                        "egrm_prior": "expert_binary",
                        "alpha_init": args.alpha_init,
                        "mask_update_every": args.mask_update_every,
                        "checkpoint_strategy": "single-best",
                        "split_seed": args.split_seed,
                        "training_seeds": list(seeds),
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "patience": args.patience,
                        "balance_mode": args.balance_mode,
                        "dgam_mode": args.dgam_mode,
                        "channels": args.channels,
                        "kan_hidden": args.kan_hidden,
                        "kan_latent": args.kan_latent,
                    },
                    "selected": asdict(current_config),
                    "selected_by_stage": {
                        str(key): value for key, value in selected_by_stage.items()
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    _, new_summary = write_new_outputs(all_results, outdir)
    master_df = make_master_table(
        new_summary=new_summary,
        selected_by_stage=selected_by_stage,
        outdir=outdir,
    )

    final_stage = new_summary[new_summary["Stage"] == 4].copy()
    final_selected_row = final_stage[
        np.isclose(
            final_stage["Candidate"].astype(float),
            selected_by_stage[4],
            rtol=0,
            atol=1e-12,
        )
    ].iloc[0]

    print("\n" + "=" * 78)
    print("FINAL VALIDATION-SELECTED CONFIGURATION")
    print("=" * 78)
    print(json.dumps(asdict(current_config), indent=2))
    print(
        "Final selected configuration metrics from Stage 4 | "
        "ValF1={:.2f}±{:.2f} | Acc={:.2f}±{:.2f} | "
        "Macro-F1={:.2f}±{:.2f} | Macro-P={:.2f}±{:.2f} | "
        "Macro-R={:.2f}±{:.2f}".format(
            final_selected_row["Best val Macro-F1 (%) Mean"],
            final_selected_row["Best val Macro-F1 (%) Std"],
            final_selected_row["Acc (%) Mean"],
            final_selected_row["Acc (%) Std"],
            final_selected_row["Macro-F1 (%) Mean"],
            final_selected_row["Macro-F1 (%) Std"],
            final_selected_row["Macro-P (%) Mean"],
            final_selected_row["Macro-P (%) Std"],
            final_selected_row["Macro-R (%) Mean"],
            final_selected_row["Macro-R (%) Std"],
        )
    )
    print("The final configuration was selected without using test metrics.")
    print("Total rows in combined master table: {}".format(len(master_df)))

    print("\nSaved:")
    for filename in [
        "new_hyperparameter_search_runs.csv",
        "new_hyperparameter_search_summary.csv",
        "new_hyperparameter_validation_ranking.csv",
        "selected_stage_trajectory.csv",
        "final_selected_configuration.json",
        "combined_all_sensitivity_results.csv",
        "combined_all_sensitivity_table.tex",
        "new_hyperparameter_sensitivity_table.tex",
        "fixed_expert_binary_egrm.csv",
    ]:
        print("  {}".format(outdir / filename))


if __name__ == "__main__":
    main()
