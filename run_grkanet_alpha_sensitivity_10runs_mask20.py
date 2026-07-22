# -*- coding: utf-8 -*-
"""
Ten-run alpha-initialization sensitivity experiment for the full GR-KANet.

Purpose
-------
Evaluate whether the initial value of the learnable EGRM scaling coefficient
alpha affects validation stability and final diagnostic performance. The model
architecture and training objective are unchanged.

Fixed protocol
--------------
- Fixed expert-binary EGRM used in the submitted manuscript.
- Fixed stratified train/validation/test split = 64/16/20.
- Fixed split seed = 42.
- Ten training seeds = 42, 43, ..., 51.
- KAN-DGAM derivative mask updated every 20 GLOBAL optimizer steps.
- StandardScaler fitted only on the training set.
- Checkpoint selected by validation Macro-F1.
- All settings except alpha initialization remain unchanged.
- The final alpha initialization must be selected from validation results, not
  from test performance.

Default alpha initializations
-----------------------------
0.10, 0.25, 0.50, 1.00, and 2.00.

Outputs include per-seed results, mean/std, t-based 95% confidence intervals,
validation-only ranking, final learned alpha, and EGRM drift.

Python compatibility: Python 3.8+.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
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
# Fixed data settings
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
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class AlphaRunResult:
    alpha_init: float
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
    learned_egrm: np.ndarray


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
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        le,
        display_labels,
        scaler,
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
        raise ValueError(
            "balance_mode must be one of: none, ce_only, sampler_only, both"
        )

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
# EGRM prior construction
# -----------------------------------------------------------------------------

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
        outs = []
        for j, f_j in enumerate(self.functions):
            x_j = x[:, j : j + 1]
            outs.append(f_j(x_j) + x_j)
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.transformed_components(x))


class KANDGAM(nn.Module):
    def __init__(
        self,
        channels: int,
        num_gases: int,
        kan_layer: KANLayer,
        mode: str = "energy",
        ema: float = 0.9,
        reference_points: int = 200,
        entropy_bins: int = 30,
    ):
        super().__init__()
        if mode not in {"energy", "entropy"}:
            raise ValueError("mode must be energy or entropy")
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
        d = self.kan_layer.in_features
        if d != C * G:
            raise ValueError(
                "KAN input dimension does not match channels*num_gases"
            )

        grid = torch.linspace(
            -3.0, 3.0, self.reference_points, device=device
        ).unsqueeze(1)
        derivs = []
        for f_j in self.kan_layer.functions:
            y = f_j(grid).squeeze(1)
            dy = torch.gradient(y)[0]
            derivs.append(dy.detach().float().cpu().numpy())

        derivs = np.asarray(derivs, dtype=np.float32).reshape(
            C, G, self.reference_points
        )

        if self.mode == "energy":
            scores = np.mean(np.square(derivs), axis=(1, 2)).astype(np.float32)
        else:
            scores_list = []
            for c in range(C):
                curve = np.abs(derivs[c].reshape(-1))
                hist, _ = np.histogram(
                    curve, bins=self.entropy_bins, density=False
                )
                p = hist.astype(np.float64)
                p = p / (p.sum() + 1e-12)
                scores_list.append(float(-np.sum(p * np.log(p + 1e-12))))
            scores = np.asarray(scores_list, dtype=np.float32)

        self.channel_scores = scores.copy()
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return torch.tensor(norm, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_mask_from_kan(self) -> None:
        new_mask = self._compute_channel_scores()
        self.kan_mask.mul_(self.ema).add_(new_mask * (1.0 - self.ema))

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        g = self.gate(x_k)
        w = self.kan_mask.view(1, -1, 1)
        return x_k * g * w


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
        if egrm_init.shape != (num_gases, channels // num_gases):
            raise ValueError(
                "EGRM shape must be ({}, {}), got {}".format(
                    num_gases, channels // num_gases, egrm_init.shape
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
        self.alpha = nn.Parameter(
            torch.tensor(alpha_init, dtype=torch.float32)
        )

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
        x2 = self.residual2(x1 + x_a)
        return x2

    def extract_kan_features(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.forward_features(x)
        return self.kan(x2.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.extract_kan_features(x)
        z = self.dropout(z)
        return self.classifier(z)


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, EvalResult]:
    model.eval()
    total_loss = 0.0
    total = 0
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
                    loss = criterion(logits, yb)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            pred = logits.argmax(dim=1)
            total_loss += float(loss.item()) * yb.size(0)
            total += yb.size(0)
            y_true.extend(yb.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    result = EvalResult(
        acc=accuracy_score(yt, yp),
        macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
        macro_p=precision_score(
            yt, yp, average="macro", zero_division=0
        ),
        macro_r=recall_score(yt, yp, average="macro", zero_division=0),
        y_true=yt,
        y_pred=yp,
    )
    return total_loss / max(total, 1), result


def train_one_seed(
    split: SplitData,
    alpha_init: float,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[EvalResult, int, float, float, float, np.ndarray]:
    """Train one full GR-KANet run for a specified alpha initialization."""
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
        dgam_mode=args.dgam_mode,
        alpha_init=alpha_init,
        dropout=args.dropout,
        channels=args.channels,
        kan_hidden=args.kan_hidden,
        kan_latent=args.kan_latent,
        dgam_ema=args.dgam_ema,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    # Initialize once, then refresh every N global optimizer steps.
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
                    raise RuntimeError(
                        "AMP is enabled but GradScaler is unavailable."
                    )
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

        _, val_res = evaluate_model(model, val_loader, criterion, device)
        val_macro_f1 = val_res.macro_f1

        if args.verbose:
            print(
                "    alpha_init={:.4f} seed={} epoch={:03d} valF1={:.4f}".format(
                    alpha_init, seed, epoch, val_macro_f1
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
        raise RuntimeError("No valid checkpoint was obtained.")

    model.load_state_dict(best_state)
    model.to(device)

    _, test_res = evaluate_model(model, test_loader, criterion, device)
    learned_egrm = model.egrm.detach().cpu().numpy().astype(np.float32)
    final_alpha = float(model.alpha.detach().cpu().item())
    egrm_change_fro = float(
        np.linalg.norm(learned_egrm - EXPERT_BINARY, ord="fro")
    )

    return (
        test_res,
        best_epoch,
        best_val_macro_f1,
        final_alpha,
        egrm_change_fro,
        learned_egrm,
    )


# -----------------------------------------------------------------------------
# Statistics and output
# -----------------------------------------------------------------------------

def parse_seed_list(text: str, n_runs: int) -> List[int]:
    if text.strip():
        seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid seed was found.")
        return seeds
    return [42 + i for i in range(n_runs)]


def parse_alpha_list(text: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not np.isfinite(value):
            raise ValueError("Every alpha initialization must be finite.")
        values.append(value)
    if not values:
        raise ValueError("--alphas must contain at least one numeric value.")
    # Preserve order while removing exact duplicates.
    return list(dict.fromkeys(values))


def alpha_label(value: float) -> str:
    text = ("{:.8f}".format(value)).rstrip("0").rstrip(".")
    return "alpha_" + text.replace("-", "m").replace(".", "p")


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
    se = std / math.sqrt(n)
    if student_t is not None:
        critical = float(student_t.ppf(0.975, df=n - 1))
    else:
        critical = 1.96
    half_width = critical * se
    return mean - half_width, mean + half_width, half_width


def summarize_results(run_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metrics = [
        "Best val Macro-F1 (%)",
        "Acc (%)",
        "Macro-F1 (%)",
        "Macro-P (%)",
        "Macro-R (%)",
        "Final alpha",
        "Alpha displacement",
        "EGRM change Frobenius",
        "Elapsed seconds",
    ]

    for alpha_init, group in run_df.groupby("Alpha init", sort=False):
        row: Dict[str, object] = {
            "Alpha init": float(alpha_init),
            "Runs": int(len(group)),
            "Best epoch Mean": float(group["Best epoch"].mean()),
            "Best epoch Std": sample_std(group["Best epoch"].values),
        }
        for metric in metrics:
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


def format_pm(mean_value: float, std_value: float, digits: int = 2) -> str:
    template = "{:." + str(digits) + "f} $\\pm$ {:." + str(digits) + "f}"
    return template.format(mean_value, std_value)


def write_latex_table(summary_df: pd.DataFrame, outdir: Path) -> None:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        (
            r"\caption{Sensitivity of the full GR-KANet model to the "
            r"initial value of the learnable EGRM scaling coefficient $\alpha$. "
            r"Results are reported as mean $\pm$ standard deviation over ten "
            r"training seeds under a fixed stratified split. Configuration "
            r"selection is based only on validation Macro-F1.}"
        ),
        r"\label{tab:alpha-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c|cccccc}",
        r"\toprule",
        (
            r"Initial $\alpha$ & Val Macro-F1 (\%) & Acc (\%) & Macro-F1 (\%) & "
            r"Macro-P (\%) & Macro-R (\%) & Final $\alpha$ \\"
        ),
        r"\midrule",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "{:.2f} & {} & {} & {} & {} & {} & {} \\\\".format(
                float(row["Alpha init"]),
                format_pm(
                    row["Best val Macro-F1 (%) Mean"],
                    row["Best val Macro-F1 (%) Std"],
                ),
                format_pm(row["Acc (%) Mean"], row["Acc (%) Std"]),
                format_pm(
                    row["Macro-F1 (%) Mean"], row["Macro-F1 (%) Std"]
                ),
                format_pm(
                    row["Macro-P (%) Mean"], row["Macro-P (%) Std"]
                ),
                format_pm(
                    row["Macro-R (%) Mean"], row["Macro-R (%) Std"]
                ),
                format_pm(
                    row["Final alpha Mean"], row["Final alpha Std"], digits=4
                ),
            )
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    (outdir / "alpha_sensitivity_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_learned_egrm_outputs(
    results: Sequence[AlphaRunResult], outdir: Path
) -> None:
    if not results:
        return

    arrays: Dict[str, np.ndarray] = {}
    by_alpha: Dict[float, List[np.ndarray]] = {}
    for result in results:
        key = "{}_seed_{}".format(alpha_label(result.alpha_init), result.seed)
        arrays[key] = result.learned_egrm.astype(np.float32)
        by_alpha.setdefault(result.alpha_init, []).append(result.learned_egrm)

    np.savez_compressed(outdir / "learned_egrm_all_runs.npz", **arrays)

    for alpha_init, matrices in by_alpha.items():
        stack = np.stack(matrices, axis=0)
        mean_matrix = stack.mean(axis=0)
        std_matrix = (
            stack.std(axis=0, ddof=1)
            if len(stack) > 1
            else np.zeros_like(mean_matrix)
        )
        label = alpha_label(alpha_init)
        pd.DataFrame(
            mean_matrix,
            index=FEATURES,
            columns=PATTERNS,
        ).to_csv(
            outdir / "learned_egrm_mean_{}.csv".format(label),
            encoding="utf-8-sig",
            float_format="%.8f",
        )
        pd.DataFrame(
            std_matrix,
            index=FEATURES,
            columns=PATTERNS,
        ).to_csv(
            outdir / "learned_egrm_std_{}.csv".format(label),
            encoding="utf-8-sig",
            float_format="%.8f",
        )


def write_outputs(
    results: Sequence[AlphaRunResult], outdir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for result in results:
        run_rows.append(
            {
                "Alpha init": result.alpha_init,
                "Run": result.run_id,
                "Seed": result.seed,
                "Best epoch": result.best_epoch,
                "Best val Macro-F1 (%)": 100.0 * result.best_val_macro_f1,
                "Acc (%)": 100.0 * result.acc,
                "Macro-F1 (%)": 100.0 * result.macro_f1,
                "Macro-P (%)": 100.0 * result.macro_p,
                "Macro-R (%)": 100.0 * result.macro_r,
                "Final alpha": result.final_alpha,
                "Alpha displacement": result.final_alpha - result.alpha_init,
                "EGRM change Frobenius": result.egrm_change_fro,
                "Elapsed seconds": result.elapsed_seconds,
            }
        )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(
        outdir / "alpha_sensitivity_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    summary_df = summarize_results(run_df)
    summary_df.to_csv(
        outdir / "alpha_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    validation_ranking = summary_df.sort_values(
        by=[
            "Best val Macro-F1 (%) Mean",
            "Best val Macro-F1 (%) Std",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)
    validation_ranking.insert(
        0,
        "Validation rank",
        np.arange(1, len(validation_ranking) + 1),
    )
    validation_ranking.to_csv(
        outdir / "alpha_sensitivity_validation_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    write_latex_table(summary_df, outdir)
    write_learned_egrm_outputs(results, outdir)
    return run_df, summary_df


# -----------------------------------------------------------------------------
# Command line and main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run alpha-initialization sensitivity experiments for the full "
            "GR-KANet model using ten fixed training seeds and "
            "mask_update_every=20."
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
        default="grkanet_alpha_sensitivity_10runs_mask20_outputs",
    )

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds. Empty means consecutive seeds from 42.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.10,0.25,0.50,1.00,2.00",
        help=(
            "Comma-separated initial values of the learnable alpha. "
            "The final setting must be selected from validation results."
        ),
    )

    # Fixed full-model settings.
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kan-hidden", type=int, default=8)
    parser.add_argument("--kan-latent", type=int, default=64)
    parser.add_argument("--mask-update-every", type=int, default=20)
    parser.add_argument("--dgam-ema", type=float, default=0.8)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive.")
    if args.mask_update_every <= 0:
        raise ValueError("--mask-update-every must be positive.")

    outdir = ensure_dir(args.outdir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    seeds = parse_seed_list(args.seeds, args.n_runs)
    alpha_values = parse_alpha_list(args.alphas)

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
    print("Alpha initializations: {}".format(alpha_values))
    print("Classes: {}".format(split.display_labels))
    print(
        "Split sizes: train={}, val={}, test={}".format(
            len(split.y_train), len(split.y_val), len(split.y_test)
        )
    )
    print(
        "Fixed full-model config | expert_binary EGRM | mask_every={} "
        "mode={} ema={} dropout={} lr={} batch_size={} epochs={} "
        "patience={} balance={}".format(
            args.mask_update_every,
            args.dgam_mode,
            args.dgam_ema,
            args.dropout,
            args.lr,
            args.batch_size,
            args.epochs,
            args.patience,
            args.balance_mode,
        )
    )
    print("Only the initial value of the learnable alpha changes.")
    print("Selection/ranking uses validation Macro-F1 only, not test metrics.")

    results: List[AlphaRunResult] = []

    for alpha_idx, alpha_init in enumerate(alpha_values, start=1):
        print(
            "\n################ Alpha {}/{}: {:.4f} ################".format(
                alpha_idx, len(alpha_values), alpha_init
            )
        )

        for run_id, seed in enumerate(seeds, start=1):
            print(
                "\n===== alpha_init={:.4f} | Run {}/{} | seed={} =====".format(
                    alpha_init, run_id, len(seeds), seed
                )
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            (
                test_res,
                best_epoch,
                best_val_macro_f1,
                final_alpha,
                egrm_change_fro,
                learned_egrm,
            ) = train_one_seed(
                split,
                alpha_init,
                seed,
                device,
                args,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            result = AlphaRunResult(
                alpha_init=alpha_init,
                run_id=run_id,
                seed=seed,
                best_epoch=best_epoch,
                best_val_macro_f1=best_val_macro_f1,
                acc=test_res.acc,
                macro_f1=test_res.macro_f1,
                macro_p=test_res.macro_p,
                macro_r=test_res.macro_r,
                elapsed_seconds=elapsed,
                final_alpha=final_alpha,
                egrm_change_fro=egrm_change_fro,
                learned_egrm=learned_egrm,
            )
            results.append(result)

            print(
                "alpha_init={:.4f} | Run {:02d} | seed={} | best_epoch={} | "
                "valF1={:.2f} | Acc={:.2f} | Macro-F1={:.2f} | "
                "Macro-P={:.2f} | Macro-R={:.2f} | final_alpha={:.4f} | "
                "delta_alpha={:+.4f} | EGRM-change={:.4f} | time={:.1f}s".format(
                    alpha_init,
                    run_id,
                    seed,
                    best_epoch,
                    100.0 * best_val_macro_f1,
                    100.0 * test_res.acc,
                    100.0 * test_res.macro_f1,
                    100.0 * test_res.macro_p,
                    100.0 * test_res.macro_r,
                    final_alpha,
                    final_alpha - alpha_init,
                    egrm_change_fro,
                    elapsed,
                )
            )

            # Preserve completed runs if execution is interrupted.
            write_outputs(results, outdir)

    _, summary_df = write_outputs(results, outdir)

    print("\n================ Alpha-sensitivity summary ================")
    for _, row in summary_df.iterrows():
        print(
            "alpha_init={:<5.2f} | ValF1={:.2f}±{:.2f} | Acc={:.2f}±{:.2f} | "
            "Macro-F1={:.2f}±{:.2f} | Macro-P={:.2f}±{:.2f} | "
            "Macro-R={:.2f}±{:.2f} | final_alpha={:.4f}±{:.4f}".format(
                row["Alpha init"],
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
                row["Final alpha Mean"],
                row["Final alpha Std"],
            )
        )

    ranked = summary_df.sort_values(
        by=[
            "Best val Macro-F1 (%) Mean",
            "Best val Macro-F1 (%) Std",
        ],
        ascending=[False, True],
    )
    best_alpha = float(ranked.iloc[0]["Alpha init"])
    print(
        "\nValidation-only first-ranked alpha initialization: {:.4f}.".format(
            best_alpha
        )
    )
    print(
        "Do not select alpha from the test metrics. After locking the value, "
        "rerun the full model and all affected ablations under the same protocol."
    )

    print("\nSaved:")
    print("  {}".format(outdir / "fixed_expert_binary_egrm.csv"))
    print("  {}".format(outdir / "alpha_sensitivity_runs.csv"))
    print("  {}".format(outdir / "alpha_sensitivity_summary.csv"))
    print("  {}".format(outdir / "alpha_sensitivity_validation_ranking.csv"))
    print("  {}".format(outdir / "alpha_sensitivity_table.tex"))
    print("  {}".format(outdir / "learned_egrm_all_runs.npz"))


if __name__ == "__main__":
    main()
