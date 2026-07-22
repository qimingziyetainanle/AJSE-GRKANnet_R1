# -*- coding: utf-8 -*-
"""
Ten-run KAN-DGAM mask-update-interval experiment for the full GR-KANet model.

Purpose:
    Evaluate how the KAN-DGAM derivative-mask update interval affects training
    stability, reproducibility, and diagnostic performance. This experiment
    trains the FULL GR-KANet only. The model architecture, fixed data split,
    preprocessing, optimizer, loss, early stopping, and all other parameters
    are held unchanged.

Default protocol:
    - Fixed stratified train/validation/test split = 64/16/20.
    - Fixed split seed = 42.
    - Ten training seeds = 42, 43, ..., 51.
    - Compared update intervals = 1, 5, 10, and 20 GLOBAL optimizer steps.
    - StandardScaler fitted only on the training set.
    - Checkpoint selected by validation Macro-F1.
    - The interval should be selected from validation results, not test results.
    - Test metrics are reported only to characterize the locked configurations.

All full-model settings other than mask_update_every remain identical to
run_grkanet_full_10runs_final.py:
    energy mask, alpha=2.0, DGAM EMA=0.8, dropout=0.0, lr=1e-3,
    batch_size=64, weighted sampler + class-weighted CE,
    max_epochs=150, patience=80.

Python compatibility:
    This script avoids Python 3.10-only type syntax and should run on Python 3.8.
"""

from __future__ import annotations

import argparse
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
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None


# -----------------------------
# Fixed data settings
# -----------------------------

FEATURES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]
ID_TO_ABBR = {1: "T12", 2: "T3", 3: "PD", 4: "D1", 5: "D2", 6: "NC"}


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
class RunResult:
    run_id: int
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float


class DGADataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path) -> Path:
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
    """Load data and create one fixed stratified 64/16/20 split."""
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
    display_labels = []
    for cid in raw_class_ids:
        try:
            display_labels.append(ID_TO_ABBR[int(cid)])
        except Exception:
            display_labels.append(str(cid))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=split_seed
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

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, le, display_labels, scaler)


def compute_class_weights_np(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-8)
    inv = inv * (len(classes) / inv.sum())
    weights = np.ones(num_classes, dtype=np.float32)
    weights[classes] = inv
    return weights


def make_loaders(split: SplitData, batch_size: int, balance_mode: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DGADataset(split.X_train, split.y_train)
    val_ds = DGADataset(split.X_val, split.y_val)
    test_ds = DGADataset(split.X_test, split.y_test)

    if balance_mode not in {"none", "ce_only", "sampler_only", "both"}:
        raise ValueError("balance_mode must be one of: none, ce_only, sampler_only, both")

    if balance_mode in {"sampler_only", "both"}:
        classes, counts = np.unique(split.y_train, return_counts=True)
        inv_count = {int(cls): 1.0 / float(cnt) for cls, cnt in zip(classes, counts)}
        sample_weights = np.asarray([inv_count[int(t)] for t in split.y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# -----------------------------
# GR-KANet model
# -----------------------------

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
        self.functions = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(in_features)
        ])
        self.linear = nn.Linear(in_features, out_features)

    def transformed_components(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for j, f_j in enumerate(self.functions):
            x_j = x[:, j:j + 1]
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
            raise ValueError("KAN input dimension does not match channels*num_gases")

        grid = torch.linspace(-3.0, 3.0, self.reference_points, device=device).unsqueeze(1)
        derivs = []
        for f_j in self.kan_layer.functions:
            y = f_j(grid).squeeze(1)
            dy = torch.gradient(y)[0]
            derivs.append(dy.detach().float().cpu().numpy())

        derivs = np.asarray(derivs, dtype=np.float32).reshape(C, G, self.reference_points)

        if self.mode == "energy":
            scores = np.mean(np.square(derivs), axis=(1, 2)).astype(np.float32)
        else:
            scores_list = []
            for c in range(C):
                curve = np.abs(derivs[c].reshape(-1))
                hist, _ = np.histogram(curve, bins=self.entropy_bins, density=False)
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

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual1 = ResidualBlock1D(channels)
        self.residual2 = ResidualBlock1D(channels)
        self.kan = KANLayer(channels * num_gases, kan_latent, hidden=kan_hidden)
        self.dgam = KANDGAM(channels, num_gases, self.kan, mode=dgam_mode, ema=dgam_ema)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(kan_latent, num_classes)

        K0 = torch.tensor([
            [1, 0, 0, 0],  # H2
            [1, 1, 0, 0],  # CH4
            [0, 1, 1, 0],  # C2H4
            [0, 1, 1, 0],  # C2H2
            [0, 0, 0, 1],  # CO
            [0, 0, 0, 1],  # CO2
            [1, 1, 1, 0],  # THC
            [0, 1, 1, 0],  # C2H6
        ], dtype=torch.float32)
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
        x2 = self.residual2(x1 + x_a)
        return x2

    def extract_kan_features(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.forward_features(x)
        return self.kan(x2.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.extract_kan_features(x)
        z = self.dropout(z)
        return self.classifier(z)


# -----------------------------
# Training and evaluation
# -----------------------------

def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, EvalResult]:
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
    res = EvalResult(
        acc=accuracy_score(yt, yp),
        macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
        macro_p=precision_score(yt, yp, average="macro", zero_division=0),
        macro_r=recall_score(yt, yp, average="macro", zero_division=0),
        y_true=yt,
        y_pred=yp,
    )
    return total_loss / max(total, 1), res


def train_one_seed(
    split: SplitData,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[EvalResult, int, float]:
    set_seed(seed)
    num_classes = len(split.display_labels)

    train_loader, val_loader, test_loader = make_loaders(
        split, batch_size=args.batch_size, balance_mode=args.balance_mode
    )

    model = GRKANet(
        num_classes=num_classes,
        dgam_mode=args.dgam_mode,
        alpha_init=args.alpha_init,
        dropout=args.dropout,
        channels=args.channels,
        kan_hidden=args.kan_hidden,
        kan_latent=args.kan_latent,
        dgam_ema=args.dgam_ema,
    ).to(device)

    if args.balance_mode in {"ce_only", "both"}:
        class_weights = torch.tensor(
            compute_class_weights_np(split.y_train, num_classes), dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda")) if GradScaler is not None else None
    amp_enabled = device.type == "cuda" and autocast is not None

    best_state = None
    best_epoch = 0
    best_val_macro_f1 = -1.0
    no_improve = 0

    # Initialize the derivative-based mask before training.
    model.dgam.update_mask_from_kan()

    # IMPORTANT: use a global optimizer-step counter. The Hunan-Plant training
    # split has only a few mini-batches per epoch, so an epoch-local counter
    # could otherwise use an inconsistent update schedule.
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
                assert scaler is not None
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
            print("    seed={} epoch={:03d} valF1={:.4f}".format(seed, epoch, val_macro_f1))

        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    _, test_res = evaluate_model(model, test_loader, criterion, device)
    return test_res, best_epoch, best_val_macro_f1



@dataclass
class IntervalRunResult:
    interval: int
    run_id: int
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float
    elapsed_seconds: float


def parse_int_list(text: str, name: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("{} must contain at least one integer.".format(name))
    if any(value <= 0 for value in values):
        raise ValueError("All values in {} must be positive integers.".format(name))
    # Preserve the user's order while removing duplicates.
    return list(dict.fromkeys(values))


def parse_seed_list(text: str, n_runs: int) -> List[int]:
    if text.strip():
        seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid seed was found.")
        return seeds
    return [42 + i for i in range(n_runs)]


def sample_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarize_results(run_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for interval, group in run_df.groupby("Mask update interval", sort=False):
        summary_rows.append({
            "Mask update interval": int(interval),
            "Runs": int(len(group)),
            "Best val Macro-F1 (%) Mean": float(group["Best val Macro-F1 (%)"].mean()),
            "Best val Macro-F1 (%) Std": sample_std(group["Best val Macro-F1 (%)"].values),
            "Best epoch Mean": float(group["Best epoch"].mean()),
            "Best epoch Std": sample_std(group["Best epoch"].values),
            "Acc (%) Mean": float(group["Acc (%)"].mean()),
            "Acc (%) Std": sample_std(group["Acc (%)"].values),
            "Macro-F1 (%) Mean": float(group["Macro-F1 (%)"].mean()),
            "Macro-F1 (%) Std": sample_std(group["Macro-F1 (%)"].values),
            "Macro-P (%) Mean": float(group["Macro-P (%)"].mean()),
            "Macro-P (%) Std": sample_std(group["Macro-P (%)"].values),
            "Macro-R (%) Mean": float(group["Macro-R (%)"].mean()),
            "Macro-R (%) Std": sample_std(group["Macro-R (%)"].values),
            "Elapsed (s/run) Mean": float(group["Elapsed seconds"].mean()),
            "Elapsed (s/run) Std": sample_std(group["Elapsed seconds"].values),
        })
    return pd.DataFrame(summary_rows)


def format_pm(mean_value: float, std_value: float) -> str:
    return "{:.2f} $\\pm$ {:.2f}".format(mean_value, std_value)


def write_latex_table(summary_df: pd.DataFrame, outdir: Path) -> None:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        (
            r"\caption{Effect of the KAN-DGAM derivative-mask update interval on the full "
            r"GR-KANet model. Results are reported as mean $\pm$ standard deviation over ten "
            r"training seeds under a fixed stratified split. The interval is selected according "
            r"to validation Macro-F1.}"
        ),
        r"\label{tab:mask-update-interval}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c|ccccc}",
        r"\toprule",
        (
            r"Interval & Val Macro-F1 (\%) & Acc (\%) & Macro-F1 (\%) & "
            r"Macro-P (\%) & Macro-R (\%) \\"
        ),
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                int(row["Mask update interval"]),
                format_pm(row["Best val Macro-F1 (%) Mean"], row["Best val Macro-F1 (%) Std"]),
                format_pm(row["Acc (%) Mean"], row["Acc (%) Std"]),
                format_pm(row["Macro-F1 (%) Mean"], row["Macro-F1 (%) Std"]),
                format_pm(row["Macro-P (%) Mean"], row["Macro-P (%) Std"]),
                format_pm(row["Macro-R (%) Mean"], row["Macro-R (%) Std"]),
            )
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    (outdir / "mask_update_interval_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results: List[IntervalRunResult], outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for result in results:
        run_rows.append({
            "Mask update interval": result.interval,
            "Run": result.run_id,
            "Seed": result.seed,
            "Best epoch": result.best_epoch,
            "Best val Macro-F1 (%)": 100.0 * result.best_val_macro_f1,
            "Acc (%)": 100.0 * result.acc,
            "Macro-F1 (%)": 100.0 * result.macro_f1,
            "Macro-P (%)": 100.0 * result.macro_p,
            "Macro-R (%)": 100.0 * result.macro_r,
            "Elapsed seconds": result.elapsed_seconds,
        })

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(
        outdir / "mask_update_interval_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    summary_df = summarize_results(run_df)
    summary_df.to_csv(
        outdir / "mask_update_interval_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # This file is the one to inspect first. Ranking uses validation only.
    validation_ranking = summary_df.sort_values(
        by=["Best val Macro-F1 (%) Mean", "Best val Macro-F1 (%) Std"],
        ascending=[False, True],
    ).reset_index(drop=True)
    validation_ranking.insert(0, "Validation rank", np.arange(1, len(validation_ranking) + 1))
    validation_ranking.to_csv(
        outdir / "mask_update_interval_validation_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    write_latex_table(summary_df, outdir)
    return run_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full GR-KANet over multiple KAN-DGAM mask-update intervals "
            "using ten fixed training seeds."
        )
    )
    parser.add_argument("--data", type=str, default="胖虎电厂数据未加测试集.xlsx")
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument("--outdir", type=str, default="grkanet_mask_interval_10runs_outputs")

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds. Empty means consecutive seeds starting from 42.",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="1,5,10,20",
        help="Comma-separated GLOBAL optimizer-step intervals to compare.",
    )

    # These defaults are copied unchanged from the final full-model script.
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--balance-mode",
        choices=["none", "ce_only", "sampler_only", "both"],
        default="both",
    )
    parser.add_argument("--dgam-mode", choices=["energy", "entropy"], default="energy")
    parser.add_argument("--alpha-init", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kan-hidden", type=int, default=8)
    parser.add_argument("--kan-latent", type=int, default=64)
    parser.add_argument("--dgam-ema", type=float, default=0.8)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive.")

    outdir = ensure_dir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    intervals = parse_int_list(args.intervals, "--intervals")
    seeds = parse_seed_list(args.seeds, args.n_runs)

    split = load_fixed_split(
        args.data,
        label_col=args.label_col,
        split_seed=args.split_seed,
    )

    print("Using device: {}".format(device))
    print("Data file: {}".format(args.data))
    print("Fixed split seed: {}".format(args.split_seed))
    print("Training seeds: {}".format(seeds))
    print("Mask update intervals: {}".format(intervals))
    print("Classes: {}".format(split.display_labels))
    print(
        "Split sizes: train={}, val={}, test={}".format(
            len(split.y_train), len(split.y_val), len(split.y_test)
        )
    )
    print(
        "Fixed full-model config | mode={} alpha={} ema={} dropout={} lr={} "
        "batch_size={} epochs={} patience={} balance={}".format(
            args.dgam_mode,
            args.alpha_init,
            args.dgam_ema,
            args.dropout,
            args.lr,
            args.batch_size,
            args.epochs,
            args.patience,
            args.balance_mode,
        )
    )
    print("Only mask_update_every changes between configurations.")

    results: List[IntervalRunResult] = []

    for interval_idx, interval in enumerate(intervals, start=1):
        print(
            "\n################ Interval {}/{}: every {} global steps ################".format(
                interval_idx, len(intervals), interval
            )
        )
        interval_args = argparse.Namespace(**vars(args))
        interval_args.mask_update_every = interval

        for run_id, seed in enumerate(seeds, start=1):
            print(
                "\n===== Interval={} | Run {}/{} | seed={} =====".format(
                    interval, run_id, len(seeds), seed
                )
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            test_res, best_epoch, best_val_macro_f1 = train_one_seed(
                split, seed, device, interval_args
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            result = IntervalRunResult(
                interval=interval,
                run_id=run_id,
                seed=seed,
                best_epoch=best_epoch,
                best_val_macro_f1=best_val_macro_f1,
                acc=test_res.acc,
                macro_f1=test_res.macro_f1,
                macro_p=test_res.macro_p,
                macro_r=test_res.macro_r,
                elapsed_seconds=elapsed,
            )
            results.append(result)

            print(
                "Interval={} | Run {:02d} | seed={} | best_epoch={} | "
                "valF1={:.2f} | Acc={:.2f} | Macro-F1={:.2f} | "
                "Macro-P={:.2f} | Macro-R={:.2f} | time={:.1f}s".format(
                    interval,
                    run_id,
                    seed,
                    best_epoch,
                    100.0 * best_val_macro_f1,
                    100.0 * test_res.acc,
                    100.0 * test_res.macro_f1,
                    100.0 * test_res.macro_p,
                    100.0 * test_res.macro_r,
                    elapsed,
                )
            )

            # Save after every run so completed results survive interruption.
            write_outputs(results, outdir)

    _, summary_df = write_outputs(results, outdir)

    print("\n================ Mask-update-interval summary ================")
    for _, row in summary_df.iterrows():
        print(
            "Every {:>2d} steps | ValF1={:.2f}±{:.2f} | Acc={:.2f}±{:.2f} | "
            "Macro-F1={:.2f}±{:.2f} | Macro-P={:.2f}±{:.2f} | "
            "Macro-R={:.2f}±{:.2f} | time={:.1f}±{:.1f}s/run".format(
                int(row["Mask update interval"]),
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
                row["Elapsed (s/run) Mean"],
                row["Elapsed (s/run) Std"],
            )
        )

    ranked = summary_df.sort_values(
        by=["Best val Macro-F1 (%) Mean", "Best val Macro-F1 (%) Std"],
        ascending=[False, True],
    )
    best_interval = int(ranked.iloc[0]["Mask update interval"])
    print(
        "\nValidation-only first-ranked interval: {} global optimizer steps.".format(
            best_interval
        )
    )
    print("Do not select the interval from the test metrics.")

    print("\nSaved:")
    print("  {}".format(outdir / "mask_update_interval_runs.csv"))
    print("  {}".format(outdir / "mask_update_interval_summary.csv"))
    print("  {}".format(outdir / "mask_update_interval_validation_ranking.csv"))
    print("  {}".format(outdir / "mask_update_interval_table.tex"))


if __name__ == "__main__":
    main()
