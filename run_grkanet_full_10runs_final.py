# -*- coding: utf-8 -*-
"""
Final ten-run experiment for GR-KANet under a fixed stratified split.

Purpose:
    This script is used for the robustness table in the paper. It keeps the
    Hunan-Plant data split fixed and repeats model training with different
    random seeds. It does not generate figures, so it is faster than the main
    experiment script.

Protocol:
    - Data split: fixed stratified 64/16/20 split controlled by --split-seed.
    - StandardScaler: fitted only on the final training set.
    - Repeated seeds: control model initialization, sampler randomness, and CUDA randomness.
    - KAN-DGAM mask: initialized once and then updated every 5 GLOBAL optimizer steps.
      The counter is not reset at the beginning of each epoch.
    - Checkpoint selection: best validation Macro-F1.
    - Metrics: Acc, Macro-F1, Macro-P, Macro-R on the fixed test set.

Final GR-KANet configuration selected using validation-only screening:
    energy mask, alpha=2.0, DGAM EMA=0.8, mask update every 20 global steps,
    dropout=0.0, lr=1e-3, batch_size=64, weighted sampler + class-weighted CE,
    max_epochs=150, patience=80.

Python compatibility:
    This script avoids Python 3.10-only type syntax and should run on Python 3.8.
"""

from __future__ import annotations

import argparse
import os
import random
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


def format_mean_std(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        std = 0.0
    else:
        std = float(values.std(ddof=1))
    return "{:.2f} $\\pm$ {:.2f}".format(float(values.mean()), std)


def write_outputs(results: List[RunResult], outdir: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "Run": r.run_id,
            "Seed": r.seed,
            "Best epoch": r.best_epoch,
            "Best val Macro-F1 (%)": round(100.0 * r.best_val_macro_f1, 2),
            "Acc (%)": round(100.0 * r.acc, 2),
            "Macro-F1 (%)": round(100.0 * r.macro_f1, 2),
            "Macro-P (%)": round(100.0 * r.macro_p, 2),
            "Macro-R (%)": round(100.0 * r.macro_r, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "repeated_runs_results.csv", index=False, encoding="utf-8-sig")

    # Compute mean and sample standard deviation from the unrounded results.
    acc = np.asarray([100.0 * r.acc for r in results], dtype=float)
    f1 = np.asarray([100.0 * r.macro_f1 for r in results], dtype=float)
    p = np.asarray([100.0 * r.macro_p for r in results], dtype=float)
    rr = np.asarray([100.0 * r.macro_r for r in results], dtype=float)

    summary = pd.DataFrame([{
        "Model": "Full GR-KANet",
        "Acc (%)": format_mean_std(acc),
        "Macro-F1 (%)": format_mean_std(f1),
        "Macro-P (%)": format_mean_std(p),
        "Macro-R (%)": format_mean_std(rr),
    }])
    summary.to_csv(outdir / "repeated_runs_summary.csv", index=False, encoding="utf-8-sig")

    latex_row = (
        "Full model & {} & {} & {} & {} \\\\".format(
            format_mean_std(acc), format_mean_std(f1), format_mean_std(p), format_mean_std(rr)
        )
    )
    (outdir / "repeated_runs_latex_row.txt").write_text(latex_row, encoding="utf-8")

    table_lines = [
        r"\begin{table}[!htbp]",
        r"	\centering",
        r"	\scriptsize",
        r"	\caption{Performance of the proposed model over ten runs with different random seeds under a fixed stratified split of the Hunan-Plant dataset.}",
        r"	\label{tab:full-ten-runs}",
        r"	\resizebox{0.8\linewidth}{!}{%",
        r"		\begin{tabular}{@{}l|cccc@{}}",
        r"			\toprule",
        "\t\t\tModel & Acc (\\%) & Macro-F1 (\\%) & Macro-P (\\%) & Macro-R (\\%) \\\\",
        r"			\midrule",
        "			" + latex_row,
        r"			\bottomrule",
        r"		\end{tabular}%",
        r"	}",
        r"\end{table}",
    ]
    table = "\n".join(table_lines)
    (outdir / "repeated_runs_table.tex").write_text(table, encoding="utf-8")


def parse_seed_list(text: str, n_runs: int) -> List[int]:
    if text.strip():
        seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
        return seeds
    return [42 + i for i in range(n_runs)]      ###########################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final ten-seed GR-KANet experiment under a fixed split.")
    parser.add_argument("--data", type=str, default="胖虎电厂数据未加测试集.xlsx")
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument("--outdir", type=str, default="grkanet_full_10runs_outputs")

    parser.add_argument("--split-seed", type=int, default=42, help="Fixed seed for train/val/test split.")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated training seeds. Leave empty to use consecutive seeds starting from 42.")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--balance-mode", choices=["none", "ce_only", "sampler_only", "both"], default="both")

    parser.add_argument("--dgam-mode", choices=["energy", "entropy"], default="energy")
    parser.add_argument("--alpha-init", type=float, default=2.0)
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
    outdir = ensure_dir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("Using device: {}".format(device))
    print("Data file: {}".format(args.data))
    print("Fixed split seed: {}".format(args.split_seed))
    print(
        "Final ten-run config | mode={} alpha={} ema={} mask_every={} dropout={} lr={} batch_size={} epochs={} patience={} balance={}".format(
            args.dgam_mode, args.alpha_init, args.dgam_ema, args.mask_update_every,
            args.dropout, args.lr, args.batch_size, args.epochs, args.patience,
            args.balance_mode
        )
    )

    split = load_fixed_split(args.data, label_col=args.label_col, split_seed=args.split_seed)
    print("Classes: {}".format(split.display_labels))
    print("Split sizes: train={}, val={}, test={}".format(len(split.y_train), len(split.y_val), len(split.y_test)))

    seeds = parse_seed_list(args.seeds, args.n_runs)
    print("Training seeds: {}".format(seeds))

    results: List[RunResult] = []
    for run_id, seed in enumerate(seeds, start=1):
        print("\n===== Run {}/{} | seed={} =====".format(run_id, len(seeds), seed))
        test_res, best_epoch, best_val_macro_f1 = train_one_seed(split, seed, device, args)
        rr = RunResult(
            run_id=run_id,
            seed=seed,
            best_epoch=best_epoch,
            best_val_macro_f1=best_val_macro_f1,
            acc=test_res.acc,
            macro_f1=test_res.macro_f1,
            macro_p=test_res.macro_p,
            macro_r=test_res.macro_r,
        )
        results.append(rr)
        print(
            "Run {:02d} | seed={} | best_epoch={} | valF1={:.2f} | Acc={:.2f} | Macro-F1={:.2f} | Macro-P={:.2f} | Macro-R={:.2f}".format(
                run_id, seed, best_epoch, 100.0 * best_val_macro_f1,
                100.0 * test_res.acc, 100.0 * test_res.macro_f1,
                100.0 * test_res.macro_p, 100.0 * test_res.macro_r
            )
        )
        write_outputs(results, outdir)

    write_outputs(results, outdir)

    print("\n===== Summary over {} runs =====".format(len(results)))
    metric_values = {
        "Acc (%)": np.asarray([100.0 * r.acc for r in results], dtype=float),
        "Macro-F1 (%)": np.asarray([100.0 * r.macro_f1 for r in results], dtype=float),
        "Macro-P (%)": np.asarray([100.0 * r.macro_p for r in results], dtype=float),
        "Macro-R (%)": np.asarray([100.0 * r.macro_r for r in results], dtype=float),
    }
    for metric, vals in metric_values.items():
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print("{} = {:.2f} ± {:.2f}".format(metric, vals.mean(), std))

    print("\nSaved: {}".format(outdir / "repeated_runs_results.csv"))
    print("Saved: {}".format(outdir / "repeated_runs_summary.csv"))
    print("Saved: {}".format(outdir / "repeated_runs_latex_row.txt"))
    print("Saved: {}".format(outdir / "repeated_runs_table.tex"))


if __name__ == "__main__":
    main()
