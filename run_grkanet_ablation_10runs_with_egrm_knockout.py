# -*- coding: utf-8 -*-
"""
Final ten-run ablation experiments for GR-KANet on the Hunan-Plant DGA dataset.

Protocol:
    - Fixed stratified train/validation/test split = 64/16/20.
    - Split seed = 42.
    - StandardScaler is fitted only on the final training set.
    - Training seeds = 42-51.
    - Maximum epochs = 150; early-stopping patience = 80.
    - Checkpoint selection = validation Macro-F1.
    - KAN-DGAM mask update interval = 20 global optimizer steps.
    - Metrics = Acc, Macro-F1, Macro-P, and Macro-R.

Ablation definitions:
    1) Full GR-KANet:
       trained normally with EGRM enabled.

    2) w/o EGRM (inference knockout):
       the validation-selected Full checkpoint is kept completely frozen,
       and EGRM is disabled only during test inference. No retraining or
       fine-tuning is performed after EGRM is disabled.

    3) w/o KAN classifier:
       replaces the KAN classifier with a linear classifier and therefore
       also removes the KAN-derived attention mask source.

    4) w/o KAN-DGAM:
       removes KAN-DGAM while retaining the EGRM-enhanced feature stream.

Important:
    The row named "w/o EGRM" in this script is an inference-time knockout,
    not a separately retrained architecture. This must be stated explicitly
    in the manuscript and response letter.
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
except Exception:  # pragma: no cover
    GradScaler = None
    autocast = None


FEATURES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]
ID_TO_ABBR = {1: "T12", 2: "T3", 3: "PD", 4: "D1", 5: "D2", 6: "NC"}


# Existing ten-run result of the full GR-KANet.
# The full model is NOT retrained by this script.
# Update these values here if the retained full-model result changes.
FULL_MODEL_SUMMARY = {
    "Variant": "Full GR-KANet",
    "Runs": 10,
    "Acc (%) Mean": 93.65,
    "Acc (%) Std": 2.73,
    "Macro-F1 (%) Mean": 94.24,
    "Macro-F1 (%) Std": 2.55,
    "Macro-P (%) Mean": 94.34,
    "Macro-P (%) Std": 2.55,
    "Macro-R (%) Mean": 94.48,
    "Macro-R (%) Std": 2.61,
    "Source": "Existing full-model ten-run experiment (mask interval = 20)",
}


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
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


class DGADataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_and_split_data(
    data_path: str | Path,
    label_col: str = "故障编码",
    seed: int = 42,
    scaler_mode: str = "train",
) -> SplitData:
    df = pd.read_excel(data_path)
    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
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

    if scaler_mode == "train":
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=seed
        )
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    elif scaler_mode == "trainval":
        scaler = StandardScaler().fit(X_trainval)
        X_trainval = scaler.transform(X_trainval).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=seed
        )
    else:
        raise ValueError("scaler_mode must be 'train' or 'trainval'")

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, display_labels, scaler)


def make_loaders(
    split: SplitData,
    batch_size: int = 64,
    balance_mode: str = "both",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DGADataset(split.X_train, split.y_train)
    val_ds = DGADataset(split.X_val, split.y_val)
    test_ds = DGADataset(split.X_test, split.y_test)

    if balance_mode in {"sampler_only", "both"}:
        classes, counts = np.unique(split.y_train, return_counts=True)
        inv_count = {cls: 1.0 / cnt for cls, cnt in zip(classes, counts)}
        sample_weights = np.array([inv_count[y] for y in split.y_train], dtype=np.float32)
        sample_weights = torch.from_numpy(sample_weights)
        # Keep the same sampler behavior as the tuning/main script.
        # The global torch seed is reset before each variant.
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-8)
    inv = inv * (len(classes) / inv.sum())
    weights = np.ones(num_classes, dtype=np.float32)
    weights[classes] = inv
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# Model modules
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
        identity = x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + identity)


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.functions = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(in_features)]
        )
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
            raise ValueError("mode must be 'energy' or 'entropy'")
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
            raise ValueError(f"KAN input dim {d} must equal channels*num_gases = {C*G}")

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
        gate = self.gate(x_k)
        return x_k * gate * self.kan_mask.view(1, -1, 1)


class FixedGateAttention(nn.Module):
    """Sample-wise gate without KAN derivative mask."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Sigmoid())

    def update_mask_from_kan(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class AblationGRKANet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dgam_mode: str = "energy",
        alpha_init: float = 2.0,
        dropout: float = 0.0,
        dgam_ema: float = 0.8,
        channels: int = 32,
        num_gases: int = 8,
        use_egrm: bool = True,
        use_dgam: bool = True,
        use_kan_derivative: bool = True,
        use_kan_classifier: bool = True,
        use_residual2: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_gases = num_gases
        self.use_egrm = use_egrm
        self.use_dgam = use_dgam
        self.use_kan_derivative = use_kan_derivative
        self.use_kan_classifier = use_kan_classifier
        self.use_residual2 = use_residual2

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual1 = ResidualBlock1D(channels)
        self.residual2 = ResidualBlock1D(channels) if use_residual2 else nn.Identity()

        K0 = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [1, 1, 1, 0],
                [0, 1, 1, 0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("egrm_init", K0)
        self.egrm_delta = nn.Parameter(torch.zeros_like(K0))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.kan = KANLayer(channels * num_gases, 64, hidden=8) if use_kan_classifier else None

        if use_dgam:
            if use_kan_derivative and use_kan_classifier:
                assert self.kan is not None
                self.dgam = KANDGAM(channels, num_gases, self.kan, mode=dgam_mode, ema=dgam_ema)
            else:
                self.dgam = FixedGateAttention(channels)
        else:
            self.dgam = None

        self.dropout = nn.Dropout(dropout)
        if use_kan_classifier:
            self.classifier = nn.Linear(64, num_classes)
        else:
            # Remove the KAN nonlinear decision layer and use a plain linear classifier.
            # This isolates the contribution of the KAN classifier rather than adding another MLP.
            self.classifier = nn.Linear(channels * num_gases, num_classes)

    @property
    def egrm(self) -> torch.Tensor:
        return self.egrm_init + self.egrm_delta

    def egrm_channel_prior(self) -> torch.Tensor:
        return self.egrm.reshape(1, self.channels, 1)

    def update_attention_mask(self) -> None:
        if self.dgam is not None and hasattr(self.dgam, "update_mask_from_kan"):
            self.dgam.update_mask_from_kan()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x.unsqueeze(1))
        x1 = self.residual1(x0)

        if self.use_egrm:
            x_k = x1 + self.alpha * self.egrm_channel_prior().to(x1.device)
        else:
            x_k = x1

        if self.use_dgam:
            assert self.dgam is not None
            x_a = self.dgam(x_k)
        else:
            # Remove KAN-DGAM reweighting while keeping the EGRM-enhanced feature stream.
            x_a = x_k

        x_in2 = x1 + x_a
        x2 = self.residual2(x_in2)
        flat = x2.flatten(1)

        if self.use_kan_classifier:
            assert self.kan is not None
            z = self.kan(flat)
            z = self.dropout(z)
            logits = self.classifier(z)
        else:
            logits = self.classifier(flat)
        return logits




# -----------------------------
# Train, summarize, and export
# -----------------------------

from math import sqrt

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


@dataclass
class Metrics:
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float


@dataclass
class RunResult:
    variant: str
    run_id: int
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float
    protocol: str


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Metrics:
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    amp_enabled = device.type == "cuda" and autocast is not None

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if amp_enabled:
                with autocast(enabled=True):
                    logits = model(xb)
            else:
                logits = model(xb)

            pred = logits.argmax(dim=1)
            ys.extend(yb.detach().cpu().numpy().tolist())
            ps.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(ys)
    y_pred = np.asarray(ps)

    return Metrics(
        acc=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_p=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_r=recall_score(y_true, y_pred, average="macro", zero_division=0),
    )


def build_criterion(
    split: SplitData,
    num_classes: int,
    balance_mode: str,
    device: torch.device,
) -> nn.Module:
    if balance_mode in {"ce_only", "both"}:
        class_weights = compute_class_weights(split.y_train, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()


def train_model(
    variant_name: str,
    model: AblationGRKANet,
    split: SplitData,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    balance_mode: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    mask_update_every: int,
    verbose: bool = False,
) -> Tuple[AblationGRKANet, int, float]:
    model = model.to(device)
    criterion = build_criterion(split, num_classes, balance_mode, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = GradScaler(enabled=(device.type == "cuda")) if GradScaler is not None else None
    amp_enabled = device.type == "cuda" and autocast is not None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0

    model.update_attention_mask()
    global_step = 0

    for epoch in range(1, epochs + 1):
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
                    raise RuntimeError("AMP is enabled but GradScaler is unavailable.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            global_step += 1
            if mask_update_every > 0 and global_step % mask_update_every == 0:
                model.update_attention_mask()

        val_metrics = evaluate(model, val_loader, device)

        if val_metrics.macro_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_metrics.macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if verbose:
            print(
                f"    {variant_name} | epoch={epoch:03d} | "
                f"val Macro-F1={val_metrics.macro_f1 * 100:.2f}"
            )

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was obtained for {variant_name}.")

    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch, best_val_f1


def parse_seed_list(text: str, n_runs: int) -> List[int]:
    if text.strip():
        seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integer seed was found.")
        return seeds
    return [42 + i for i in range(n_runs)]


def sample_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean_value = float(arr.mean())
    if len(arr) <= 1:
        return mean_value, mean_value, mean_value

    sem = float(arr.std(ddof=1) / sqrt(len(arr)))
    if scipy_stats is not None:
        critical = float(scipy_stats.t.ppf(0.975, df=len(arr) - 1))
    else:
        critical = 2.262

    half_width = critical * sem
    return mean_value, mean_value - half_width, mean_value + half_width


def summarize_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variant, group in run_df.groupby("Variant", sort=False):
        row = {
            "Variant": variant,
            "Runs": int(len(group)),
            "Protocol": str(group["Protocol"].iloc[0]),
        }

        for metric in ["Acc (%)", "Macro-F1 (%)", "Macro-P (%)", "Macro-R (%)"]:
            mean_value, ci_low, ci_high = mean_ci95(group[metric].to_numpy(dtype=float))
            row[f"{metric} Mean"] = mean_value
            row[f"{metric} Std"] = sample_std(group[metric].to_numpy(dtype=float))
            row[f"{metric} 95% CI Low"] = ci_low
            row[f"{metric} 95% CI High"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.2f} \\pm {std_value:.2f}"


def latex_table(summary_df: pd.DataFrame) -> str:
    metric_pairs = [
        ("Acc (%) Mean", "Acc (%) Std"),
        ("Macro-F1 (%) Mean", "Macro-F1 (%) Std"),
        ("Macro-P (%) Mean", "Macro-P (%) Std"),
        ("Macro-R (%) Mean", "Macro-R (%) Std"),
    ]

    ranks = {}
    for mean_col, _ in metric_pairs:
        unique_means = sorted(
            {round(float(v), 12) for v in summary_df[mean_col].values},
            reverse=True,
        )
        ranks[mean_col] = (
            unique_means[0],
            unique_means[1] if len(unique_means) > 1 else None,
        )

    lines = [
        r"\begin{table}[!htbp]",
        r"    \centering",
        r"    \scriptsize",
        (
            r"    \caption{Ablation results of GR-KANet on the Hunan-Plant dataset. "
            r"Results are reported as mean $\pm$ standard deviation over ten runs. "
            r"The w/o EGRM row is obtained by disabling EGRM at inference in the "
            r"same frozen validation-selected Full checkpoint, without retraining.}"
        ),
        r"    \label{tab:ablation-results}",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tabular}{@{}l|cccc@{}}",
        r"            \toprule",
        r"            Variant & Acc (\%) & Macro-F1 (\%) & Macro-P (\%) & Macro-R (\%) \\",
        r"            \midrule",
    ]

    for _, row in summary_df.iterrows():
        cells = []
        for mean_col, std_col in metric_pairs:
            mean_value = float(row[mean_col])
            std_value = float(row[std_col])
            cell = format_mean_std(mean_value, std_value)

            best, second = ranks[mean_col]
            if abs(mean_value - best) < 1e-9:
                cell = f"\\mathbf{{{cell}}}"
            elif second is not None and abs(mean_value - second) < 1e-9:
                cell = f"\\underline{{{cell}}}"

            cells.append(f"${cell}$")

        lines.append(
            "            {} & {} & {} & {} & {} \\\\".format(
                row["Variant"], *cells
            )
        )

    lines.extend(
        [
            r"            \bottomrule",
            r"        \end{tabular}%",
            r"    }",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def paired_knockout_statistics(run_df: pd.DataFrame) -> pd.DataFrame:
    full = run_df[run_df["Variant"] == "Full GR-KANet"].sort_values("Seed")
    ko = run_df[run_df["Variant"] == "w/o EGRM"].sort_values("Seed")

    if full["Seed"].tolist() != ko["Seed"].tolist():
        raise RuntimeError("Full and w/o EGRM seeds do not match for paired analysis.")

    rows = []
    for metric in ["Acc (%)", "Macro-F1 (%)", "Macro-P (%)", "Macro-R (%)"]:
        diffs = full[metric].to_numpy(dtype=float) - ko[metric].to_numpy(dtype=float)
        mean_diff, ci_low, ci_high = mean_ci95(diffs)

        t_p = np.nan
        wilcoxon_p = np.nan
        if scipy_stats is not None and len(diffs) > 1:
            try:
                t_p = float(scipy_stats.ttest_1samp(diffs, popmean=0.0).pvalue)
            except Exception:
                pass
            try:
                if not np.allclose(diffs, 0.0):
                    wilcoxon_p = float(
                        scipy_stats.wilcoxon(
                            diffs,
                            zero_method="wilcox",
                            alternative="two-sided",
                        ).pvalue
                    )
            except Exception:
                pass

        rows.append(
            {
                "Metric": metric,
                "Runs": len(diffs),
                "Mean paired drop (Full - w/o EGRM, pp)": mean_diff,
                "Drop Std (pp)": sample_std(diffs),
                "95% CI Low (pp)": ci_low,
                "95% CI High (pp)": ci_high,
                "Full better runs": int(np.sum(diffs > 1e-12)),
                "w/o EGRM better runs": int(np.sum(diffs < -1e-12)),
                "Ties": int(np.sum(np.abs(diffs) <= 1e-12)),
                "Paired t-test p": t_p,
                "Wilcoxon p": wilcoxon_p,
            }
        )

    return pd.DataFrame(rows)


def write_outputs(results: List[RunResult], outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_df = pd.DataFrame(
        [
            {
                "Variant": r.variant,
                "Run": r.run_id,
                "Seed": r.seed,
                "Best epoch": r.best_epoch,
                "Best val Macro-F1 (%)": r.best_val_macro_f1 * 100.0,
                "Acc (%)": r.acc * 100.0,
                "Macro-F1 (%)": r.macro_f1 * 100.0,
                "Macro-P (%)": r.macro_p * 100.0,
                "Macro-R (%)": r.macro_r * 100.0,
                "Protocol": r.protocol,
            }
            for r in results
        ]
    )

    run_df.to_csv(
        outdir / "ablation_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    summary_df = summarize_runs(run_df)
    summary_df.to_csv(
        outdir / "ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    paired_df = paired_knockout_statistics(run_df)
    paired_df.to_csv(
        outdir / "egrm_knockout_paired_statistics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    table_text = latex_table(summary_df)
    (outdir / "ablation_table_10runs.tex").write_text(table_text, encoding="utf-8")
    (outdir / "ablation_results_for_latex.txt").write_text(table_text, encoding="utf-8")

    return run_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Full GR-KANet, inference-time EGRM knockout, "
            "w/o KAN classifier, and w/o KAN-DGAM over ten seeds."
        )
    )
    parser.add_argument("--data", type=str, default="胖虎电厂数据未加测试集.xlsx")
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument(
        "--outdir",
        type=str,
        default="grkanet_ablation_10runs_with_egrm_knockout_outputs",
    )

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated training seeds. Empty means 42-51.",
    )

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scaler-mode", choices=["train", "trainval"], default="train")
    parser.add_argument(
        "--balance-mode",
        choices=["none", "ce_only", "sampler_only", "both"],
        default="both",
    )
    parser.add_argument("--mask-update-every", type=int, default=20)
    parser.add_argument("--dgam-ema", type=float, default=0.8)
    parser.add_argument("--alpha-init", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    outdir = ensure_dir(args.outdir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    split = load_and_split_data(
        args.data,
        label_col=args.label_col,
        seed=args.split_seed,
        scaler_mode=args.scaler_mode,
    )

    _, val_loader, test_loader = make_loaders(
        split,
        batch_size=args.batch_size,
        balance_mode="none",
        seed=args.split_seed,
    )

    num_classes = len(split.display_labels)
    training_seeds = parse_seed_list(args.seeds, args.n_runs)

    print(f"Using device: {device}")
    print(f"Classes: {split.display_labels}")
    print(
        f"Fixed split sizes: train={len(split.X_train)}, "
        f"val={len(split.X_val)}, test={len(split.X_test)}"
    )
    print(f"Split seed: {args.split_seed}")
    print(f"Training seeds: {training_seeds}")
    print(
        "Config | "
        f"epochs={args.epochs}, patience={args.patience}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, batch_size={args.batch_size}, "
        f"balance={args.balance_mode}, scaler={args.scaler_mode}, "
        f"dgam_ema={args.dgam_ema}, alpha={args.alpha_init}, "
        f"dropout={args.dropout}, mask_update_every={args.mask_update_every}"
    )

    results: List[RunResult] = []

    # 1) Train Full once per seed, then evaluate the same checkpoint with EGRM ON/OFF.
    print("\n========== Full GR-KANet + w/o EGRM inference knockout ==========")
    for run_id, seed in enumerate(training_seeds, start=1):
        set_seed(seed)
        train_loader, _, _ = make_loaders(
            split,
            batch_size=args.batch_size,
            balance_mode=args.balance_mode,
            seed=seed,
        )

        full_model = AblationGRKANet(
            num_classes=num_classes,
            dgam_mode="energy",
            alpha_init=args.alpha_init,
            dropout=args.dropout,
            dgam_ema=args.dgam_ema,
            channels=32,
            num_gases=8,
            use_egrm=True,
            use_dgam=True,
            use_kan_derivative=True,
            use_kan_classifier=True,
            use_residual2=True,
        )

        full_model, best_epoch, best_val_f1 = train_model(
            variant_name="Full GR-KANet",
            model=full_model,
            split=split,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            balance_mode=args.balance_mode,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mask_update_every=args.mask_update_every,
            verbose=args.verbose,
        )

        full_model.use_egrm = True
        full_metrics = evaluate(full_model, test_loader, device)

        full_model.use_egrm = False
        knockout_metrics = evaluate(full_model, test_loader, device)

        full_model.use_egrm = True

        results.extend(
            [
                RunResult(
                    variant="Full GR-KANet",
                    run_id=run_id,
                    seed=seed,
                    best_epoch=best_epoch,
                    best_val_macro_f1=best_val_f1,
                    acc=full_metrics.acc,
                    macro_f1=full_metrics.macro_f1,
                    macro_p=full_metrics.macro_p,
                    macro_r=full_metrics.macro_r,
                    protocol="Normally trained Full model",
                ),
                RunResult(
                    variant="w/o EGRM",
                    run_id=run_id,
                    seed=seed,
                    best_epoch=best_epoch,
                    best_val_macro_f1=best_val_f1,
                    acc=knockout_metrics.acc,
                    macro_f1=knockout_metrics.macro_f1,
                    macro_p=knockout_metrics.macro_p,
                    macro_r=knockout_metrics.macro_r,
                    protocol=(
                        "Inference-time EGRM knockout in the same frozen "
                        "validation-selected Full checkpoint; no retraining"
                    ),
                ),
            ]
        )

        print(
            f"Run {run_id:02d}/{len(training_seeds):02d} | seed={seed} | "
            f"best_epoch={best_epoch:03d} | ValF1={best_val_f1 * 100:.2f}\n"
            f"  Full       | Acc={full_metrics.acc * 100:.2f} | "
            f"F1={full_metrics.macro_f1 * 100:.2f} | "
            f"P={full_metrics.macro_p * 100:.2f} | "
            f"R={full_metrics.macro_r * 100:.2f}\n"
            f"  w/o EGRM   | Acc={knockout_metrics.acc * 100:.2f} | "
            f"F1={knockout_metrics.macro_f1 * 100:.2f} | "
            f"P={knockout_metrics.macro_p * 100:.2f} | "
            f"R={knockout_metrics.macro_r * 100:.2f} | "
            f"F1 drop={(full_metrics.macro_f1 - knockout_metrics.macro_f1) * 100:+.2f} pp"
        )

        write_outputs(results, outdir)
        del full_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2) Retrained architectural ablations.
    retrained_variants = [
        (
            "w/o KAN classifier",
            dict(
                dgam_mode="energy",
                use_egrm=True,
                use_dgam=True,
                use_kan_derivative=False,
                use_kan_classifier=False,
                use_residual2=True,
            ),
        ),
        (
            "w/o KAN-DGAM",
            dict(
                dgam_mode="energy",
                use_egrm=True,
                use_dgam=False,
                use_kan_derivative=False,
                use_kan_classifier=True,
                use_residual2=True,
            ),
        ),
    ]

    for variant_name, cfg in retrained_variants:
        print(f"\n========== {variant_name} ==========")

        for run_id, seed in enumerate(training_seeds, start=1):
            set_seed(seed)
            train_loader, _, _ = make_loaders(
                split,
                batch_size=args.batch_size,
                balance_mode=args.balance_mode,
                seed=seed,
            )

            model = AblationGRKANet(
                num_classes=num_classes,
                alpha_init=args.alpha_init,
                dropout=args.dropout,
                dgam_ema=args.dgam_ema,
                **cfg,
            )

            model, best_epoch, best_val_f1 = train_model(
                variant_name=variant_name,
                model=model,
                split=split,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_classes=num_classes,
                balance_mode=args.balance_mode,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                mask_update_every=args.mask_update_every,
                verbose=args.verbose,
            )

            metrics = evaluate(model, test_loader, device)

            results.append(
                RunResult(
                    variant=variant_name,
                    run_id=run_id,
                    seed=seed,
                    best_epoch=best_epoch,
                    best_val_macro_f1=best_val_f1,
                    acc=metrics.acc,
                    macro_f1=metrics.macro_f1,
                    macro_p=metrics.macro_p,
                    macro_r=metrics.macro_r,
                    protocol="Retrained architectural ablation",
                )
            )

            print(
                f"Run {run_id:02d}/{len(training_seeds):02d} | seed={seed} | "
                f"best_epoch={best_epoch:03d} | "
                f"Acc={metrics.acc * 100:.2f} | "
                f"Macro-F1={metrics.macro_f1 * 100:.2f} | "
                f"Macro-P={metrics.macro_p * 100:.2f} | "
                f"Macro-R={metrics.macro_r * 100:.2f}"
            )

            write_outputs(results, outdir)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    run_df, summary_df = write_outputs(results, outdir)

    print("\n===== Final ten-run ablation summary =====")
    for _, row in summary_df.iterrows():
        print(
            f"{row['Variant']:24s} | "
            f"Acc={row['Acc (%) Mean']:.2f}±{row['Acc (%) Std']:.2f} | "
            f"Macro-F1={row['Macro-F1 (%) Mean']:.2f}±{row['Macro-F1 (%) Std']:.2f} | "
            f"Macro-P={row['Macro-P (%) Mean']:.2f}±{row['Macro-P (%) Std']:.2f} | "
            f"Macro-R={row['Macro-R (%) Mean']:.2f}±{row['Macro-R (%) Std']:.2f}"
        )

    print("\nSaved:")
    print(f"  {outdir / 'ablation_runs.csv'}")
    print(f"  {outdir / 'ablation_summary.csv'}")
    print(f"  {outdir / 'egrm_knockout_paired_statistics.csv'}")
    print(f"  {outdir / 'ablation_table_10runs.tex'}")
    print(f"  {outdir / 'ablation_results_for_latex.txt'}")


if __name__ == "__main__":
    main()
