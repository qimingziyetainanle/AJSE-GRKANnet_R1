# -*- coding: utf-8 -*-
"""
Validation-first search over multiple theoretically constrained binary K0 matrices
for the FULL GR-KANet model.

Purpose
-------
Search a diverse library of binary 8x4 EGRM initial matrices while keeping the
published GR-KANet architecture and all non-K0 settings unchanged.

Protocol
--------
1. Fixed 64/16/20 stratified split, split seed = 42.
2. Training seeds = 42, ..., 51 by default.
3. Search-stage ranking uses validation Macro-F1 ONLY.
4. Test metrics are not calculated for all candidate matrices.
5. After all candidates are ranked, only the validation-first candidate is
   retrained with the same ten seeds and evaluated on the test set.
6. The script is resumable: completed candidate/seed runs are skipped.

Default workload
----------------
- 36 K0 candidates x 10 seeds = 360 validation-search runs.
- Selected K0 x 10 seeds = 10 final test runs.
- Total = 370 runs.

The candidate library preserves the manuscript's four coarse pattern groups:
P1 light-gas, P2 broad hydrocarbon, P3 C2-hydrocarbon, P4 carbon-oxide.
H2, CO, and CO2 are kept as anchors. Only ambiguous hydrocarbon memberships
are varied using a predefined set of physically interpretable binary choices.

Important
---------
Changing K0 does not affect a strict w/o-EGRM model implemented with
use_egrm=False. It does affect the Full model and any other variant that still
retains the EGRM branch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


FEATURES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]
PATTERNS = ["P1_light_gas", "P2_broad_hydrocarbon", "P3_C2_hydrocarbon", "P4_carbon_oxide"]
ID_TO_ABBR = {1: "T12", 2: "T3", 3: "PD", 4: "D1", 5: "D2", 6: "NC"}

ORIGINAL_K0 = np.asarray(
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
    display_labels: List[str]
    scaler: StandardScaler


@dataclass
class EvalMetrics:
    acc: float
    macro_f1: float
    macro_p: float
    macro_r: float


@dataclass
class Candidate:
    candidate_id: str
    matrix: np.ndarray
    ch4_choice: str
    c2h4_choice: str
    c2h2_choice: str
    thc_choice: str
    c2h6_choice: str
    hamming_from_original: int
    ones: int


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


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_seed_list(text: str, n_runs: int) -> List[int]:
    if text.strip():
        values = [int(v.strip()) for v in text.split(",") if v.strip()]
        if not values:
            raise ValueError("--seeds does not contain valid integers.")
        return values
    return [42 + i for i in range(n_runs)]


def load_fixed_split(
    data_path: str,
    label_col: str,
    split_seed: int,
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

    raw_ids = le.inverse_transform(np.arange(len(le.classes_)))
    display_labels: List[str] = []
    for cid in raw_ids:
        try:
            display_labels.append(ID_TO_ABBR[int(cid)])
        except Exception:
            display_labels.append(str(cid))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=split_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
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
        display_labels=display_labels,
        scaler=scaler,
    )


def make_loaders(
    split: SplitData,
    batch_size: int,
    balance_mode: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DGADataset(split.X_train, split.y_train)
    val_ds = DGADataset(split.X_val, split.y_val)
    test_ds = DGADataset(split.X_test, split.y_test)

    if balance_mode in {"sampler_only", "both"}:
        classes, counts = np.unique(split.y_train, return_counts=True)
        inv = {int(cls): 1.0 / float(cnt) for cls, cnt in zip(classes, counts)}
        weights = np.asarray([inv[int(y)] for y in split.y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-8)
    inv = inv * (len(classes) / inv.sum())
    values = np.ones(num_classes, dtype=np.float32)
    values[classes] = inv
    return torch.tensor(values, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Candidate K0 library
# -----------------------------------------------------------------------------

CHOICES: Dict[str, Dict[str, np.ndarray]] = {
    "CH4": {
        "P1P2": np.asarray([1, 1, 0, 0], dtype=np.float32),
        "P2": np.asarray([0, 1, 0, 0], dtype=np.float32),
    },
    "C2H4": {
        "P2P3": np.asarray([0, 1, 1, 0], dtype=np.float32),
        "P3": np.asarray([0, 0, 1, 0], dtype=np.float32),
    },
    "C2H2": {
        "P2P3": np.asarray([0, 1, 1, 0], dtype=np.float32),
        "P1P3": np.asarray([1, 0, 1, 0], dtype=np.float32),
        "P3": np.asarray([0, 0, 1, 0], dtype=np.float32),
        "P1P2P3": np.asarray([1, 1, 1, 0], dtype=np.float32),
    },
    "THC": {
        "P1P2P3": np.asarray([1, 1, 1, 0], dtype=np.float32),
        "P2P3": np.asarray([0, 1, 1, 0], dtype=np.float32),
        "P2": np.asarray([0, 1, 0, 0], dtype=np.float32),
    },
    "C2H6": {
        "P2P3": np.asarray([0, 1, 1, 0], dtype=np.float32),
        "P2": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "P3": np.asarray([0, 0, 1, 0], dtype=np.float32),
    },
}


def _matrix_key(matrix: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in matrix.reshape(-1).tolist())


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(int) != b.astype(int)))


def build_all_theory_candidates() -> List[Candidate]:
    all_candidates: List[Candidate] = []
    seen = set()

    for ch4_name, c2h4_name, c2h2_name, thc_name, c2h6_name in itertools.product(
        CHOICES["CH4"].keys(),
        CHOICES["C2H4"].keys(),
        CHOICES["C2H2"].keys(),
        CHOICES["THC"].keys(),
        CHOICES["C2H6"].keys(),
    ):
        matrix = np.stack(
            [
                np.asarray([1, 0, 0, 0], dtype=np.float32),  # H2 anchor
                CHOICES["CH4"][ch4_name],
                CHOICES["C2H4"][c2h4_name],
                CHOICES["C2H2"][c2h2_name],
                np.asarray([0, 0, 0, 1], dtype=np.float32),  # CO anchor
                np.asarray([0, 0, 0, 1], dtype=np.float32),  # CO2 anchor
                CHOICES["THC"][thc_name],
                CHOICES["C2H6"][c2h6_name],
            ],
            axis=0,
        ).astype(np.float32)

        # Every pattern must be represented at least once.
        if np.any(matrix.sum(axis=0) == 0):
            continue

        key = _matrix_key(matrix)
        if key in seen:
            continue
        seen.add(key)

        all_candidates.append(
            Candidate(
                candidate_id="",
                matrix=matrix,
                ch4_choice=ch4_name,
                c2h4_choice=c2h4_name,
                c2h2_choice=c2h2_name,
                thc_choice=thc_name,
                c2h6_choice=c2h6_name,
                hamming_from_original=_hamming(matrix, ORIGINAL_K0),
                ones=int(matrix.sum()),
            )
        )

    return all_candidates


def select_diverse_candidates(all_candidates: List[Candidate], n_candidates: int) -> List[Candidate]:
    if n_candidates <= 0:
        raise ValueError("--num-candidates must be positive.")
    if n_candidates > len(all_candidates):
        raise ValueError(
            f"Requested {n_candidates} candidates, but only {len(all_candidates)} valid candidates exist."
        )

    original_idx = None
    for idx, candidate in enumerate(all_candidates):
        if np.array_equal(candidate.matrix, ORIGINAL_K0):
            original_idx = idx
            break
    if original_idx is None:
        raise RuntimeError("Original K0 was not generated by the candidate library.")

    selected = [all_candidates[original_idx]]
    remaining = [c for i, c in enumerate(all_candidates) if i != original_idx]

    # Deterministic farthest-point selection gives broad coverage without random search.
    while len(selected) < n_candidates:
        scored = []
        for candidate in remaining:
            min_distance = min(_hamming(candidate.matrix, s.matrix) for s in selected)
            mean_distance = float(
                np.mean([_hamming(candidate.matrix, s.matrix) for s in selected])
            )
            scored.append(
                (
                    min_distance,
                    mean_distance,
                    -abs(candidate.ones - int(ORIGINAL_K0.sum())),
                    -candidate.hamming_from_original,
                    _matrix_key(candidate.matrix),
                    candidate,
                )
            )
        scored.sort(key=lambda item: item[:-1], reverse=True)
        chosen = scored[0][-1]
        selected.append(chosen)
        remaining = [item for item in remaining if item is not chosen]

    for idx, candidate in enumerate(selected):
        prefix = "K00_original" if idx == 0 else f"K{idx:02d}"
        candidate.candidate_id = prefix
    return selected


def save_candidate_library(candidates: Sequence[Candidate], outdir: Path) -> None:
    metadata_rows = []
    long_rows = []
    text_blocks = []

    for c in candidates:
        metadata_rows.append(
            {
                "Candidate": c.candidate_id,
                "CH4": c.ch4_choice,
                "C2H4": c.c2h4_choice,
                "C2H2": c.c2h2_choice,
                "THC": c.thc_choice,
                "C2H6": c.c2h6_choice,
                "Ones": c.ones,
                "Hamming from original": c.hamming_from_original,
            }
        )
        text_blocks.append(f"{c.candidate_id}\n{c.matrix.astype(int)}\n")
        for i, gas in enumerate(FEATURES):
            for j, pattern in enumerate(PATTERNS):
                long_rows.append(
                    {
                        "Candidate": c.candidate_id,
                        "Gas": gas,
                        "Pattern": pattern,
                        "Value": int(c.matrix[i, j]),
                    }
                )

        pd.DataFrame(c.matrix.astype(int), index=FEATURES, columns=PATTERNS).to_csv(
            outdir / f"{c.candidate_id}_K0.csv",
            encoding="utf-8-sig",
        )

    pd.DataFrame(metadata_rows).to_csv(
        outdir / "k0_candidate_metadata.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(long_rows).to_csv(
        outdir / "k0_candidate_matrices_long.csv", index=False, encoding="utf-8-sig"
    )
    (outdir / "k0_candidate_matrices.txt").write_text(
        "\n".join(text_blocks), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# GR-KANet
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
        self.functions = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                for _ in range(in_features)
            ]
        )
        self.linear = nn.Linear(in_features, out_features)

    def transformed_components(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for j, f_j in enumerate(self.functions):
            x_j = x[:, j : j + 1]
            parts.append(f_j(x_j) + x_j)
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.transformed_components(x))


class KANDGAM(nn.Module):
    def __init__(
        self,
        channels: int,
        num_gases: int,
        kan_layer: KANLayer,
        mode: str,
        ema: float,
        reference_points: int = 200,
        entropy_bins: int = 30,
    ):
        super().__init__()
        if mode not in {"energy", "entropy"}:
            raise ValueError("mode must be 'energy' or 'entropy'.")
        self.channels = channels
        self.num_gases = num_gases
        self.kan_layer = kan_layer
        self.mode = mode
        self.ema = ema
        self.reference_points = reference_points
        self.entropy_bins = entropy_bins
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Sigmoid())
        self.register_buffer("kan_mask", torch.ones(channels))

    @torch.no_grad()
    def _compute_channel_scores(self) -> torch.Tensor:
        device = self.kan_mask.device
        grid = torch.linspace(-3.0, 3.0, self.reference_points, device=device).unsqueeze(1)
        derivs = []
        for f_j in self.kan_layer.functions:
            y = f_j(grid).squeeze(1)
            dy = torch.gradient(y)[0]
            derivs.append(dy.detach().float().cpu().numpy())
        derivs = np.asarray(derivs, dtype=np.float32).reshape(
            self.channels, self.num_gases, self.reference_points
        )

        if self.mode == "energy":
            scores = np.mean(np.square(derivs), axis=(1, 2)).astype(np.float32)
        else:
            values = []
            for c in range(self.channels):
                curve = np.abs(derivs[c].reshape(-1))
                hist, _ = np.histogram(curve, bins=self.entropy_bins, density=False)
                p = hist.astype(np.float64)
                p = p / (p.sum() + 1e-12)
                values.append(float(-np.sum(p * np.log(p + 1e-12))))
            scores = np.asarray(values, dtype=np.float32)

        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return torch.tensor(norm, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_mask_from_kan(self) -> None:
        new_mask = self._compute_channel_scores()
        self.kan_mask.mul_(self.ema).add_(new_mask * (1.0 - self.ema))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        return x * gate * self.kan_mask.view(1, -1, 1)


class GRKANet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        egrm_init: np.ndarray,
        channels: int,
        num_gases: int,
        kan_hidden: int,
        kan_latent: int,
        dgam_mode: str,
        dgam_ema: float,
        alpha_init: float,
        dropout: float,
    ):
        super().__init__()
        if egrm_init.shape != (8, 4):
            raise ValueError(f"K0 must have shape (8, 4), got {egrm_init.shape}.")
        if channels != 32:
            raise ValueError("This published K0 flattening implementation requires channels=32.")

        self.channels = channels
        self.num_gases = num_gases
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual1 = ResidualBlock1D(channels)
        self.residual2 = ResidualBlock1D(channels)
        self.kan = KANLayer(channels * num_gases, kan_latent, hidden=kan_hidden)
        self.dgam = KANDGAM(channels, num_gases, self.kan, dgam_mode, dgam_ema)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x.unsqueeze(1))
        x1 = self.residual1(x0)
        xk = x1 + self.alpha * self.egrm_channel_prior().to(x1.device)
        xa = self.dgam(xk)
        x2 = self.residual2(x1 + xa)
        z = self.kan(x2.flatten(1))
        z = self.dropout(z)
        return self.classifier(z)


# -----------------------------------------------------------------------------
# Training and statistics
# -----------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalMetrics:
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
    return EvalMetrics(
        acc=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_p=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_r=recall_score(y_true, y_pred, average="macro", zero_division=0),
    )


def train_one(
    split: SplitData,
    matrix: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    evaluate_test: bool,
) -> Dict[str, float]:
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        split, args.batch_size, args.balance_mode
    )
    num_classes = len(split.display_labels)

    model = GRKANet(
        num_classes=num_classes,
        egrm_init=matrix,
        channels=args.channels,
        num_gases=8,
        kan_hidden=args.kan_hidden,
        kan_latent=args.kan_latent,
        dgam_mode=args.dgam_mode,
        dgam_ema=args.dgam_ema,
        alpha_init=args.alpha_init,
        dropout=args.dropout,
    ).to(device)

    if args.balance_mode in {"ce_only", "both"}:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights(split.y_train, num_classes).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=(device.type == "cuda")) if GradScaler is not None else None
    amp_enabled = device.type == "cuda" and autocast is not None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_f1 = -1.0
    no_improve = 0
    global_step = 0
    model.dgam.update_mask_from_kan()

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
                    raise RuntimeError("AMP enabled but GradScaler unavailable.")
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

        val_metrics = evaluate(model, val_loader, criterion, device)
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
            if no_improve >= args.patience:
                break

        if args.verbose:
            print(f"      seed={seed} epoch={epoch:03d} valF1={100*val_metrics.macro_f1:.2f}")

    if best_state is None:
        raise RuntimeError("No valid checkpoint obtained.")

    model.load_state_dict(best_state)
    model.to(device)
    learned = model.egrm.detach().cpu().numpy()

    row: Dict[str, float] = {
        "Seed": int(seed),
        "Best epoch": int(best_epoch),
        "Best val Macro-F1 (%)": 100.0 * best_val_f1,
        "Final alpha": float(model.alpha.detach().cpu().item()),
        "EGRM change Frobenius": float(np.linalg.norm(learned - matrix, ord="fro")),
    }

    if evaluate_test:
        metrics = evaluate(model, test_loader, criterion, device)
        row.update(
            {
                "Acc (%)": 100.0 * metrics.acc,
                "Macro-F1 (%)": 100.0 * metrics.macro_f1,
                "Macro-P (%)": 100.0 * metrics.macro_p,
                "Macro-R (%)": 100.0 * metrics.macro_r,
            }
        )
    return row


def sample_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, 0.0
    se = float(arr.std(ddof=1)) / math.sqrt(len(arr))
    critical = float(student_t.ppf(0.975, len(arr) - 1)) if student_t is not None else 1.96
    half = critical * se
    return mean - half, mean + half, half


def summarize_validation(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in run_df.groupby("Candidate", sort=False):
        vals = group["Best val Macro-F1 (%)"].astype(float).values
        lo, hi, half = ci95(vals)
        rows.append(
            {
                "Candidate": candidate,
                "Runs": len(group),
                "Val Macro-F1 (%) Mean": float(vals.mean()),
                "Val Macro-F1 (%) Std": sample_std(vals),
                "Val Macro-F1 (%) 95% CI Low": lo,
                "Val Macro-F1 (%) 95% CI High": hi,
                "Val Macro-F1 (%) 95% CI Half-width": half,
                "Best epoch Mean": float(group["Best epoch"].mean()),
                "Final alpha Mean": float(group["Final alpha"].mean()),
                "EGRM change Frobenius Mean": float(group["EGRM change Frobenius"].mean()),
                "Elapsed seconds Mean": float(group["Elapsed seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_test(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in run_df.groupby("Candidate", sort=False):
        row: Dict[str, object] = {"Candidate": candidate, "Runs": len(group)}
        for col in [
            "Best val Macro-F1 (%)",
            "Acc (%)",
            "Macro-F1 (%)",
            "Macro-P (%)",
            "Macro-R (%)",
        ]:
            values = group[col].astype(float).values
            lo, hi, half = ci95(values)
            row[f"{col} Mean"] = float(values.mean())
            row[f"{col} Std"] = sample_std(values)
            row[f"{col} 95% CI Low"] = lo
            row[f"{col} 95% CI High"] = hi
            row[f"{col} 95% CI Half-width"] = half
        rows.append(row)
    return pd.DataFrame(rows)


def load_existing(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(required_columns))
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Existing file {path} is missing columns: {missing}")
    return df


def write_validation_outputs(run_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    run_df = run_df.sort_values(["Candidate", "Seed"]).reset_index(drop=True)
    run_df.to_csv(
        outdir / "k0_validation_search_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    summary = summarize_validation(run_df)
    summary.to_csv(
        outdir / "k0_validation_search_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    ranking = summary.sort_values(
        ["Val Macro-F1 (%) Mean", "Val Macro-F1 (%) Std", "Candidate"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Validation rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(
        outdir / "k0_validation_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return ranking


def write_test_outputs(run_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    run_df = run_df.sort_values(["Candidate", "Seed"]).reset_index(drop=True)
    run_df.to_csv(
        outdir / "selected_k0_test_runs.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    summary = summarize_test(run_df)
    summary.to_csv(
        outdir / "selected_k0_test_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search diverse theory-constrained K0 matrices using validation Macro-F1 only."
    )
    parser.add_argument("--data", type=str, default="胖虎电厂数据未加测试集.xlsx")
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument(
        "--outdir",
        type=str,
        default="grkanet_k0_candidate_search_10runs_outputs",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=36,
        help="Number of diverse K0 candidates, including the original K0.",
    )

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--balance-mode",
        choices=["none", "ce_only", "sampler_only", "both"],
        default="both",
    )
    parser.add_argument("--dgam-mode", choices=["energy", "entropy"], default="energy")
    parser.add_argument("--dgam-ema", type=float, default=0.8)
    parser.add_argument("--alpha-init", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kan-hidden", type=int, default=8)
    parser.add_argument("--kan-latent", type=int, default=64)
    parser.add_argument("--mask-update-every", type=int, default=20)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Stop after validation ranking and do not retrain/evaluate the selected K0 on test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive.")
    if args.mask_update_every <= 0:
        raise ValueError("--mask-update-every must be positive.")

    outdir = ensure_dir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seeds = parse_seed_list(args.seeds, args.n_runs)
    split = load_fixed_split(args.data, args.label_col, args.split_seed)

    all_candidates = build_all_theory_candidates()
    candidates = select_diverse_candidates(all_candidates, args.num_candidates)
    candidate_map = {c.candidate_id: c for c in candidates}
    save_candidate_library(candidates, outdir)

    print(f"Using device: {device}")
    print(f"Data: {args.data}")
    print(f"Split seed: {args.split_seed}")
    print(f"Training seeds: {seeds}")
    print(f"Valid theory-constrained library size: {len(all_candidates)}")
    print(f"Selected diverse candidates: {len(candidates)}")
    print(
        "Fixed config | alpha={} mask_every={} ema={} lr={} wd={} dropout={} "
        "epochs={} patience={} batch={} balance={}".format(
            args.alpha_init,
            args.mask_update_every,
            args.dgam_ema,
            args.lr,
            args.weight_decay,
            args.dropout,
            args.epochs,
            args.patience,
            args.batch_size,
            args.balance_mode,
        )
    )
    print("Search ranking uses validation Macro-F1 only. Test is reserved for the selected K0.")
    print("The run is resumable from k0_validation_search_runs.csv.")

    validation_path = outdir / "k0_validation_search_runs.csv"
    validation_cols = [
        "Candidate",
        "Seed",
        "Best epoch",
        "Best val Macro-F1 (%)",
        "Final alpha",
        "EGRM change Frobenius",
        "Elapsed seconds",
    ]
    validation_df = load_existing(validation_path, validation_cols)
    completed = {
        (str(row["Candidate"]), int(row["Seed"]))
        for _, row in validation_df.iterrows()
    }
    validation_rows = validation_df.to_dict("records")

    for c_idx, candidate in enumerate(candidates, start=1):
        print(
            f"\n################ Candidate {c_idx}/{len(candidates)}: "
            f"{candidate.candidate_id} ################"
        )
        print(candidate.matrix.astype(int))
        print(
            "Choices | CH4={} C2H4={} C2H2={} THC={} C2H6={} | ones={} | Hamming={}".format(
                candidate.ch4_choice,
                candidate.c2h4_choice,
                candidate.c2h2_choice,
                candidate.thc_choice,
                candidate.c2h6_choice,
                candidate.ones,
                candidate.hamming_from_original,
            )
        )

        for run_id, seed in enumerate(seeds, start=1):
            key = (candidate.candidate_id, int(seed))
            if key in completed:
                print(f"  Skip completed | {candidate.candidate_id} seed={seed}")
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = train_one(
                split=split,
                matrix=candidate.matrix,
                seed=seed,
                args=args,
                device=device,
                evaluate_test=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            row = {
                "Candidate": candidate.candidate_id,
                **result,
                "Elapsed seconds": elapsed,
            }
            validation_rows.append(row)
            completed.add(key)
            validation_df = pd.DataFrame(validation_rows, columns=validation_cols)
            ranking = write_validation_outputs(validation_df, outdir)

            print(
                f"  Run {run_id:02d}/{len(seeds):02d} | seed={seed} | "
                f"best_epoch={int(result['Best epoch']):03d} | "
                f"ValF1={result['Best val Macro-F1 (%)']:.2f} | "
                f"alpha={result['Final alpha']:.4f} | time={elapsed:.1f}s"
            )

    validation_df = pd.DataFrame(validation_rows, columns=validation_cols)
    ranking = write_validation_outputs(validation_df, outdir)

    if len(ranking) != len(candidates):
        raise RuntimeError(
            "Search did not complete every candidate. Rerun the same command to resume."
        )

    selected_id = str(ranking.iloc[0]["Candidate"])
    selected = candidate_map[selected_id]
    selection_payload = {
        "selection_basis": "Highest mean validation Macro-F1; lower validation std as tie-breaker",
        "selected_candidate": selected_id,
        "selected_matrix": selected.matrix.astype(int).tolist(),
        "validation_mean": float(ranking.iloc[0]["Val Macro-F1 (%) Mean"]),
        "validation_std": float(ranking.iloc[0]["Val Macro-F1 (%) Std"]),
        "fixed_configuration": {
            "split_seed": args.split_seed,
            "training_seeds": seeds,
            "alpha_init": args.alpha_init,
            "mask_update_every": args.mask_update_every,
            "dgam_ema": args.dgam_ema,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "checkpoint": "single-best validation Macro-F1",
        },
    }
    (outdir / "selected_k0.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(selected.matrix.astype(int), index=FEATURES, columns=PATTERNS).to_csv(
        outdir / "selected_k0_matrix.csv", encoding="utf-8-sig"
    )

    print("\n================ Validation-first K0 ranking (top 10) ================")
    print(
        ranking[
            ["Validation rank", "Candidate", "Val Macro-F1 (%) Mean", "Val Macro-F1 (%) Std"]
        ].head(10).to_string(index=False)
    )
    print(f"\nSelected K0 from validation only: {selected_id}")
    print(selected.matrix.astype(int))

    if args.search_only:
        print("--search-only was used; final selected-K0 test evaluation was skipped.")
        return

    test_path = outdir / "selected_k0_test_runs.csv"
    test_cols = [
        "Candidate",
        "Seed",
        "Best epoch",
        "Best val Macro-F1 (%)",
        "Final alpha",
        "EGRM change Frobenius",
        "Acc (%)",
        "Macro-F1 (%)",
        "Macro-P (%)",
        "Macro-R (%)",
        "Elapsed seconds",
    ]
    test_df = load_existing(test_path, test_cols)

    # If a partially completed test file belongs to another selected candidate,
    # keep it separate rather than mixing incompatible results.
    if not test_df.empty and any(test_df["Candidate"].astype(str) != selected_id):
        backup = outdir / f"selected_k0_test_runs_previous_{int(time.time())}.csv"
        test_df.to_csv(backup, index=False, encoding="utf-8-sig")
        test_df = pd.DataFrame(columns=test_cols)

    test_completed = {int(v) for v in test_df.get("Seed", pd.Series(dtype=int)).tolist()}
    test_rows = test_df.to_dict("records")

    print("\n================ Final selected-K0 test evaluation ================")
    for run_id, seed in enumerate(seeds, start=1):
        if seed in test_completed:
            print(f"  Skip completed final test | seed={seed}")
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = train_one(
            split=split,
            matrix=selected.matrix,
            seed=seed,
            args=args,
            device=device,
            evaluate_test=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        row = {"Candidate": selected_id, **result, "Elapsed seconds": elapsed}
        test_rows.append(row)
        test_df = pd.DataFrame(test_rows, columns=test_cols)
        test_summary = write_test_outputs(test_df, outdir)

        print(
            f"  Run {run_id:02d}/{len(seeds):02d} | seed={seed} | "
            f"ValF1={result['Best val Macro-F1 (%)']:.2f} | "
            f"Acc={result['Acc (%)']:.2f} | F1={result['Macro-F1 (%)']:.2f} | "
            f"P={result['Macro-P (%)']:.2f} | R={result['Macro-R (%)']:.2f} | "
            f"time={elapsed:.1f}s"
        )

    test_df = pd.DataFrame(test_rows, columns=test_cols)
    test_summary = write_test_outputs(test_df, outdir)
    row = test_summary.iloc[0]

    print("\n================ Selected K0 final summary ================")
    print(
        "{} | ValF1={:.2f}±{:.2f} | Acc={:.2f}±{:.2f} | "
        "Macro-F1={:.2f}±{:.2f} | Macro-P={:.2f}±{:.2f} | "
        "Macro-R={:.2f}±{:.2f}".format(
            selected_id,
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

    print("\nSaved:")
    for name in [
        "k0_candidate_metadata.csv",
        "k0_candidate_matrices.txt",
        "k0_validation_search_runs.csv",
        "k0_validation_search_summary.csv",
        "k0_validation_ranking.csv",
        "selected_k0.json",
        "selected_k0_matrix.csv",
        "selected_k0_test_runs.csv",
        "selected_k0_test_summary.csv",
    ]:
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
