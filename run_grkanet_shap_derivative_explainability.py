# -*- coding: utf-8 -*-
"""
Unified SHAP + sample-level KAN derivative analysis for GR-KANet.

This script addresses two reviewer comments:
1) Add SHAP or another explainability technique to support interpretability.
2) Show case studies in which KAN derivative curves help explain why a sample
   was assigned to a specific fault class.

Core design
-----------
- Input-level SHAP:
  Uses SHAP GradientExplainer on the eight standardized DGA gas inputs.
  The explained model output is the class logit, not the softmax probability.
- Global explanation:
  Exports global mean absolute SHAP importance and a class-by-gas SHAP matrix.
  Multiple seeds can be aggregated without filtering any seed.
- Local case studies:
  Deterministically selects:
      a representative correctly classified discharge sample,
      a representative correctly classified thermal sample,
      and an ambiguous sample (misclassified sample preferred).
- KAN derivative explanation:
  For the predicted class c, the two linear layers after the component-wise
  KAN transformations are combined:
      q_c = W_classifier[c, :] @ W_KAN
  For internal KAN component j:
      contribution_j,c = q_j,c * h_j(x_j)
      sensitivity_j,c  = q_j,c * h'_j(x_j)
  The script ranks internal components by |sensitivity_j,c| and exports the
  corresponding KAN response and derivative curves with the sample operating
  point marked.

Interpretation boundary
-----------------------
SHAP values describe model-attributed input contributions. KAN derivatives
describe local sensitivity of the learned internal mapping. Neither quantity
is claimed to prove physical causality.

Final GR-KANet protocol
-----------------------
- Fixed stratified 64/16/20 split, split seed 42.
- StandardScaler fitted only on the training set.
- alpha = 2.0
- DGAM EMA = 0.8
- mask updated every 20 GLOBAL optimizer steps during training.
- dropout = 0.0
- lr = 1e-3
- batch size = 64
- weighted sampler + class-weighted cross-entropy
- checkpoint selected using validation Macro-F1
- test set used only after model selection and for post-hoc explanation

Required files in the same folder
---------------------------------
run_grkanet_full_10runs_final.py
胖虎电厂数据未加测试集.xlsx

Required packages
-----------------
pip install shap matplotlib pandas numpy scikit-learn torch openpyxl

Quick run using only seed 42
----------------------------
python run_grkanet_shap_derivative_explainability.py --seeds 42

Recommended final run aggregating all ten seeds
-----------------------------------------------
python run_grkanet_shap_derivative_explainability.py ^
    --seeds 42,43,44,45,46,47,48,49,50,51

The script reuses saved checkpoints in the output checkpoint folder. If a
checkpoint is absent, it trains that seed once and saves the validation-selected
checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except Exception as exc:
    raise RuntimeError(
        "The 'shap' package is required. Install it with: pip install shap"
    ) from exc

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None


# Keep SVG text editable in Adobe Illustrator/Inkscape.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10

DEFAULT_FEATURES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]
DISCHARGE_CLASSES = ["D1", "D2", "PD"]
THERMAL_CLASSES = ["T12", "T3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP and sample-level KAN derivative explanations."
    )
    parser.add_argument(
        "--source-module",
        type=str,
        default="run_grkanet_full_10runs_final",
        help="Module that defines GRKANet, load_fixed_split, and make_loaders.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="胖虎电厂数据未加测试集.xlsx",
    )
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated training seeds. Use 42,...,51 for ten-model aggregation.",
    )
    parser.add_argument(
        "--case-seed",
        type=int,
        default=42,
        help="Pre-specified model seed used for local case studies.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="grkanet_explainability_outputs",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-update-every", type=int, default=20)
    parser.add_argument("--alpha-init", type=float, default=2.0)
    parser.add_argument("--dgam-ema", type=float, default=0.8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kan-hidden", type=int, default=8)
    parser.add_argument("--kan-latent", type=int, default=64)
    parser.add_argument(
        "--background-size",
        type=int,
        default=48,
        help="Maximum class-balanced training samples used as SHAP background.",
    )
    parser.add_argument(
        "--shap-nsamples",
        type=int,
        default=300,
        help="Expected-gradient samples used by SHAP GradientExplainer.",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--top-kan-components",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--case-classes",
        type=str,
        default="D1,T3",
        help="Preferred representative classes: one discharge and one thermal class.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_seed_list(text: str) -> List[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("No valid seed was provided.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds were provided. The script will not filter them.")
    return seeds


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(
    source: Any,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    model = source.GRKANet(
        num_classes=num_classes,
        dgam_mode="energy",
        alpha_init=args.alpha_init,
        dropout=args.dropout,
        channels=args.channels,
        kan_hidden=args.kan_hidden,
        kan_latent=args.kan_latent,
        dgam_ema=args.dgam_ema,
    )
    return model.to(device)


def checkpoint_path(checkpoint_dir: Path, seed: int) -> Path:
    return checkpoint_dir / "grkanet_seed_{:d}_mask20.pt".format(seed)


def extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "best_state"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]

        # A raw state_dict is also a dictionary.
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload

    raise ValueError("Unsupported checkpoint format.")


def evaluate_validation(
    source: Any,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    loss, result = source.evaluate_model(model, loader, criterion, device)
    return float(loss), float(result.macro_f1)


def train_or_load_model(
    source: Any,
    split: Any,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a seed checkpoint when available; otherwise train it once.

    No test metric is used for checkpoint selection.
    """
    ckpt_path = checkpoint_path(checkpoint_dir, seed)
    num_classes = len(split.display_labels)
    model = create_model(source, num_classes, args, device)

    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(extract_state_dict(payload))
        model.to(device)
        model.eval()

        metadata = dict(payload) if isinstance(payload, dict) else {}
        metadata["checkpoint_path"] = str(ckpt_path)
        metadata["loaded_existing_checkpoint"] = True
        return model, metadata

    set_seed(seed)
    train_loader, val_loader, _ = source.make_loaders(
        split,
        batch_size=args.batch_size,
        balance_mode="both",
    )

    class_weights_np = source.compute_class_weights_np(
        split.y_train,
        num_classes,
    )
    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    amp_enabled = device.type == "cuda" and autocast is not None
    scaler = (
        GradScaler(enabled=True)
        if amp_enabled and GradScaler is not None
        else None
    )

    # Final revised protocol: global counter and interval 20.
    model.dgam.update_mask_from_kan()
    global_step = 0

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_f1 = -1.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                if scaler is None:
                    raise RuntimeError("AMP is enabled but GradScaler is unavailable.")
                with autocast(enabled=True):
                    logits = model(xb)
                    loss = criterion(logits, yb)
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

        _, val_f1 = evaluate_validation(
            source,
            model,
            val_loader,
            criterion,
            device,
        )

        improved = val_f1 > best_val_f1 + 1e-6
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if args.verbose:
            print(
                "seed={} epoch={:03d} val Macro-F1={:.2f}%{}".format(
                    seed,
                    epoch,
                    100.0 * val_f1,
                    " | BEST" if improved else "",
                )
            )

        if no_improve >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was obtained for seed {}.".format(seed))

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    payload = {
        "model_state_dict": best_state,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "config": {
            "split_seed": args.split_seed,
            "mask_update_every": args.mask_update_every,
            "alpha_init": args.alpha_init,
            "dgam_ema": args.dgam_ema,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "balance_mode": "both",
        },
        "display_labels": list(split.display_labels),
        "feature_names": list(getattr(source, "FEATURES", DEFAULT_FEATURES)),
    }
    torch.save(payload, ckpt_path)

    payload["checkpoint_path"] = str(ckpt_path)
    payload["loaded_existing_checkpoint"] = False
    return model, payload


def class_balanced_background_indices(
    y_train: np.ndarray,
    max_size: int,
    seed: int,
) -> np.ndarray:
    """
    Select a deterministic class-balanced background from the training set only.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    per_class = max(1, max_size // len(classes))
    selected: List[int] = []

    for class_id in classes:
        indices = np.where(y_train == class_id)[0]
        take = min(per_class, len(indices))
        chosen = rng.choice(indices, size=take, replace=False)
        selected.extend(chosen.tolist())

    if len(selected) < min(max_size, len(y_train)):
        remaining = np.setdiff1d(
            np.arange(len(y_train)),
            np.asarray(selected, dtype=int),
            assume_unique=False,
        )
        extra_size = min(max_size - len(selected), len(remaining))
        if extra_size > 0:
            extra = rng.choice(remaining, size=extra_size, replace=False)
            selected.extend(extra.tolist())

    return np.asarray(sorted(selected), dtype=int)


def normalize_shap_values(
    raw_values: Any,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """
    Convert different SHAP-version outputs to [samples, features, classes].
    """
    if isinstance(raw_values, list):
        if len(raw_values) != n_classes:
            raise ValueError(
                "Unexpected SHAP list length: {} instead of {}.".format(
                    len(raw_values), n_classes
                )
            )
        array = np.stack(
            [np.asarray(item, dtype=float) for item in raw_values],
            axis=-1,
        )
        return array

    array = np.asarray(raw_values, dtype=float)

    if array.shape == (n_samples, n_features, n_classes):
        return array
    if array.shape == (n_classes, n_samples, n_features):
        return np.transpose(array, (1, 2, 0))
    if array.shape == (n_samples, n_classes, n_features):
        return np.transpose(array, (0, 2, 1))

    raise ValueError(
        "Unsupported SHAP output shape {}. Expected a multiclass array.".format(
            array.shape
        )
    )


def compute_shap_values(
    model: nn.Module,
    background_np: np.ndarray,
    explained_np: np.ndarray,
    device: torch.device,
    nsamples: int,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected-gradient SHAP values for class logits.

    Returns
    -------
    shap_values: [samples, features, classes]
    baseline_logits: [classes]
    """
    model.eval()

    background_tensor = torch.tensor(
        background_np,
        dtype=torch.float32,
        device=device,
    )
    explained_tensor = torch.tensor(
        explained_np,
        dtype=torch.float32,
        device=device,
    )

    explainer = shap.GradientExplainer(model, background_tensor)
    raw_values = explainer.shap_values(
        explained_tensor,
        nsamples=nsamples,
    )

    values = normalize_shap_values(
        raw_values,
        n_samples=len(explained_np),
        n_features=explained_np.shape[1],
        n_classes=n_classes,
    )

    with torch.no_grad():
        baseline_logits = (
            model(background_tensor)
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

    return values, baseline_logits


def predict_all(
    model: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            batch = torch.tensor(
                x_np[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits_list.append(
                model(batch).detach().cpu().numpy()
            )

    logits = np.concatenate(logits_list, axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    return logits, probabilities, predictions


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(
        str(stem.with_suffix(".svg")),
        format="svg",
        bbox_inches="tight",
    )
    fig.savefig(
        str(stem.with_suffix(".pdf")),
        format="pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        str(stem.with_suffix(".png")),
        format="png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_global_importance(
    feature_names: Sequence[str],
    mean_values: np.ndarray,
    std_values: np.ndarray,
    output_stem: Path,
) -> None:
    order = np.argsort(mean_values)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    positions = np.arange(len(order))

    ax.barh(
        positions,
        mean_values[order],
        xerr=std_values[order] if np.any(std_values > 0) else None,
        capsize=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([feature_names[index] for index in order])
    ax.set_xlabel("Mean absolute SHAP value (class logit)")
    ax.set_ylabel("DGA gas")
    ax.set_title("Global input importance")
    ax.grid(axis="x", alpha=0.25)

    save_figure(fig, output_stem)


def plot_classwise_heatmap(
    feature_names: Sequence[str],
    class_names: Sequence[str],
    matrix: np.ndarray,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    image = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("DGA gas")
    ax.set_ylabel("Fault class")
    ax.set_title("Class-specific mean absolute SHAP values")

    threshold = float(np.nanmedian(matrix))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                "{:.3f}".format(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, label="Mean |SHAP| (class logit)")
    save_figure(fig, output_stem)


def choose_class_representative(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_index: int,
    excluded: Sequence[int],
) -> Optional[int]:
    """
    Correctly classified sample whose confidence is closest to the class median.
    """
    excluded_set = set(int(item) for item in excluded)
    candidates = np.where(
        (true_labels == class_index) & (predictions == class_index)
    )[0]
    candidates = np.asarray(
        [index for index in candidates if int(index) not in excluded_set],
        dtype=int,
    )
    if len(candidates) == 0:
        return None

    confidence = probabilities[candidates, class_index]
    median_confidence = np.median(confidence)
    local = int(np.argmin(np.abs(confidence - median_confidence)))
    return int(candidates[local])


def choose_group_representative(
    group_names: Sequence[str],
    preferred_name: Optional[str],
    class_names: Sequence[str],
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    excluded: Sequence[int],
) -> Tuple[int, str]:
    """
    Prefer the user-specified class. If unavailable, use the group class with
    the largest number of correct test predictions.
    """
    candidate_names = []
    if preferred_name and preferred_name in group_names:
        candidate_names.append(preferred_name)
    candidate_names.extend(
        [name for name in group_names if name not in candidate_names]
    )

    class_to_correct_count: Dict[str, int] = {}
    for name in candidate_names:
        if name not in class_names:
            continue
        class_id = class_names.index(name)
        count = int(
            np.sum(
                (true_labels == class_id)
                & (predictions == class_id)
            )
        )
        class_to_correct_count[name] = count

    ordered_names = sorted(
        class_to_correct_count,
        key=lambda name: (
            0 if name == preferred_name else 1,
            -class_to_correct_count[name],
            name,
        ),
    )

    for name in ordered_names:
        class_id = class_names.index(name)
        selected = choose_class_representative(
            true_labels,
            predictions,
            probabilities,
            class_id,
            excluded,
        )
        if selected is not None:
            return selected, name

    raise RuntimeError(
        "No correctly classified representative sample was found for group {}.".format(
            list(group_names)
        )
    )


def choose_ambiguous_sample(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    excluded: Sequence[int],
) -> Tuple[int, str]:
    """
    Prefer a misclassified sample with the smallest top-1/top-2 probability
    margin. If no misclassification exists, choose the lowest-margin remaining
    sample.
    """
    excluded_set = set(int(item) for item in excluded)
    sorted_probs = np.sort(probabilities, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    all_indices = np.arange(len(true_labels))
    remaining = np.asarray(
        [index for index in all_indices if int(index) not in excluded_set],
        dtype=int,
    )

    misclassified = remaining[
        predictions[remaining] != true_labels[remaining]
    ]
    if len(misclassified) > 0:
        selected = int(misclassified[np.argmin(margins[misclassified])])
        return selected, "ambiguous_misclassified"

    selected = int(remaining[np.argmin(margins[remaining])])
    return selected, "ambiguous_low_margin"


def parse_preferred_case_classes(text: str) -> Tuple[Optional[str], Optional[str]]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    discharge = names[0] if len(names) >= 1 else None
    thermal = names[1] if len(names) >= 2 else None
    return discharge, thermal


def select_case_indices(
    class_names: Sequence[str],
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    case_classes_text: str,
) -> List[Dict[str, Any]]:
    discharge_preferred, thermal_preferred = parse_preferred_case_classes(
        case_classes_text
    )

    cases: List[Dict[str, Any]] = []
    used: List[int] = []

    discharge_index, discharge_class = choose_group_representative(
        DISCHARGE_CLASSES,
        discharge_preferred,
        class_names,
        true_labels,
        predictions,
        probabilities,
        used,
    )
    used.append(discharge_index)
    cases.append({
        "case_type": "representative_discharge",
        "test_index": discharge_index,
        "selection_class": discharge_class,
        "selection_rule": (
            "Correctly classified sample with predicted-class confidence "
            "closest to the median among correct samples of the selected class."
        ),
    })

    thermal_index, thermal_class = choose_group_representative(
        THERMAL_CLASSES,
        thermal_preferred,
        class_names,
        true_labels,
        predictions,
        probabilities,
        used,
    )
    used.append(thermal_index)
    cases.append({
        "case_type": "representative_thermal",
        "test_index": thermal_index,
        "selection_class": thermal_class,
        "selection_rule": (
            "Correctly classified sample with predicted-class confidence "
            "closest to the median among correct samples of the selected class."
        ),
    })

    ambiguous_index, ambiguous_type = choose_ambiguous_sample(
        true_labels,
        predictions,
        probabilities,
        used,
    )
    cases.append({
        "case_type": ambiguous_type,
        "test_index": ambiguous_index,
        "selection_class": "",
        "selection_rule": (
            "Misclassified test sample with the smallest top-1/top-2 "
            "probability margin; if none exists, the remaining sample with "
            "the smallest margin."
        ),
    })

    return cases


def plot_probability_case(
    probabilities: np.ndarray,
    class_names: Sequence[str],
    true_class: str,
    predicted_class: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    positions = np.arange(len(class_names))
    ax.bar(positions, probabilities)
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "True: {} | Predicted: {}".format(true_class, predicted_class)
    )
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, output_stem)


def plot_standardized_inputs(
    standardized_values: np.ndarray,
    raw_values: np.ndarray,
    feature_names: Sequence[str],
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    positions = np.arange(len(feature_names))
    ax.bar(positions, standardized_values)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(feature_names, rotation=35, ha="right")
    ax.set_ylabel("Standardized input")
    ax.set_title("DGA input profile")

    for index, raw_value in enumerate(raw_values):
        ax.text(
            index,
            standardized_values[index],
            "{:.2f}".format(raw_value),
            ha="center",
            va="bottom" if standardized_values[index] >= 0 else "top",
            fontsize=7,
            rotation=90,
        )

    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_stem)


def plot_local_shap_bar(
    shap_values: np.ndarray,
    raw_values: np.ndarray,
    feature_names: Sequence[str],
    predicted_class: str,
    output_stem: Path,
) -> None:
    order = np.argsort(np.abs(shap_values))
    labels = [
        "{} = {:.2f}".format(feature_names[index], raw_values[index])
        for index in order
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    positions = np.arange(len(order))
    ax.barh(positions, shap_values[order])
    ax.axvline(0.0, linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("SHAP contribution to {} logit".format(predicted_class))
    ax.set_title("Local input-level SHAP explanation")
    ax.grid(axis="x", alpha=0.25)

    save_figure(fig, output_stem)


def plot_native_shap_waterfall(
    local_values: np.ndarray,
    baseline_value: float,
    raw_values: np.ndarray,
    feature_names: Sequence[str],
    predicted_class: str,
    output_stem: Path,
) -> bool:
    """
    Export an editable native SHAP waterfall plot when supported by the
    installed SHAP version. The custom local SHAP bar is always exported too.
    """
    try:
        explanation = shap.Explanation(
            values=local_values,
            base_values=baseline_value,
            data=raw_values,
            feature_names=list(feature_names),
        )
        shap.plots.waterfall(
            explanation,
            max_display=len(feature_names),
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(7.2, 4.7)
        plt.title(
            "Local SHAP waterfall for predicted class {}".format(
                predicted_class
            )
        )
        save_figure(fig, output_stem)
        return True
    except Exception as exc:
        output_stem.with_suffix(".error.txt").write_text(
            str(exc),
            encoding="utf-8",
        )
        plt.close("all")
        return False


def collect_internal_flat_features(
    model: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    rows: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            batch = torch.tensor(
                x_np[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            features = model.forward_features(batch).flatten(1)
            rows.append(features.detach().cpu().numpy())

    return np.concatenate(rows, axis=0)


def effective_component_weights(
    model: nn.Module,
) -> torch.Tensor:
    """
    q = W_classifier @ W_KAN, shape [classes, 256].
    """
    return model.classifier.weight @ model.kan.linear.weight


def component_value_and_derivative(
    component_function: nn.Module,
    x_value: float,
    device: torch.device,
) -> Tuple[float, float]:
    x_tensor = torch.tensor(
        [[x_value]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    h_value = component_function(x_tensor) + x_tensor
    derivative = torch.autograd.grad(
        h_value,
        x_tensor,
        grad_outputs=torch.ones_like(h_value),
        create_graph=False,
        retain_graph=False,
    )[0]

    return (
        float(h_value.detach().cpu().item()),
        float(derivative.detach().cpu().item()),
    )


def rank_kan_components_for_case(
    model: nn.Module,
    standardized_sample: np.ndarray,
    predicted_class_id: int,
    feature_names: Sequence[str],
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Rank internal KAN components by class-specific local sensitivity.
    """
    model.eval()
    sample_tensor = torch.tensor(
        standardized_sample.reshape(1, -1),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        internal_flat = (
            model.forward_features(sample_tensor)
            .flatten(1)
            .detach()
            .cpu()
            .numpy()[0]
        )

    q = (
        effective_component_weights(model)
        .detach()
        .cpu()
        .numpy()
    )
    q_class = q[predicted_class_id]

    rows: List[Dict[str, Any]] = []
    num_gases = len(feature_names)

    for component_index, x_value in enumerate(internal_flat):
        h_value, derivative = component_value_and_derivative(
            model.kan.functions[component_index],
            float(x_value),
            device,
        )
        weight = float(q_class[component_index])
        contribution = weight * h_value
        weighted_sensitivity = weight * derivative
        channel_index = component_index // num_gases
        gas_index = component_index % num_gases

        rows.append({
            "Component index": component_index,
            "Channel index": channel_index,
            "Gas index": gas_index,
            "Gas": feature_names[gas_index],
            "Internal coordinate": float(x_value),
            "Effective class weight q": weight,
            "KAN component h(x)": h_value,
            "Local derivative h_prime(x)": derivative,
            "Class-specific contribution q*h(x)": contribution,
            "Class-specific sensitivity q*h_prime(x)": weighted_sensitivity,
            "Absolute class-specific sensitivity": abs(weighted_sensitivity),
        })

    dataframe = pd.DataFrame(rows).sort_values(
        "Absolute class-specific sensitivity",
        ascending=False,
    )
    return dataframe, internal_flat


def curve_range(
    reference_values: np.ndarray,
    operating_point: float,
) -> Tuple[float, float]:
    lower = float(np.percentile(reference_values, 1.0))
    upper = float(np.percentile(reference_values, 99.0))
    lower = min(lower, operating_point)
    upper = max(upper, operating_point)

    span = upper - lower
    if span < 1e-6:
        span = max(abs(operating_point), 1.0)

    margin = 0.15 * span
    return lower - margin, upper + margin


def calculate_component_curve(
    component_function: nn.Module,
    x_grid: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    x_tensor = torch.tensor(
        x_grid.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    h_tensor = component_function(x_tensor) + x_tensor
    derivative_tensor = torch.autograd.grad(
        h_tensor,
        x_tensor,
        grad_outputs=torch.ones_like(h_tensor),
        create_graph=False,
        retain_graph=False,
    )[0]

    return (
        h_tensor.detach().cpu().numpy().reshape(-1),
        derivative_tensor.detach().cpu().numpy().reshape(-1),
    )


def plot_component_response(
    x_grid: np.ndarray,
    h_values: np.ndarray,
    operating_x: float,
    operating_h: float,
    component_row: pd.Series,
    predicted_class: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(x_grid, h_values)
    ax.scatter([operating_x], [operating_h], zorder=3)
    ax.axvline(operating_x, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Internal coordinate x")
    ax.set_ylabel("KAN component response h(x)")
    ax.set_title(
        "{} | channel {} / {} | contribution={:.3f}".format(
            predicted_class,
            int(component_row["Channel index"]),
            component_row["Gas"],
            float(component_row["Class-specific contribution q*h(x)"]),
        )
    )
    ax.grid(alpha=0.25)

    save_figure(fig, output_stem)


def plot_component_derivative(
    x_grid: np.ndarray,
    derivative_values: np.ndarray,
    operating_x: float,
    operating_derivative: float,
    component_row: pd.Series,
    predicted_class: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(x_grid, derivative_values)
    ax.scatter([operating_x], [operating_derivative], zorder=3)
    ax.axvline(operating_x, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linewidth=0.8)

    ax.set_xlabel("Internal coordinate x")
    ax.set_ylabel("Local derivative h'(x)")
    ax.set_title(
        "{} | channel {} / {} | q h'(x)={:.3f}".format(
            predicted_class,
            int(component_row["Channel index"]),
            component_row["Gas"],
            float(component_row["Class-specific sensitivity q*h_prime(x)"]),
        )
    )
    ax.grid(alpha=0.25)

    save_figure(fig, output_stem)


def write_protocol(
    output_path: Path,
    args: argparse.Namespace,
    seeds: Sequence[int],
    feature_names: Sequence[str],
    class_names: Sequence[str],
) -> None:
    lines = [
        "GR-KANet explainability protocol",
        "=" * 36,
        "",
        "Reviewer questions addressed:",
        "1. SHAP/Grad-CAM or other explainability support.",
        "2. Sample-level derivative-curve case studies.",
        "",
        "Model/data protocol:",
        "- Fixed split seed: {}".format(args.split_seed),
        "- Training seeds: {}".format(list(seeds)),
        "- Features: {}".format(list(feature_names)),
        "- Classes: {}".format(list(class_names)),
        "- StandardScaler fitted only on training data.",
        "- SHAP background drawn only from training data.",
        "- Test set used only for post-hoc explanation.",
        "- Checkpoint selected by validation Macro-F1.",
        "- KAN-DGAM mask update interval: 20 global optimizer steps.",
        "",
        "SHAP:",
        "- Method: SHAP GradientExplainer / expected gradients.",
        "- Explained output: class logit.",
        "- Background size: {}".format(args.background_size),
        "- SHAP nsamples: {}".format(args.shap_nsamples),
        "",
        "Case selection:",
        "- Representative discharge: correct sample nearest class median confidence.",
        "- Representative thermal: correct sample nearest class median confidence.",
        "- Ambiguous case: misclassified sample with minimum top1-top2 margin; "
        "otherwise minimum-margin remaining sample.",
        "- Local cases use pre-specified seed {}.".format(args.case_seed),
        "",
        "Interpretation boundary:",
        "- SHAP values are model-attributed contributions, not physical causality.",
        "- KAN derivatives are local sensitivities of internal learned mappings.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.mask_update_every != 20:
        raise ValueError(
            "The final revised protocol requires --mask-update-every 20."
        )

    seeds = parse_seed_list(args.seeds)
    if args.case_seed not in seeds:
        raise ValueError(
            "--case-seed must be included in --seeds so that the same "
            "pre-specified model is available for local explanations."
        )

    outdir = ensure_dir(Path(args.outdir))
    checkpoint_dir = ensure_dir(outdir / "checkpoints")
    global_dir = ensure_dir(outdir / "global_shap")
    case_dir = ensure_dir(outdir / "case_studies")

    source = importlib.import_module(args.source_module)
    feature_names = list(getattr(source, "FEATURES", DEFAULT_FEATURES))

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print("Device:", device)
    print("Seeds:", seeds)
    print("Case-study seed:", args.case_seed)
    print("Data:", args.data)

    split = source.load_fixed_split(
        args.data,
        label_col=args.label_col,
        split_seed=args.split_seed,
    )
    class_names = list(split.display_labels)
    num_classes = len(class_names)

    background_indices = class_balanced_background_indices(
        split.y_train,
        max_size=args.background_size,
        seed=args.split_seed,
    )
    background_np = split.X_train[background_indices]

    pd.DataFrame({
        "Training index": background_indices,
        "Encoded class": split.y_train[background_indices],
        "Class": [
            class_names[int(value)]
            for value in split.y_train[background_indices]
        ],
    }).to_csv(
        outdir / "shap_background_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_global_rows: List[Dict[str, Any]] = []
    seed_classwise_arrays: List[np.ndarray] = []
    seed_overall_arrays: List[np.ndarray] = []

    case_model: Optional[nn.Module] = None
    case_shap_values: Optional[np.ndarray] = None
    case_baseline_logits: Optional[np.ndarray] = None
    case_logits: Optional[np.ndarray] = None
    case_probabilities: Optional[np.ndarray] = None
    case_predictions: Optional[np.ndarray] = None

    model_metadata_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        print("\n===== Explainability model seed {} =====".format(seed))
        model, metadata = train_or_load_model(
            source=source,
            split=split,
            seed=seed,
            args=args,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        model_metadata_rows.append({
            "Seed": seed,
            "Checkpoint": metadata.get("checkpoint_path", ""),
            "Loaded existing checkpoint": metadata.get(
                "loaded_existing_checkpoint", False
            ),
            "Best epoch": metadata.get("best_epoch", ""),
            "Best validation Macro-F1": metadata.get(
                "best_val_macro_f1", ""
            ),
        })

        shap_values, baseline_logits = compute_shap_values(
            model=model,
            background_np=background_np,
            explained_np=split.X_test,
            device=device,
            nsamples=args.shap_nsamples,
            n_classes=num_classes,
        )

        classwise_mean_abs = np.mean(np.abs(shap_values), axis=0).T
        overall_mean_abs = np.mean(classwise_mean_abs, axis=0)

        seed_classwise_arrays.append(classwise_mean_abs)
        seed_overall_arrays.append(overall_mean_abs)

        for feature_index, feature_name in enumerate(feature_names):
            seed_global_rows.append({
                "Seed": seed,
                "Feature": feature_name,
                "Mean absolute SHAP": overall_mean_abs[feature_index],
            })

        if seed == args.case_seed:
            case_model = model
            case_shap_values = shap_values
            case_baseline_logits = baseline_logits
            case_logits, case_probabilities, case_predictions = predict_all(
                model,
                split.X_test,
                device,
            )
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pd.DataFrame(model_metadata_rows).to_csv(
        outdir / "model_checkpoint_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_overall_stack = np.stack(seed_overall_arrays, axis=0)
    seed_classwise_stack = np.stack(seed_classwise_arrays, axis=0)

    global_mean = seed_overall_stack.mean(axis=0)
    global_std = (
        seed_overall_stack.std(axis=0, ddof=1)
        if len(seeds) > 1
        else np.zeros_like(global_mean)
    )
    classwise_mean = seed_classwise_stack.mean(axis=0)
    classwise_std = (
        seed_classwise_stack.std(axis=0, ddof=1)
        if len(seeds) > 1
        else np.zeros_like(classwise_mean)
    )

    global_summary = pd.DataFrame({
        "Feature": feature_names,
        "Mean absolute SHAP across seeds": global_mean,
        "Sample std across seeds": global_std,
        "Number of model seeds": len(seeds),
    }).sort_values(
        "Mean absolute SHAP across seeds",
        ascending=False,
    )
    global_summary.to_csv(
        global_dir / "global_shap_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(seed_global_rows).to_csv(
        global_dir / "global_shap_seedwise.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classwise_rows: List[Dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        for feature_index, feature_name in enumerate(feature_names):
            classwise_rows.append({
                "Class": class_name,
                "Feature": feature_name,
                "Mean absolute SHAP across seeds": classwise_mean[
                    class_index, feature_index
                ],
                "Sample std across seeds": classwise_std[
                    class_index, feature_index
                ],
                "Number of model seeds": len(seeds),
            })
    pd.DataFrame(classwise_rows).to_csv(
        global_dir / "classwise_shap_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_global_importance(
        feature_names,
        global_mean,
        global_std,
        global_dir / "fig_global_shap_importance",
    )
    plot_classwise_heatmap(
        feature_names,
        class_names,
        classwise_mean,
        global_dir / "fig_classwise_shap_heatmap",
    )

    if (
        case_model is None
        or case_shap_values is None
        or case_baseline_logits is None
        or case_logits is None
        or case_probabilities is None
        or case_predictions is None
    ):
        raise RuntimeError("The pre-specified case-study model was not retained.")

    raw_test_values = split.scaler.inverse_transform(split.X_test)
    selected_cases = select_case_indices(
        class_names,
        split.y_test,
        case_predictions,
        case_probabilities,
        args.case_classes,
    )

    prediction_rows: List[Dict[str, Any]] = []
    for test_index in range(len(split.X_test)):
        sorted_prob = np.sort(case_probabilities[test_index])
        row: Dict[str, Any] = {
            "Test index": test_index,
            "True class": class_names[int(split.y_test[test_index])],
            "Predicted class": class_names[int(case_predictions[test_index])],
            "Correct": int(
                split.y_test[test_index] == case_predictions[test_index]
            ),
            "Top1-top2 margin": sorted_prob[-1] - sorted_prob[-2],
        }
        for class_index, class_name in enumerate(class_names):
            row["P({})".format(class_name)] = case_probabilities[
                test_index, class_index
            ]
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(
        outdir / "test_predictions_case_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Internal-feature reference ranges use training data only.
    internal_training_features = collect_internal_flat_features(
        case_model,
        split.X_train,
        device,
    )

    case_manifest_rows: List[Dict[str, Any]] = []
    component_detail_frames: List[pd.DataFrame] = []

    for case_number, case in enumerate(selected_cases, start=1):
        test_index = int(case["test_index"])
        true_id = int(split.y_test[test_index])
        predicted_id = int(case_predictions[test_index])
        true_name = class_names[true_id]
        predicted_name = class_names[predicted_id]
        probability = float(
            case_probabilities[test_index, predicted_id]
        )

        case_name = "case_{:02d}_{}".format(
            case_number,
            case["case_type"],
        )
        current_dir = ensure_dir(case_dir / case_name)

        standardized_sample = split.X_test[test_index]
        raw_sample = raw_test_values[test_index]
        local_shap = case_shap_values[
            test_index, :, predicted_id
        ]

        plot_probability_case(
            case_probabilities[test_index],
            class_names,
            true_name,
            predicted_name,
            current_dir / "predicted_probabilities",
        )
        plot_standardized_inputs(
            standardized_sample,
            raw_sample,
            feature_names,
            current_dir / "dga_input_profile",
        )
        plot_local_shap_bar(
            local_shap,
            raw_sample,
            feature_names,
            predicted_name,
            current_dir / "local_shap_contributions",
        )
        waterfall_created = plot_native_shap_waterfall(
            local_values=local_shap,
            baseline_value=float(
                case_baseline_logits[predicted_id]
            ),
            raw_values=raw_sample,
            feature_names=feature_names,
            predicted_class=predicted_name,
            output_stem=current_dir / "local_shap_waterfall",
        )

        component_df, internal_flat = rank_kan_components_for_case(
            case_model,
            standardized_sample,
            predicted_id,
            feature_names,
            device,
        )
        component_df.insert(0, "Case", case_name)
        component_df.insert(1, "Test index", test_index)
        component_df.insert(2, "True class", true_name)
        component_df.insert(3, "Predicted class", predicted_name)
        component_df.to_csv(
            current_dir / "kan_component_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
        component_detail_frames.append(component_df)

        top_components = component_df.head(
            args.top_kan_components
        ).copy()

        for rank, (_, component_row) in enumerate(
            top_components.iterrows(),
            start=1,
        ):
            component_index = int(
                component_row["Component index"]
            )
            operating_x = float(
                component_row["Internal coordinate"]
            )

            lower, upper = curve_range(
                internal_training_features[:, component_index],
                operating_x,
            )
            x_grid = np.linspace(
                lower,
                upper,
                args.curve_points,
                dtype=np.float32,
            )
            h_values, derivative_values = calculate_component_curve(
                case_model.kan.functions[component_index],
                x_grid,
                device,
            )
            operating_h = float(
                component_row["KAN component h(x)"]
            )
            operating_derivative = float(
                component_row["Local derivative h_prime(x)"]
            )

            stem_suffix = "rank_{:02d}_component_{:03d}_ch{:02d}_{}".format(
                rank,
                component_index,
                int(component_row["Channel index"]),
                component_row["Gas"],
            )
            plot_component_response(
                x_grid,
                h_values,
                operating_x,
                operating_h,
                component_row,
                predicted_name,
                current_dir / (
                    "kan_response_" + stem_suffix
                ),
            )
            plot_component_derivative(
                x_grid,
                derivative_values,
                operating_x,
                operating_derivative,
                component_row,
                predicted_name,
                current_dir / (
                    "kan_derivative_" + stem_suffix
                ),
            )

            pd.DataFrame({
                "Grid x": x_grid,
                "KAN response h(x)": h_values,
                "Derivative h_prime(x)": derivative_values,
            }).to_csv(
                current_dir / (
                    "curve_data_" + stem_suffix + ".csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

        sorted_probability = np.sort(
            case_probabilities[test_index]
        )
        case_manifest_rows.append({
            "Case": case_name,
            "Case type": case["case_type"],
            "Test index": test_index,
            "Selection class": case["selection_class"],
            "Selection rule": case["selection_rule"],
            "True class": true_name,
            "Predicted class": predicted_name,
            "Correct": int(true_id == predicted_id),
            "Predicted probability": probability,
            "Top1-top2 margin": (
                sorted_probability[-1] - sorted_probability[-2]
            ),
            "Native SHAP waterfall created": waterfall_created,
            "Output folder": str(current_dir),
        })

        gas_table = pd.DataFrame({
            "Feature": feature_names,
            "Raw gas value": raw_sample,
            "Standardized input": standardized_sample,
            "SHAP value for predicted-class logit": local_shap,
        })
        gas_table.to_csv(
            current_dir / "case_input_and_shap.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(case_manifest_rows).to_csv(
        case_dir / "selected_case_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        component_detail_frames,
        ignore_index=True,
    ).to_csv(
        case_dir / "all_case_kan_component_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_protocol(
        outdir / "explainability_protocol.txt",
        args,
        seeds,
        feature_names,
        class_names,
    )

    summary = {
        "seeds": seeds,
        "case_seed": args.case_seed,
        "split_seed": args.split_seed,
        "feature_names": feature_names,
        "class_names": class_names,
        "background_size_actual": int(len(background_indices)),
        "shap_n_samples": args.shap_nsamples,
        "mask_update_every": args.mask_update_every,
        "global_figures": [
            str(global_dir / "fig_global_shap_importance.svg"),
            str(global_dir / "fig_classwise_shap_heatmap.svg"),
        ],
        "case_manifest": str(
            case_dir / "selected_case_manifest.csv"
        ),
    }
    (outdir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFinished.")
    print(
        "Global SHAP SVG:",
        global_dir / "fig_global_shap_importance.svg",
    )
    print(
        "Classwise SHAP SVG:",
        global_dir / "fig_classwise_shap_heatmap.svg",
    )
    print(
        "Case manifest:",
        case_dir / "selected_case_manifest.csv",
    )
    print(
        "All figures were also exported as editable SVG, PDF, and 600-dpi PNG."
    )


if __name__ == "__main__":
    main()
