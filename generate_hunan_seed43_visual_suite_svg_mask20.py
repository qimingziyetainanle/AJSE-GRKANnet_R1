# -*- coding: utf-8 -*-
"""
Clean visual-generation script for the Hunan-Plant dataset (SVG version, mask-update-every=20).

This script uses:
  - fixed split seed = 42
  - shared training seed = 43

Outputs:
  1. 13 confusion matrices (all methods)
  2. GR-KANet visualization figures:
       - channel mask at convergence
       - KAN response functions
       - KAN derivative curves
       - channel scores by energy
       - channel scores by entropy
  3. 13 decision-space t-SNE plots (all methods)

It does NOT redraw the raw-feature t-SNE figure, because the user plans to keep
that panel unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


METHOD_ORDER: List[Tuple[str, str]] = [
    ("01", "SVM"),
    ("02", "XGBoost"),
    ("03", "1D-CNN"),
    ("04", "SMOTE-GBDT"),
    ("05", "SMOTE-ENN-XGBoost"),
    ("06", "AdaBoost-TCNN"),
    ("07", "Lightweight ResNet"),
    ("08", "MTF-GhostNetV2"),
    ("09", "Graph-enhanced DGA"),
    ("10", "DE-IPFL"),
    ("11", "Knowledge-filtered DSL"),
    ("12", "Knowledge-CapsNet"),
    ("13", "Ours (GR-KANet)"),
]


@dataclass
class MethodArtifact:
    name: str
    y_pred: np.ndarray
    decision_vectors: np.ndarray


def resolve_existing_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    candidates = [raw, Path.cwd() / raw, Path(__file__).resolve().parent / raw]
    for c in candidates:
        if c.exists():
            return c.resolve()
    attempted = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"Cannot find file: {path_text}\nAttempted:\n{attempted}")


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Python module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_filename(name: str) -> str:
    name = name.replace("Ours (GR-KANet)", "GR-KANet")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name.strip("_")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "Acc (%)": 100.0 * accuracy_score(y_true, y_pred),
        "Macro-F1 (%)": 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro-P (%)": 100.0 * precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro-R (%)": 100.0 * recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_path: Path,
    show_ticks: bool = True,
    show_title: bool = False,
    title: str = "",
    show_axis_labels: bool = False,
    show_colorbar: bool = False,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    if show_ticks:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show_axis_labels:
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

    if show_title and title:
        ax.set_title(title, fontsize=10, pad=4)

    vmax = float(cm.max()) if cm.size else 0.0
    threshold = vmax / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            value = int(cm[r, c])
            ax.text(c, r, str(value), ha="center", va="center", fontsize=7,
                    color="white" if value > threshold else "black")
    ax.set_ylim(len(labels) - 0.5, -0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_tsne_plot(
    features: np.ndarray,
    y_true: np.ndarray,
    labels: List[str],
    output_path: Path,
    seed: int,
    show_title: bool = False,
    title: str = "",
) -> None:
    if features.ndim != 2:
        raise ValueError("t-SNE features must be 2-D")
    n_samples = features.shape[0]
    perplexity = min(15, max(5, n_samples // 4))
    if perplexity >= n_samples:
        perplexity = max(2, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    Z = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for cls_idx, cls_name in enumerate(labels):
        mask = (y_true == cls_idx)
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=24, alpha=0.85,
            color=TSNE_CLASS_COLORS[cls_idx % len(TSNE_CLASS_COLORS)],
            label=cls_name,
        )
    if show_title and title:
        ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_tsne_legend(labels: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    handles = []
    for cls_idx, cls_name in enumerate(labels):
        h = ax.scatter([], [], s=40, color=TSNE_CLASS_COLORS[cls_idx % len(TSNE_CLASS_COLORS)], label=cls_name)
        handles.append(h)
    ax.legend(handles=handles, labels=labels, loc="center", frameon=False, ncol=2, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def predict_torch_logits(model: nn.Module, X: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X), dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    logits_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []
    amp_enabled = device.type == "cuda" and getattr(torch.cuda.amp, "autocast", None) is not None
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(xb)
            else:
                logits = model(xb)
            logits_np = logits.detach().cpu().numpy()
            logits_list.append(logits_np)
            preds_list.append(np.argmax(logits_np, axis=1))
    logits_all = np.concatenate(logits_list, axis=0)
    preds_all = np.concatenate(preds_list, axis=0)
    return preds_all, logits_all


def train_torch_model_return_model(
    comp,
    model: nn.Module,
    split,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    weighted_sampler: bool,
    class_weighted_ce: bool,
    verbose: bool,
):
    num_classes = len(split.display_labels)
    train_loader, val_loader, _ = comp.make_torch_loaders(
        split.X_train, split.y_train,
        split.X_val, split.y_val,
        split.X_test, split.y_test,
        batch_size=batch_size,
        weighted_sampler=weighted_sampler,
    )
    model = model.to(device)
    if class_weighted_ce:
        w = torch.tensor(comp.class_weights_np(split.y_train, num_classes), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"

    best_state = None
    best_val_f1 = -1.0
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=True):
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

        yv, pv = comp.evaluate_torch(model, val_loader, device)
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
    return model


class FeaturePreferenceMLPWithFeatures(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = 8):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(in_dim))
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logits).view(1, -1)
        z = x * gate
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.extract_features(x)
        return self.fc3(z)


class CNN1DWithFeatures(nn.Module):
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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class LightweightResNet1DWithFeatures(nn.Module):
    def __init__(self, comp, num_classes: int, channels: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            comp.ResidualBlock1D(channels),
            comp.ResidualBlock1D(channels),
            comp.ResidualBlock1D(channels),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x.unsqueeze(1))
        z = self.blocks(z)
        z = self.pool(z).squeeze(-1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


def train_grkanet_return_model(gr, split, seed: int, device: torch.device, args: argparse.Namespace):
    gr.set_seed(seed)
    num_classes = len(split.display_labels)
    train_loader, val_loader, test_loader = gr.make_loaders(split, batch_size=args.batch_size, balance_mode=args.balance_mode)
    model = gr.GRKANet(
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
        class_weights = torch.tensor(gr.compute_class_weights_np(split.y_train, num_classes), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"
    best_state = None
    best_epoch = 0
    best_val_macro_f1 = -1.0
    no_improve = 0
    model.dgam.update_mask_from_kan()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=True):
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
        _, val_res = gr.evaluate_model(model, val_loader, criterion, device)
        val_f1 = val_res.macro_f1
        if args.verbose:
            print(f"  GR-KANet epoch {epoch:03d} | val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_f1
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
    _, test_res = gr.evaluate_model(model, test_loader, criterion, device)
    return model, test_res, best_epoch, best_val_macro_f1


def get_grkanet_logits_and_features(model, X: np.ndarray, device: torch.device):
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X), dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    logits_all = []
    feat_all = []
    pred_all = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            feats = model.extract_kan_features(xb)
            logits_np = logits.detach().cpu().numpy()
            pred_all.append(np.argmax(logits_np, axis=1))
            logits_all.append(logits_np)
            feat_all.append(feats.detach().cpu().numpy())
    return np.concatenate(pred_all), np.concatenate(logits_all), np.concatenate(feat_all)


def train_all_method_artifacts(comp, gr, split_comp, split_gr, device: torch.device, train_seed: int, verbose: bool) -> Tuple[Dict[str, MethodArtifact], nn.Module]:
    artifacts: Dict[str, MethodArtifact] = {}
    num_classes = len(split_comp.display_labels)
    labels = split_comp.display_labels

    # 1 SVM
    comp.set_seed(train_seed)
    svm = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=train_seed)
    svm.fit(split_comp.X_train, split_comp.y_train)
    artifacts["SVM"] = MethodArtifact("SVM", svm.predict(split_comp.X_test), svm.predict_proba(split_comp.X_test))

    # 2 XGBoost
    comp.set_seed(train_seed)
    xgb = comp.make_xgb(num_classes, seed=train_seed)
    sw = comp.sample_weights_np(split_comp.y_train, num_classes)
    try:
        xgb.fit(split_comp.X_train, split_comp.y_train, sample_weight=sw)
    except TypeError:
        xgb.fit(split_comp.X_train, split_comp.y_train)
    artifacts["XGBoost"] = MethodArtifact("XGBoost", xgb.predict(split_comp.X_test), xgb.predict_proba(split_comp.X_test))

    # 3 1D-CNN
    comp.set_seed(train_seed)
    cnn = train_torch_model_return_model(comp, CNN1DWithFeatures(num_classes), split_comp, device,
                                         epochs=150, patience=80, lr=1e-3, batch_size=64,
                                         weighted_sampler=True, class_weighted_ce=True, verbose=verbose)
    pred, logits = predict_torch_logits(cnn, split_comp.X_test, device)
    artifacts["1D-CNN"] = MethodArtifact("1D-CNN", pred, F.softmax(torch.tensor(logits), dim=1).numpy())

    # 4 SMOTE-GBDT
    comp.set_seed(train_seed)
    y_train = split_comp.y_train
    counts = np.bincount(y_train)
    positive_counts = counts[counts > 0]
    min_count = int(positive_counts.min()) if len(positive_counts) else 0
    if comp.SMOTE is None or min_count <= 1:
        X_res, y_res = comp.random_oversample(split_comp.X_train, y_train, train_seed)
    else:
        k_neighbors = min(5, min_count - 1)
        X_res, y_res = comp.SMOTE(random_state=train_seed, k_neighbors=k_neighbors).fit_resample(split_comp.X_train, y_train)
    gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                      min_samples_leaf=2, subsample=0.90, random_state=train_seed)
    gbdt.fit(X_res, y_res)
    artifacts["SMOTE-GBDT"] = MethodArtifact("SMOTE-GBDT", gbdt.predict(split_comp.X_test), gbdt.predict_proba(split_comp.X_test))

    # 5 SMOTE-ENN-XGBoost
    comp.set_seed(train_seed)
    if comp.SMOTEENN is None:
        X_res, y_res = comp.random_oversample(split_comp.X_train, split_comp.y_train, train_seed)
    else:
        resampler = comp.SMOTEENN(random_state=train_seed)
        X_res, y_res = resampler.fit_resample(split_comp.X_train, split_comp.y_train)
    xgb2 = comp.make_xgb(num_classes, seed=train_seed)
    xgb2.fit(X_res, y_res)
    artifacts["SMOTE-ENN-XGBoost"] = MethodArtifact("SMOTE-ENN-XGBoost", xgb2.predict(split_comp.X_test), xgb2.predict_proba(split_comp.X_test))

    # 6 AdaBoost-TCNN
    comp.set_seed(train_seed)
    Xtr = comp.tcnn_feature_map(split_comp.X_train)
    Xte = comp.tcnn_feature_map(split_comp.X_test)
    X_res, y_res = comp.random_oversample(Xtr, split_comp.y_train, train_seed)
    try:
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=train_seed),
                                 n_estimators=120, learning_rate=0.05, random_state=train_seed, algorithm="SAMME")
    except TypeError:
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=train_seed),
                                 n_estimators=120, learning_rate=0.05, random_state=train_seed, algorithm="SAMME")
    ada.fit(X_res, y_res)
    artifacts["AdaBoost-TCNN"] = MethodArtifact("AdaBoost-TCNN", ada.predict(Xte), ada.predict_proba(Xte))

    # 7 Lightweight ResNet
    comp.set_seed(train_seed)
    lres = train_torch_model_return_model(comp, LightweightResNet1DWithFeatures(comp, num_classes, channels=16), split_comp, device,
                                          epochs=100, patience=30, lr=5e-4, batch_size=32,
                                          weighted_sampler=False, class_weighted_ce=True, verbose=verbose)
    pred, logits = predict_torch_logits(lres, split_comp.X_test, device)
    artifacts["Lightweight ResNet"] = MethodArtifact("Lightweight ResNet", pred, F.softmax(torch.tensor(logits), dim=1).numpy())

    # 8 MTF-GhostNetV2
    comp.set_seed(train_seed)
    Xtr = comp.mtf_feature_map(split_comp.X_train)
    Xte = comp.mtf_feature_map(split_comp.X_test)
    clf = comp.fit_xgb_or_rf(Xtr, split_comp.y_train, num_classes, train_seed + 23, n_estimators=160)
    artifacts["MTF-GhostNetV2"] = MethodArtifact("MTF-GhostNetV2", clf.predict(Xte), clf.predict_proba(Xte))

    # 9 Graph-enhanced DGA
    comp.set_seed(train_seed)
    Xtr = comp.graph_feature_map(split_comp.X_train)
    Xte = comp.graph_feature_map(split_comp.X_test)
    clf = comp.fit_xgb_or_rf(Xtr, split_comp.y_train, num_classes, train_seed + 41, n_estimators=140)
    artifacts["Graph-enhanced DGA"] = MethodArtifact("Graph-enhanced DGA", clf.predict(Xte), clf.predict_proba(Xte))

    # 10 DE-IPFL
    comp.set_seed(train_seed)
    X_aug, y_aug = comp.augment_gaussian(split_comp.X_train, split_comp.y_train, copies=3, noise_std=0.04, seed=train_seed)
    aug_split = comp.SplitData(X_aug, split_comp.X_val, split_comp.X_test, y_aug, split_comp.y_val, split_comp.y_test,
                               split_comp.display_labels, split_comp.scaler)
    de_model = train_torch_model_return_model(comp, FeaturePreferenceMLPWithFeatures(num_classes), aug_split, device,
                                              epochs=150, patience=80, lr=1e-3, batch_size=64,
                                              weighted_sampler=True, class_weighted_ce=True, verbose=verbose)
    pred, logits = predict_torch_logits(de_model, split_comp.X_test, device)
    artifacts["DE-IPFL"] = MethodArtifact("DE-IPFL", pred, F.softmax(torch.tensor(logits), dim=1).numpy())

    # 11 Knowledge-filtered DSL
    comp.set_seed(train_seed)
    Xtr = comp.knowledge_features(split_comp.X_train)
    Xte = comp.knowledge_features(split_comp.X_test)
    ytr = split_comp.y_train
    if comp.SMOTE is not None:
        X_res, y_res = comp.SMOTE(random_state=train_seed, k_neighbors=3).fit_resample(Xtr, ytr)
    else:
        X_res, y_res = comp.random_oversample(Xtr, ytr, train_seed)
    estimators = [
        ("svm", SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=train_seed)),
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=None, class_weight="balanced", random_state=train_seed)),
        ("xgb", comp.make_xgb(num_classes, seed=train_seed, n_estimators=180)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=train_seed)),
    ]
    dsl = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=train_seed),
        stack_method="auto", cv=3, n_jobs=1,
    )
    dsl.fit(X_res, y_res)
    artifacts["Knowledge-filtered DSL"] = MethodArtifact("Knowledge-filtered DSL", dsl.predict(Xte), dsl.predict_proba(Xte))

    # 12 Knowledge-CapsNet
    comp.set_seed(train_seed)
    kcaps = train_torch_model_return_model(comp, comp.KnowledgeCapsNet(num_classes), split_comp, device,
                                           epochs=150, patience=80, lr=1e-3, batch_size=64,
                                           weighted_sampler=True, class_weighted_ce=True, verbose=verbose)
    pred, logits = predict_torch_logits(kcaps, split_comp.X_test, device)
    artifacts["Knowledge-CapsNet"] = MethodArtifact("Knowledge-CapsNet", pred, F.softmax(torch.tensor(logits), dim=1).numpy())

    # 13 GR-KANet
    gr_args = argparse.Namespace(
        batch_size=64, balance_mode="both", dgam_mode="energy", alpha_init=2.0,
        dropout=0.0, channels=32, kan_hidden=8, kan_latent=64, dgam_ema=0.8,
        lr=1e-3, epochs=150, patience=80, mask_update_every=20, verbose=verbose,
    )
    gr_model, gr_test_res, best_epoch, best_val_f1 = train_grkanet_return_model(gr, split_gr, train_seed, device, gr_args)
    pred, logits, gr_feats = get_grkanet_logits_and_features(gr_model, split_gr.X_test, device)
    artifacts["Ours (GR-KANet)"] = MethodArtifact("Ours (GR-KANet)", pred, F.softmax(torch.tensor(logits), dim=1).numpy())
    print(f"GR-KANet | seed={train_seed} | best_epoch={best_epoch} | valF1={100*best_val_f1:.2f} | test Acc={100*gr_test_res.acc:.2f} | test Macro-F1={100*gr_test_res.macro_f1:.2f}")

    return artifacts, gr_model


def compute_grkanet_scores(model, mode: str = "energy", reference_points: int = 200, entropy_bins: int = 30):
    C = model.channels
    G = model.num_gases

    # Put the reference grid on the same device and with the same dtype as
    # the trained model. Otherwise a CUDA model cannot process a CPU grid.
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype
    grid = torch.linspace(
        -3.0,
        3.0,
        reference_points,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)

    model.eval()
    derivs = []
    responses = []
    for f_j in model.kan.functions:
        with torch.no_grad():
            y = f_j(grid).squeeze(1)

        resp = (y + grid.squeeze(1)).detach().cpu().numpy()
        dy = torch.gradient(y)[0].detach().cpu().numpy()
        responses.append(resp)
        derivs.append(dy)

    responses = np.asarray(responses, dtype=np.float32).reshape(C, G, reference_points)
    derivs = np.asarray(derivs, dtype=np.float32).reshape(C, G, reference_points)
    if mode == "energy":
        scores = np.mean(np.square(derivs), axis=(1, 2)).astype(np.float32)
    else:
        scores_list = []
        for c in range(C):
            curve = np.abs(derivs[c].reshape(-1))
            hist, _ = np.histogram(curve, bins=entropy_bins, density=False)
            p = hist.astype(np.float64)
            p = p / (p.sum() + 1e-12)
            scores_list.append(float(-np.sum(p * np.log(p + 1e-12))))
        scores = np.asarray(scores_list, dtype=np.float32)
    return grid.squeeze(1).detach().cpu().numpy(), responses, derivs, scores


def plot_grkanet_mask(model, output_path: Path):
    mask = model.dgam.kan_mask.detach().cpu().numpy().astype(float)
    norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    x = np.arange(1, len(norm) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(x, norm, width=0.8)
    ax.set_xlabel("Latent channel index")
    ax.set_ylabel("Normalized weight")
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


GAS_NAMES = ["H2", "CH4", "C2H4", "C2H2", "CO", "CO2", "THC", "C2H6"]

TSNE_CLASS_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def plot_grkanet_functions_and_derivatives(model, output_func_dir: Path, output_deriv_dir: Path):
    grid, responses, derivs, energy_scores = compute_grkanet_scores(model, mode="energy")
    top_channel = int(np.argmax(energy_scores))

    output_func_dir.mkdir(parents=True, exist_ok=True)
    output_deriv_dir.mkdir(parents=True, exist_ok=True)

    for g_idx, gas in enumerate(GAS_NAMES):
        # Response-function subplot saved as an independent figure
        fig1, ax1 = plt.subplots(figsize=(4.2, 3.6))
        ax1.plot(grid, responses[top_channel, g_idx], linewidth=1.8)
        ax1.set_title(gas, fontsize=12)
        ax1.grid(alpha=0.25)
        fig1.tight_layout(pad=0.3)
        fig1.savefig(
            output_func_dir / f"response_{g_idx + 1:02d}_{gas}.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig1)

        # Derivative-curve subplot saved as an independent figure
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
        ax2.plot(grid, derivs[top_channel, g_idx], linewidth=1.8)
        ax2.set_title(gas, fontsize=12)
        ax2.grid(alpha=0.25)
        fig2.tight_layout(pad=0.3)
        fig2.savefig(
            output_deriv_dir / f"derivative_{g_idx + 1:02d}_{gas}.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig2)


def plot_grkanet_scores(model, output_energy: Path, output_entropy: Path):
    _, _, _, energy = compute_grkanet_scores(model, mode="energy")
    _, _, _, entropy = compute_grkanet_scores(model, mode="entropy")
    for scores, path, ylabel in [
        (energy, output_energy, "Score by derivative energy"),
        (entropy, output_entropy, "Score by entropy"),
    ]:
        x = np.arange(1, len(scores) + 1)
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.bar(x, scores, width=0.8)
        ax.set_xlabel("Latent channel index")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        fig.tight_layout()
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a clean Hunan seed-43 visual suite.")
    parser.add_argument("--data", type=str, default="胖虎电厂数据未加测试集.xlsx")
    parser.add_argument("--label-col", type=str, default="故障编码")
    parser.add_argument("--comparison-script", type=str, default="run_ajse_hunan_fixedsplit_10runs.py")
    parser.add_argument("--grkanet-script", type=str, default="run_grkanet_full_10runs_final.py")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=43)
    parser.add_argument("--outdir", type=str, default="hunan_seed43_visual_suite")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-cm-titles", action="store_true")
    parser.add_argument("--show-cm-axis-labels", action="store_true")
    parser.add_argument("--show-cm-colorbar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_existing_path(args.data)
    comparison_path = resolve_existing_path(args.comparison_script)
    grkanet_path = resolve_existing_path(args.grkanet_script)

    outdir = Path(args.outdir).expanduser().resolve()
    (outdir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    (outdir / "grkanet_visuals").mkdir(parents=True, exist_ok=True)
    (outdir / "decision_tsne").mkdir(parents=True, exist_ok=True)

    comp = import_module_from_path("visual_comp_hunan", comparison_path)
    gr = import_module_from_path("visual_gr_hunan", grkanet_path)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {data_path}")
    print(f"Fixed split seed: {args.split_seed}")
    print(f"Shared training seed: {args.train_seed}")

    split_comp = comp.load_and_split_data(data_path, label_col=args.label_col, seed=args.split_seed)
    split_gr = gr.load_fixed_split(str(data_path), label_col=args.label_col, split_seed=args.split_seed)

    if not np.array_equal(split_comp.y_test, split_gr.y_test):
        raise RuntimeError("The two formal scripts do not produce the same test labels.")

    artifacts, gr_model = train_all_method_artifacts(comp, gr, split_comp, split_gr, device, args.train_seed, args.verbose)
    labels = list(split_comp.display_labels)
    y_true = split_comp.y_test

    metric_rows = []
    pred_table = pd.DataFrame({"True index": y_true, "True label": [labels[int(v)] for v in y_true]})

    # Raw-feature t-SNE panel (the additional initial plot)
    raw_tsne_path = outdir / "decision_tsne" / "00_raw_gas_features_tsne.svg"
    save_tsne_plot(
        split_comp.X_test,
        y_true,
        labels,
        raw_tsne_path,
        seed=args.train_seed,
        show_title=False,
        title="Raw gas features",
    )

    for order, method in METHOD_ORDER:
        art = artifacts[method]
        cm_path = outdir / "confusion_matrices" / f"{order}_{safe_filename(method)}_confusion_matrix.svg"
        save_confusion_matrix(
            y_true, art.y_pred, labels, cm_path,
            show_ticks=True, show_title=args.show_cm_titles, title=method,
            show_axis_labels=args.show_cm_axis_labels, show_colorbar=args.show_cm_colorbar,
        )
        tsne_path = outdir / "decision_tsne" / f"{order}_{safe_filename(method)}_decision_tsne.svg"
        save_tsne_plot(art.decision_vectors, y_true, labels, tsne_path, seed=args.train_seed,
                       show_title=False, title=method)
        metrics = compute_metrics(y_true, art.y_pred)
        metrics.update({"Order": int(order), "Method": method, "Training seed": args.train_seed})
        metric_rows.append(metrics)
        pred_table[f"{method} index"] = art.y_pred
        pred_table[f"{method} label"] = [labels[int(v)] for v in art.y_pred]
        print(f"{method:28s} | Acc={metrics['Acc (%)']:.2f} | Macro-F1={metrics['Macro-F1 (%)']:.2f}")

    # GR-KANet-only visual figures
    plot_grkanet_mask(gr_model, outdir / "grkanet_visuals" / "grkanet_seed43_channel_mask.svg")
    plot_grkanet_functions_and_derivatives(
        gr_model,
        outdir / "grkanet_visuals" / "kan_response_functions",
        outdir / "grkanet_visuals" / "kan_derivative_curves",
    )
    plot_grkanet_scores(
        gr_model,
        outdir / "grkanet_visuals" / "grkanet_seed43_channel_scores_energy.svg",
        outdir / "grkanet_visuals" / "grkanet_seed43_channel_scores_entropy.svg",
    )

    # 15th t-SNE panel: legend only
    save_tsne_legend(labels, outdir / "decision_tsne" / "15_tsne_legend.svg")

    metrics_df = pd.DataFrame(metric_rows).sort_values("Order")
    metrics_df.to_csv(outdir / "seed43_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pred_table.to_csv(outdir / "seed43_predictions.csv", index=False, encoding="utf-8-sig")

    readme = [
        "Hunan seed-43 visual suite",
        f"Dataset: {data_path}",
        f"Fixed split seed: {args.split_seed}",
        f"Shared training seed: {args.train_seed}",
        "",
        "Subfolders:",
        "  confusion_matrices/   -> 13 confusion matrices",
        "  grkanet_visuals/      -> Fig. 6/7/8-style GR-KANet visuals (including 8 individual response plots and 8 individual derivative plots)",
        "  decision_tsne/        -> 14 t-SNE plots (1 raw-feature + 13 method-level decision-space plots) plus 1 standalone legend panel",
        "",
        "Note:",
        "  decision_tsne/ includes 00_raw_gas_features_tsne.svg, 01-13 method-level t-SNE plots, and 15_tsne_legend.png.",
    ]
    (outdir / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    print(f"\nSaved all outputs to: {outdir}")


if __name__ == "__main__":
    main()
