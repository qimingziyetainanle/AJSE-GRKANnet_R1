# -*- coding: utf-8 -*-
"""
GR-KANet computational-cost benchmark for revised Table 7.

Table 7 remains compact and reports only:
    Model | Params (K) | Train time (s/epoch) | Inference (ms/sample)

Hardware/software information is saved separately for the reviewer response,
but is not added to the LaTeX table.

Final protocol:
    - fixed split seed: 42
    - training seed: 42
    - mask update interval: 20 global optimizer steps
    - DGAM mode: energy
    - alpha: 2.0
    - DGAM EMA: 0.8
    - dropout: 0.0
    - learning rate: 1e-3
    - batch size: 64
    - weighted sampler + class-weighted cross-entropy
    - checkpoint selected by validation Macro-F1
    - batch-size-1 inference with the converged fixed mask

Place this script in the same folder as:
    run_grkanet_full_10runs_final.py
    胖虎电厂数据未加测试集.xlsx

Run:
    python run_grkanet_table7_benchmark.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GR-KANet parameter count, training time, and inference latency."
    )
    parser.add_argument(
        "--source-module",
        type=str,
        default="run_grkanet_full_10runs_final",
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
        default="grkanet_table7_benchmark_outputs",
    )

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--mask-update-every", type=int, default=20)
    parser.add_argument("--dgam-ema", type=float, default=0.8)
    parser.add_argument("--alpha-init", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kan-hidden", type=int, default=8)
    parser.add_argument("--kan-latent", type=int, default=64)

    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training and inference to run on CPU.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def create_model(source, num_classes: int, args, device: torch.device) -> nn.Module:
    return source.GRKANet(
        num_classes=num_classes,
        dgam_mode="energy",
        alpha_init=args.alpha_init,
        dropout=args.dropout,
        channels=args.channels,
        kan_hidden=args.kan_hidden,
        kan_latent=args.kan_latent,
        dgam_ema=args.dgam_ema,
    ).to(device)


def train_and_time(
    source,
    split,
    args,
    device: torch.device,
    outdir: Path,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float], Path]:
    """
    Train one validation-selected final model.

    Training time per epoch includes:
        forward pass, backward pass, optimizer update, and mask updates.

    Validation time is not included.
    """
    set_seed(args.training_seed)

    train_loader, val_loader, _ = source.make_loaders(
        split,
        batch_size=args.batch_size,
        balance_mode="both",
    )

    num_classes = len(split.display_labels)
    model = create_model(source, num_classes, args, device)

    class_weights = torch.tensor(
        source.compute_class_weights_np(split.y_train, num_classes),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    amp_enabled = device.type == "cuda" and autocast is not None
    scaler = (
        GradScaler(enabled=True)
        if amp_enabled and GradScaler is not None
        else None
    )

    model.dgam.update_mask_from_kan()
    global_step = 0

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_f1 = -1.0
    no_improve = 0
    epoch_rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        synchronize(device)
        start = time.perf_counter()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                assert scaler is not None
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

        synchronize(device)
        train_time = time.perf_counter() - start

        _, val_result = source.evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )
        val_f1 = float(val_result.macro_f1)

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

        epoch_rows.append(
            {
                "Epoch": epoch,
                "Training time (s)": train_time,
                "Validation Macro-F1 (%)": 100.0 * val_f1,
                "Global optimizer steps": global_step,
                "Mask update interval": args.mask_update_every,
                "Best checkpoint": int(improved),
            }
        )

        if args.verbose:
            print(
                "Epoch {:03d} | train={:.4f} s | val F1={:.2f}%{}".format(
                    epoch,
                    train_time,
                    100.0 * val_f1,
                    " | BEST" if improved else "",
                )
            )

        if no_improve >= args.patience:
            print("Early stopping at epoch {}.".format(epoch))
            break

    if best_state is None:
        raise RuntimeError("No validation-selected checkpoint was obtained.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    checkpoint_path = outdir / "best_grkanet_mask20_seed42.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "mask_update_every": args.mask_update_every,
            "training_seed": args.training_seed,
            "split_seed": args.split_seed,
        },
        checkpoint_path,
    )

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv(
        outdir / "training_epoch_times.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    all_times = epoch_df["Training time (s)"].astype(float).tolist()
    summary_times = all_times[1:] if len(all_times) > 1 else all_times

    timing_summary = {
        "Mean train time (s/epoch)": statistics.fmean(summary_times),
        "Sample std train time (s/epoch)": sample_std(summary_times),
        "Median train time (s/epoch)": statistics.median(summary_times),
        "Completed epochs": int(len(epoch_df)),
        "Best epoch": int(best_epoch),
        "Best validation Macro-F1 (%)": 100.0 * best_val_f1,
    }

    return model, epoch_df, timing_summary, checkpoint_path


def benchmark_inference(
    model: nn.Module,
    split,
    args,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Measure batch-size-1 model inference latency.

    Scope:
        standardized single-sample tensor
        -> GR-KANet forward pass
        -> argmax prediction

    The converged mask is fixed during inference.
    """
    model.eval()

    sample = torch.tensor(
        split.X_test[0:1],
        dtype=torch.float32,
        device=device,
    )

    rows: List[Dict[str, float]] = []

    with torch.no_grad():
        for _ in range(args.warmup):
            logits = model(sample)
            _ = logits.argmax(dim=1)
        synchronize(device)

        for repeat in range(1, args.repeats + 1):
            synchronize(device)
            start = time.perf_counter()

            logits = model(sample)
            _ = logits.argmax(dim=1)

            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rows.append(
                {
                    "Repeat": repeat,
                    "Inference latency (ms/sample)": elapsed_ms,
                }
            )

    latency_df = pd.DataFrame(rows)
    values = latency_df["Inference latency (ms/sample)"].astype(float).tolist()

    summary = {
        "Mean inference (ms/sample)": statistics.fmean(values),
        "Sample std inference (ms/sample)": sample_std(values),
        "Median inference (ms/sample)": statistics.median(values),
        "P95 inference (ms/sample)": percentile(values, 95.0),
        "Min inference (ms/sample)": min(values),
        "Max inference (ms/sample)": max(values),
        "Throughput (samples/s)": 1000.0 / statistics.fmean(values),
        "Warm-up repetitions": int(args.warmup),
        "Recorded repetitions": int(args.repeats),
    }

    return latency_df, summary


def get_hardware_info(device: torch.device) -> Dict[str, object]:
    info: Dict[str, object] = {
        "Operating system": platform.platform(),
        "Processor": platform.processor(),
        "Logical CPU cores": os.cpu_count(),
        "Python version": sys.version.replace("\n", " "),
        "PyTorch version": torch.__version__,
        "Benchmark device": str(device),
        "CUDA available": torch.cuda.is_available(),
        "CUDA version": torch.version.cuda,
    }

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["GPU"] = torch.cuda.get_device_name(device)
        info["GPU memory (GB)"] = properties.total_memory / (1024.0 ** 3)

    return info


def write_outputs(
    model: nn.Module,
    training_summary: Dict[str, float],
    inference_summary: Dict[str, float],
    latency_df: pd.DataFrame,
    checkpoint_path: Path,
    args,
    device: torch.device,
    outdir: Path,
) -> None:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_k = params / 1000.0

    summary_row = {
        "Model": "Ours (GR-KANet)",
        "Params (K)": params_k,
        **training_summary,
        **inference_summary,
        "Mask update interval during training": args.mask_update_every,
        "Inference batch size": 1,
        "Inference mask": "Converged fixed mask",
        "Device": str(device),
    }

    pd.DataFrame([summary_row]).to_csv(
        outdir / "table7_timing_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    latency_df.to_csv(
        outdir / "inference_latency_raw.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    hardware = get_hardware_info(device)
    (outdir / "hardware_info.txt").write_text(
        "\n".join("{}: {}".format(k, v) for k, v in hardware.items()),
        encoding="utf-8",
    )
    (outdir / "hardware_info.json").write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_mean = float(training_summary["Mean train time (s/epoch)"])
    inference_mean = float(inference_summary["Mean inference (ms/sample)"])

    latex_row = (
        "Ours (GR-KANet) & {:.1f} & {:.2f} & {:.1f} \\\\".format(
            params_k,
            train_mean,
            inference_mean,
        )
    )
    (outdir / "table7_latex_row.txt").write_text(
        latex_row,
        encoding="utf-8",
    )

    latex_table = r"""\begin{table}[!htbp]
    \centering
    \scriptsize
    \caption{Computational cost of GR-KANet on the Hunan-Plant dataset under the final experimental configuration. The derivative-guided mask is updated every 20 global optimizer steps during training, and inference latency is measured with batch size 1 using the converged fixed mask.}
    \label{tab:computational-cost}
    \begin{tabular}{@{}l|ccc@{}}
        \toprule
        Model & Params (K) & Train time (s/epoch) & Inference (ms/sample) \\
        \midrule
        """ + latex_row + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    (outdir / "table7_updated.tex").write_text(
        latex_table,
        encoding="utf-8",
    )

    report_lines = [
        "GR-KANet Table 7 benchmark",
        "=" * 32,
        "",
        "Final configuration:",
        "- Mask update interval: 20 global optimizer steps",
        "- Training seed: {}".format(args.training_seed),
        "- Split seed: {}".format(args.split_seed),
        "- Checkpoint: selected by validation Macro-F1",
        "- Inference batch size: 1",
        "- Inference mask: converged and fixed",
        "",
        "Table 7 values:",
        "- Params (K): {:.1f}".format(params_k),
        "- Train time (s/epoch): {:.4f}".format(train_mean),
        "- Inference (ms/sample): {:.4f}".format(inference_mean),
        "",
        "Additional timing record for reviewer response:",
        "- Training-time sample std: {:.4f} s/epoch".format(
            training_summary["Sample std train time (s/epoch)"]
        ),
        "- Inference sample std: {:.4f} ms/sample".format(
            inference_summary["Sample std inference (ms/sample)"]
        ),
        "- Median inference: {:.4f} ms/sample".format(
            inference_summary["Median inference (ms/sample)"]
        ),
        "- P95 inference: {:.4f} ms/sample".format(
            inference_summary["P95 inference (ms/sample)"]
        ),
        "- Throughput: {:.2f} samples/s".format(
            inference_summary["Throughput (samples/s)"]
        ),
        "",
        "Hardware details are stored in hardware_info.txt and are not included in Table 7.",
        "Checkpoint: {}".format(checkpoint_path),
    ]
    (outdir / "table7_benchmark_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.mask_update_every != 20:
        raise ValueError(
            "The revised final protocol requires --mask-update-every 20."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    source = importlib.import_module(args.source_module)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("Mask update interval:", args.mask_update_every)
    print("Data:", args.data)

    split = source.load_fixed_split(
        args.data,
        label_col=args.label_col,
        split_seed=args.split_seed,
    )
    print(
        "Split sizes: train={}, val={}, test={}".format(
            len(split.y_train),
            len(split.y_val),
            len(split.y_test),
        )
    )

    model, _, training_summary, checkpoint_path = train_and_time(
        source=source,
        split=split,
        args=args,
        device=device,
        outdir=outdir,
    )

    latency_df, inference_summary = benchmark_inference(
        model=model,
        split=split,
        args=args,
        device=device,
    )

    write_outputs(
        model=model,
        training_summary=training_summary,
        inference_summary=inference_summary,
        latency_df=latency_df,
        checkpoint_path=checkpoint_path,
        args=args,
        device=device,
        outdir=outdir,
    )

    print("\nFinished.")
    print(
        "Train time = {:.4f} s/epoch".format(
            training_summary["Mean train time (s/epoch)"]
        )
    )
    print(
        "Inference = {:.4f} ms/sample".format(
            inference_summary["Mean inference (ms/sample)"]
        )
    )
    print("Saved:", outdir / "table7_updated.tex")
    print("Saved:", outdir / "table7_timing_summary.csv")
    print("Saved:", outdir / "hardware_info.txt")
    print("Saved:", outdir / "table7_benchmark_report.txt")


if __name__ == "__main__":
    main()
