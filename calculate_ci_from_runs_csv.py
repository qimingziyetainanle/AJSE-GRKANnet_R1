# -*- coding: utf-8 -*-
"""
Recursively find per-run CSV files, calculate mean, sample standard deviation,
and two-sided 95% confidence intervals, verify existing summary CSVs when
possible, and export CSV/LaTeX results.

Default behavior
----------------
1. Scan the current directory and all subdirectories.
2. Process CSV files whose names contain "run" or "runs".
3. Skip summary/ranking/metadata/output CSV files.
4. Use Student's t distribution:
      n = 10 -> df = 9
      n = 5  -> df = 4
   Other run counts are also supported automatically with df = n - 1.
5. Never delete, filter, or replace any random seed.

Recommended placement
---------------------
Put this script in the parent folder containing folders such as:
    repeated_runs/
    comparison_results/
    ablation_results/
Then run:
    python calculate_ci_from_runs_csv.py

Optional examples
-----------------
    python calculate_ci_from_runs_csv.py --root "D:/GRKANet_Final_Revision_10runs"
    python calculate_ci_from_runs_csv.py --root . --outdir ci_results
    python calculate_ci_from_runs_csv.py --root . --include-all-csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None


# Exact two-sided 95% critical values requested by the user.
T_CRITICAL_95 = {
    5: 2.7764451051977987,   # df = 4
    10: 2.2621571627409915,  # df = 9
}

GROUP_COLUMN_CANDIDATES = [
    "Variant",
    "Method",
    "Model",
    "Setting",
    "Configuration",
    "Config",
    "Approach",
    "Dataset",
    "Experiment",
    "Prior",
    "Alpha",
    "Strategy",
]

EXCLUDE_NUMERIC_COLUMN_TOKENS = [
    "run",
    "seed",
    "epoch",
    "time",
    "second",
    "minute",
    "id",
    "index",
    "fold",
    "split",
    "source",
    "rank",
    "parameter",
    "param",
]

SKIP_FILE_TOKENS = [
    "summary",
    "ranking",
    "metadata",
    "matrix",
    "matrices",
    "selected",
    "configuration",
    "config",
    "paired_statistics",
    "confidence_interval",
    "ci_results",
    "verification",
    "master_statistics",
    "latex",
]


def normalize_name(value: object) -> str:
    """Normalize a name for tolerant column/file matching."""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with common encodings."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(
        f"Could not read {path}. Tried utf-8-sig, utf-8, gb18030, gbk.\n"
        + "\n".join(errors)
    )


def discover_csv_files(root: Path, include_all_csv: bool) -> List[Path]:
    """Find candidate per-run CSV files recursively."""
    files: List[Path] = []
    for path in root.rglob("*.csv"):
        stem = path.stem.lower()

        if any(token in stem for token in SKIP_FILE_TOKENS):
            continue

        if include_all_csv or "run" in stem:
            files.append(path)

    return sorted(files)


def detect_group_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect grouping columns.

    Usually there is one grouping column such as Variant or Method. If none is
    found, the whole file is treated as one group.
    """
    normalized_to_actual = {normalize_name(col): col for col in df.columns}

    found: List[str] = []
    for candidate in GROUP_COLUMN_CANDIDATES:
        key = normalize_name(candidate)
        if key in normalized_to_actual:
            found.append(normalized_to_actual[key])

    # Prefer the first clear semantic grouping column.
    return found[:1]


def is_metric_column(series: pd.Series, column_name: str) -> bool:
    """Return True when a numeric column looks like an evaluation metric."""
    if not pd.api.types.is_numeric_dtype(series):
        return False

    normalized = normalize_name(column_name)

    if any(token in normalized for token in EXCLUDE_NUMERIC_COLUMN_TOKENS):
        return False

    # Keep common metric names and other numeric outcome columns.
    metric_tokens = [
        "acc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro",
        "micro",
        "auc",
        "far",
        "specificity",
        "sensitivity",
        "valf1",
        "score",
    ]
    if any(token in normalized for token in metric_tokens):
        return True

    # Conservative fallback: numeric columns expressed as percentages.
    if "%" in str(column_name):
        return True

    return False


def detect_metric_columns(df: pd.DataFrame) -> List[str]:
    metrics = [
        col for col in df.columns
        if is_metric_column(df[col], str(col))
    ]

    if not metrics:
        raise ValueError(
            "No metric columns were detected. Expected numeric columns such as "
            "Acc (%), Macro-F1 (%), Macro-P (%), or Macro-R (%)."
        )
    return metrics


def t_critical_95(n: int) -> float:
    """Two-sided 95% Student-t critical value."""
    if n < 2:
        raise ValueError("At least two runs are required for a confidence interval.")

    if n in T_CRITICAL_95:
        return T_CRITICAL_95[n]

    if student_t is None:
        raise RuntimeError(
            f"Run count n={n} is not 5 or 10 and scipy is unavailable. "
            "Install scipy or use 5/10 runs."
        )

    return float(student_t.ppf(0.975, df=n - 1))


def calculate_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Calculate mean, sample std, and two-sided 95% t confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    n = int(arr.size)
    if n < 2:
        raise ValueError(f"Only {n} valid values were found; at least 2 are required.")

    mean_value = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1))
    df = n - 1
    t_value = t_critical_95(n)
    standard_error = sample_std / math.sqrt(n)
    half_width = t_value * standard_error

    return {
        "N": n,
        "df": df,
        "t critical (95%)": t_value,
        "Mean": mean_value,
        "Sample Std": sample_std,
        "Standard Error": standard_error,
        "95% CI Half-width": half_width,
        "95% CI Lower": mean_value - half_width,
        "95% CI Upper": mean_value + half_width,
    }


def iter_groups(
    df: pd.DataFrame,
    group_columns: Sequence[str],
) -> Iterable[Tuple[str, pd.DataFrame]]:
    if not group_columns:
        yield "All runs", df
        return

    group_col = group_columns[0]
    for group_value, group_df in df.groupby(group_col, sort=False, dropna=False):
        yield str(group_value), group_df


def process_run_file(path: Path, root: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Calculate statistics for one per-run CSV."""
    df = read_csv_robust(path)
    group_columns = detect_group_columns(df)
    metric_columns = detect_metric_columns(df)

    rows: List[Dict[str, object]] = []

    for group_name, group_df in iter_groups(df, group_columns):
        for metric in metric_columns:
            numeric_values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            if len(numeric_values) < 2:
                continue

            stats = calculate_statistics(numeric_values.to_numpy(dtype=float))
            rows.append(
                {
                    "Source File": str(path.relative_to(root)),
                    "Group": group_name,
                    "Metric": metric,
                    **stats,
                }
            )

    if not rows:
        raise ValueError("No group/metric combination contained at least two valid runs.")

    metadata = {
        "path": path,
        "group_columns": group_columns,
        "metric_columns": metric_columns,
        "raw_rows": len(df),
    }
    return pd.DataFrame(rows), metadata


def find_existing_summary(run_path: Path) -> Optional[Path]:
    """Find a likely sibling summary CSV."""
    run_stem_norm = normalize_name(run_path.stem)
    candidates = []

    for path in run_path.parent.glob("*.csv"):
        if path == run_path:
            continue
        if "summary" not in path.stem.lower():
            continue

        summary_norm = normalize_name(path.stem)

        # Common-prefix score after removing run/result/summary words.
        base_run = re.sub(r"(runs?|results?|summary)", "", run_stem_norm)
        base_summary = re.sub(r"(runs?|results?|summary)", "", summary_norm)
        score = len(set(base_run) & set(base_summary))
        if base_run and base_summary and (base_run in base_summary or base_summary in base_run):
            score += 100

        candidates.append((score, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return candidates[0][1]


def find_summary_column(
    summary_df: pd.DataFrame,
    metric_name: str,
    stat_kind: str,
) -> Optional[str]:
    """
    Find a summary column corresponding to a metric mean or std.

    Examples:
        Acc (%) Mean
        Macro-F1 (%) Std
        Accuracy Mean
    """
    metric_norm = normalize_name(metric_name)
    stat_tokens = {
        "mean": ("mean", "average", "avg"),
        "std": ("std", "stdev", "standarddeviation", "sd"),
    }[stat_kind]

    matches: List[Tuple[int, str]] = []
    for col in summary_df.columns:
        col_norm = normalize_name(col)
        if metric_norm and metric_norm not in col_norm:
            continue
        if not any(token in col_norm for token in stat_tokens):
            continue
        score = len(metric_norm)
        matches.append((score, col))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], item[1]))
    return matches[0][1]


def detect_summary_group_column(
    summary_df: pd.DataFrame,
    computed_groups: Sequence[str],
) -> Optional[str]:
    if len(computed_groups) == 1 and computed_groups[0] == "All runs":
        return None

    normalized_groups = {normalize_name(group) for group in computed_groups}

    for col in summary_df.columns:
        values = {
            normalize_name(value)
            for value in summary_df[col].dropna().astype(str).tolist()
        }
        if normalized_groups & values:
            return col

    return None


def verify_existing_summary(
    stats_df: pd.DataFrame,
    run_path: Path,
    tolerance: float,
) -> pd.DataFrame:
    """
    Compare recomputed mean/std against an existing sibling summary CSV.

    The verification is best-effort because summary schemas vary.
    """
    summary_path = find_existing_summary(run_path)
    rows: List[Dict[str, object]] = []

    if summary_path is None:
        for _, row in stats_df.iterrows():
            rows.append(
                {
                    "Run File": run_path.name,
                    "Summary File": "",
                    "Group": row["Group"],
                    "Metric": row["Metric"],
                    "Status": "No sibling summary CSV found",
                }
            )
        return pd.DataFrame(rows)

    summary_df = read_csv_robust(summary_path)
    groups = stats_df["Group"].astype(str).unique().tolist()
    summary_group_col = detect_summary_group_column(summary_df, groups)

    for _, computed in stats_df.iterrows():
        group_name = str(computed["Group"])
        metric_name = str(computed["Metric"])

        subset = summary_df
        if summary_group_col is not None:
            wanted = normalize_name(group_name)
            mask = summary_df[summary_group_col].astype(str).map(normalize_name) == wanted
            subset = summary_df.loc[mask]

        if subset.empty:
            rows.append(
                {
                    "Run File": run_path.name,
                    "Summary File": summary_path.name,
                    "Group": group_name,
                    "Metric": metric_name,
                    "Status": "Group not found in summary",
                }
            )
            continue

        mean_col = find_summary_column(summary_df, metric_name, "mean")
        std_col = find_summary_column(summary_df, metric_name, "std")

        if mean_col is None or std_col is None:
            rows.append(
                {
                    "Run File": run_path.name,
                    "Summary File": summary_path.name,
                    "Group": group_name,
                    "Metric": metric_name,
                    "Status": "Mean/std columns not detected in summary",
                }
            )
            continue

        existing_mean = pd.to_numeric(subset.iloc[0][mean_col], errors="coerce")
        existing_std = pd.to_numeric(subset.iloc[0][std_col], errors="coerce")

        calc_mean = float(computed["Mean"])
        calc_std = float(computed["Sample Std"])

        mean_diff = (
            float(existing_mean) - calc_mean
            if pd.notna(existing_mean) else np.nan
        )
        std_diff = (
            float(existing_std) - calc_std
            if pd.notna(existing_std) else np.nan
        )

        passed = (
            pd.notna(existing_mean)
            and pd.notna(existing_std)
            and abs(mean_diff) <= tolerance
            and abs(std_diff) <= tolerance
        )

        rows.append(
            {
                "Run File": run_path.name,
                "Summary File": summary_path.name,
                "Group": group_name,
                "Metric": metric_name,
                "Existing Mean": existing_mean,
                "Recomputed Mean": calc_mean,
                "Mean Difference": mean_diff,
                "Existing Std": existing_std,
                "Recomputed Sample Std": calc_std,
                "Std Difference": std_diff,
                "Tolerance": tolerance,
                "Status": "PASS" if passed else "CHECK",
            }
        )

    return pd.DataFrame(rows)


def latex_escape(text: object) -> str:
    value = str(text)
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
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def make_long_latex_table(stats_df: pd.DataFrame) -> str:
    """Create one long LaTeX table containing every computed statistic."""
    lines = [
        r"\begin{longtable}{llllll}",
        r"\caption{Repeated-run statistics with two-sided 95\% Student's $t$ confidence intervals.}\\",
        r"\toprule",
        r"File & Group & Metric & $n$ & Mean $\pm$ SD & 95\% CI \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"File & Group & Metric & $n$ & Mean $\pm$ SD & 95\% CI \\",
        r"\midrule",
        r"\endhead",
    ]

    for _, row in stats_df.iterrows():
        lines.append(
            "{} & {} & {} & {} & ${:.2f} \\pm {:.2f}$ & "
            "$[{:.2f},\\,{:.2f}]$ \\\\".format(
                latex_escape(row["Source File"]),
                latex_escape(row["Group"]),
                latex_escape(row["Metric"]),
                int(row["N"]),
                float(row["Mean"]),
                float(row["Sample Std"]),
                float(row["95% CI Lower"]),
                float(row["95% CI Upper"]),
            )
        )

    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def make_wide_latex_table(
    file_stats: pd.DataFrame,
    source_label: str,
) -> str:
    """
    Create a paper-friendly table with one row per group and one column per metric.

    Each cell contains:
        mean ± sample SD [95% CI lower, upper]
    """
    metrics = file_stats["Metric"].drop_duplicates().tolist()
    groups = file_stats["Group"].drop_duplicates().tolist()

    col_spec = "l" + "c" * len(metrics)
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Variant & " + " & ".join(latex_escape(metric) for metric in metrics) + r" \\",
        r"\midrule",
    ]

    for group in groups:
        group_df = file_stats[file_stats["Group"] == group]
        cells = []
        for metric in metrics:
            match = group_df[group_df["Metric"] == metric]
            if match.empty:
                cells.append("--")
                continue
            row = match.iloc[0]
            cells.append(
                "${:.2f} \\pm {:.2f}\\;[{:.2f},\\,{:.2f}]$".format(
                    float(row["Mean"]),
                    float(row["Sample Std"]),
                    float(row["95% CI Lower"]),
                    float(row["95% CI Upper"]),
                )
            )

        lines.append(
            latex_escape(group) + " & " + " & ".join(cells) + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            rf"% Source: {latex_escape(source_label)}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find per-run CSV files and calculate mean, sample standard "
            "deviation, and two-sided 95% Student-t confidence intervals."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan recursively. Default: current directory.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="confidence_interval_outputs",
        help="Output directory. Relative paths are created under --root.",
    )
    parser.add_argument(
        "--include-all-csv",
        action="store_true",
        help="Process every non-summary CSV, not only filenames containing 'run'.",
    )
    parser.add_argument(
        "--verification-tolerance",
        type=float,
        default=0.01,
        help=(
            "Absolute tolerance used to verify existing mean/std values. "
            "Default 0.01 matches values rounded to two decimals."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        path for path in discover_csv_files(root, args.include_all_csv)
        if outdir not in path.parents
    ]

    if not csv_files:
        raise FileNotFoundError(
            "No candidate per-run CSV files were found. "
            "Place this script in the parent folder or pass --root."
        )

    all_stats: List[pd.DataFrame] = []
    verification_frames: List[pd.DataFrame] = []
    processing_log: List[Dict[str, object]] = []

    print(f"Root: {root}")
    print(f"Output: {outdir}")
    print(f"Candidate CSV files: {len(csv_files)}")

    for path in csv_files:
        try:
            stats_df, metadata = process_run_file(path, root)
            all_stats.append(stats_df)

            verification_df = verify_existing_summary(
                stats_df=stats_df,
                run_path=path,
                tolerance=args.verification_tolerance,
            )
            verification_frames.append(verification_df)

            safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)
            latex_text = make_wide_latex_table(
                stats_df,
                source_label=str(path.relative_to(root)),
            )
            (outdir / f"{safe_stem}_statistics_table.tex").write_text(
                latex_text,
                encoding="utf-8",
            )

            processing_log.append(
                {
                    "File": str(path.relative_to(root)),
                    "Status": "Processed",
                    "Raw Rows": metadata["raw_rows"],
                    "Group Columns": ", ".join(metadata["group_columns"]) or "(none)",
                    "Metric Columns": ", ".join(metadata["metric_columns"]),
                }
            )
            print(f"[OK] {path.relative_to(root)}")

        except Exception as exc:
            processing_log.append(
                {
                    "File": str(path.relative_to(root)),
                    "Status": f"Skipped: {exc}",
                    "Raw Rows": "",
                    "Group Columns": "",
                    "Metric Columns": "",
                }
            )
            print(f"[SKIP] {path.relative_to(root)} -> {exc}")

    if not all_stats:
        raise RuntimeError("No CSV file could be processed successfully.")

    master_df = pd.concat(all_stats, ignore_index=True)
    master_df.to_csv(
        outdir / "master_statistics_with_95CI.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    verification_master = pd.concat(
        verification_frames,
        ignore_index=True,
    )
    verification_master.to_csv(
        outdir / "existing_summary_verification.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    pd.DataFrame(processing_log).to_csv(
        outdir / "processing_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_latex = make_long_latex_table(master_df)
    (outdir / "master_statistics_with_95CI.tex").write_text(
        long_latex,
        encoding="utf-8",
    )

    # Plain-text rows for direct copying.
    text_lines = []
    for _, row in master_df.iterrows():
        text_lines.append(
            "{} | {} | {} | n={} | {:.2f}±{:.2f} | 95% CI [{:.2f}, {:.2f}]".format(
                row["Source File"],
                row["Group"],
                row["Metric"],
                int(row["N"]),
                float(row["Mean"]),
                float(row["Sample Std"]),
                float(row["95% CI Lower"]),
                float(row["95% CI Upper"]),
            )
        )

    (outdir / "statistics_for_direct_copy.txt").write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Processed statistical rows: {len(master_df)}")
    print(f"Main output: {outdir / 'master_statistics_with_95CI.csv'}")
    print(f"Verification: {outdir / 'existing_summary_verification.csv'}")
    print(f"LaTeX: {outdir / 'master_statistics_with_95CI.tex'}")
    print(f"Direct-copy text: {outdir / 'statistics_for_direct_copy.txt'}")


if __name__ == "__main__":
    main()
