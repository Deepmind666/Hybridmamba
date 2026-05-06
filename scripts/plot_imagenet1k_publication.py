#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "runs"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "figures" / "imagenet1k"
DEFAULT_STEM = "imagenet1k_validation_curves"

# AERIS-like publication palette
COLORS = {
    "TinyViM-B": "#2D83BD",
    "MobileMamba-B1": "#36A657",
    "Reference": "#5A5A5A",
    "Best": "#D15B9A",
    "Axis": "#111111",
    "Grid": "#CFCFCF",
}


@dataclass
class Point:
    epoch: int
    top1: float | None
    top5: float | None


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": COLORS["Axis"],
            "axes.labelcolor": COLORS["Axis"],
            "axes.titlecolor": COLORS["Axis"],
            "xtick.color": COLORS["Axis"],
            "ytick.color": COLORS["Axis"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": COLORS["Grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
            "font.size": 6.9,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def maybe_number(value: object) -> float | int | None:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        except ValueError:
            return None
    return None


def flatten_dict(obj: object, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_dict(value, child))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            child = f"{prefix}.{index}" if prefix else str(index)
            flat.update(flatten_dict(value, child))
    else:
        flat[prefix] = obj
    return flat


def normalize_percent(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def extract_from_flat(flat: dict[str, object], needle: str) -> float | int | None:
    target = needle.lower()
    for key, value in flat.items():
        if target == key.lower() or key.lower().endswith(target):
            num = maybe_number(value)
            if num is not None:
                return num
    return None


def parse_record(line: str) -> Point | None:
    raw = line.strip()
    if not raw:
        return None

    payload = raw
    if "{" in raw and "}" in raw:
        payload = raw[raw.find("{") : raw.rfind("}") + 1]

    obj = None
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(payload)
            break
        except Exception:
            continue

    epoch = None
    top1 = None
    top5 = None

    if isinstance(obj, dict):
        flat = flatten_dict(obj)
        epoch = extract_from_flat(flat, "epoch")
        top1 = extract_from_flat(flat, "test_acc1") or extract_from_flat(flat, "acc1") or extract_from_flat(flat, "top1")
        top5 = extract_from_flat(flat, "test_acc5") or extract_from_flat(flat, "acc5") or extract_from_flat(flat, "top5")

    if epoch is None:
        m_epoch = re.search(r"(?:^|[^A-Za-z0-9_])epoch(?:[^0-9+\-]*)(-?\d+)", raw, flags=re.IGNORECASE)
        if m_epoch:
            epoch = int(m_epoch.group(1))

    if top1 is None:
        m_top1 = re.search(r"(?:test_acc1|acc@?1|top1|top-1)[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
        if m_top1:
            top1 = float(m_top1.group(1))

    if top5 is None:
        m_top5 = re.search(r"(?:test_acc5|acc@?5|top5|top-5)[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
        if m_top5:
            top5 = float(m_top5.group(1))

    if top1 is None or top5 is None:
        m_top1_cnt = re.search(r"top1_cnt[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
        m_top5_cnt = re.search(r"top5_cnt[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
        m_top_all = re.search(r"top_all[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
        if m_top_all:
            total = float(m_top_all.group(1))
            if total > 0:
                if top1 is None and m_top1_cnt:
                    top1 = float(m_top1_cnt.group(1)) * 100.0 / total
                if top5 is None and m_top5_cnt:
                    top5 = float(m_top5_cnt.group(1)) * 100.0 / total

    if epoch is None or (top1 is None and top5 is None):
        return None

    return Point(epoch=int(epoch), top1=normalize_percent(top1), top5=normalize_percent(top5))


def choose_log_file(path: Path) -> Path:
    if path.is_file():
        return path

    priority = ["log.txt", "log_train.txt", "train.log", "log_val.txt"]
    for name in priority:
        match = next(path.rglob(name), None)
        if match is not None:
            return match

    candidates = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".log", ".jsonl"}]
    if not candidates:
        raise FileNotFoundError(f"No log file found under {path}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def infer_label(path: Path) -> str:
    text = str(path).lower()
    if "tinyvim" in text:
        return "TinyViM-B"
    if "mobilemamba" in text:
        return "MobileMamba-B1"
    return path.parent.name or path.stem


def discover_series(root: Path) -> list[tuple[str, Path]]:
    hits: dict[str, tuple[float, Path]] = {}
    if not root.exists():
        return []
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        lower = str(file).lower()
        if "imagenet1k" not in lower and "mobilemamba" not in lower and "tinyvim" not in lower:
            continue
        if file.suffix.lower() not in {".txt", ".log", ".jsonl"} and file.name not in {"log.txt", "log_train.txt"}:
            continue
        label = infer_label(file)
        mtime = file.stat().st_mtime
        current = hits.get(label)
        if current is None or mtime > current[0]:
            hits[label] = (mtime, file)
    series = [(label, path) for label, (_, path) in hits.items()]
    series.sort(key=lambda item: item[0])
    return series


def load_series(label: str, source: Path) -> list[Point]:
    log_file = choose_log_file(source)
    if "mobilemamba" in str(log_file).lower() or "mobilemamba" in label.lower():
        return load_mobilemamba_series(label, log_file)
    points: dict[int, Point] = {}
    with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            point = parse_record(line)
            if point is None:
                continue
            if point.top1 is None and point.top5 is None:
                continue
            existing = points.get(point.epoch)
            if existing is None:
                points[point.epoch] = point
            else:
                points[point.epoch] = Point(
                    epoch=point.epoch,
                    top1=point.top1 if point.top1 is not None else existing.top1,
                    top5=point.top5 if point.top5 is not None else existing.top5,
                )
    if not points:
        raise RuntimeError(f"No validation records parsed from {log_file}")
    ordered = [points[epoch] for epoch in sorted(points)]
    print(f"[{label}] {log_file} -> {len(ordered)} validation points")
    return ordered


def load_mobilemamba_series(label: str, log_file: Path) -> list[Point]:
    points: dict[int, Point] = {}
    pending_counts: Point | None = None
    with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue

            m_counts = re.search(
                r"top1_cnt[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)\s*\].*top5_cnt[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)\s*\].*top_all[^0-9+\-]*([0-9]+(?:\.[0-9]+)?)",
                raw,
                flags=re.IGNORECASE,
            )
            if m_counts:
                total = float(m_counts.group(3))
                if total > 0:
                    pending_counts = Point(
                        epoch=-1,
                        top1=float(m_counts.group(1)) * 100.0 / total,
                        top5=float(m_counts.group(2)) * 100.0 / total,
                    )
                continue

            if "max [" not in raw.lower() or "epoch:" not in raw.lower():
                continue

            m_epoch = re.search(r"epoch:\s*(\d+)", raw, flags=re.IGNORECASE)
            if not m_epoch:
                continue
            epoch = int(m_epoch.group(1))

            m_top1_ema = re.search(r"top1-ema:\s*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
            m_top1 = re.search(r"top1:\s*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
            top1 = float(m_top1_ema.group(1)) if m_top1_ema else (float(m_top1.group(1)) if m_top1 else None)
            top5 = pending_counts.top5 if pending_counts is not None else None
            if top1 is None and pending_counts is not None:
                top1 = pending_counts.top1

            if top1 is not None or top5 is not None:
                existing = points.get(epoch)
                candidate = Point(epoch=epoch, top1=top1, top5=top5)
                if existing is None:
                    points[epoch] = candidate
                else:
                    points[epoch] = Point(
                        epoch=epoch,
                        top1=candidate.top1 if candidate.top1 is not None else existing.top1,
                        top5=candidate.top5 if candidate.top5 is not None else existing.top5,
                    )
            pending_counts = None

    if not points:
        raise RuntimeError(f"No validation records parsed from {log_file}")
    ordered = [points[epoch] for epoch in sorted(points)]
    print(f"[{label}] {log_file} -> {len(ordered)} validation points")
    return ordered


def summarize_series(points: list[Point]) -> dict[str, float | int]:
    top1_points = [p for p in points if p.top1 is not None]
    best_top1 = max(top1_points, key=lambda p: p.top1) if top1_points else None
    top5_points = [p for p in points if p.top5 is not None]
    best_top5 = max(top5_points, key=lambda p: p.top5) if top5_points else None
    last = points[-1]
    return {
        "best_epoch_top1": best_top1.epoch if best_top1 is not None else -1,
        "best_top1": float(best_top1.top1) if best_top1 is not None else float("nan"),
        "best_epoch_top5": best_top5.epoch if best_top5 is not None else -1,
        "best_top5": float(best_top5.top5) if best_top5 is not None else float("nan"),
        "last_epoch": last.epoch,
        "last_top1": float(last.top1) if last.top1 is not None else float("nan"),
        "last_top5": float(last.top5) if last.top5 is not None else float("nan"),
    }


def plot_panel(ax: plt.Axes, metric: str, series_items: list[tuple[str, list[Point]]]) -> None:
    for label, points in series_items:
        xs = [p.epoch for p in points if getattr(p, metric) is not None]
        ys = [getattr(p, metric) for p in points if getattr(p, metric) is not None]
        if not xs:
            continue
        color = COLORS.get(label, COLORS["Reference"])
        ax.plot(xs, ys, color=color, linewidth=1.35, marker="o", markersize=2.6, markeredgewidth=0.0, label=label)
        best_index = max(range(len(ys)), key=lambda i: ys[i])
        ax.scatter([xs[best_index]], [ys[best_index]], s=26, facecolors="white", edgecolors=color, linewidths=0.95, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=14, color=color, zorder=4)

    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(left=0)
    ax.margins(x=0.02, y=0.06)


def write_summary(out_dir: Path, series_items: list[tuple[str, list[Point]]]) -> None:
    rows = []
    for label, points in series_items:
        summary = summarize_series(points)
        rows.append({"series": label, **summary})

    csv_path = out_dir / f"{DEFAULT_STEM}_summary.csv"
    md_path = out_dir / f"{DEFAULT_STEM}_summary.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Series | Best epoch (top1) | Best top1 | Best epoch (top5) | Best top5 | Last epoch | Last top1 | Last top5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {series} | {best_epoch_top1} | {best_top1:.2f} | {best_epoch_top5} | {best_top5:.2f} | {last_epoch} | {last_top1:.2f} | {last_top5:.2f} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_series(args: argparse.Namespace) -> list[tuple[str, list[Point]]]:
    series: list[tuple[str, Path]] = []
    if args.series:
        for token in args.series:
            if "=" in token:
                label, raw_path = token.split("=", 1)
            else:
                raw_path = token
                label = None
            source = Path(raw_path).expanduser()
            if label is None or not label.strip():
                label = infer_label(source)
            series.append((label.strip(), source))
    else:
        series = discover_series(args.root)

    if not series:
        raise SystemExit("No series found. Pass --series Label=PATH or place Imagenet1K logs under the default run root.")

    ordered: list[tuple[str, list[Point]]] = []
    for label, source in series:
        ordered.append((label, load_series(label, source)))
    return ordered


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ImageNet-1K validation curves in a publication style.")
    parser.add_argument("--series", action="append", default=[], help="Series entry in the form Label=PATH. Can be a file or a directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT, help="Auto-discovery root when --series is not provided.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory where figures and summaries will be written.")
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Filename stem for exported figures.")
    args = parser.parse_args()

    setup_style()
    series_items = build_series(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)
    plot_panel(axes[0], "top1", series_items)
    plot_panel(axes[1], "top5", series_items)

    axes[0].set_title("Top-1 validation accuracy", loc="left", pad=4)
    axes[1].set_title("Top-5 validation accuracy", loc="left", pad=4)
    axes[0].text(-0.13, 1.05, "a", transform=axes[0].transAxes, fontsize=9.0, weight="bold", va="top", ha="left")
    axes[1].text(-0.13, 1.05, "b", transform=axes[1].transAxes, fontsize=9.0, weight="bold", va="top", ha="left")

    if series_items:
        axes[0].legend(frameon=False, loc="lower right", handlelength=1.8, handletextpad=0.4, borderaxespad=0.2)

    save_figure(fig, args.out_dir, args.stem)
    write_summary(args.out_dir, series_items)
    print(f"Saved figures to {args.out_dir / (args.stem + '.pdf')}")


if __name__ == "__main__":
    main()
