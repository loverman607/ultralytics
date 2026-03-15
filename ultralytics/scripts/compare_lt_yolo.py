from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_val(weights: Path, data: Path, split: str, imgsz: int, conf: float, device: str | None, project: Path, name: str, plots: bool):
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        conf=conf,
        device=device,
        plots=plots,
        project=str(project),
        name=name,
    )
    return metrics


def _summary_by_class(metrics) -> dict[str, dict]:
    # metrics.summary() returns list of dicts keyed by "Class" name
    summary = metrics.summary()
    return {row["Class"]: row for row in summary}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _num(x):
    return x if isinstance(x, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-class AP and metrics between two checkpoints.")
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\yoloresearch\ultralytics\runs\detect\train4\weights\best.pt"),
        help="Baseline weights (.pt).",
    )
    parser.add_argument(
        "--improved-weights",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\yoloresearch\ultralytics\runs\detect\train\weights\best.pt"),
        help="Improved weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\datasets\MACO\data.yaml"),
        help="Path to data.yaml.",
    )
    parser.add_argument("--split", choices=["test", "val", "train"], default="test", help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default=None, help="Device string, e.g. '0' or 'cpu'.")
    parser.add_argument("--project", type=Path, default=Path("runs/compare"), help="Project dir for outputs.")
    parser.add_argument("--name", default="lt_compare", help="Run name for outputs.")
    parser.add_argument("--baseline-name", default="baseline_val", help="Run name for baseline validation outputs.")
    parser.add_argument("--improved-name", default="improved_val", help="Run name for improved validation outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Disable metrics plots.")
    args = parser.parse_args()

    if not args.baseline_weights.exists():
        raise FileNotFoundError(f"Baseline weights not found: {args.baseline_weights}")
    if not args.improved_weights.exists():
        raise FileNotFoundError(f"Improved weights not found: {args.improved_weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    out_dir = args.project / args.name
    _ensure_dir(out_dir)

    plots = not args.no_plots
    baseline_metrics = _run_val(
        args.baseline_weights,
        args.data,
        args.split,
        args.imgsz,
        args.conf,
        args.device,
        out_dir,
        args.baseline_name,
        plots,
    )
    improved_metrics = _run_val(
        args.improved_weights,
        args.data,
        args.split,
        args.imgsz,
        args.conf,
        args.device,
        out_dir,
        args.improved_name,
        plots,
    )

    base = _summary_by_class(baseline_metrics)
    imp = _summary_by_class(improved_metrics)
    classes = sorted(set(base.keys()) | set(imp.keys()))

    rows = []
    for cls in classes:
        b = base.get(cls, {})
        i = imp.get(cls, {})
        row = {
            "Class": cls,
            "Images": b.get("Images", i.get("Images")),
            "Instances": b.get("Instances", i.get("Instances")),
            "Baseline-Box-P": b.get("Box-P"),
            "Improved-Box-P": i.get("Box-P"),
            "Delta-Box-P": None,
            "Baseline-Box-R": b.get("Box-R"),
            "Improved-Box-R": i.get("Box-R"),
            "Delta-Box-R": None,
            "Baseline-Box-F1": b.get("Box-F1"),
            "Improved-Box-F1": i.get("Box-F1"),
            "Delta-Box-F1": None,
            "Baseline-mAP50": b.get("mAP50"),
            "Improved-mAP50": i.get("mAP50"),
            "Delta-mAP50": None,
            "Baseline-mAP50-95": b.get("mAP50-95"),
            "Improved-mAP50-95": i.get("mAP50-95"),
            "Delta-mAP50-95": None,
        }
        for k in ["Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]:
            b_val = _num(b.get(k))
            i_val = _num(i.get(k))
            if b_val is not None and i_val is not None:
                row[f"Delta-{k}"] = round(i_val - b_val, 6)
        rows.append(row)

    _write_csv(out_dir / "class_metrics_comparison.csv", rows)

    overall = {
        "baseline": baseline_metrics.results_dict,
        "improved": improved_metrics.results_dict,
        "delta": {
            k: (improved_metrics.results_dict.get(k) - baseline_metrics.results_dict.get(k))
            for k in baseline_metrics.results_dict.keys()
        },
    }
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
