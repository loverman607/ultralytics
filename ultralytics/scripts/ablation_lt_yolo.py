from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from ultralytics import YOLO


KEY_MAP = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "fitness": "fitness",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


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


def _short_metrics(results_dict: dict) -> dict[str, float]:
    out = {}
    for k, short in KEY_MAP.items():
        if k in results_dict:
            v = results_dict[k]
            out[short] = float(v) if hasattr(v, "__float__") else v
    return out


def _parse_ablation_list(items: list[str]) -> list[dict]:
    ablations = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --ablation '{item}'. Use name=path.")
        name, path = item.split("=", 1)
        ablations.append({"name": name.strip(), "weights": Path(path.strip())})
    return ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study: compare baseline, full, and ablations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ultralytics/ultralytics/scripts/ablation_config.yaml"),
        help="Path to ablation config YAML.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\yoloresearch\ultralytics\runs\detect\train4\weights\best.pt"),
        help="Baseline weights (.pt).",
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\yoloresearch\ultralytics\runs\detect\train\weights\best.pt"),
        help="Full model weights (.pt).",
    )
    parser.add_argument("--ablation", action="append", help="Ablation entry as name=path. Can be repeated.")
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
    parser.add_argument("--project", type=Path, default=Path("runs/ablation"), help="Project dir for outputs.")
    parser.add_argument("--name", default="lt_ablation", help="Run name for outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Disable metrics plots.")
    args = parser.parse_args()

    cfg = _load_cfg(args.config) if args.config.exists() else {}

    baseline = Path(cfg.get("baseline", args.baseline))
    full = Path(cfg.get("full", args.full))
    data = Path(cfg.get("data", args.data))
    split = cfg.get("split", args.split)
    imgsz = int(cfg.get("imgsz", args.imgsz))
    conf = float(cfg.get("conf", args.conf))
    device = cfg.get("device", args.device)
    project = Path(cfg.get("project", args.project))
    name = cfg.get("name", args.name)
    plots = not args.no_plots if "plots" not in cfg else bool(cfg.get("plots"))

    ablations = cfg.get("ablations", None) or []
    ablations.extend(_parse_ablation_list(args.ablation))

    if not baseline.exists():
        raise FileNotFoundError(f"Baseline weights not found: {baseline}")
    if not full.exists():
        raise FileNotFoundError(f"Full weights not found: {full}")
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    out_dir = project / name
    _ensure_dir(out_dir)

    baseline_metrics = _run_val(baseline, data, split, imgsz, conf, device, out_dir, f"{name}_baseline", plots)
    full_metrics = _run_val(full, data, split, imgsz, conf, device, out_dir, f"{name}_full", plots)

    base_summary = _summary_by_class(baseline_metrics)
    full_summary = _summary_by_class(full_metrics)

    overall_rows = []
    overall_rows.append({"run": "baseline", **_short_metrics(baseline_metrics.results_dict)})
    overall_rows.append({"run": "full", **_short_metrics(full_metrics.results_dict)})

    ablation_rows = []
    for ab in ablations:
        ab_name = ab.get("name", "ablation")
        ab_weights = Path(ab["weights"])
        if not ab_weights.exists():
            raise FileNotFoundError(f"Ablation weights not found: {ab_weights}")
        ab_metrics = _run_val(ab_weights, data, split, imgsz, conf, device, out_dir, f"{name}_{ab_name}", plots)
        ab_summary = _summary_by_class(ab_metrics)

        overall_rows.append({"run": f"ablation_{ab_name}", **_short_metrics(ab_metrics.results_dict)})

        base_short = _short_metrics(baseline_metrics.results_dict)
        full_short = _short_metrics(full_metrics.results_dict)
        ab_short = _short_metrics(ab_metrics.results_dict)
        contrib = {"ablation": ab_name}
        for k in KEY_MAP.values():
            b = base_short.get(k)
            f = full_short.get(k)
            a = ab_short.get(k)
            contrib[f"ablation_{k}"] = a
            contrib[f"gain_vs_baseline_{k}"] = (a - b) if a is not None and b is not None else None
            contrib[f"drop_vs_full_{k}"] = (f - a) if a is not None and f is not None else None
        ablation_rows.append(contrib)

        # per-class delta CSV for this ablation
        class_rows = []
        classes = sorted(set(base_summary.keys()) | set(full_summary.keys()) | set(ab_summary.keys()))
        for cls in classes:
            b = base_summary.get(cls, {})
            f = full_summary.get(cls, {})
            a = ab_summary.get(cls, {})
            row = {
                "Class": cls,
                "Baseline-mAP50": b.get("mAP50"),
                "Baseline-mAP50-95": b.get("mAP50-95"),
                "Full-mAP50": f.get("mAP50"),
                "Full-mAP50-95": f.get("mAP50-95"),
                "Ablation-mAP50": a.get("mAP50"),
                "Ablation-mAP50-95": a.get("mAP50-95"),
            }
            if f.get("mAP50") is not None and a.get("mAP50") is not None:
                row["Drop_vs_Full-mAP50"] = round(f["mAP50"] - a["mAP50"], 6)
            if f.get("mAP50-95") is not None and a.get("mAP50-95") is not None:
                row["Drop_vs_Full-mAP50-95"] = round(f["mAP50-95"] - a["mAP50-95"], 6)
            if a.get("mAP50") is not None and b.get("mAP50") is not None:
                row["Gain_vs_Baseline-mAP50"] = round(a["mAP50"] - b["mAP50"], 6)
            if a.get("mAP50-95") is not None and b.get("mAP50-95") is not None:
                row["Gain_vs_Baseline-mAP50-95"] = round(a["mAP50-95"] - b["mAP50-95"], 6)
            class_rows.append(row)
        _write_csv(out_dir / f"class_delta_{ab_name}.csv", class_rows)

    _write_csv(out_dir / "overall_metrics.csv", overall_rows)
    _write_csv(out_dir / "ablation_contributions.csv", ablation_rows)

    overall = {
        "baseline": baseline_metrics.results_dict,
        "full": full_metrics.results_dict,
    }
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
