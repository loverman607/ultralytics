from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO


def _get_data_from_weights(weights: Path) -> Path | None:
    try:
        import torch

        ckpt = torch.load(weights, map_location="cpu")
        if isinstance(ckpt, dict):
            data = ckpt.get("train_args", {}).get("data")
            return Path(data) if data else None
    except Exception:
        return None
    return None


def _resolve_split(data: dict, split: str) -> Path:
    base = Path(data.get("path", "")) if data.get("path") else None
    src = data.get(split) or data.get("val") or data.get("train")
    if not src:
        raise ValueError(f"No '{split}', 'val', or 'train' split found in data.yaml")
    src_path = Path(src)
    if base and not src_path.is_absolute():
        src_path = base / src_path
    return src_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LT YOLO inference using data.yaml splits.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\yoloresearch\ultralytics\runs\detect\train2\weights\best.pt"),
        help="Path to model weights (.pt). Required.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"C:\Users\ifeol\Documents\MACO\datasets\MACO\data.yaml"),
        help="Path to data.yaml.",
    )
    parser.add_argument("--split", choices=["test", "val", "train"], default="test", help="Dataset split to run on.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device string, e.g. '0' or 'cpu'.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/infer"),
        help="Project directory for outputs.",
    )
    parser.add_argument("--name", default="lt_infer", help="Run name for outputs.")
    parser.add_argument("--predict", dest="predict", action="store_true", help="Run prediction on the split.")
    parser.add_argument("--no-predict", dest="predict", action="store_false", help="Skip prediction.")
    parser.add_argument("--metrics", dest="metrics", action="store_true", help="Compute metrics (mAP, etc.).")
    parser.add_argument("--no-metrics", dest="metrics", action="store_false", help="Skip metrics computation.")
    parser.add_argument("--val", dest="metrics", action="store_true", help="Alias for --metrics.")
    parser.add_argument(
        "--val-name",
        default=None,
        help="Run name for validation outputs (defaults to '{name}_val').",
    )
    parser.set_defaults(predict=True, metrics=True)
    args = parser.parse_args()

    weights = args.weights
    if weights is None:
        raise FileNotFoundError("Weights not provided. Pass --weights.")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    data_path = args.data
    if not data_path.exists():
        inferred = _get_data_from_weights(weights)
        if inferred and inferred.exists():
            data_path = inferred
        else:
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    source = _resolve_split(data, args.split)

    print(f"Using weights: {weights}")
    print(f"Using data: {data_path}")
    model = YOLO(str(weights))
    if args.predict:
        model.predict(
            source=str(source),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=True,
            project=str(args.project),
            name=args.name,
        )

    if args.metrics:
        val_name = args.val_name or f"{args.name}_val"
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            plots=True,
            project=str(args.project),
            name=val_name,
        )
        metrics_dict = metrics if isinstance(metrics, dict) else getattr(metrics, "results_dict", None)
        if metrics_dict:
            metrics_path = Path(args.project) / val_name / "metrics.yaml"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(yaml.safe_dump(metrics_dict, sort_keys=False), encoding="utf-8")
            print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
