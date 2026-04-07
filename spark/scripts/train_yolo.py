"""Train YOLO models for pulmonary nodule detection on NODE21 dataset.

Supports both YOLOv8 and YOLO26 variants. The script auto-detects the model
version from the model name and adjusts training parameters accordingly.

YOLOv8s: baseline detector, NODE21 score ~0.93
YOLO26s: latest Ultralytics release with STAL, NMS-free detection, ProgLoss

Usage:
    python train_yolo.py --model yolov8s.pt --name yolov8s_nodule
    python train_yolo.py --model yolo26s.pt --name yolo26s_nodule
    python train_yolo.py --model yolov8m.pt --epochs 150 --batch 2
    python train_yolo.py --model yolo26m.pt --device 1 --name yolo26m_exp
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def is_yolo26(model_name: str) -> bool:
    """Check if the model is a YOLO26 variant based on its name."""
    name_lower = model_name.lower()
    return "yolo26" in name_lower or "26" in name_lower


def build_train_args(args: argparse.Namespace) -> dict:
    """Build the keyword arguments dict for model.train() based on model version."""

    yolo26 = is_yolo26(args.model)

    # --- common training hyperparameters ---
    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=1e-5,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        workers=4,
        # conservative augmentation for medical imaging
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.5,
        flipud=0.0,
        scale=0.1,
        translate=0.05,
    )

    # --- version-specific loss weights ---
    if not yolo26:
        # YOLOv8 loss weights
        train_kwargs["box"] = 7.5
        train_kwargs["cls"] = 0.3
        train_kwargs["dfl"] = 1.5
    # YOLO26 uses ProgLoss and MuSGD by default; do NOT pass dfl.

    return train_kwargs


def copy_best_weights(run_dir: Path, model_name: str) -> None:
    """Copy the best checkpoint to checkpoints/yolo/{model_name}/."""
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[WARN] best.pt not found at {best_pt}, skipping copy.")
        return

    stem = Path(model_name).stem  # e.g. yolov8s
    dest_dir = Path("checkpoints/yolo") / stem
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / "best.pt"
    shutil.copy2(best_pt, dest_path)
    print(f"[INFO] Best weights copied to {dest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO for pulmonary nodule detection (YOLOv8 / YOLO26)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model file (yolov8s.pt, yolov8m.pt, yolo26s.pt, yolo26m.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/yolo_dataset/node21.yaml",
        help="Path to YOLO dataset YAML",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (auto-generated from model name if omitted)",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saving runs",
    )

    args = parser.parse_args()

    # Auto-generate experiment name if not provided
    if args.name is None:
        args.name = Path(args.model).stem + "_nodule"

    yolo26 = is_yolo26(args.model)
    version_tag = "YOLO26" if yolo26 else "YOLOv8"

    print(f"[INFO] Detected model version: {version_tag}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Data:  {args.data}")
    print(f"[INFO] Image size: {args.imgsz}")
    print(f"[INFO] Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"[INFO] Experiment: {args.project}/{args.name}")
    if yolo26:
        print("[INFO] YOLO26 mode: ProgLoss + MuSGD defaults, no DFL parameter")
    else:
        print("[INFO] YOLOv8 mode: box=7.5, cls=0.3, dfl=1.5")

    # Load model
    model = YOLO(args.model)

    # Build training arguments and launch training
    train_kwargs = build_train_args(args)
    results = model.train(**train_kwargs)

    # Copy best weights to checkpoints directory
    # Ultralytics saves to save_dir which may differ from project/name
    run_dir = Path(args.project) / args.name
    # Also check the actual save_dir from training results
    if hasattr(results, "save_dir"):
        actual_dir = Path(results.save_dir)
        if actual_dir.exists():
            run_dir = actual_dir
    copy_best_weights(run_dir, args.model)

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
