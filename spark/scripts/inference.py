#!/usr/bin/env python3
"""
Hospital Inference Tool for Pulmonary Nodule Detection.

Runs available models (Faster R-CNN VinDr, YOLOv8, YOLO26) on chest X-ray
images and fuses predictions with Weighted Box Fusion (WBF).

Supports PNG, DICOM (.dcm), and MHA input formats.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".mha", ".mhd"}


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """Load a CXR image from PNG, DICOM, or MHA and return a uint8 grayscale
    numpy array (H, W)."""
    ext = Path(image_path).suffix.lower()

    if ext in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img

    elif ext in (".dcm",):
        import SimpleITK as sitk
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        sitk_img = reader.Execute()
        arr = sitk.GetArrayFromImage(sitk_img)
        # DICOM may be (1, H, W) or (H, W)
        if arr.ndim == 3:
            arr = arr[0]
        # Normalize to 0-255 uint8
        arr = arr.astype(np.float64)
        mn, mx = arr.min(), arr.max()
        if mx - mn > 0:
            arr = (arr - mn) / (mx - mn) * 255.0
        return arr.astype(np.uint8)

    elif ext in (".mha", ".mhd"):
        import SimpleITK as sitk
        sitk_img = sitk.ReadImage(image_path)
        arr = sitk.GetArrayFromImage(sitk_img)
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float64)
        mn, mx = arr.min(), arr.max()
        if mx - mn > 0:
            arr = (arr - mn) / (mx - mn) * 255.0
        return arr.astype(np.uint8)

    else:
        raise ValueError(f"Unsupported image format: {ext}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_to_float(img_gray: np.ndarray, target_size: int = 1024):
    """Resize to target_size x target_size and normalize to [0, 1] float32."""
    img_resized = cv2.resize(img_gray, (target_size, target_size),
                             interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    return img_resized, img_float


def build_frcnn_input(img_uint8: np.ndarray, img_float: np.ndarray) -> torch.Tensor:
    """Create 3-channel input for Faster R-CNN:
       Ch0 = Original, Ch1 = CLAHE, Ch2 = Unsharp Mask.
       Returns tensor (3, H, W) in [0, 1]."""
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(img_uint8).astype(np.float32) / 255.0

    # Unsharp mask
    blurred = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=3)
    ch_unsharp = cv2.addWeighted(img_uint8, 1.5, blurred, -0.5, 0)
    ch_unsharp = ch_unsharp.astype(np.float32) / 255.0

    tensor = np.stack([img_float, ch_clahe, ch_unsharp], axis=0)  # (3, H, W)
    return torch.from_numpy(tensor)


def build_yolo_input(img_uint8: np.ndarray) -> np.ndarray:
    """Create 3-channel BGR image for YOLO (grayscale replicated)."""
    return cv2.merge([img_uint8, img_uint8, img_uint8])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_frcnn(weights_path: str, device: torch.device):
    """Load Faster R-CNN ResNet50-FPN with 2 classes (background + nodule)."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    num_classes=2)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    # Support both raw state_dict and wrapped checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded Faster R-CNN from {weights_path}")
    return model


def load_yolo(weights_path: str, model_name: str = "YOLOv8"):
    """Load a YOLO model via ultralytics."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    print(f"[INFO] Loaded {model_name} from {weights_path}")
    return model


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.2):
    """Standard greedy NMS. boxes: (N, 4) xyxy, scores: (N,)."""
    if len(boxes) == 0:
        return np.array([]), np.array([])
    keep = torchvision.ops.nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        iou_thr,
    ).numpy()
    return boxes[keep], scores[keep]


def infer_frcnn(model, img_tensor: torch.Tensor, device: torch.device,
                confidence: float, img_size: int = 1024):
    """Run Faster R-CNN inference. Returns boxes (N,4) xyxy and scores (N,)."""
    inp = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(inp)[0]
    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    # Filter by confidence
    mask = scores >= confidence
    boxes, scores = boxes[mask], scores[mask]
    # NMS
    boxes, scores = nms(boxes, scores, iou_thr=0.2)
    return boxes, scores


def infer_yolo(model, img_bgr: np.ndarray, confidence: float,
               img_size: int = 1024, apply_nms: bool = True):
    """Run YOLO inference. Returns boxes (N,4) xyxy and scores (N,)."""
    results = model.predict(img_bgr, imgsz=img_size, conf=confidence,
                            verbose=False)
    boxes_all = []
    scores_all = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            boxes_all.append(r.boxes.xyxy.cpu().numpy())
            scores_all.append(r.boxes.conf.cpu().numpy())
    if len(boxes_all) == 0:
        return np.empty((0, 4)), np.empty((0,))
    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    if apply_nms and len(boxes) > 0:
        boxes, scores = nms(boxes, scores, iou_thr=0.2)
    return boxes, scores


# ---------------------------------------------------------------------------
# Weighted Box Fusion
# ---------------------------------------------------------------------------

def run_wbf(all_boxes, all_scores, all_weights, img_size: int = 1024,
            iou_thr: float = 0.2, skip_box_thr: float = 0.1):
    """Run WBF across model predictions.
    all_boxes: list of (N_i, 4) arrays in xyxy pixel coords.
    all_scores: list of (N_i,) arrays.
    all_weights: list of floats, one per model.
    Returns fused boxes (M, 4) in pixel coords and scores (M,).
    """
    from ensemble_boxes import weighted_boxes_fusion

    # Normalize boxes to [0, 1]
    norm_boxes = []
    norm_scores = []
    norm_labels = []
    for b, s in zip(all_boxes, all_scores):
        if len(b) == 0:
            norm_boxes.append(np.empty((0, 4)))
            norm_scores.append(np.empty((0,)))
            norm_labels.append(np.empty((0,)))
            continue
        nb = b.copy().astype(np.float64)
        nb[:, [0, 2]] /= img_size  # x
        nb[:, [1, 3]] /= img_size  # y
        nb = np.clip(nb, 0.0, 1.0)
        norm_boxes.append(nb.tolist())
        norm_scores.append(s.tolist())
        norm_labels.append([0] * len(s))  # single class

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        norm_boxes, norm_scores, norm_labels,
        weights=all_weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    # Denormalize
    fused_boxes = np.array(fused_boxes)
    fused_scores = np.array(fused_scores)
    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= img_size
        fused_boxes[:, [1, 3]] *= img_size

    return fused_boxes, fused_scores


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def draw_boxes(img_gray_uint8: np.ndarray, boxes: np.ndarray,
               scores: np.ndarray, color=(0, 0, 255), thickness=2):
    """Draw bounding boxes on a BGR copy of the grayscale image."""
    vis = cv2.cvtColor(img_gray_uint8, cv2.COLOR_GRAY2BGR)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"NODULE {score:.2f}"
        # Text background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return vis


def build_json_report(image_name: str, per_model_dets: dict,
                      ensemble_boxes_arr: np.ndarray,
                      ensemble_scores_arr: np.ndarray):
    """Build a JSON-serialisable report dict."""
    detections = []
    for model_name, (boxes, scores) in per_model_dets.items():
        for box, sc in zip(boxes, scores):
            detections.append({
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "score": float(sc),
                "model_source": model_name,
            })

    ensemble_dets = []
    for box, sc in zip(ensemble_boxes_arr, ensemble_scores_arr):
        ensemble_dets.append({
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "score": float(sc),
        })

    report = {
        "image_name": image_name,
        "num_detections": len(ensemble_dets),
        "detections": detections,
        "ensemble_detections": ensemble_dets,
    }
    return report


# ---------------------------------------------------------------------------
# Main inference pipeline for a single image
# ---------------------------------------------------------------------------

def process_single_image(image_path: str, models: dict, args):
    """Run full pipeline on one image. Returns (report_dict, annotated_img)."""
    image_name = Path(image_path).name
    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {image_name}")
    print(f"{'='*60}")

    device = torch.device(args.device)
    t0 = time.time()

    # 1. Load
    img_gray = load_image(image_path)
    print(f"  Loaded image: {img_gray.shape}")

    # 2. Preprocess
    img_resized_uint8, img_float = preprocess_to_float(img_gray, target_size=1024)
    print(f"  Preprocessed to 1024x1024, range [{img_float.min():.2f}, {img_float.max():.2f}]")

    # 3-6. Run available models
    per_model_dets = {}
    model_boxes_for_wbf = []
    model_scores_for_wbf = []
    wbf_weights = []

    weight_map = {"frcnn": 2, "yolov8": 1, "yolo26": 1}

    if "frcnn" in models:
        frcnn_tensor = build_frcnn_input(img_resized_uint8, img_float)
        boxes, scores = infer_frcnn(models["frcnn"], frcnn_tensor, device,
                                    confidence=args.confidence)
        per_model_dets["frcnn"] = (boxes, scores)
        model_boxes_for_wbf.append(boxes)
        model_scores_for_wbf.append(scores)
        wbf_weights.append(weight_map["frcnn"])
        print(f"  Faster R-CNN: {len(boxes)} detections")

    if "yolov8" in models:
        yolo_img = build_yolo_input(img_resized_uint8)
        boxes, scores = infer_yolo(models["yolov8"], yolo_img,
                                   confidence=args.confidence,
                                   apply_nms=True)
        per_model_dets["yolov8"] = (boxes, scores)
        model_boxes_for_wbf.append(boxes)
        model_scores_for_wbf.append(scores)
        wbf_weights.append(weight_map["yolov8"])
        print(f"  YOLOv8: {len(boxes)} detections")

    if "yolo26" in models:
        yolo_img = build_yolo_input(img_resized_uint8)
        boxes, scores = infer_yolo(models["yolo26"], yolo_img,
                                   confidence=args.confidence,
                                   apply_nms=False)  # YOLO26 is NMS-free
        per_model_dets["yolo26"] = (boxes, scores)
        model_boxes_for_wbf.append(boxes)
        model_scores_for_wbf.append(scores)
        wbf_weights.append(weight_map["yolo26"])
        print(f"  YOLO26: {len(boxes)} detections")

    # 7. Ensemble
    num_models = len(model_boxes_for_wbf)
    if num_models == 0:
        print("[WARN] No models available. Skipping.")
        return None, None

    if num_models >= 2 and args.ensemble:
        print(f"  Running WBF ensemble ({num_models} models, weights={wbf_weights})")
        final_boxes, final_scores = run_wbf(
            model_boxes_for_wbf, model_scores_for_wbf, wbf_weights,
            img_size=1024, iou_thr=0.2, skip_box_thr=0.1,
        )
    else:
        # Single model or ensemble disabled: concatenate all detections
        if num_models == 1:
            print("  Single model -- skipping WBF")
        else:
            print("  Ensemble disabled -- using raw detections")
        final_boxes = np.concatenate(model_boxes_for_wbf, axis=0) if model_boxes_for_wbf else np.empty((0, 4))
        final_scores = np.concatenate(model_scores_for_wbf, axis=0) if model_scores_for_wbf else np.empty((0,))

    # Filter final by confidence again (WBF can change scores)
    if len(final_scores) > 0:
        mask = final_scores >= args.confidence
        final_boxes = final_boxes[mask]
        final_scores = final_scores[mask]

    elapsed = time.time() - t0
    print(f"  Ensemble detections: {len(final_boxes)} (took {elapsed:.2f}s)")

    # 8. Output
    annotated = draw_boxes(img_resized_uint8, final_boxes, final_scores)
    report = build_json_report(image_name, per_model_dets,
                               final_boxes, final_scores)

    # Console summary
    print(f"\n  --- Detection Summary for {image_name} ---")
    print(f"  Total ensemble detections: {report['num_detections']}")
    for det in report["ensemble_detections"]:
        print(f"    NODULE  score={det['score']:.3f}  "
              f"box=[{det['x1']:.0f}, {det['y1']:.0f}, "
              f"{det['x2']:.0f}, {det['y2']:.0f}]")

    return report, annotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pulmonary Nodule Detection -- Hospital Inference Tool")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str,
                       help="Path to a single CXR image (PNG/DICOM/MHA)")
    group.add_argument("--image_dir", type=str,
                       help="Directory with CXR images for batch processing")

    parser.add_argument("--frcnn_weights", type=str, default=None,
                        help="Path to Faster R-CNN checkpoint")
    parser.add_argument("--yolov8_weights", type=str, default=None,
                        help="Path to YOLOv8 checkpoint")
    parser.add_argument("--yolo26_weights", type=str, default=None,
                        help="Path to YOLO26 checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="results/inference/",
                        help="Output directory (default: results/inference/)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Minimum confidence threshold (default: 0.3)")
    parser.add_argument("--ensemble", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use WBF ensemble (default: True, --no-ensemble to disable)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device (default: cuda:0)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate that at least one model is provided
    if not any([args.frcnn_weights, args.yolov8_weights, args.yolo26_weights]):
        print("[ERROR] At least one model weight must be provided "
              "(--frcnn_weights, --yolov8_weights, or --yolo26_weights)")
        sys.exit(1)

    # Setup device
    device = torch.device(args.device)
    if "cuda" in args.device and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = torch.device("cpu")
        args.device = "cpu"

    # Load models
    models = {}
    if args.frcnn_weights:
        models["frcnn"] = load_frcnn(args.frcnn_weights, device)
    if args.yolov8_weights:
        models["yolov8"] = load_yolo(args.yolov8_weights, model_name="YOLOv8")
    if args.yolo26_weights:
        models["yolo26"] = load_yolo(args.yolo26_weights, model_name="YOLO26")

    print(f"\n[INFO] Models loaded: {list(models.keys())}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Confidence threshold: {args.confidence}")
    print(f"[INFO] Ensemble (WBF): {args.ensemble}")

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        dirpath = Path(args.image_dir)
        if not dirpath.is_dir():
            print(f"[ERROR] Not a directory: {args.image_dir}")
            sys.exit(1)
        for f in sorted(dirpath.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(str(f))
        print(f"[INFO] Found {len(image_paths)} images in {args.image_dir}")

    if not image_paths:
        print("[ERROR] No valid images found.")
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process
    all_reports = []
    for img_path in image_paths:
        report, annotated = process_single_image(img_path, models, args)
        if report is None:
            continue

        stem = Path(img_path).stem

        # Save annotated image
        img_out = out_dir / f"{stem}_detections.png"
        cv2.imwrite(str(img_out), annotated)
        print(f"  Saved annotated image: {img_out}")

        # Save JSON report
        json_out = out_dir / f"{stem}_report.json"
        with open(json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved JSON report: {json_out}")

        all_reports.append(report)

    # Final summary
    total = sum(r["num_detections"] for r in all_reports)
    print(f"\n{'='*60}")
    print(f"[DONE] Processed {len(all_reports)} image(s), "
          f"{total} total ensemble detection(s)")
    print(f"[DONE] Results saved to: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
