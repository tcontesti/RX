"""Evaluate pulmonary nodule detection models: Faster R-CNN, YOLOv8, YOLO26.

Computes FROC curves, NODE21 score, image-level AUROC, Competition Metric,
per-FP-level sensitivity, and inference time. Produces a CSV report,
overlaid FROC plot, and example detection visualisations.

Usage:
    python evaluate.py \
        --frcnn_checkpoint weights/frcnn_best.pth \
        --yolov8_checkpoint weights/yolov8s_best.pt \
        --yolo26_checkpoint weights/yolo26s_best.pt \
        --data_csv data/val.csv \
        --img_dir data/images/ \
        --output_dir results/
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODE21_FP_LEVELS = [0.25, 0.5, 1, 2, 4, 8]
IMG_SIZE = 1024

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_froc(gt_boxes_list, pred_boxes_list, pred_scores_list,
                 num_images, iou_thresh=0.2):
    """Compute the FROC curve from per-image ground-truth and predictions.

    Args:
        gt_boxes_list:   list of np.ndarray (N_i, 4) per image.
        pred_boxes_list: list of np.ndarray (M_i, 4) per image.
        pred_scores_list: list of np.ndarray (M_i,) per image.
        num_images: total number of images evaluated.
        iou_thresh: IoU threshold for a true-positive match.

    Returns:
        fps_per_image: 1-D array, average false positives per image.
        sensitivities: 1-D array, sensitivity at each operating point.
    """
    all_tp_scores = []
    all_fp_scores = []
    num_lesions = 0

    for gt_boxes, pred_boxes, pred_scores in zip(
            gt_boxes_list, pred_boxes_list, pred_scores_list):
        num_gt = len(gt_boxes)
        num_lesions += num_gt
        matched_gt = set()

        # Sort predictions by descending score
        if len(pred_scores) == 0:
            continue
        order = np.argsort(-pred_scores)
        for idx in order:
            best_iou = 0.0
            best_gt = -1
            for g, gt_box in enumerate(gt_boxes):
                if g in matched_gt:
                    continue
                iou_val = compute_iou(pred_boxes[idx], gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = g
            if best_iou >= iou_thresh and best_gt >= 0:
                all_tp_scores.append(pred_scores[idx])
                matched_gt.add(best_gt)
            else:
                all_fp_scores.append(pred_scores[idx])

    all_tp_scores = np.array(all_tp_scores, dtype=np.float64)
    all_fp_scores = np.array(all_fp_scores, dtype=np.float64)

    if num_lesions == 0:
        return np.array([0.0]), np.array([0.0])

    # Build curve over all unique thresholds
    all_scores = sorted(set(all_tp_scores.tolist() + all_fp_scores.tolist()))
    fps_list = []
    sens_list = []
    for thresh in all_scores:
        fps_list.append((all_fp_scores >= thresh).sum())
        sens_list.append((all_tp_scores >= thresh).sum())
    # Append the point where threshold is above all scores (0 detections)
    fps_list.append(0)
    sens_list.append(0)

    fps_per_image = np.asarray(fps_list, dtype=np.float64) / float(num_images)
    sensitivities = np.asarray(sens_list, dtype=np.float64) / float(num_lesions)

    return fps_per_image, sensitivities


def node21_score(fps_per_image, sensitivities,
                 fp_levels=None):
    """Interpolate sensitivity at the official NODE21 FP/image levels.

    Returns:
        score: mean sensitivity (the NODE21 metric).
        per_level: dict mapping each FP level to its interpolated sensitivity.
    """
    if fp_levels is None:
        fp_levels = NODE21_FP_LEVELS
    # np.interp expects x-coords in increasing order
    interp_sens = np.interp(fp_levels,
                            fps_per_image[::-1],
                            sensitivities[::-1])
    per_level = {fp: float(s) for fp, s in zip(fp_levels, interp_sens)}
    return float(np.mean(interp_sens)), per_level


def compute_image_auroc(gt_labels, pred_max_scores):
    """Image-level AUROC (binary: has nodule yes/no).

    Args:
        gt_labels: array of 0/1 per image.
        pred_max_scores: max detection confidence per image.

    Returns:
        AUROC value (float), or None if only one class present.
    """
    gt_labels = np.asarray(gt_labels, dtype=np.int32)
    pred_max_scores = np.asarray(pred_max_scores, dtype=np.float64)
    if len(np.unique(gt_labels)) < 2:
        print("[WARN] Only one class present in gt_labels; AUROC undefined.")
        return None
    return float(roc_auc_score(gt_labels, pred_max_scores))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_csv, img_dir):
    """Load the evaluation CSV and group annotations by image.

    Returns:
        image_names: list of unique image file names.
        gt_boxes_dict: dict  img_name -> np.ndarray (N, 4) in xyxy.
        gt_labels_dict: dict img_name -> int (1 if any nodule, else 0).
    """
    df = pd.read_csv(data_csv)
    image_names = df["img_name"].unique().tolist()

    gt_boxes_dict = {}
    gt_labels_dict = {}
    for img_name in image_names:
        rows = df[df["img_name"] == img_name]
        label = int(rows["label"].values[0])
        gt_labels_dict[img_name] = 1 if label == 1 else 0
        boxes = []
        if label == 1:
            for _, row in rows.iterrows():
                x1 = float(row["x"])
                y1 = float(row["y"])
                x2 = x1 + float(row["width"])
                y2 = y1 + float(row["height"])
                boxes.append([x1, y1, x2, y2])
        gt_boxes_dict[img_name] = np.array(boxes, dtype=np.float64).reshape(-1, 4)

    return image_names, gt_boxes_dict, gt_labels_dict


def load_image(img_path, multichannel=False):
    """Load a PNG image and return a float32 numpy array normalised to [0, 1].

    If multichannel is True, the single-channel image is stacked to 3 channels
    (required by Faster R-CNN with a ResNet backbone pretrained on ImageNet).
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    # Ensure 2-D
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    if multichannel:
        img = np.stack([img, img, img], axis=0)  # (3, H, W)
    else:
        img = img[np.newaxis, ...]  # (1, H, W)
    return img


# ---------------------------------------------------------------------------
# Faster R-CNN evaluation
# ---------------------------------------------------------------------------

def _build_frcnn_model(num_classes=2):
    """Rebuild Faster R-CNN with the same architecture used in training."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    return model


def evaluate_frcnn(model_path, data_csv, img_dir, device, multichannel=True):
    """Evaluate Faster R-CNN on the validation set.

    Returns:
        dict with keys: model_name, fps_per_image, sensitivities,
            node21, auroc, competition_metric, per_fp_sens,
            inference_time_ms, num_images.
    """
    print("\n" + "=" * 60)
    print("Evaluating Faster R-CNN")
    print("=" * 60)

    image_names, gt_boxes_dict, gt_labels_dict = load_dataset(data_csv, img_dir)
    num_images = len(image_names)

    # Build and load model
    model = _build_frcnn_model(num_classes=2)
    ckpt = torch.load(model_path, map_location="cpu")
    # Handle checkpoints saved as {"model_state_dict": ...} or raw state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Lightning-style checkpoint
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            # Strip prefix like "model." that Lightning adds
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            state_dict[new_key] = v
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    gt_boxes_list = []
    pred_boxes_list = []
    pred_scores_list = []
    gt_labels = []
    pred_max_scores = []
    all_pred_info = []  # for visualisation ranking
    total_time = 0.0

    score_thresh = 0.005  # very low to capture full FROC curve

    with torch.no_grad():
        for img_name in image_names:
            fname = img_name if img_name.endswith(".png") else f"{img_name}.png"
            img_path = os.path.join(img_dir, fname)
            img_np = load_image(img_path, multichannel=multichannel)
            tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, C, H, W)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            preds = model(tensor)[0]
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            total_time += (t1 - t0)

            scores = preds["scores"].cpu().numpy()
            boxes = preds["boxes"].cpu().numpy()

            # Filter by score threshold
            keep = scores >= score_thresh
            scores = scores[keep]
            boxes = boxes[keep]

            gt_boxes_list.append(gt_boxes_dict[img_name])
            pred_boxes_list.append(boxes)
            pred_scores_list.append(scores)
            gt_labels.append(gt_labels_dict[img_name])
            pred_max_scores.append(float(scores.max()) if len(scores) > 0 else 0.0)

            all_pred_info.append({
                "img_name": img_name,
                "gt_boxes": gt_boxes_dict[img_name],
                "pred_boxes": boxes,
                "pred_scores": scores,
                "gt_label": gt_labels_dict[img_name],
                "max_score": pred_max_scores[-1],
            })

    # Compute metrics
    fps_pi, sens = compute_froc(gt_boxes_list, pred_boxes_list,
                                pred_scores_list, num_images)
    n21, per_fp = node21_score(fps_pi, sens)
    auroc = compute_image_auroc(gt_labels, pred_max_scores)
    froc_025 = per_fp.get(0.25, 0.0)
    cm = 0.75 * (auroc if auroc is not None else 0.0) + 0.25 * froc_025
    avg_time_ms = (total_time / num_images) * 1000.0

    return {
        "model_name": "Faster R-CNN",
        "fps_per_image": fps_pi,
        "sensitivities": sens,
        "node21": n21,
        "auroc": auroc,
        "competition_metric": cm,
        "per_fp_sens": per_fp,
        "inference_time_ms": avg_time_ms,
        "num_images": num_images,
        "pred_info": all_pred_info,
    }


# ---------------------------------------------------------------------------
# YOLO (v8 / 26) evaluation
# ---------------------------------------------------------------------------

def evaluate_yolo(model_path, data_csv, img_dir, device,
                  model_name="YOLOv8"):
    """Evaluate a YOLO model (v8 or 26) using the Ultralytics API.

    Returns:
        dict with the same keys as evaluate_frcnn.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    image_names, gt_boxes_dict, gt_labels_dict = load_dataset(data_csv, img_dir)
    num_images = len(image_names)

    model = YOLO(model_path)

    gt_boxes_list = []
    pred_boxes_list = []
    pred_scores_list = []
    gt_labels = []
    pred_max_scores = []
    all_pred_info = []
    total_time = 0.0

    for img_name in image_names:
        fname = img_name if img_name.endswith(".png") else f"{img_name}.png"
        img_path = os.path.join(img_dir, fname)

        torch.cuda.synchronize() if "cuda" in str(device) else None
        t0 = time.perf_counter()
        results = model.predict(
            source=img_path,
            conf=0.001,
            iou=0.5,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
        )
        torch.cuda.synchronize() if "cuda" in str(device) else None
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        gt_boxes_list.append(gt_boxes_dict[img_name])
        pred_boxes_list.append(boxes)
        pred_scores_list.append(scores)
        gt_labels.append(gt_labels_dict[img_name])
        pred_max_scores.append(float(scores.max()) if len(scores) > 0 else 0.0)

        all_pred_info.append({
            "img_name": img_name,
            "gt_boxes": gt_boxes_dict[img_name],
            "pred_boxes": boxes,
            "pred_scores": scores,
            "gt_label": gt_labels_dict[img_name],
            "max_score": pred_max_scores[-1],
        })

    # Compute metrics
    fps_pi, sens = compute_froc(gt_boxes_list, pred_boxes_list,
                                pred_scores_list, num_images)
    n21, per_fp = node21_score(fps_pi, sens)
    auroc = compute_image_auroc(gt_labels, pred_max_scores)
    froc_025 = per_fp.get(0.25, 0.0)
    cm = 0.75 * (auroc if auroc is not None else 0.0) + 0.25 * froc_025
    avg_time_ms = (total_time / num_images) * 1000.0

    return {
        "model_name": model_name,
        "fps_per_image": fps_pi,
        "sensitivities": sens,
        "node21": n21,
        "auroc": auroc,
        "competition_metric": cm,
        "per_fp_sens": per_fp,
        "inference_time_ms": avg_time_ms,
        "num_images": num_images,
        "pred_info": all_pred_info,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_froc_comparison(results_dict, save_path):
    """Plot overlaid FROC curves for all evaluated models.

    Args:
        results_dict: dict model_name -> result dict from evaluate_*.
        save_path: file path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, res) in enumerate(results_dict.items()):
        fps = res["fps_per_image"]
        sens = res["sensitivities"]
        n21 = res["node21"]
        color = colors[idx % len(colors)]
        ax.plot(fps, sens, label=f"{name}  (NODE21={n21:.4f})", color=color,
                linewidth=2)

        # Mark official FP levels
        per_fp = res["per_fp_sens"]
        for fp_level, s_val in per_fp.items():
            ax.plot(fp_level, s_val, "o", color=color, markersize=6)
            ax.annotate(f"{s_val:.2f}", (fp_level, s_val),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color)

    ax.set_xscale("log")
    ax.set_xlim(left=0.1, right=64)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Average FP per Image", fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.set_title("FROC Curve Comparison - Pulmonary Nodule Detection", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Draw vertical lines at official FP levels
    for fp_level in NODE21_FP_LEVELS:
        ax.axvline(x=fp_level, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] FROC comparison plot saved to {save_path}")


def plot_detections(image, gt_boxes, pred_boxes, scores, save_path,
                    threshold=0.3):
    """Draw GT (green) and predicted (red) bounding boxes on an image.

    Args:
        image: numpy array (H, W) or (H, W, 3), float [0,1] or uint8.
        gt_boxes: (N, 4) array in xyxy.
        pred_boxes: (M, 4) array in xyxy.
        scores: (M,) array of confidence scores.
        save_path: output file path.
        threshold: minimum score to draw a prediction box.
    """
    if image.dtype != np.uint8:
        vis = (image * 255).astype(np.uint8)
    else:
        vis = image.copy()

    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # GT boxes in green
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "GT", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Predictions in red
    for box, sc in zip(pred_boxes, scores):
        if sc < threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{sc:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_report(results_dict, output_dir):
    """Save evaluation_report.csv with all metrics per model."""
    rows = []
    for name, res in results_dict.items():
        row = {
            "model": name,
            "NODE21_score": res["node21"],
            "AUROC": res["auroc"],
            "Competition_Metric": res["competition_metric"],
            "Inference_ms_per_img": res["inference_time_ms"],
            "num_images": res["num_images"],
        }
        for fp_level in NODE21_FP_LEVELS:
            row[f"Sens@FP={fp_level}"] = res["per_fp_sens"].get(fp_level, None)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "evaluation_report.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Report saved to {csv_path}")
    return df


def print_summary(results_dict):
    """Print a formatted summary table to the console."""
    header = (
        f"{'Model':<16} {'NODE21':>8} {'AUROC':>8} {'CM':>8} "
        f"{'ms/img':>8} "
    )
    for fp in NODE21_FP_LEVELS:
        header += f"{'S@' + str(fp):>8}"
    print("\n" + "=" * len(header))
    print("EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, res in results_dict.items():
        auroc_str = f"{res['auroc']:.4f}" if res["auroc"] is not None else "  N/A "
        line = (
            f"{name:<16} {res['node21']:>8.4f} {auroc_str:>8} "
            f"{res['competition_metric']:>8.4f} "
            f"{res['inference_time_ms']:>8.1f} "
        )
        for fp in NODE21_FP_LEVELS:
            s = res["per_fp_sens"].get(fp, 0.0)
            line += f"{s:>8.4f}"
        print(line)
    print("=" * len(header))


def save_visualisations(results_dict, img_dir, output_dir, num_vis=30):
    """Save example detection images: top-scoring (best) and lowest-scoring
    positive images (worst).

    For each model we save up to num_vis images total:
      - 2/3 are "best" (highest max pred score on positive images)
      - 1/3 are "worst" (lowest max pred score on positive images)
    """
    det_dir = os.path.join(output_dir, "detections")
    os.makedirs(det_dir, exist_ok=True)

    num_best = int(num_vis * 2 / 3)
    num_worst = num_vis - num_best

    for model_name, res in results_dict.items():
        safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
        pred_info = res["pred_info"]

        # Rank positive images by max score
        positive = [p for p in pred_info if p["gt_label"] == 1]
        positive.sort(key=lambda p: p["max_score"], reverse=True)

        best = positive[:num_best]
        worst = list(reversed(positive[-num_worst:])) if len(positive) >= num_worst else list(reversed(positive))

        for tag, subset in [("best", best), ("worst", worst)]:
            for i, info in enumerate(subset):
                fname = info["img_name"]
                if not fname.endswith(".png"):
                    fname = f"{fname}.png"
                img_path = os.path.join(img_dir, fname)
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.astype(np.float32)
                max_val = img.max()
                if max_val > 0:
                    img = img / max_val

                fname = f"{safe_name}_{tag}_{i:02d}_{info['img_name']}"
                save_path = os.path.join(det_dir, fname)
                plot_detections(
                    img, info["gt_boxes"], info["pred_boxes"],
                    info["pred_scores"], save_path, threshold=0.3
                )
        print(f"[INFO] Visualisations for {model_name} saved to {det_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nodule detection models (Faster R-CNN, YOLOv8, YOLO26)"
    )
    parser.add_argument("--frcnn_checkpoint", type=str, default=None,
                        help="Path to Faster R-CNN checkpoint (.pth)")
    parser.add_argument("--yolov8_checkpoint", type=str, default=None,
                        help="Path to YOLOv8 checkpoint best.pt")
    parser.add_argument("--yolo26_checkpoint", type=str, default=None,
                        help="Path to YOLO26 checkpoint best.pt")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to validation CSV")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing PNG images")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Output directory for reports and plots")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference")
    parser.add_argument("--num_vis", type=int, default=30,
                        help="Number of visualisation images to save")
    parser.add_argument("--multichannel", action="store_true",
                        help="Use 3-channel input for Faster R-CNN (stacked grayscale)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Check that at least one checkpoint is provided
    checkpoints = {
        "Faster R-CNN": args.frcnn_checkpoint,
        "YOLOv8": args.yolov8_checkpoint,
        "YOLO26": args.yolo26_checkpoint,
    }
    active = {k: v for k, v in checkpoints.items() if v is not None}
    if not active:
        parser.error("At least one checkpoint must be provided.")

    print(f"[INFO] Models to evaluate: {list(active.keys())}")

    # Run evaluation for each model
    results_dict = {}
    for name, ckpt_path in active.items():
        if name == "Faster R-CNN":
            res = evaluate_frcnn(ckpt_path, args.data_csv, args.img_dir,
                                 device, multichannel=args.multichannel)
        elif name in ("YOLOv8", "YOLO26"):
            res = evaluate_yolo(ckpt_path, args.data_csv, args.img_dir,
                                device, model_name=name)
        else:
            continue
        results_dict[name] = res

    # Generate outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. CSV report
    save_report(results_dict, args.output_dir)

    # 2. FROC comparison plot
    froc_path = os.path.join(args.output_dir, "froc_comparison.png")
    plot_froc_comparison(results_dict, froc_path)

    # 3. Detection visualisations
    save_visualisations(results_dict, args.img_dir, args.output_dir,
                        num_vis=args.num_vis)

    # 4. Console summary
    print_summary(results_dict)

    print(f"\n[INFO] All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
