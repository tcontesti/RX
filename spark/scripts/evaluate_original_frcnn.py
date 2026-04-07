#!/usr/bin/env python3
"""Evaluate original pre-trained Faster R-CNN VinDr checkpoints alongside
YOLOv8 and YOLO26 results. Produces NODE21 score, AUROC, CM, per-FP
sensitivity, and a 4-model FROC comparison plot."""

import argparse
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

NODE21_FP_LEVELS = [0.25, 0.5, 1, 2, 4, 8]


# ------------------------------------------------------------------
# Model loading (matches the user's exact recipe)
# ------------------------------------------------------------------
def load_frcnn(checkpoint_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Remap keys: backbone.body.body.X -> backbone.body.X (double-body pattern)
    has_double_body = any("backbone.body.body." in k for k in state_dict)
    if has_double_body:
        remapped = {}
        skipped = []
        for k, v in state_dict.items():
            new_key = k.replace("backbone.body.body.", "backbone.body.", 1)
            # Skip CBAM keys that don't exist in standard model
            if "cbam" in new_key:
                skipped.append(k)
                continue
            remapped[new_key] = v
        state_dict = remapped
        print(f"  Remapped {len(remapped)} keys (double-body -> single-body)")
        if skipped:
            print(f"  Skipped {len(skipped)} CBAM-only keys: {skipped}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} — {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} — {unexpected[:5]}...")

    model.roi_heads.score_thresh = 0.005
    model.roi_heads.nms_thresh = 0.2
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {checkpoint_path}")
    print(f"  Parameters: {n_params:,}  score_thresh=0.005  nms_thresh=0.2")
    return model


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    image_names = df["img_name"].unique().tolist()
    gt_boxes_dict = {}
    gt_labels_dict = {}
    for name in image_names:
        rows = df[df["img_name"] == name]
        has_nodule = int(rows["label"].max())
        gt_labels_dict[name] = has_nodule
        boxes = []
        if has_nodule:
            for _, r in rows.iterrows():
                if r["label"] == 1 and r["width"] > 0 and r["height"] > 0:
                    boxes.append([
                        float(r["x"]), float(r["y"]),
                        float(r["x"]) + float(r["width"]),
                        float(r["y"]) + float(r["height"]),
                    ])
        gt_boxes_dict[name] = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    return image_names, gt_boxes_dict, gt_labels_dict


def load_image_gray3ch(img_path):
    """Load grayscale PNG, normalise to [0,1] float32, replicate to 3 channels."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    img = img.astype(np.float32) / 255.0
    return np.stack([img, img, img], axis=0)  # (3, H, W)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_froc(gt_boxes_list, pred_boxes_list, pred_scores_list,
                 num_images, iou_thresh=0.2):
    all_tp_scores, all_fp_scores = [], []
    num_lesions = 0
    for gt_boxes, pred_boxes, pred_scores in zip(
            gt_boxes_list, pred_boxes_list, pred_scores_list):
        num_lesions += len(gt_boxes)
        matched_gt = set()
        if len(pred_scores) == 0:
            continue
        order = np.argsort(-pred_scores)
        for idx in order:
            best_iou, best_g = 0.0, -1
            for g, gb in enumerate(gt_boxes):
                if g in matched_gt:
                    continue
                iou = compute_iou(pred_boxes[idx], gb)
                if iou > best_iou:
                    best_iou = iou; best_g = g
            if best_iou >= iou_thresh and best_g >= 0:
                all_tp_scores.append(pred_scores[idx])
                matched_gt.add(best_g)
            else:
                all_fp_scores.append(pred_scores[idx])

    all_tp = np.asarray(all_tp_scores, dtype=np.float64)
    all_fp = np.asarray(all_fp_scores, dtype=np.float64)
    if num_lesions == 0:
        return np.array([0.0]), np.array([0.0])

    thresholds = sorted(set(all_tp.tolist() + all_fp.tolist()))
    fps_list, sens_list = [], []
    for t in thresholds:
        fps_list.append((all_fp >= t).sum())
        sens_list.append((all_tp >= t).sum())
    fps_list.append(0); sens_list.append(0)

    fps_per_image = np.asarray(fps_list, dtype=np.float64) / num_images
    sensitivities = np.asarray(sens_list, dtype=np.float64) / num_lesions
    return fps_per_image, sensitivities


def node21_score(fps_per_image, sensitivities):
    interp = np.interp(NODE21_FP_LEVELS,
                       fps_per_image[::-1], sensitivities[::-1])
    per_level = {fp: float(s) for fp, s in zip(NODE21_FP_LEVELS, interp)}
    return float(np.mean(interp)), per_level


# ------------------------------------------------------------------
# Evaluate one FRCNN checkpoint
# ------------------------------------------------------------------
def evaluate_frcnn(model, image_names, gt_boxes_dict, gt_labels_dict,
                   img_dir, device):
    gt_boxes_list, pred_boxes_list, pred_scores_list = [], [], []
    gt_labels, pred_max_scores = [], []
    total_time = 0.0

    with torch.no_grad():
        for name in image_names:
            fname = name if name.endswith(".png") else f"{name}.png"
            path = os.path.join(img_dir, fname)
            arr = load_image_gray3ch(path)                     # (3,H,W)
            tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,3,H,W)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = model(tensor)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            scores = preds["scores"].cpu().numpy()
            boxes  = preds["boxes"].cpu().numpy()

            gt_boxes_list.append(gt_boxes_dict[name])
            pred_boxes_list.append(boxes)
            pred_scores_list.append(scores)
            gt_labels.append(gt_labels_dict[name])
            pred_max_scores.append(float(scores.max()) if len(scores) else 0.0)

    num_images = len(image_names)
    fps_pi, sens = compute_froc(gt_boxes_list, pred_boxes_list,
                                pred_scores_list, num_images)
    n21, per_fp = node21_score(fps_pi, sens)

    gt_arr = np.array(gt_labels)
    pm_arr = np.array(pred_max_scores)
    auroc = float(roc_auc_score(gt_arr, pm_arr)) if len(np.unique(gt_arr)) > 1 else None
    froc025 = per_fp.get(0.25, 0.0)
    cm = 0.75 * (auroc if auroc else 0.0) + 0.25 * froc025
    ms_img = (total_time / num_images) * 1000.0

    return dict(fps_per_image=fps_pi, sensitivities=sens,
                node21=n21, auroc=auroc, cm=cm, per_fp=per_fp,
                ms_img=ms_img, num_images=num_images)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_froc_comparison(results, save_path):
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, r) in enumerate(results.items()):
        fps = r["fps_per_image"]; sens = r["sensitivities"]
        ax.plot(fps, sens, linewidth=2, color=colors[i % len(colors)],
                label=f"{name}  (NODE21={r['node21']:.4f})")
        for fp_val in NODE21_FP_LEVELS:
            s = np.interp(fp_val, fps[::-1], sens[::-1])
            ax.plot(fp_val, s, "o", color=colors[i % len(colors)], markersize=6)
            ax.annotate(f"{s:.2f}", (fp_val, s), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=colors[i % len(colors)])

    for fp_val in NODE21_FP_LEVELS:
        ax.axvline(x=fp_val, color="gray", linestyle="--", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Average FP per Image", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("FROC Curve Comparison — Pulmonary Nodule Detection", fontsize=14)
    ax.set_xlim([0.1, 30])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] FROC plot saved to {save_path}")


def print_table(results):
    header = (f"{'Model':<35} {'NODE21':>7} {'AUROC':>7} {'CM':>7} "
              f"{'ms/img':>7}  {'S@0.25':>6} {'S@0.5':>6} {'S@1':>6} "
              f"{'S@2':>6} {'S@4':>6} {'S@8':>6}")
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("EVALUATION SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for name, r in results.items():
        pf = r["per_fp"]
        print(f"{name:<35} {r['node21']:>7.4f} "
              f"{r['auroc']:>7.4f} {r['cm']:>7.4f} {r['ms_img']:>7.1f}  "
              f"{pf[0.25]:>6.3f} {pf[0.5]:>6.3f} {pf[1]:>6.3f} "
              f"{pf[2]:>6.3f} {pf[4]:>6.3f} {pf[8]:>6.3f}")
    print(f"{'='*len(header)}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate original FRCNN checkpoints + compare with YOLO")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to FRCNN .pth checkpoints")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each checkpoint")
    parser.add_argument("--yolov8_checkpoint", default=None)
    parser.add_argument("--yolo26_checkpoint", default=None)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", default="results_original/")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    image_names, gt_boxes_dict, gt_labels_dict = load_dataset(args.data_csv)
    print(f"[INFO] Validation set: {len(image_names)} images")

    results = {}

    # --- Evaluate each FRCNN checkpoint ---
    names = args.names or [os.path.basename(c) for c in args.checkpoints]
    for ckpt_path, display_name in zip(args.checkpoints, names):
        print(f"\n{'='*60}")
        print(f"Evaluating: {display_name}")
        print(f"{'='*60}")
        model = load_frcnn(ckpt_path, device)
        r = evaluate_frcnn(model, image_names, gt_boxes_dict,
                           gt_labels_dict, args.img_dir, device)
        results[display_name] = r
        print(f"  NODE21={r['node21']:.4f}  AUROC={r['auroc']:.4f}  "
              f"CM={r['cm']:.4f}  {r['ms_img']:.1f} ms/img")
        del model
        torch.cuda.empty_cache()

    # --- Evaluate YOLOv8 ---
    if args.yolov8_checkpoint:
        print(f"\n{'='*60}")
        print("Evaluating: YOLOv8s")
        print(f"{'='*60}")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolov8_checkpoint)
        gt_bl, pb_l, ps_l, gt_lab, pm_s = [], [], [], [], []
        total_t = 0.0
        for name in image_names:
            fname = name if name.endswith(".png") else f"{name}.png"
            path = os.path.join(args.img_dir, fname)
            t0 = time.perf_counter()
            res = yolo_model.predict(source=path, conf=0.001, iou=0.2,
                                     imgsz=1024, verbose=False, device=device)
            total_t += time.perf_counter() - t0
            boxes = res[0].boxes.xyxy.cpu().numpy() if len(res[0].boxes) else np.zeros((0,4))
            scores = res[0].boxes.conf.cpu().numpy() if len(res[0].boxes) else np.zeros(0)
            gt_bl.append(gt_boxes_dict[name]); pb_l.append(boxes); ps_l.append(scores)
            gt_lab.append(gt_labels_dict[name])
            pm_s.append(float(scores.max()) if len(scores) else 0.0)
        fps_pi, sens = compute_froc(gt_bl, pb_l, ps_l, len(image_names))
        n21, pf = node21_score(fps_pi, sens)
        auroc = float(roc_auc_score(gt_lab, pm_s))
        results["YOLOv8s"] = dict(fps_per_image=fps_pi, sensitivities=sens,
                                  node21=n21, auroc=auroc,
                                  cm=0.75*auroc+0.25*pf[0.25],
                                  per_fp=pf, ms_img=(total_t/len(image_names))*1000,
                                  num_images=len(image_names))
        print(f"  NODE21={n21:.4f}  AUROC={auroc:.4f}")

    # --- Evaluate YOLO26 ---
    if args.yolo26_checkpoint:
        print(f"\n{'='*60}")
        print("Evaluating: YOLO26s")
        print(f"{'='*60}")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo26_checkpoint)
        gt_bl, pb_l, ps_l, gt_lab, pm_s = [], [], [], [], []
        total_t = 0.0
        for name in image_names:
            fname = name if name.endswith(".png") else f"{name}.png"
            path = os.path.join(args.img_dir, fname)
            t0 = time.perf_counter()
            res = yolo_model.predict(source=path, conf=0.001, iou=0.2,
                                     imgsz=1024, verbose=False, device=device)
            total_t += time.perf_counter() - t0
            boxes = res[0].boxes.xyxy.cpu().numpy() if len(res[0].boxes) else np.zeros((0,4))
            scores = res[0].boxes.conf.cpu().numpy() if len(res[0].boxes) else np.zeros(0)
            gt_bl.append(gt_boxes_dict[name]); pb_l.append(boxes); ps_l.append(scores)
            gt_lab.append(gt_labels_dict[name])
            pm_s.append(float(scores.max()) if len(scores) else 0.0)
        fps_pi, sens = compute_froc(gt_bl, pb_l, ps_l, len(image_names))
        n21, pf = node21_score(fps_pi, sens)
        auroc = float(roc_auc_score(gt_lab, pm_s))
        results["YOLO26s"] = dict(fps_per_image=fps_pi, sensitivities=sens,
                                  node21=n21, auroc=auroc,
                                  cm=0.75*auroc+0.25*pf[0.25],
                                  per_fp=pf, ms_img=(total_t/len(image_names))*1000,
                                  num_images=len(image_names))
        print(f"  NODE21={n21:.4f}  AUROC={auroc:.4f}")

    # --- Summary ---
    print_table(results)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "evaluation_report.csv")
    rows = []
    for name, r in results.items():
        row = {"model": name, "NODE21": r["node21"], "AUROC": r["auroc"],
               "CM": r["cm"], "ms_img": r["ms_img"]}
        for fp in NODE21_FP_LEVELS:
            row[f"S@FP={fp}"] = r["per_fp"][fp]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[INFO] Report saved to {csv_path}")

    # FROC plot
    plot_path = os.path.join(args.output_dir, "froc_comparison.png")
    plot_froc_comparison(results, plot_path)


if __name__ == "__main__":
    main()
