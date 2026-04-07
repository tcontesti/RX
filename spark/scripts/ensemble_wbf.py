#!/usr/bin/env python3
"""WBF Ensemble of FRCNN + YOLOv8 for pulmonary nodule detection.

Evaluates individual models and their WBF ensemble on the same val set.
Produces FROC comparison, detection visualizations, and CSV report.
"""

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
from ensemble_boxes import weighted_boxes_fusion

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

NODE21_FP_LEVELS = [0.25, 0.5, 1, 2, 4, 8]


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_frcnn(path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.roi_heads.score_thresh = 0.005
    model.roi_heads.nms_thresh = 0.2
    model.to(device).eval()
    print(f"[FRCNN] Loaded {path}")
    return model


def load_yolo(path, device):
    from ultralytics import YOLO
    model = YOLO(path)
    print(f"[YOLO]  Loaded {path}")
    return model


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    names = df["img_name"].unique().tolist()
    gt_boxes, gt_labels = {}, {}
    for n in names:
        rows = df[df["img_name"] == n]
        has = int(rows["label"].max())
        gt_labels[n] = has
        bxs = []
        if has:
            for _, r in rows.iterrows():
                if r["label"] == 1 and r["width"] > 0 and r["height"] > 0:
                    bxs.append([float(r["x"]), float(r["y"]),
                                float(r["x"]) + float(r["width"]),
                                float(r["y"]) + float(r["height"])])
        gt_boxes[n] = np.array(bxs, dtype=np.float64).reshape(-1, 4)
    return names, gt_boxes, gt_labels


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
@torch.no_grad()
def infer_frcnn(model, img_path, device):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 4)), np.zeros(0)
    f = img.astype(np.float32)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    t = torch.from_numpy(np.stack([f, f, f])).unsqueeze(0).to(device)
    out = model(t)[0]
    return out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()


def infer_yolo(model, img_path, device):
    res = model.predict(source=img_path, conf=0.001, iou=0.2,
                        imgsz=1024, verbose=False, device=device)
    if len(res[0].boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.conf.cpu().numpy()


# ------------------------------------------------------------------
# WBF
# ------------------------------------------------------------------
def apply_wbf(boxes_a, scores_a, boxes_b, scores_b,
              img_size, weights, iou_thr=0.2, skip_thr=0.05):
    def norm(b):
        if len(b) == 0:
            return np.zeros((0, 4))
        return np.clip(b / img_size, 0, 1)

    bl = [norm(boxes_a).tolist(), norm(boxes_b).tolist()]
    sl = [scores_a.tolist() if len(scores_a) else [],
          scores_b.tolist() if len(scores_b) else []]
    ll = [[0] * len(scores_a), [0] * len(scores_b)]

    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll,
                                       weights=weights,
                                       iou_thr=iou_thr,
                                       skip_box_thr=skip_thr)
    return np.array(fb) * img_size, np.array(fs), np.array(fl)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def compute_froc(gt_list, pred_list, score_list, n_imgs, iou_t=0.2):
    all_det = []
    total_gt = 0
    for gt, pb, sc in zip(gt_list, pred_list, score_list):
        total_gt += len(gt)
        matched = set()
        if len(sc) == 0:
            continue
        order = np.argsort(-sc)
        for i in order:
            best_iou, best_g = 0.0, -1
            for g in range(len(gt)):
                if g in matched:
                    continue
                iou = compute_iou(pb[i], gt[g])
                if iou > best_iou:
                    best_iou, best_g = iou, g
            if best_iou >= iou_t and best_g >= 0:
                all_det.append((sc[i], True))
                matched.add(best_g)
            else:
                all_det.append((sc[i], False))

    if total_gt == 0 or not all_det:
        return np.array([0.0]), np.array([0.0])

    all_det.sort(key=lambda x: -x[0])
    tp = fp = 0
    fps_l, sens_l = [], []
    for _, is_tp in all_det:
        if is_tp:
            tp += 1
        else:
            fp += 1
        fps_l.append(fp / n_imgs)
        sens_l.append(tp / total_gt)
    return np.array(fps_l), np.array(sens_l)


def node21_score(fps, sens):
    per = {}
    for tfp in NODE21_FP_LEVELS:
        s = 0.0
        for f, si in zip(fps, sens):
            if f <= tfp:
                s = si
            else:
                break
        per[tfp] = s
    return float(np.mean(list(per.values()))), per


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
def plot_froc(results, save_path):
    colors = {"FRCNN": "#1f77b4", "YOLOv8s": "#ff7f0e",
              "Ensemble WBF": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, r in results.items():
        c = colors.get(name, "#333")
        ax.plot(r["fps"], r["sens"], lw=2, color=c,
                label=f"{name} (NODE21={r['n21']:.4f})")
        for fp in NODE21_FP_LEVELS:
            s = np.interp(fp, r["fps"], r["sens"])
            ax.plot(fp, s, "o", color=c, ms=6)
            ax.annotate(f"{s:.2f}", (fp, s), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=c)
    for fp in NODE21_FP_LEVELS:
        ax.axvline(fp, color="gray", ls="--", alpha=0.3)
    ax.set_xscale("log"); ax.set_xlim([0.1, 30]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("Average FP per Image", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("FROC — FRCNN vs YOLOv8 vs Ensemble WBF", fontsize=14)
    ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)
    print(f"[INFO] FROC plot saved to {save_path}")


def save_detections(vis_data, img_dir, out_dir, n_best=20, n_worst=10):
    os.makedirs(out_dir, exist_ok=True)
    # Sort by max ensemble score descending
    vis_data.sort(key=lambda x: -x["max_score"])
    selected = vis_data[:n_best] + vis_data[-n_worst:]

    for i, v in enumerate(selected):
        fname = v["name"] if v["name"].endswith(".png") else f"{v['name']}.png"
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        # GT (green)
        for b in v["gt"]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 0), 2)
        # Predictions (red)
        for b, s in zip(v["boxes"], v["scores"]):
            if s >= 0.3:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 0, 255), 2)
                cv2.putText(img, f"{s:.2f}", (int(b[0]), int(b[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tag = "best" if i < n_best else "worst"
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{i:03d}_{v['name']}.png"), img)
    print(f"[INFO] {len(selected)} visualizations saved to {out_dir}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frcnn", default="checkpoints/frcnn_corrected/best_node21.pth")
    p.add_argument("--yolo", default="checkpoints/yolo/yolov8s/best.pt")
    p.add_argument("--data_csv", default="data/splits/val_fold0.csv")
    p.add_argument("--img_dir", default="data/png_images")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wbf_weights", nargs=2, type=float, default=[0.90, 0.91])
    p.add_argument("--iou_thr", type=float, default=0.2)
    p.add_argument("--skip_thr", type=float, default=0.05)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    names, gt_boxes, gt_labels = load_dataset(args.data_csv)
    n_imgs = len(names)
    print(f"[INFO] {n_imgs} validation images")

    frcnn = load_frcnn(args.frcnn, device)
    yolo = load_yolo(args.yolo, device)

    # Storage for each model + ensemble
    res = {"FRCNN": {"gt": [], "pb": [], "sc": [], "gl": [], "ms": []},
           "YOLOv8s": {"gt": [], "pb": [], "sc": [], "gl": [], "ms": []},
           "Ensemble WBF": {"gt": [], "pb": [], "sc": [], "gl": [], "ms": []}}
    vis_data = []

    print(f"[INFO] Running inference on {n_imgs} images...")
    for i, name in enumerate(names):
        fname = name if name.endswith(".png") else f"{name}.png"
        path = os.path.join(args.img_dir, fname)
        gt = gt_boxes[name]
        gl = gt_labels[name]
        h, w = 1024, 1024  # NODE21 images are 1024x1024

        # Read actual size
        img_check = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_check is not None:
            h, w = img_check.shape[:2]

        # FRCNN
        fb, fs = infer_frcnn(frcnn, path, device)
        # YOLOv8
        yb, ys = infer_yolo(yolo, path, device)
        # WBF
        eb, es, _ = apply_wbf(fb, fs, yb, ys, max(h, w),
                              args.wbf_weights, args.iou_thr, args.skip_thr)

        for key, boxes, scores in [("FRCNN", fb, fs),
                                   ("YOLOv8s", yb, ys),
                                   ("Ensemble WBF", eb, es)]:
            res[key]["gt"].append(gt)
            res[key]["pb"].append(boxes)
            res[key]["sc"].append(scores)
            res[key]["gl"].append(gl)
            res[key]["ms"].append(float(scores.max()) if len(scores) else 0.0)

        vis_data.append({
            "name": name, "gt": gt,
            "boxes": eb, "scores": es,
            "max_score": float(es.max()) if len(es) else 0.0,
        })

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_imgs}")

    # Compute metrics
    print("\n" + "=" * 110)
    print("EVALUATION SUMMARY")
    print("=" * 110)
    header = (f"{'Model':<20} {'NODE21':>7} {'AUROC':>7} {'CM':>7} "
              f"{'S@0.25':>7} {'S@0.5':>7} {'S@1':>7} {'S@2':>7} {'S@4':>7} {'S@8':>7}")
    print(header)
    print("-" * 110)

    plot_data = {}
    rows = []
    for model_name in ["FRCNN", "YOLOv8s", "Ensemble WBF"]:
        r = res[model_name]
        fps, sens = compute_froc(r["gt"], r["pb"], r["sc"], n_imgs, args.iou_thr)
        n21, per = node21_score(fps, sens)

        gl_arr = np.array(r["gl"])
        ms_arr = np.array(r["ms"])
        auroc = float(roc_auc_score(gl_arr, ms_arr)) if len(np.unique(gl_arr)) > 1 else 0.0
        cm = 0.75 * auroc + 0.25 * per[0.25]

        plot_data[model_name] = {"fps": fps, "sens": sens, "n21": n21}

        print(f"{model_name:<20} {n21:>7.4f} {auroc:>7.4f} {cm:>7.4f} "
              f"{per[0.25]:>7.3f} {per[0.5]:>7.3f} {per[1]:>7.3f} "
              f"{per[2]:>7.3f} {per[4]:>7.3f} {per[8]:>7.3f}")

        row = {"model": model_name, "NODE21": n21, "AUROC": auroc, "CM": cm}
        for fp in NODE21_FP_LEVELS:
            row[f"S@FP={fp}"] = per[fp]
        rows.append(row)

    print("=" * 110)

    # Save
    csv_path = os.path.join(args.output_dir, "ensemble_report.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[INFO] Report saved to {csv_path}")

    plot_froc(plot_data, os.path.join(args.output_dir, "froc_ensemble.png"))
    save_detections(vis_data, args.img_dir,
                    os.path.join(args.output_dir, "ensemble_detections"))


if __name__ == "__main__":
    main()
