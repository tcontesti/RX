#!/usr/bin/env python3
"""Reproduce Faster R-CNN VinDr training from Colab.

Version A: exact replica (simple split, model.train() in val, save by val_loss)
Version B: corrected (grouped split, model.eval() in val, save by NODE21)

Usage:
    # Version A (replica)
    python scripts/train_frcnn_reproduce.py --csv data/metadata_augmented_spark.csv \
        --output_dir checkpoints/frcnn_reproduce/ --version A

    # Version B (corrected)
    python scripts/train_frcnn_reproduce.py --csv data/metadata_augmented_spark.csv \
        --output_dir checkpoints/frcnn_corrected/ --version B --fold 0
"""

import argparse
import math
import os
import sys
import time
from collections import OrderedDict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ------------------------------------------------------------------
# Dataset (exact replica of original)
# ------------------------------------------------------------------
class Node21DetectionDatasetPNG(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df[self.df["file_path"].apply(os.path.exists)]
        self.images = self.df["file_path"].unique().tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self._empty(idx)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img3 = np.stack([img, img, img], axis=0)  # grayscale x3

        rows = self.df[self.df["file_path"] == path]
        boxes, labels = [], []
        for _, r in rows.iterrows():
            if r["label"] == 1:
                x1, y1 = float(r["x"]), float(r["y"])
                x2, y2 = x1 + float(r["width"]), y1 + float(r["height"])
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target = {
            "boxes": boxes_t, "labels": labels_t, "area": area,
            "iscrowd": torch.zeros(len(labels_t), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return torch.tensor(img3, dtype=torch.float32), target

    def _empty(self, idx):
        return (
            torch.zeros((3, 224, 224), dtype=torch.float32),
            {"boxes": torch.zeros((0, 4), dtype=torch.float32),
             "labels": torch.zeros((0,), dtype=torch.int64),
             "area": torch.zeros((0,), dtype=torch.float32),
             "iscrowd": torch.zeros((0,), dtype=torch.int64),
             "image_id": torch.tensor([idx])},
        )


def detection_collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


# ------------------------------------------------------------------
# Model (exact replica of original)
# ------------------------------------------------------------------
def build_frcnn_vindr(weights_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )

    vindr_state = torch.load(weights_path, map_location=device, weights_only=False)

    # Remove old head keys
    for k in ["roi_heads.box_predictor.cls_score.weight",
              "roi_heads.box_predictor.cls_score.bias",
              "roi_heads.box_predictor.bbox_pred.weight",
              "roi_heads.box_predictor.bbox_pred.bias"]:
        if k in vindr_state:
            del vindr_state[k]

    model.load_state_dict(vindr_state, strict=False)

    # Freeze backbone body, unfreeze layer4
    for param in model.backbone.body.parameters():
        param.requires_grad = False
    for name, param in model.backbone.body.named_parameters():
        if name.startswith("layer4"):
            param.requires_grad = True
    # FPN trainable
    for name, param in model.backbone.named_parameters():
        if "fpn" in name:
            param.requires_grad = True

    # New 2-class head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {total:,} total, {trainable:,} trainable, "
          f"{total - trainable:,} frozen")

    return model.to(device)


# ------------------------------------------------------------------
# Early stopping (same as original)
# ------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ------------------------------------------------------------------
# FROC / NODE21 evaluation
# ------------------------------------------------------------------
def compute_froc_score(model, dataset, device, iou_thresh=0.2):
    """Compute NODE21 score on a dataset."""
    model.eval()
    prev_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.005
    model.roi_heads.nms_thresh = 0.2

    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        collate_fn=detection_collate, num_workers=2)

    all_det = []  # (score, is_tp)
    total_gt = 0
    num_images = len(dataset)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt = tgt["boxes"].numpy()
                total_gt += len(gt)
                pred_boxes = out["boxes"].cpu().numpy()
                pred_scores = out["scores"].cpu().numpy()
                matched = set()
                order = np.argsort(-pred_scores)
                for j in order:
                    pb = pred_boxes[j]
                    best_iou, best_g = 0.0, -1
                    for g in range(len(gt)):
                        if g in matched:
                            continue
                        iou = _iou(pb, gt[g])
                        if iou > best_iou:
                            best_iou = iou; best_g = g
                    if best_iou >= iou_thresh and best_g >= 0:
                        all_det.append((pred_scores[j], True))
                        matched.add(best_g)
                    else:
                        all_det.append((pred_scores[j], False))

    model.roi_heads.score_thresh = prev_thresh

    if total_gt == 0:
        return 0.0, [], []

    all_det.sort(key=lambda x: -x[0])
    tp = fp = 0
    fps_list, sens_list = [], []
    for score, is_tp in all_det:
        if is_tp:
            tp += 1
        else:
            fp += 1
        fps_list.append(fp / num_images)
        sens_list.append(tp / total_gt)

    # NODE21: mean sensitivity at FP = [0.25, 0.5, 1, 2, 4, 8]
    target_fps = [0.25, 0.5, 1, 2, 4, 8]
    sens_at = []
    for tfp in target_fps:
        s = 0.0
        for fpi, si in zip(fps_list, sens_list):
            if fpi <= tfp:
                s = si
            else:
                break
        sens_at.append(s)

    return float(np.mean(sens_at)), fps_list, sens_list


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------
# Split functions
# ------------------------------------------------------------------
def split_version_a(df):
    """Simple split (replica of original — has data leakage)."""
    images = df["file_path"].unique()
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    return df[df["file_path"].isin(train_imgs)], df[df["file_path"].isin(val_imgs)]


def split_version_b(df, fold=0, n_splits=5):
    """Grouped split — no leakage between base and augmented images."""
    # Extract base image name (strip _aug suffix)
    df = df.copy()
    df["base_name"] = df["img_name"].apply(
        lambda x: x.split("_aug")[0].replace(".mha", "").replace(".png", "")
    )

    # Per-image label: 1 if any nodule
    img_df = df.groupby("file_path").agg(
        label_max=("label", "max"),
        base_name=("base_name", "first"),
    ).reset_index()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = img_df["file_path"].values
    y = img_df["label_max"].values
    groups = img_df["base_name"].values

    for i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        if i == fold:
            train_paths = set(X[train_idx])
            val_paths = set(X[val_idx])
            break

    train_df = df[df["file_path"].isin(train_paths)].drop(columns=["base_name"])
    val_df = df[df["file_path"].isin(val_paths)].drop(columns=["base_name"])
    return train_df, val_df


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/metadata_augmented_spark.csv")
    parser.add_argument("--weights", default="weights/fastercnn50.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="checkpoints/frcnn_reproduce/")
    parser.add_argument("--version", choices=["A", "B"], default="A",
                        help="A=replica with leakage, B=corrected no leakage")
    parser.add_argument("--fold", type=int, default=0, help="Fold for version B")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Faster R-CNN VinDr Reproduction — Version {args.version}")
    print(f"  {'REPLICA (simple split, model.train() val)' if args.version == 'A' else 'CORRECTED (grouped split, model.eval() val)'}")
    print("=" * 70)

    # Load data
    df = pd.read_csv(args.csv)
    print(f"CSV: {len(df)} rows, {df['img_name'].nunique()} unique images")

    # Split
    if args.version == "A":
        train_df, val_df = split_version_a(df)
    else:
        train_df, val_df = split_version_b(df, fold=args.fold)

    print(f"Train: {len(train_df)} rows, {train_df['file_path'].nunique()} images "
          f"({(train_df['label']==1).sum()} pos)")
    print(f"Val:   {len(val_df)} rows, {val_df['file_path'].nunique()} images "
          f"({(val_df['label']==1).sum()} pos)")

    # Check leakage
    if args.version == "A":
        train_bases = set(
            p.split("/")[-1].split("_aug")[0].replace(".png", "")
            for p in train_df["file_path"].unique()
        )
        val_bases = set(
            p.split("/")[-1].split("_aug")[0].replace(".png", "")
            for p in val_df["file_path"].unique()
        )
        leak = train_bases & val_bases
        print(f"  Base image overlap (leakage): {len(leak)} images")
    else:
        train_bases = set(
            p.split("/")[-1].split("_aug")[0].replace(".png", "")
            for p in train_df["file_path"].unique()
        )
        val_bases = set(
            p.split("/")[-1].split("_aug")[0].replace(".png", "")
            for p in val_df["file_path"].unique()
        )
        leak = train_bases & val_bases
        print(f"  Base image overlap (should be 0): {len(leak)}")

    # Datasets & loaders
    train_dataset = Node21DetectionDatasetPNG(train_df)
    val_dataset = Node21DetectionDatasetPNG(val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=detection_collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=detection_collate, num_workers=2, pin_memory=True)

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset:   {len(val_dataset)} images")

    # Model
    model = build_frcnn_vindr(args.weights, device)

    # Optimizer (exact same as original)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    # Early stopping on val_loss (for A) or NODE21 (for B)
    if args.version == "A":
        early_stopper = EarlyStopping(patience=15, min_delta=0.0001, mode="min")
    else:
        early_stopper = EarlyStopping(patience=20, min_delta=0.001, mode="max")

    best_val_loss = float("inf")
    best_node21 = 0.0

    print(f"\nStarting training ({args.epochs} epochs)...\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validate (loss) — Faster R-CNN only computes loss in train mode
        # Version B "correction" applies to FROC eval (model.eval()), not loss calc
        model.train()

        val_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    v_batches += 1
        val_loss /= max(v_batches, 1)

        elapsed = time.time() - t0

        # NODE21 evaluation every 2 epochs
        node21 = 0.0
        if (epoch + 1) % 2 == 0:
            node21, _, _ = compute_froc_score(model, val_dataset, device)

        print(f"Epoch [{epoch+1:3d}/{args.epochs}]  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"NODE21={node21:.4f}  time={elapsed:.1f}s")

        # Save by val_loss (as original did)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_valloss.pth"))
            print(f"  >> BEST val_loss={best_val_loss:.4f}")

        # Also save by NODE21
        if node21 > best_node21:
            best_node21 = node21
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_node21.pth"))
            print(f"  >> BEST NODE21={best_node21:.4f}")

        # Early stopping
        if args.version == "A":
            if early_stopper.step(val_loss):
                print(f"\nEarly stopping at epoch {epoch+1} (val_loss)")
                break
        else:
            if (epoch + 1) % 2 == 0 and early_stopper.step(node21):
                print(f"\nEarly stopping at epoch {epoch+1} (NODE21)")
                break

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final evaluation on best_node21 checkpoint:")
    best_path = os.path.join(args.output_dir, "best_node21.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False))
        score, fps, sens = compute_froc_score(model, val_dataset, device)
        print(f"  NODE21 score: {score:.4f}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best NODE21: {best_node21:.4f}")
    print(f"{'='*70}")
    print("Training complete.")


if __name__ == "__main__":
    main()
