#!/usr/bin/env python3
"""
Train Faster R-CNN with VinDr pre-trained weights for pulmonary nodule detection.

v2 fixes:
  - Save checkpoint by NODE21 score (not val_loss)
  - Lower score_thresh (0.005) and nms_thresh (0.2) at inference
  - ReduceLROnPlateau on NODE21 score
  - FROC eval every 2 epochs
  - Improved augmentation
  - Early stopping on NODE21 score (patience=20)

Usage:
    python scripts/train_frcnn_vindr.py \
        --data_csv data/splits/train_fold0.csv \
        --val_csv data/splits/val_fold0.csv \
        --img_dir data/png_images/ \
        --weights_path weights/fastercnn50.pth \
        --multichannel \
        --epochs 50 \
        --batch_size 2
"""

import argparse
import math
import os
import sys
import time
import warnings

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output so background runs show progress
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Node21DetectionDataset(torch.utils.data.Dataset):
    """Dataset for NODE21-style detection CSVs."""

    def __init__(self, csv_path, img_dir, transforms=None, multichannel=False):
        self.img_dir = img_dir
        self.transforms = transforms
        self.multichannel = multichannel

        df = pd.read_csv(csv_path)
        self.image_names = df["img_name"].unique().tolist()
        self.annotations = {}
        for name in self.image_names:
            rows = df[df["img_name"] == name]
            boxes = []
            for _, row in rows.iterrows():
                if row["label"] == 1 and not (
                    pd.isna(row["x"]) or pd.isna(row["y"])
                    or pd.isna(row["width"]) or pd.isna(row["height"])
                ):
                    x1 = float(row["x"])
                    y1 = float(row["y"])
                    w = float(row["width"])
                    h = float(row["height"])
                    if w > 0 and h > 0:
                        boxes.append([x1, y1, x1 + w, y1 + h])
            self.annotations[name] = boxes

    def __len__(self):
        return len(self.image_names)

    def _make_multichannel(self, gray):
        """Create 3-channel image: Original + CLAHE + Unsharp Mask."""
        ch0 = gray.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_u8 = (gray * 255).astype(np.uint8)
        ch1 = clahe.apply(gray_u8).astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ch2 = np.clip(1.5 * gray + (-0.5) * blurred, 0.0, 1.0).astype(np.float32)
        return np.stack([ch0, ch1, ch2], axis=-1)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        fname = img_name if img_name.endswith(".png") else f"{img_name}.png"
        img_path = os.path.join(self.img_dir, fname)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        gray = gray.astype(np.float32) / 255.0

        if self.multichannel:
            image = self._make_multichannel(gray)
        else:
            image = np.stack([gray, gray, gray], axis=-1)

        boxes = self.annotations[img_name]

        if len(boxes) > 0:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.ones(len(boxes), dtype=np.int64)
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes_arr.tolist() if len(boxes_arr) > 0 else [],
                labels=labels_arr.tolist() if len(labels_arr) > 0 else [],
            )
            image = transformed["image"]
            if len(transformed["bboxes"]) > 0:
                boxes_arr = np.array(transformed["bboxes"], dtype=np.float32)
                labels_arr = np.array(transformed["labels"], dtype=np.int64)
            else:
                boxes_arr = np.zeros((0, 4), dtype=np.float32)
                labels_arr = np.zeros((0,), dtype=np.int64)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        boxes_tensor = torch.as_tensor(boxes_arr, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels_arr, dtype=torch.int64)

        if boxes_tensor.numel() > 0:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target


# ---------------------------------------------------------------------------
# Augmentations (improved)
# ---------------------------------------------------------------------------
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(weights_path, num_classes=2, device="cuda:0"):
    """Build Faster R-CNN with VinDr pre-trained weights.

    Key: set low score_thresh and nms_thresh for high-recall inference.
    """
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # Load VinDr weights (remove old head)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    keys_to_remove = [k for k in state_dict if k.startswith("roi_heads.box_predictor.")]
    for k in keys_to_remove:
        del state_dict[k]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    total_loaded = len(state_dict) - len(unexpected)
    total_model = len(dict(model.named_parameters())) + len(
        dict(model.named_buffers())
    )
    print(f"[Model] Loaded VinDr weights from {weights_path}")
    print(f"  Keys loaded: {total_loaded}, Missing: {len(missing)} (new head), "
          f"Unexpected: {len(unexpected)}")

    # Replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # LOW thresholds for high-recall inference during FROC evaluation
    model.roi_heads.score_thresh = 0.005
    model.roi_heads.nms_thresh = 0.2

    # Freeze strategy
    freeze_prefixes = [
        "backbone.body.conv1",
        "backbone.body.bn1",
        "backbone.body.layer1",
        "backbone.body.layer2",
        "backbone.body.layer3",
    ]
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_prefixes):
            param.requires_grad = False

    # Verification printout
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"  Trainable: {trainable:,} / {total:,}  (frozen: {frozen:,})")

    # Verify freeze correctness
    for check_name in ["backbone.body.layer1.0.conv1.weight",
                       "backbone.body.layer3.0.conv1.weight"]:
        for n, p in model.named_parameters():
            if n == check_name:
                assert not p.requires_grad, f"{check_name} should be frozen!"
                break
    for check_name in ["backbone.body.layer4.0.conv1.weight",
                       "backbone.fpn.inner_blocks.0.weight"]:
        for n, p in model.named_parameters():
            if n == check_name:
                assert p.requires_grad, f"{check_name} should be trainable!"
                break
    print("  Freeze verification: PASSED (layer1-3 frozen, layer4+FPN trainable)")

    print(f"  score_thresh={model.roi_heads.score_thresh}, "
          f"nms_thresh={model.roi_heads.nms_thresh}")

    return model.to(device)


# ---------------------------------------------------------------------------
# FROC Evaluation
# ---------------------------------------------------------------------------
def compute_froc(model, dataloader, device, iou_threshold=0.2):
    """Compute FROC curve and NODE21 score."""
    model.eval()
    all_det_scores = []
    all_num_gt = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                num_gt = len(gt_boxes)
                all_num_gt.append(num_gt)

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                gt_matched = np.zeros(num_gt, dtype=bool)
                order = np.argsort(-pred_scores)

                for j in order:
                    pb = pred_boxes[j]
                    score = pred_scores[j]
                    matched = False

                    if num_gt > 0:
                        ious = _iou_single_vs_many(pb, gt_boxes)
                        best_gt = np.argmax(ious)
                        if ious[best_gt] >= iou_threshold and not gt_matched[best_gt]:
                            gt_matched[best_gt] = True
                            matched = True

                    all_det_scores.append((score, matched))

    num_images = len(all_num_gt)
    total_gt = sum(all_num_gt)

    if total_gt == 0 or num_images == 0:
        return 0.0, [], []

    all_det_scores.sort(key=lambda x: -x[0])

    fps_per_image = []
    sensitivities = []
    tp = 0
    fp = 0

    for score, matched in all_det_scores:
        if matched:
            tp += 1
        else:
            fp += 1
        fps_per_image.append(fp / num_images)
        sensitivities.append(tp / total_gt)

    target_fps = [0.25, 0.5, 1, 2, 4, 8]
    sens_at_fp = []
    for target_fp in target_fps:
        s = 0.0
        for fpi, si in zip(fps_per_image, sensitivities):
            if fpi <= target_fp:
                s = si
            else:
                break
        sens_at_fp.append(s)

    node21_score = float(np.mean(sens_at_fp))
    return node21_score, fps_per_image, sensitivities


def _iou_single_vs_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-6)


def plot_froc(fps_per_image, sensitivities, node21_score, epoch, output_dir):
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fps_per_image, sensitivities, linewidth=2)
    ax.set_xlabel("Average FP per image")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"FROC Curve  |  Epoch {epoch}  |  NODE21 Score: {node21_score:.4f}")
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    for fp_val in [0.25, 0.5, 1, 2, 4, 8]:
        ax.axvline(x=fp_val, color="gray", linestyle="--", alpha=0.4)
    fig.tight_layout()
    epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    fig.savefig(os.path.join(results_dir, f"froc_epoch_{epoch_str}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not math.isfinite(losses.item()):
            print(f"  [WARN] NaN/Inf loss, skipping batch.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if math.isfinite(losses.item()):
            total_loss += losses.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("=" * 70)
    print("  Faster R-CNN v2 | VinDr Pre-trained | NODE21 Score Checkpoint")
    print("=" * 70)
    print(f"  multichannel={args.multichannel}, lr={args.lr}, "
          f"bs={args.batch_size}, epochs={args.epochs}, fold={args.fold}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint criterion: NODE21 score (FROC every 2 epochs)")
    print(f"  Early stopping: patience=20 on NODE21 score")
    print(f"  LR scheduler: ReduceLROnPlateau(patience=5, factor=0.5)")
    print()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    # Datasets
    train_transforms = get_train_transforms()
    train_dataset = Node21DetectionDataset(
        args.data_csv, args.img_dir, train_transforms, args.multichannel
    )
    val_dataset = Node21DetectionDataset(
        args.val_csv, args.img_dir, None, args.multichannel
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"  Train images: {len(train_dataset)}")
    print(f"  Val   images: {len(val_dataset)}")
    print()

    # Model
    model = build_model(args.weights_path, num_classes=2, device=device)

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau monitors NODE21 score (mode=max)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    # Early stopping based on NODE21 score
    best_node21 = 0.0
    patience_counter = 0
    patience = 20
    last_node21 = 0.0

    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = validate(model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # FROC evaluation every 2 epochs
        if epoch % 2 == 0 or epoch == 1:
            node21, fps_list, sens_list = compute_froc(
                model, val_loader, device, iou_threshold=0.2
            )
            last_node21 = node21

            print(
                f"Epoch [{epoch:3d}/{args.epochs}]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"NODE21={node21:.4f}  lr={current_lr:.2e}  time={elapsed:.1f}s"
            )

            if len(fps_list) > 0:
                plot_froc(fps_list, sens_list, node21, epoch, args.output_dir)

            # Step scheduler with NODE21 score
            scheduler.step(node21)

            # Save best model by NODE21 score
            if node21 > best_node21:
                best_node21 = node21
                patience_counter = 0
                save_path = os.path.join(
                    args.output_dir, f"best_model_fold{args.fold}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "node21_score": node21,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                    },
                    save_path,
                )
                print(f"  >> NEW BEST NODE21={node21:.4f} -> {save_path}")
            else:
                patience_counter += 1
                print(f"  >> No improvement ({patience_counter}/{patience}), "
                      f"best={best_node21:.4f}")
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no NODE21 improvement for {patience} eval cycles).")
                    break
        else:
            print(
                f"Epoch [{epoch:3d}/{args.epochs}]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={current_lr:.2e}  time={elapsed:.1f}s"
            )

    # Final FROC on best model
    best_ckpt = os.path.join(args.output_dir, f"best_model_fold{args.fold}.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        node21, fps_list, sens_list = compute_froc(
            model, val_loader, device, iou_threshold=0.2
        )
        print(f"\n{'='*70}")
        print(f"Final FROC (best model, epoch {checkpoint['epoch']}):")
        print(f"  NODE21 score: {node21:.4f}")
        print(f"{'='*70}")
        if len(fps_list) > 0:
            plot_froc(fps_list, sens_list, node21, "final", args.output_dir)

    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN v2 with VinDr weights for nodule detection"
    )
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, default="weights/fastercnn50.pth")
    parser.add_argument("--multichannel", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/frcnn_vindr/")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)
