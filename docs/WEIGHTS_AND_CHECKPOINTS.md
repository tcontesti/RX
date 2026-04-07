# Weights and Checkpoints Inventory

## Pre-trained Weights (weights/)

| File | Size | Source | Description |
|------|------|--------|-------------|
| `fastercnn50.pth` | 159MB | VinDr-CXR | Faster R-CNN ResNet50-FPN pre-trained on VinDr-CXR (14 pathologies) |
| `yolo5x_vindr.pt` | 170MB | VinDr-CXR | YOLOv5x pre-trained on VinDr-CXR |
| `F1_E79_ModelX_v4_T0.325_V0.410.ckpt` | 41MB | VinDr-CXR | EfficientDet pre-trained on VinDr-CXR |

### Loading fastercnn50.pth (VinDr)
```python
import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
state_dict = torch.load("weights/fastercnn50.pth", map_location="cpu")
# Remove old head (VinDr had 15 classes)
for k in list(state_dict.keys()):
    if k.startswith("roi_heads.box_predictor."):
        del state_dict[k]
model.load_state_dict(state_dict, strict=False)
# New head for your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
```

---

## Trained Checkpoints

### Original Checkpoints (from Colab project)

| File | Size | NODE21 | AUROC | Notes |
|------|------|--------|-------|-------|
| `checkpoints/frcnn_original/checkpoint_3canal_frcnn_vindn_epoch_11_20251231_003142.pth` | 159MB | **0.9596** | 0.9904 | Best original, grayscale x3, includes leakage in training split |
| `checkpoints/frcnn_original/checkpoint_canal_frcnn_vindn_attention_cbam_epoch_16_20260101_104415.pth` | 161MB | 0.9181 | 0.9586 | CBAM attention variant |

#### Loading original checkpoints
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.roi_heads.score_thresh = 0.005  # IMPORTANT for FROC evaluation
model.roi_heads.nms_thresh = 0.2
```

**Note for CBAM checkpoint:** Keys have `backbone.body.body.` prefix (double body). Remap:
```python
remapped = {}
for k, v in state_dict.items():
    new_k = k.replace("backbone.body.body.", "backbone.body.", 1)
    if "cbam" not in new_k:
        remapped[new_k] = v
model.load_state_dict(remapped, strict=False)
```

### Reproduced Checkpoints (Spark)

| File | Size | NODE21 | AUROC | Notes |
|------|------|--------|-------|-------|
| `checkpoints/frcnn_reproduce/best_node21.pth` | 159MB | **0.9695** | 0.9936 | Replica A (with leakage), selected by NODE21 |
| `checkpoints/frcnn_reproduce/best_valloss.pth` | 159MB | 0.9651 | 0.9744 | Replica A, selected by val_loss |
| `checkpoints/frcnn_corrected/best_node21.pth` | 159MB | **0.9025** | 0.9460 | Version B (NO leakage), honest score |
| `checkpoints/frcnn_corrected/best_valloss.pth` | 159MB | 0.9025 | 0.9460 | Version B, selected by val_loss |

Loading is identical to original checkpoints (same architecture).

### FRCNN v2 (trained on original splits without augmentation)

| File | Size | NODE21 | Notes |
|------|------|--------|-------|
| `checkpoints/frcnn_vindr_v2/best_model_fold0.pth` | 409MB | 0.8544 | Includes optimizer state |
| `checkpoints/frcnn_vindr/best_model_fold0.pth` | 409MB | 0.4546 | v1, saved by val_loss (bad criterion) |

Loading v2 checkpoints (includes optimizer state):
```python
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
```

### YOLO Checkpoints

| File | Size | NODE21 | AUROC | Notes |
|------|------|--------|-------|-------|
| `checkpoints/yolo/yolov8s/best.pt` | 22MB | **0.9103** | 0.9686 | YOLOv8s, best epoch 26 |
| `checkpoints/yolo/yolo26s/best.pt` | 20MB | 0.7929 | 0.9557 | YOLO26s v2, best epoch 73 |

#### Loading YOLO checkpoints
```python
from ultralytics import YOLO
model = YOLO("checkpoints/yolo/yolov8s/best.pt")
results = model.predict(source="image.png", conf=0.001, iou=0.2, imgsz=1024)
```

---

## Evaluation Summary (all on val_fold0.csv, 977 original images)

| Model | NODE21 | AUROC | CM | S@0.25 | S@1 | S@8 | ms/img |
|-------|--------|-------|----|--------|-----|-----|--------|
| FRCNN Replica-A (best_node21) | **0.9695** | 0.9936 | 0.9827 | 0.950 | 0.970 | 0.977 | 35 |
| FRCNN Original (Colab) | 0.9596 | 0.9904 | 0.9795 | 0.947 | 0.963 | 0.963 | 36 |
| YOLOv8s | 0.9103 | 0.9686 | 0.9283 | 0.821 | 0.924 | 0.957 | 16 |
| FRCNN Corrected-B (no leakage) | 0.9025 | 0.9460 | 0.9146 | 0.821 | 0.890 | 0.973 | 35 |
| YOLO26s | 0.7929 | 0.9557 | 0.8754 | 0.635 | 0.791 | 0.887 | 19 |

**Key finding:** Data leakage inflates NODE21 by ~6.7 points (0.9695 vs 0.9025).
The honest score without leakage is **0.9025** (FRCNN) and **0.9103** (YOLOv8s).

---

## Ensemble WBF Results (2026-04-03)

**Config:** FRCNN Corrected-B + YOLOv8s, weights=[0.90, 0.91], iou_thr=0.2, skip_box_thr=0.05

| Metric | Ensemble WBF | YOLOv8s | FRCNN | Gain |
|--------|-------------|---------|-------|------|
| NODE21 | **0.9391** | 0.9103 | 0.9025 | +2.6% |
| AUROC | 0.9683 | 0.9686 | 0.9460 | — |
| CM | **0.9447** | 0.9316 | 0.9146 | +1.3% |
| S@0.25 FP | **0.874** | 0.821 | 0.821 | +5.3% |
| S@1 FP | **0.944** | 0.924 | 0.890 | +2.0% |
| S@8 FP | **0.973** | 0.957 | 0.973 | — |

Script: scripts/ensemble_wbf.py (355 lines)
Results: results/ensemble_report.csv, results/froc_ensemble.png
Visualizations: results/ensemble_detections/ (30 images)

All scores are honest (no data leakage, StratifiedGroupKFold split).
