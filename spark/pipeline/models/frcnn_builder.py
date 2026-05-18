"""Faster R-CNN model builder with optional CBAM attention and flexible weight loading."""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CBAM (Convolutional Block Attention Module)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])  # (B, C)
        max_pool = x.amax(dim=[2, 3])  # (B, C)
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        attention = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# Backbone wrapper that injects CBAM after layer4
# ---------------------------------------------------------------------------

class ResNetBackboneWithCBAM(nn.Module):
    """Wraps a ResNet backbone to insert a CBAM module after layer4."""

    def __init__(self, backbone: nn.Module, cbam: CBAM):
        super().__init__()
        self.backbone = backbone
        self.cbam = cbam
        # Expose out_channels so FPN / FasterRCNN can read it
        if hasattr(backbone, "out_channels"):
            self.out_channels = backbone.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard ResNet forward through body layers
        x = self.backbone.body.conv1(x)
        x = self.backbone.body.bn1(x)
        x = self.backbone.body.relu(x)
        x = self.backbone.body.maxpool(x)
        x = self.backbone.body.layer1(x)
        x = self.backbone.body.layer2(x)
        x = self.backbone.body.layer3(x)
        x = self.backbone.body.layer4(x)
        x = self.cbam(x)
        return x


# ---------------------------------------------------------------------------
# Layer freezing helper
# ---------------------------------------------------------------------------

def _freeze_layers(model: nn.Module, freeze_layers: list):
    """Freeze named parameters matching any prefix in *freeze_layers*.

    Prefixes are matched both as-is and with 'backbone.body.' prepended,
    so config can use short names like 'layer1' or full paths.
    """
    if not freeze_layers:
        return
    # Build expanded prefixes to match both short and full parameter paths
    expanded = []
    for prefix in freeze_layers:
        expanded.append(prefix)
        expanded.append(f"backbone.body.{prefix}")
    for name, param in model.named_parameters():
        for prefix in expanded:
            if name.startswith(prefix):
                param.requires_grad = False
                break


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_frcnn(cfg: Dict[str, Any], device: torch.device) -> FasterRCNN:
    """Build a Faster R-CNN model from a nested config dict.

    Expected config keys (dot-separated shown for clarity):
        model.num_classes          - int, number of classes (including background)
        model.pretrained_weights   - str, path to .pth file (for vindr) or ignored
        model.pretrained_source    - str, one of "vindr", "coco", "none"
        model.freeze_layers        - list[str], parameter name prefixes to freeze
        model.trainable_layers     - int, backbone trainable layers (0-5)
        model.attention            - str or None, "cbam" or null
        model.cbam_channels        - int, channels for CBAM (default 2048)
        model.cbam_reduction       - int, reduction ratio for CBAM (default 16)
        model.score_thresh         - float, minimum score for detections
        model.nms_thresh           - float, NMS IoU threshold
    """
    mcfg = cfg["model"]
    num_classes: int = mcfg["num_classes"]
    pretrained_source: str = mcfg.get("pretrained_source", "none")
    pretrained_weights: Optional[str] = mcfg.get("pretrained_weights", None)
    freeze_layers: list = mcfg.get("freeze_layers", [])
    trainable_layers: int = mcfg.get("trainable_layers", 3)
    attention: Optional[str] = mcfg.get("attention", None)
    cbam_channels: int = mcfg.get("cbam_channels", 2048)
    cbam_reduction: int = mcfg.get("cbam_reduction", 16)
    score_thresh: float = mcfg.get("score_thresh", 0.05)
    nms_thresh: float = mcfg.get("nms_thresh", 0.5)

    # ---- Build base model -------------------------------------------------
    if pretrained_source == "coco":
        logger.info("Loading Faster R-CNN with COCO pretrained weights.")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=trainable_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
        )
    elif pretrained_source == "vindr":
        logger.info("Loading Faster R-CNN and applying VinDr weights from %s", pretrained_weights)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            trainable_backbone_layers=trainable_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
        )
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights, map_location=device)
            # Handle wrapped state dicts (e.g. {"model": ...})
            if "model" in state_dict:
                state_dict = state_dict["model"]
            # Remove box predictor keys so head can be replaced freely
            keys_to_remove = [
                k for k in state_dict.keys() if k.startswith("roi_heads.box_predictor.")
            ]
            for k in keys_to_remove:
                del state_dict[k]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info("VinDr weight loading - missing: %d, unexpected: %d", len(missing), len(unexpected))
    else:
        logger.info("Building Faster R-CNN without pretrained weights.")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            trainable_backbone_layers=trainable_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
        )

    # ---- Replace classification head with correct num_classes -------------
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ---- Optional CBAM attention after layer4 -----------------------------
    if attention and attention.lower() == "cbam":
        logger.info("Adding CBAM attention module (channels=%d, reduction=%d)", cbam_channels, cbam_reduction)
        cbam = CBAM(in_channels=cbam_channels, reduction=cbam_reduction)
        model.backbone.body.layer4 = nn.Sequential(
            model.backbone.body.layer4,
            cbam,
        )

    # ---- Freeze requested layers ------------------------------------------
    _freeze_layers(model, freeze_layers)

    model.to(device)
    return model
