"""YOLO model builder supporting YOLOv8 and YOLO26."""

import inspect
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _is_yolo26(model) -> bool:
    """Heuristic: YOLO26 models lack the 'dfl' parameter and use ProgLoss."""
    try:
        head = model.model.model[-1]  # detection head
        if not hasattr(head, "dfl"):
            return True
    except (AttributeError, IndexError):
        pass
    return False


def build_yolo(cfg: Dict[str, Any]):
    """Build and return a YOLO model from config.

    Expected config keys:
        model.yolo_variant   - str, e.g. "yolov8n.pt", "yolo26s.pt"
        model.pretrained_weights - str or None, path to .pt checkpoint
        model.num_classes    - int
    """
    from ultralytics import YOLO

    mcfg = cfg["model"]
    variant: str = mcfg.get("yolo_variant", "yolov8n.pt")
    pretrained_weights = mcfg.get("pretrained_weights", None)

    if pretrained_weights:
        logger.info("Loading YOLO from checkpoint: %s", pretrained_weights)
        model = YOLO(pretrained_weights)
    else:
        logger.info("Loading YOLO variant: %s", variant)
        model = YOLO(variant)

    is_v26 = _is_yolo26(model)
    logger.info("Detected YOLO26: %s", is_v26)

    return model


def get_yolo_train_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a kwargs dict for model.train() from config.

    Applies conservative augmentation settings suitable for medical imaging
    (no mosaic, mixup, or aggressive HSV augmentation).
    """
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})

    params: Dict[str, Any] = {
        # Data
        "data": dcfg.get("yolo_yaml", "data.yaml"),
        "imgsz": dcfg.get("img_size", 1024),
        # Training schedule
        "epochs": tcfg.get("epochs", 100),
        "batch": tcfg.get("batch_size", 8),
        "lr0": tcfg.get("lr", 0.001),
        "lrf": tcfg.get("lr_final_ratio", 0.01),
        "weight_decay": tcfg.get("weight_decay", 0.0005),
        "warmup_epochs": tcfg.get("warmup_epochs", 3),
        "patience": tcfg.get("patience", 20),
        # Conservative medical imaging augmentation
        "mosaic": 0.0,
        "mixup": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "flipud": tcfg.get("flipud", 0.5),
        "fliplr": tcfg.get("fliplr", 0.5),
        "degrees": tcfg.get("degrees", 10.0),
        "translate": tcfg.get("translate", 0.1),
        "scale": tcfg.get("scale", 0.2),
        "shear": tcfg.get("shear", 0.0),
        # Output
        "project": tcfg.get("project", "runs"),
        "name": tcfg.get("name", "yolo_train"),
        "exist_ok": True,
        "verbose": True,
        "save": True,
        "save_period": tcfg.get("save_period", -1),
        "device": tcfg.get("device", 0),
        "workers": tcfg.get("workers", 4),
        "seed": tcfg.get("seed", 42),
    }

    # Auto-detect YOLO26 and add ProgLoss if supported
    try:
        from ultralytics import YOLO

        variant = mcfg.get("yolo_variant", "yolov8n.pt")
        pretrained = mcfg.get("pretrained_weights", None)
        probe_model = YOLO(pretrained or variant)
        if _is_yolo26(probe_model):
            logger.info("YOLO26 detected - enabling ProgLoss configuration")
            # YOLO26 does not use dfl; ProgLoss is used internally
            # No extra param needed; the model handles it automatically
    except Exception:
        pass

    return params


def train_yolo(model, cfg: Dict[str, Any]):
    """Run YOLO training using parameters from config.

    Args:
        model: YOLO model instance (from build_yolo).
        cfg: Full pipeline config dict.

    Returns:
        Training results object from ultralytics.
    """
    params = get_yolo_train_params(cfg)
    logger.info("Starting YOLO training with params: %s", params)
    results = model.train(**params)
    return results
