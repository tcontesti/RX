"""Weighted Boxes Fusion (WBF) ensemble for combining predictions from multiple models."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def ensemble_wbf(
    all_predictions: List[List[Dict]],
    img_size: int = 1024,
    iou_thr: float = 0.2,
    skip_box_thr: float = 0.05,
    weights: Optional[List[float]] = None,
) -> List[Dict]:
    """Ensemble predictions from multiple models using Weighted Boxes Fusion.

    Args:
        all_predictions: List of model predictions. Each element is a list
            of per-image dicts (one per image), where each dict contains:
                - "boxes": np.ndarray of shape (N, 4) in [x1, y1, x2, y2] pixel coords
                - "scores": np.ndarray of shape (N,)
                - "labels": np.ndarray of shape (N,) (integer class ids)
        img_size: Image size used for normalization (assumes square images).
        iou_thr: IoU threshold for WBF merging.
        skip_box_thr: Minimum score threshold; boxes below this are discarded.
        weights: Optional per-model weights. If None, all models weighted equally.

    Returns:
        List of per-image dicts with ensembled "boxes", "scores", "labels".
    """
    from ensemble_boxes import weighted_boxes_fusion

    num_models = len(all_predictions)
    if num_models == 0:
        return []

    num_images = len(all_predictions[0])
    if weights is None:
        weights = [1.0] * num_models

    results: List[Dict] = []

    for img_idx in range(num_images):
        boxes_list = []
        scores_list = []
        labels_list = []

        for model_idx in range(num_models):
            pred = all_predictions[model_idx][img_idx]
            boxes = np.array(pred.get("boxes", [])).reshape(-1, 4).astype(np.float32)
            scores = np.array(pred.get("scores", [])).ravel().astype(np.float32)
            labels = np.array(pred.get("labels", [])).ravel().astype(np.int32)

            # Normalize boxes to [0, 1]
            if len(boxes) > 0:
                norm_boxes = boxes / img_size
                # Clip to valid range
                norm_boxes = np.clip(norm_boxes, 0.0, 1.0)
            else:
                norm_boxes = np.zeros((0, 4), dtype=np.float32)

            boxes_list.append(norm_boxes.tolist())
            scores_list.append(scores.tolist())
            labels_list.append(labels.tolist())

        # Run WBF
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        # Denormalize boxes back to pixel coordinates
        fused_boxes = np.array(fused_boxes, dtype=np.float32) * img_size

        results.append({
            "boxes": fused_boxes,
            "scores": np.array(fused_scores, dtype=np.float32),
            "labels": np.array(fused_labels, dtype=np.int32),
        })

    logger.info("WBF ensemble: %d models, %d images processed.", num_models, num_images)
    return results
