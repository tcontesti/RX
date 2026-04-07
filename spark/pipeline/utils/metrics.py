"""Evaluation metrics for object detection: IoU, FROC, NODE21, AUROC."""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box_a: shape (4,)
        box_b: shape (4,)

    Returns:
        IoU value (float).
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_froc(
    detections: List[Dict],
    iou_thresh: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Free-Response ROC (FROC) curve.

    Args:
        detections: List of dicts, one per image, each containing:
            - "gt_boxes": np.ndarray of shape (M, 4) or empty
            - "pred_boxes": np.ndarray of shape (N, 4) or empty
            - "scores": np.ndarray of shape (N,) or empty
        iou_thresh: IoU threshold for a true-positive match.

    Returns:
        fps: 1-D array of average false positives per image.
        sens: 1-D array of sensitivity (recall) at each threshold.
    """
    all_scores: List[float] = []
    all_tp: List[bool] = []
    total_gt = 0
    num_images = len(detections)

    for det in detections:
        gt_boxes = np.array(det.get("gt_boxes", [])).reshape(-1, 4)
        pred_boxes = np.array(det.get("pred_boxes", [])).reshape(-1, 4)
        scores = np.array(det.get("scores", [])).ravel()

        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions by descending score
        order = np.argsort(-scores)
        pred_boxes = pred_boxes[order]
        scores = scores[order]

        matched_gt = set()
        for i, (pb, sc) in enumerate(zip(pred_boxes, scores)):
            is_tp = False
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou_val = compute_iou(pb, gb)
                if iou_val >= iou_thresh and iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0:
                matched_gt.add(best_j)
                is_tp = True

            all_scores.append(float(sc))
            all_tp.append(is_tp)

    if total_gt == 0 or len(all_scores) == 0:
        return np.array([0.0]), np.array([0.0])

    # Sort globally by score descending
    order = np.argsort(-np.array(all_scores))
    tp_arr = np.array(all_tp)[order]

    cum_tp = np.cumsum(tp_arr).astype(np.float64)
    cum_fp = np.cumsum(~tp_arr).astype(np.float64)

    sens = cum_tp / total_gt
    fps = cum_fp / num_images

    return fps, sens


def compute_node21_score(
    fps: np.ndarray,
    sens: np.ndarray,
    fp_levels: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> float:
    """Compute the NODE21 challenge score from an FROC curve.

    The score is the average sensitivity at specified FP-per-image levels.

    Args:
        fps: 1-D array of FP per image (from compute_froc).
        sens: 1-D array of sensitivity values (from compute_froc).
        fp_levels: FP-per-image levels to interpolate sensitivity at.

    Returns:
        NODE21 score (float in [0, 1]).
    """
    sensitivities = []
    for fp_level in fp_levels:
        # Find sensitivity at this FP level via interpolation
        idx = np.searchsorted(fps, fp_level, side="right") - 1
        if idx < 0:
            sensitivities.append(0.0)
        elif idx >= len(fps) - 1:
            sensitivities.append(float(sens[-1]))
        else:
            # Linear interpolation
            frac = (fp_level - fps[idx]) / (fps[idx + 1] - fps[idx] + 1e-12)
            s = sens[idx] + frac * (sens[idx + 1] - sens[idx])
            sensitivities.append(float(s))

    score = float(np.mean(sensitivities))
    logger.info("NODE21 score: %.4f (sensitivities at FP levels: %s)", score, sensitivities)
    return score


def compute_auroc(detections: List[Dict]) -> float:
    """Compute image-level AUROC for detection (binary: has lesion or not).

    Each image is positive if it has at least one GT box.
    The image-level score is the max prediction score (0 if no predictions).

    Args:
        detections: Same format as compute_froc.

    Returns:
        AUROC value (float).
    """
    from sklearn.metrics import roc_auc_score

    y_true = []
    y_score = []

    for det in detections:
        gt_boxes = np.array(det.get("gt_boxes", [])).reshape(-1, 4)
        scores = np.array(det.get("scores", [])).ravel()

        is_positive = 1 if len(gt_boxes) > 0 else 0
        max_score = float(scores.max()) if len(scores) > 0 else 0.0

        y_true.append(is_positive)
        y_score.append(max_score)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Need both classes present
    if len(np.unique(y_true)) < 2:
        logger.warning("AUROC undefined: only one class present in ground truth.")
        return 0.0

    auroc = float(roc_auc_score(y_true, y_score))
    logger.info("AUROC: %.4f", auroc)
    return auroc


def compute_competition_metric(auroc: float, node21_score: float) -> float:
    """Compute the final competition metric: 0.75 * AUROC + 0.25 * NODE21.

    Args:
        auroc: Image-level AUROC.
        node21_score: NODE21 FROC-based score.

    Returns:
        Combined competition metric (float).
    """
    metric = 0.75 * auroc + 0.25 * node21_score
    logger.info(
        "Competition metric: %.4f (AUROC=%.4f, NODE21=%.4f)",
        metric, auroc, node21_score,
    )
    return metric
