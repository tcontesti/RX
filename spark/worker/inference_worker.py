#!/usr/bin/env python3
"""
CXR Inference Worker — GPU-accelerated pulmonary nodule detection.

Runs on the Spark (Beelink with RTX 3060 Ti). Consumes inference tasks from
RabbitMQ queue ``cxr.inference``, runs a two-model ensemble (Faster R-CNN +
YOLOv8) fused with Weighted Box Fusion (WBF), and publishes results to
``cxr.results``.

Architecture overview::

    RabbitMQ (cxr.inference)
        │
        ▼
    callback()  ─── per-message ───►  process_task()
        │                                 ├─ decode_image()
        │                                 ├─ preprocess()        → 1024×1024, [0,1]
        │                                 ├─ infer_frcnn()       → boxes, scores
        │                                 ├─ infer_yolo()        → boxes, scores
        │                                 ├─ ensemble_wbf()      → fused boxes
        │                                 ├─ generate_annotated_image()
        │                                 └─ return result dict
        ▼
    RabbitMQ (cxr.results)

Models are loaded **once** at startup and kept in GPU memory. Each task
processes a single CXR image (~50 ms on GPU). The worker reconnects
automatically to RabbitMQ with exponential backoff if the connection drops.

Input message format (JSON on ``cxr.inference``)::

    {
        "study_id":    "abc-123",          # unique study identifier
        "image_data":  "<base64 PNG>",     # or an absolute file path
        "format":      "png"               # png | jpg | dcm (optional)
    }

Output message format (JSON on ``cxr.results``)::

    {
        "study_id":          "abc-123",
        "status":            "completed" | "error",
        "num_detections":    2,
        "detections":        [{"x1", "y1", "x2", "y2", "score", "label"}, ...],
        "model_details":     {"frcnn": {...}, "yolov8": {...}},
        "inference_time_ms": 48.3,
        "annotated_image_base64": "<base64 PNG with drawn boxes>"
    }

Usage::

    python inference_worker.py --config config.yaml
    python inference_worker.py --config config.yaml --test   # no RabbitMQ
"""

import argparse
import base64
import json
import logging
import os
import signal
import sys
import time
import traceback
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np
import pika
import torch
import torchvision
import yaml
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# =========================================================================
# LOGGING
# =========================================================================

def setup_logging(config):
    """Configure root logger with console + optional rotating file output.

    Uses ``RotatingFileHandler`` to cap disk usage at 50 MB total
    (10 MB per file x 5 backups). This prevents the log from filling
    the disk after weeks of unattended operation.

    Args:
        config: Full YAML config dict.  Reads ``config.logging.level``
            (default ``"INFO"``) and ``config.logging.file`` (optional path).

    Returns:
        logging.Logger: Named logger ``"inference_worker"``.
    """
    level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file = config.get("logging", {}).get("file", None)

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Rotate at 10 MB, keep 5 old files → max 50 MB on disk
        handlers.append(RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        ))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("inference_worker")


# =========================================================================
# MODEL LOADING (una vez al inicio)
# =========================================================================

def load_frcnn(config, device):
    """Load Faster R-CNN with VinDr-CXR fine-tuned weights.

    The model is a standard ``fasterrcnn_resnet50_fpn`` with 2 classes
    (background + nodule).  Weights come from training on VinDr-CXR /
    NODE21 and may be stored as either a raw ``state_dict`` or wrapped
    inside ``{"model_state_dict": ...}`` — both formats are supported.

    After loading, the RoI head thresholds are patched to very low values
    so that WBF downstream can decide what to keep (low score_thresh lets
    more candidate boxes through for ensemble fusion).

    Args:
        config: Full YAML config dict. Reads ``config.models.frcnn``.
        device: ``torch.device`` to place the model on.

    Returns:
        The loaded model in eval mode, or ``None`` if disabled in config.
    """
    cfg = config["models"]["frcnn"]
    if not cfg.get("enabled", False):
        return None

    # 2 classes: background (0) + nodule (1)
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    state_dict = torch.load(cfg["weights"], map_location=device, weights_only=False)

    # Support both raw state_dict and checkpoint wrapper {"model_state_dict": ...}
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Patch RoI thresholds — keep them low so WBF receives all candidates
    model.roi_heads.score_thresh = cfg.get("score_thresh", 0.005)
    model.roi_heads.nms_thresh = cfg.get("nms_thresh", 0.2)

    total = sum(p.numel() for p in model.parameters())
    logging.getLogger("inference_worker").info(
        f"FRCNN loaded: {total:,} params, score_thresh={cfg['score_thresh']}"
    )
    return model


def load_yolo(config):
    """Load YOLOv8 with NODE21 fine-tuned weights.

    Uses the Ultralytics YOLO API. The ``.pt`` file is a full Ultralytics
    checkpoint (architecture + weights), so no manual architecture setup
    is needed.  Import is deferred to avoid loading ultralytics when only
    FRCNN is enabled.

    Args:
        config: Full YAML config dict. Reads ``config.models.yolov8``.

    Returns:
        Ultralytics YOLO model, or ``None`` if disabled in config.
    """
    cfg = config["models"]["yolov8"]
    if not cfg.get("enabled", False):
        return None

    from ultralytics import YOLO  # deferred — heavy import
    model = YOLO(cfg["weights"])
    logging.getLogger("inference_worker").info(f"YOLOv8 loaded from {cfg['weights']}")
    return model


# =========================================================================
# IMAGE PROCESSING
# =========================================================================

def decode_image(image_data, image_format="png"):
    """Decode a CXR image from either a base64 string or an absolute file path.

    The ``image_data`` field in the RabbitMQ task can be either:
    - An absolute path to a file on the Spark's filesystem (used when the
      API server and worker share a volume), or
    - A base64-encoded PNG/JPG byte string (used when the image is sent
      inline through the message queue).

    Both paths return a single-channel uint8 grayscale numpy array.

    Args:
        image_data: Base64 string or absolute file path.
        image_format: Hint for the source format (currently unused by cv2).

    Returns:
        np.ndarray: Grayscale image, shape ``(H, W)``, dtype ``uint8``.

    Raises:
        ValueError: If decoding fails (corrupt data or missing file).
    """
    if os.path.isfile(str(image_data)):
        img = cv2.imread(str(image_data), cv2.IMREAD_GRAYSCALE)
    else:
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Failed to decode image")
    return img


def preprocess(img_gray, target_size=1024):
    """Resize to square and normalize pixel intensities to [0, 1].

    Both models expect 1024x1024 input. The float version is for FRCNN
    (which needs [0,1] tensors) and the uint8 version is for YOLO (which
    handles its own normalization internally).

    Args:
        img_gray: Grayscale image, any size, dtype uint8.
        target_size: Side length of the output square (default 1024).

    Returns:
        tuple: ``(img_uint8, img_float)`` — resized uint8 and min-max
        normalized float32 arrays, both shape ``(target_size, target_size)``.
    """
    img_resized = cv2.resize(img_gray, (target_size, target_size))
    img_float = img_resized.astype(np.float32)
    # Min-max normalize to [0, 1]; epsilon avoids division by zero on blank images
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    return img_resized, img_float


# =========================================================================
# INFERENCE
# =========================================================================

def infer_frcnn(model, img_float, device):
    """Run Faster R-CNN inference on a single preprocessed image.

    FRCNN expects a 3-channel ``[C, H, W]`` float tensor. Since CXR images
    are grayscale, the single channel is replicated three times to fill the
    RGB slots (the backbone was pre-trained on ImageNet RGB).

    Args:
        model: Loaded FRCNN model in eval mode.
        img_float: Normalized float32 array, shape ``(H, W)``, range [0, 1].
        device: ``torch.device`` the model lives on.

    Returns:
        dict: ``{"boxes": np.ndarray (N,4), "scores": np.ndarray (N,),
        "labels": np.ndarray (N,)}`` in xyxy pixel coordinates.
    """
    # Grayscale → pseudo-RGB by tripling the single channel
    img3 = np.stack([img_float, img_float, img_float], axis=0)
    tensor = torch.from_numpy(img3).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)[0]

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    return {"boxes": boxes, "scores": scores, "labels": labels}


def infer_yolo(model, img_uint8, config):
    """Run YOLOv8 inference on a single preprocessed image.

    Ultralytics expects a BGR uint8 image; since we have grayscale,
    we merge into a 3-channel BGR array. The confidence threshold is
    set very low (0.001) to let WBF decide the final cutoff.

    Args:
        model: Loaded Ultralytics YOLO model.
        img_uint8: Resized uint8 array, shape ``(H, W)``.
        config: Full YAML config dict. Reads ``config.models.yolov8``.

    Returns:
        dict: ``{"boxes": np.ndarray (N,4), "scores": np.ndarray (N,),
        "labels": np.ndarray (N,)}`` in xyxy pixel coordinates.
        All labels are ``1`` (nodule class).
    """
    cfg = config["models"]["yolov8"]
    # Grayscale → BGR (Ultralytics expects 3-channel input)
    img_bgr = cv2.merge([img_uint8, img_uint8, img_uint8])

    results = model.predict(
        source=img_bgr,
        conf=cfg.get("conf_thresh", 0.001),
        iou=0.2,
        imgsz=cfg.get("imgsz", 1024),
        verbose=False,
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    # Single-class detection — all boxes are nodule (class 1)
    labels = np.ones(len(scores), dtype=int)

    return {"boxes": boxes, "scores": scores, "labels": labels}


def ensemble_wbf(predictions, config, img_size=1024):
    """Fuse detections from multiple models using Weighted Box Fusion.

    WBF merges overlapping boxes by averaging their coordinates weighted by
    confidence, rather than discarding duplicates like NMS. This produces
    better-localised boxes when two models agree on a region.

    The ``ensemble_boxes`` library expects boxes normalised to [0, 1], so
    we divide by ``img_size`` before fusion and multiply back afterwards.

    Config knobs (``config.ensemble``):
    - ``weights``: per-model importance, e.g. ``[1.5, 1.0]`` for FRCNN-heavy.
    - ``iou_thr``:  IoU above which two boxes are considered the same object.
    - ``skip_box_thr``: minimum score for a box to enter fusion at all.

    Args:
        predictions: List of dicts from each model, each with keys
            ``"boxes"`` (N,4 xyxy), ``"scores"`` (N,), ``"labels"`` (N,).
        config: Full YAML config dict.
        img_size: Side length of the square input image (for normalisation).

    Returns:
        dict: Fused ``{"boxes", "scores", "labels"}`` in pixel coordinates.
    """
    from ensemble_boxes import weighted_boxes_fusion  # deferred import

    cfg = config["ensemble"]
    boxes_list, scores_list, labels_list = [], [], []

    for pred in predictions:
        if len(pred["boxes"]) == 0:
            boxes_list.append(np.zeros((0, 4)))
            scores_list.append(np.zeros(0))
            labels_list.append(np.zeros(0, dtype=int))
        else:
            # Normalise xyxy coords to [0, 1] as required by WBF
            boxes_norm = np.clip(pred["boxes"] / img_size, 0, 1)
            boxes_list.append(boxes_norm)
            scores_list.append(pred["scores"].astype(np.float32))
            # All class 0 for WBF (single-class problem, label is just a grouping key)
            labels_list.append(np.zeros(len(pred["scores"]), dtype=int))

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=cfg.get("weights", [1.0] * len(predictions)),
        iou_thr=cfg.get("iou_thr", 0.2),
        skip_box_thr=cfg.get("skip_box_thr", 0.05),
    )

    # Scale back to pixel coordinates
    fused_boxes = fused_boxes * img_size
    return {"boxes": fused_boxes, "scores": fused_scores, "labels": fused_labels}


def generate_annotated_image(img_uint8, detections, confidence_thresh=0.3):
    """Draw bounding boxes on the CXR and return the result as base64 PNG.

    This server-side annotation is a fallback for clients that cannot draw
    their own SVG overlay.  The frontend currently uses its own SVG boxes,
    but this image is still stored for quick preview in the history view.

    Args:
        img_uint8: Resized grayscale image, shape ``(H, W)``, dtype uint8.
        detections: Dict with ``"boxes"`` (N,4) and ``"scores"`` (N,).
        confidence_thresh: Only draw boxes above this score.

    Returns:
        str: Base64-encoded PNG image with green bounding boxes and labels.
    """
    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    for i, (box, score) in enumerate(zip(detections["boxes"], detections["scores"])):
        if score < confidence_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # green
        cv2.rectangle(img_color, (x1, y1), (x2, y2), color, 2)
        label = f"Nodule {score:.2f}"
        cv2.putText(img_color, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    _, buffer = cv2.imencode(".png", img_color)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================================
# TASK PROCESSING
# =========================================================================

def process_task(task, frcnn_model, yolo_model, config, device, logger):
    """Execute the full inference pipeline for a single CXR study.

    Pipeline steps:
    1. Decode image (base64 or file path)
    2. Preprocess to 1024x1024, normalise
    3. Run each enabled model independently
    4. Fuse predictions with WBF (if >1 model)
    5. Apply output confidence threshold
    6. Optionally render annotated image
    7. Build and return result dict

    The result dict matches the output message schema documented in the
    module docstring and is published as-is to ``cxr.results``.

    Args:
        task: Decoded JSON dict from the input queue. Must contain
            ``"study_id"`` and ``"image_data"``; ``"format"`` is optional.
        frcnn_model: Loaded FRCNN model or ``None``.
        yolo_model: Loaded YOLO model or ``None``.
        config: Full YAML config dict.
        device: ``torch.device`` for GPU tensors.
        logger: Logger instance.

    Returns:
        dict: Result payload ready for JSON serialisation and publishing.
    """
    start = time.time()

    study_id = task.get("study_id")
    image_data = task.get("image_data")  # base64 string or absolute path
    image_format = task.get("format", "png")
    img_size = config["inference"].get("input_size", 1024)

    logger.info(f"Processing study_id={study_id}")

    # --- Step 1: Decode ---
    img_gray = decode_image(image_data, image_format)

    # --- Step 2: Preprocess ---
    img_uint8, img_float = preprocess(img_gray, img_size)

    # --- Step 3: Per-model inference ---
    predictions = []
    model_results = {}

    if frcnn_model is not None:
        t0 = time.time()
        frcnn_pred = infer_frcnn(frcnn_model, img_float, device)
        model_results["frcnn"] = {
            "num_detections": len(frcnn_pred["boxes"]),
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
        predictions.append(frcnn_pred)
        logger.debug(f"  FRCNN: {len(frcnn_pred['boxes'])} detections")

    if yolo_model is not None:
        t0 = time.time()
        yolo_pred = infer_yolo(yolo_model, img_uint8, config)
        model_results["yolov8"] = {
            "num_detections": len(yolo_pred["boxes"]),
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
        predictions.append(yolo_pred)
        logger.debug(f"  YOLOv8: {len(yolo_pred['boxes'])} detections")

    # --- Step 4: Ensemble WBF ---
    # With 2 models → fuse; with 1 → pass through; with 0 → empty
    if len(predictions) > 1 and config["ensemble"].get("enabled", True):
        ensemble_pred = ensemble_wbf(predictions, config, img_size)
    elif len(predictions) == 1:
        ensemble_pred = predictions[0]
    else:
        ensemble_pred = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0)}

    # --- Step 5: Output confidence filter ---
    conf_thresh = config["inference"].get("output_confidence_thresh", 0.3)
    keep = ensemble_pred["scores"] >= conf_thresh
    final_detections = {
        "boxes": ensemble_pred["boxes"][keep],
        "scores": ensemble_pred["scores"][keep],
        "labels": ensemble_pred["labels"][keep],
    }

    # --- Step 6: Annotated image (server-side rendering, optional) ---
    annotated_image_b64 = None
    if config["inference"].get("save_annotated_image", True):
        annotated_image_b64 = generate_annotated_image(img_uint8, final_detections, conf_thresh)

    elapsed = round((time.time() - start) * 1000, 1)

    # --- Step 7: Build result payload ---
    result = {
        "study_id": study_id,
        "status": "completed",
        "num_detections": int(len(final_detections["boxes"])),
        "detections": [
            {
                "x1": round(float(b[0]), 1),
                "y1": round(float(b[1]), 1),
                "x2": round(float(b[2]), 1),
                "y2": round(float(b[3]), 1),
                "score": round(float(s), 4),
                "label": "nodule",
            }
            for b, s in zip(final_detections["boxes"], final_detections["scores"])
        ],
        "model_details": model_results,
        "inference_time_ms": elapsed,
        "ensemble": config["ensemble"].get("enabled", True),
        "confidence_threshold": conf_thresh,
    }

    if annotated_image_b64:
        result["annotated_image_base64"] = annotated_image_b64

    # Free GPU memory between tasks to prevent VRAM creep
    torch.cuda.empty_cache()

    logger.info(
        f"  Done: {result['num_detections']} detections, {elapsed}ms"
    )
    return result


# =========================================================================
# RABBITMQ CONSUMER
# =========================================================================

def run_worker(config, frcnn_model, yolo_model, device, logger):
    """Main consumer loop with automatic RabbitMQ reconnection.

    Reconnection strategy (exponential backoff)::

        connect ok ──► consume messages (blocks) ──► connection lost
                           ▲                              │
                           │   sleep(retry_delay)         │
                           │   retry_delay *= 2           │
                           │   cap at 60 s                │
                           └──────────────────────────────┘

    On successful connect the backoff resets to 5 s.  The loop exits
    cleanly on ``KeyboardInterrupt`` or when the ``_shutdown`` flag is
    set by the signal handler.

    Message handling per callback:

    - **Success**: process → publish result → ACK.
    - **Malformed JSON**: ACK (drop the poison message).
    - **Processing error**: publish an error result → ACK.  The message
      is always ACKed to prevent infinite redelivery of a bad image.
    - **Finally**: ``torch.cuda.empty_cache()`` after every task to
      prevent VRAM fragmentation over hundreds of inferences.

    Args:
        config: Full YAML config dict. Reads ``config.rabbitmq``.
        frcnn_model: Loaded FRCNN model (or ``None``).
        yolo_model: Loaded YOLO model (or ``None``).
        device: ``torch.device`` for GPU tensors.
        logger: Logger instance.
    """
    rmq = config["rabbitmq"]

    credentials = pika.PlainCredentials(rmq["user"], rmq["password"])
    parameters = pika.ConnectionParameters(
        host=rmq["host"],
        port=rmq.get("port", 5672),
        credentials=credentials,
        heartbeat=rmq.get("heartbeat", 600),
        blocked_connection_timeout=rmq.get("blocked_connection_timeout", 300),
    )

    retry_delay = 5  # seconds — doubles on each failure, caps at 60

    while not _shutdown:
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            # Durable queues survive RabbitMQ restarts
            channel.queue_declare(queue=rmq["queue_input"], durable=True)
            channel.queue_declare(queue=rmq["queue_output"], durable=True)
            # prefetch_count=1 → process one image at a time (GPU is the bottleneck)
            channel.basic_qos(prefetch_count=rmq.get("prefetch_count", 1))

            logger.info(f"Connected to RabbitMQ {rmq['host']}:{rmq['port']}")
            logger.info(f"Consuming from: {rmq['queue_input']}")
            retry_delay = 5  # reset backoff on successful connect

            def callback(ch, method, properties, body):
                """Per-message handler: decode → infer → publish → ACK."""
                try:
                    task = json.loads(body)
                    result = process_task(task, frcnn_model, yolo_model, config, device, logger)

                    # Publish completed result (persistent delivery_mode=2)
                    channel.basic_publish(
                        exchange="",
                        routing_key=rmq["queue_output"],
                        body=json.dumps(result),
                        properties=pika.BasicProperties(delivery_mode=2),
                    )
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                except json.JSONDecodeError as e:
                    # Poison message — ACK to discard, cannot be retried
                    logger.error(f"Invalid JSON message: {e}")
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                except Exception as e:
                    logger.error(f"Error processing task: {e}\n{traceback.format_exc()}")
                    # Best-effort: notify the API that this study failed
                    try:
                        task_data = json.loads(body) if body else {}
                        error_result = {
                            "study_id": task_data.get("study_id", "unknown"),
                            "status": "error",
                            "error": str(e),
                        }
                        channel.basic_publish(
                            exchange="",
                            routing_key=rmq["queue_output"],
                            body=json.dumps(error_result),
                            properties=pika.BasicProperties(delivery_mode=2),
                        )
                    except:
                        pass
                    # ACK even on error — redelivery would fail the same way
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                finally:
                    # Release cached GPU memory after every task
                    torch.cuda.empty_cache()

            channel.basic_consume(queue=rmq["queue_input"], on_message_callback=callback)
            channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("Shutting down worker...")
            try:
                channel.stop_consuming()
                connection.close()
            except:
                pass
            break

        except Exception as e:
            # Connection lost — enter exponential backoff reconnect loop
            logger.error(f"RabbitMQ connection lost: {e}")
            logger.info(f"Reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  # cap at 60 s


# =========================================================================
# TEST MODE (sin RabbitMQ)
# =========================================================================

def run_test(config, frcnn_model, yolo_model, device, logger):
    """Run a single inference without RabbitMQ for smoke-testing.

    Looks for a known NODE21 image (``n0239.png``, which contains a nodule)
    or falls back to any PNG in the dataset directory. Builds a fake task
    dict identical to what the queue would deliver, runs the full pipeline,
    prints the result JSON, and saves the annotated image to disk.

    Args:
        config: Full YAML config dict.
        frcnn_model: Loaded FRCNN model (or ``None``).
        yolo_model: Loaded YOLO model (or ``None``).
        device: ``torch.device`` for GPU tensors.
        logger: Logger instance.

    Returns:
        dict: Result payload, or ``None`` if no test image was found.
    """
    import glob

    # Prefer n0239.png (known positive case), fall back to any image
    test_images = glob.glob(os.path.expanduser("~/nodule_detection/data/png_images/n0239.png"))
    if not test_images:
        test_images = glob.glob(os.path.expanduser("~/nodule_detection/data/png_images/*.png"))

    if not test_images:
        logger.error("No test images found!")
        return

    test_path = test_images[0]
    logger.info(f"Test image: {test_path}")

    # Build a task dict identical to what RabbitMQ would deliver
    with open(test_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    task = {
        "study_id": "TEST_001",
        "image_data": image_b64,
        "format": "png",
    }

    result = process_task(task, frcnn_model, yolo_model, config, device, logger)

    # Print result without the large base64 image field
    result_display = {k: v for k, v in result.items() if k != "annotated_image_base64"}
    print("\n" + "=" * 60)
    print("TEST RESULT:")
    print(json.dumps(result_display, indent=2))
    print("=" * 60)

    # Save annotated image for visual inspection
    if result.get("annotated_image_base64"):
        out_path = os.path.expanduser("~/nodule_detection/worker/test_result.png")
        img_bytes = base64.b64decode(result["annotated_image_base64"])
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        logger.info(f"Annotated image saved to: {out_path}")

    return result


# =========================================================================
# MAIN
# =========================================================================

# Set to True by signal_handler to break the reconnection loop in run_worker.
_shutdown = False


def main():
    """Entry point: parse args, load models once, then run worker or test.

    Startup sequence:
    1. Resolve config path (CWD-relative → script-relative → absolute)
    2. Configure logging (console + rotating file)
    3. Detect GPU, select device
    4. Load FRCNN and/or YOLOv8 into GPU memory (one-time cost)
    5. Warmup: run a dummy forward pass so CUDA kernels are compiled
    6. Register SIGINT/SIGTERM for graceful shutdown
    7. Enter ``run_worker()`` (production) or ``run_test()`` (smoke test)
    8. On exit: delete models and free GPU memory
    """
    global _shutdown

    parser = argparse.ArgumentParser(description="CXR Inference Worker")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--test", action="store_true", help="Run test inference (no RabbitMQ)")
    args = parser.parse_args()

    # Config resolution: absolute path → CWD-relative → script-relative
    if os.path.isabs(args.config):
        config_path = args.config
    elif os.path.exists(args.config):
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)

    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("CXR Inference Worker starting...")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 60)

    # --- GPU device selection ---
    device_str = config["inference"].get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load models (one-time, stays in GPU memory for the process lifetime) ---
    logger.info("Loading models...")
    frcnn_model = load_frcnn(config, device)
    yolo_model = load_yolo(config)

    if frcnn_model is None and yolo_model is None:
        logger.error("No models enabled! Check config.yaml")
        sys.exit(1)

    logger.info("All models loaded successfully.")

    # --- GPU warmup: first inference compiles CUDA kernels and is slower ---
    logger.info("GPU warmup...")
    dummy = torch.randn(1, 3, 1024, 1024).to(device)
    if frcnn_model:
        with torch.no_grad():
            frcnn_model([dummy[0]])
    del dummy
    torch.cuda.empty_cache()
    logger.info("GPU warmup done.")

    # --- Graceful shutdown via _shutdown flag (no sys.exit) ---
    def signal_handler(sig, frame):
        """Set _shutdown flag so run_worker's reconnect loop exits cleanly."""
        global _shutdown
        logger.info(f"Signal {sig} received, shutting down gracefully...")
        _shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Run ---
    if args.test:
        run_test(config, frcnn_model, yolo_model, device, logger)
    else:
        run_worker(config, frcnn_model, yolo_model, device, logger)

    # --- Cleanup: release GPU memory before process exits ---
    logger.info("Cleaning up GPU...")
    del frcnn_model
    del yolo_model
    torch.cuda.empty_cache()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
