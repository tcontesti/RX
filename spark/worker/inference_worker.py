#!/usr/bin/env python3
"""
CXR Inference Worker — Corre en la Spark con GPU.

Consume tareas de RabbitMQ (cola cxr.inference),
ejecuta inferencia con FRCNN + YOLOv8 + WBF Ensemble,
y publica resultados en cxr.results.

Los modelos se cargan UNA SOLA VEZ al inicio.
Cada tarea procesa UNA imagen (~50ms en GPU).

Uso:
    python inference_worker.py --config config.yaml
    python inference_worker.py --config config.yaml --test  # Test sin RabbitMQ
"""

import argparse
import base64
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import socket
import sys
import time
import traceback
from io import BytesIO
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
    level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file = config.get("logging", {}).get("file", None)

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
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
    """Carga Faster R-CNN con pesos entrenados."""
    cfg = config["models"]["frcnn"]
    if not cfg.get("enabled", False):
        return None

    weights_path = cfg["weights"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"FRCNN weights not found: {weights_path}")

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)

    # Soportar state_dict directo o envuelto
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.roi_heads.score_thresh = cfg.get("score_thresh", 0.005)
    model.roi_heads.nms_thresh = cfg.get("nms_thresh", 0.2)

    total = sum(p.numel() for p in model.parameters())
    logging.getLogger("inference_worker").info(
        f"FRCNN loaded: {total:,} params, score_thresh={cfg['score_thresh']}"
    )
    return model


def load_yolo(config):
    """Carga YOLOv8 con pesos entrenados."""
    cfg = config["models"]["yolov8"]
    if not cfg.get("enabled", False):
        return None

    weights_path = cfg["weights"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"YOLOv8 weights not found: {weights_path}")

    from ultralytics import YOLO
    model = YOLO(weights_path)
    logging.getLogger("inference_worker").info(f"YOLOv8 loaded from {cfg['weights']}")
    return model


# =========================================================================
# IMAGE PROCESSING
# =========================================================================

def decode_image(image_data, image_format="png"):
    """Decodifica imagen desde base64 o path."""
    if os.path.isfile(str(image_data)):
        # Es un path a archivo
        img = cv2.imread(str(image_data), cv2.IMREAD_GRAYSCALE)
    else:
        # Es base64
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Failed to decode image")

    # Validate image dimensions
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        raise ValueError(f"Image too small: {w}x{h} (minimum 10x10)")
    if h > 16384 or w > 16384:
        raise ValueError(f"Image too large: {w}x{h} (maximum 16384x16384)")

    return img


def preprocess(img_gray, target_size=1024):
    """Resize y normaliza a [0,1]."""
    img_resized = cv2.resize(img_gray, (target_size, target_size))
    img_float = img_resized.astype(np.float32)
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    return img_resized, img_float


# =========================================================================
# INFERENCE
# =========================================================================

def infer_frcnn(model, img_float, device):
    """Inferencia con Faster R-CNN. Input: grayscale x 3 canales."""
    img3 = np.stack([img_float, img_float, img_float], axis=0)
    tensor = torch.from_numpy(img3).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)[0]

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    return {"boxes": boxes, "scores": scores, "labels": labels}


def infer_yolo(model, img_uint8, config):
    """Inferencia con YOLOv8."""
    cfg = config["models"]["yolov8"]
    device_str = config["inference"].get("device", "cuda:0")
    img_bgr = cv2.merge([img_uint8, img_uint8, img_uint8])

    results = model.predict(
        source=img_bgr,
        conf=cfg.get("conf_thresh", 0.001),
        iou=0.2,
        imgsz=cfg.get("imgsz", 1024),
        device=device_str,
        verbose=False,
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = np.ones(len(scores), dtype=int)  # clase 1 = nodulo

    return {"boxes": boxes, "scores": scores, "labels": labels}


def ensemble_wbf(predictions, config, img_size=1024):
    """Aplica Weighted Box Fusion sobre predicciones de multiples modelos."""
    from ensemble_boxes import weighted_boxes_fusion

    cfg = config["ensemble"]
    boxes_list, scores_list, labels_list = [], [], []

    total_detections = sum(len(pred["boxes"]) for pred in predictions)
    if total_detections == 0:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0)}

    for pred in predictions:
        if len(pred["boxes"]) == 0:
            boxes_list.append(np.zeros((0, 4)))
            scores_list.append(np.zeros(0))
            labels_list.append(np.zeros(0, dtype=int))
        else:
            boxes_norm = np.clip(pred["boxes"] / img_size, 0, 1)
            boxes_list.append(boxes_norm)
            scores_list.append(pred["scores"].astype(np.float32))
            labels_list.append(np.zeros(len(pred["scores"]), dtype=int))

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=cfg.get("weights", [1.0] * len(predictions)),
        iou_thr=cfg.get("iou_thr", 0.2),
        skip_box_thr=cfg.get("skip_box_thr", 0.05),
    )

    fused_boxes = fused_boxes * img_size
    return {"boxes": fused_boxes, "scores": fused_scores, "labels": fused_labels}


def generate_annotated_image(img_uint8, detections, confidence_thresh=0.3):
    """Genera imagen con bounding boxes dibujados. Retorna PNG en bytes."""
    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    for i, (box, score) in enumerate(zip(detections["boxes"], detections["scores"])):
        if score < confidence_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # verde
        cv2.rectangle(img_color, (x1, y1), (x2, y2), color, 2)
        label = f"Nodule {score:.2f}"
        cv2.putText(img_color, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    _, buffer = cv2.imencode(".png", img_color)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================================
# TASK PROCESSING
# =========================================================================

def process_task(task, frcnn_model, yolo_model, config, device, logger):
    """Procesa una tarea de inferencia completa."""
    start = time.time()

    study_id = task.get("study_id")
    image_data = task.get("image_data")  # base64 o path
    image_format = task.get("format", "png")
    img_size = config["inference"].get("input_size", 1024)

    logger.info(f"Processing study_id={study_id}")

    # 1. Decodificar imagen
    img_gray = decode_image(image_data, image_format)

    # 2. Preprocesar
    img_uint8, img_float = preprocess(img_gray, img_size)

    # 3. Inferencia con cada modelo
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

    # 4. Ensemble WBF
    if len(predictions) > 1 and config["ensemble"].get("enabled", True):
        ensemble_pred = ensemble_wbf(predictions, config, img_size)
    elif len(predictions) == 1:
        ensemble_pred = predictions[0]
    else:
        ensemble_pred = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0)}

    # 5. Filtrar por confidence threshold
    conf_thresh = config["inference"].get("output_confidence_thresh", 0.3)
    keep = ensemble_pred["scores"] >= conf_thresh
    final_detections = {
        "boxes": ensemble_pred["boxes"][keep],
        "scores": ensemble_pred["scores"][keep],
        "labels": ensemble_pred["labels"][keep],
    }

    # 6. Generar imagen anotada (opcional)
    annotated_image_b64 = None
    if config["inference"].get("save_annotated_image", True):
        annotated_image_b64 = generate_annotated_image(img_uint8, final_detections, conf_thresh)

    elapsed = round((time.time() - start) * 1000, 1)

    # 7. Construir resultado
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

    logger.info(
        f"  Done: {result['num_detections']} detections, {elapsed}ms"
    )
    return result


# =========================================================================
# RABBITMQ CONSUMER
# =========================================================================

def resolve_host(hostname, logger, max_retries=5, delay=5):
    """Resolve hostname with retries (mDNS puede tardar tras cambio de red)."""
    for attempt in range(max_retries):
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            if attempt < max_retries - 1:
                logger.warning(f"Cannot resolve {hostname}, retry {attempt+1}/{max_retries} in {delay}s...")
                time.sleep(delay)
            else:
                raise


def run_worker(config, frcnn_model, yolo_model, device, logger):
    """Bucle principal con reconexión automática a RabbitMQ."""
    rmq = config["rabbitmq"]
    _shutdown = False
    _channel = None
    _connection = None

    def signal_handler(sig, frame):
        nonlocal _shutdown, _channel, _connection
        logger.info("Signal received, shutting down...")
        _shutdown = True
        # Interrupt blocking start_consuming()
        try:
            if _channel and _channel.is_open:
                _channel.stop_consuming()
        except Exception:
            pass

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    reconnect_delay = 10  # initial delay in seconds
    max_delay = 300       # max delay (5 min)

    while not _shutdown:
        connection = None
        try:
            resolved_host = resolve_host(rmq["host"], logger)
            logger.info(f"Resolved {rmq['host']} -> {resolved_host}")
            credentials = pika.PlainCredentials(rmq["user"], rmq["password"])
            parameters = pika.ConnectionParameters(
                host=resolved_host,
                port=rmq.get("port", 5672),
                credentials=credentials,
                heartbeat=rmq.get("heartbeat", 600),
                blocked_connection_timeout=rmq.get("blocked_connection_timeout", 300),
            )

            connection = pika.BlockingConnection(parameters)
            _connection = connection
            channel = connection.channel()
            _channel = channel

            # Declarar colas
            channel.queue_declare(queue=rmq["queue_input"], durable=True)
            channel.queue_declare(queue=rmq["queue_output"], durable=True)

            # 1 tarea a la vez (GPU)
            channel.basic_qos(prefetch_count=rmq.get("prefetch_count", 1))

            # Reset backoff on successful connection
            reconnect_delay = 10

            logger.info(f"Connected to RabbitMQ {rmq['host']}:{rmq['port']}")
            logger.info(f"Consuming from: {rmq['queue_input']}")
            logger.info(f"Publishing to: {rmq['queue_output']}")
            logger.info("Waiting for tasks...")

            def callback(ch, method, properties, body):
                task = None
                try:
                    task = json.loads(body)

                    # Validate required fields
                    if not task.get("study_id") or not task.get("image_data"):
                        raise ValueError(f"Missing required fields (study_id/image_data) in task: {list(task.keys())}")

                    result = process_task(task, frcnn_model, yolo_model, config, device, logger)

                    # Publicar resultado
                    channel.basic_publish(
                        exchange="",
                        routing_key=rmq["queue_output"],
                        body=json.dumps(result),
                        properties=pika.BasicProperties(delivery_mode=2),
                    )

                except Exception as e:
                    logger.error(f"Error processing task: {e}\n{traceback.format_exc()}")
                    study_id = task.get("study_id") if isinstance(task, dict) else None
                    error_result = {
                        "study_id": study_id,
                        "status": "error",
                        "error": str(e),
                    }
                    try:
                        channel.basic_publish(
                            exchange="",
                            routing_key=rmq["queue_output"],
                            body=json.dumps(error_result),
                            properties=pika.BasicProperties(delivery_mode=2),
                        )
                    except Exception as pub_err:
                        logger.error(f"Failed to publish error result: {pub_err}")
                finally:
                    try:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            channel.basic_consume(queue=rmq["queue_input"], on_message_callback=callback)
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"RabbitMQ connection lost: {e}")
        except pika.exceptions.AMQPChannelError as e:
            logger.warning(f"RabbitMQ channel error: {e}")
        except Exception as e:
            if _shutdown:
                break
            logger.error(f"Unexpected error in worker loop: {e}\n{traceback.format_exc()}")
        finally:
            if connection and not connection.is_closed:
                try:
                    connection.close()
                except Exception:
                    pass

        if not _shutdown:
            logger.info(f"Reconnecting to RabbitMQ in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_delay)

    logger.info("Worker stopped.")


# =========================================================================
# TEST MODE (sin RabbitMQ)
# =========================================================================

def run_test(config, frcnn_model, yolo_model, device, logger):
    """Test de inferencia con una imagen del dataset, sin RabbitMQ."""
    import glob

    # Buscar una imagen con nodulo del val set
    test_images = glob.glob(os.path.expanduser("~/nodule_detection/data/png_images/n0239.png"))
    if not test_images:
        test_images = glob.glob(os.path.expanduser("~/nodule_detection/data/png_images/*.png"))

    if not test_images:
        logger.error("No test images found!")
        return

    test_path = test_images[0]
    logger.info(f"Test image: {test_path}")

    # Simular tarea como si viniera de RabbitMQ
    with open(test_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    task = {
        "study_id": "TEST_001",
        "image_data": image_b64,
        "format": "png",
    }

    result = process_task(task, frcnn_model, yolo_model, config, device, logger)

    # Mostrar resultado (sin la imagen anotada en base64 que es muy larga)
    result_display = {k: v for k, v in result.items() if k != "annotated_image_base64"}
    print("\n" + "=" * 60)
    print("TEST RESULT:")
    print(json.dumps(result_display, indent=2))
    print("=" * 60)

    # Guardar imagen anotada
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

def main():
    parser = argparse.ArgumentParser(description="CXR Inference Worker")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--test", action="store_true", help="Run test inference (no RabbitMQ)")
    args = parser.parse_args()

    # Cargar config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("CXR Inference Worker starting...")
    logger.info("=" * 60)

    # Device
    device_str = config["inference"].get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Cargar modelos (UNA VEZ)
    logger.info("Loading models...")
    frcnn_model = load_frcnn(config, device)
    yolo_model = load_yolo(config)

    if frcnn_model is None and yolo_model is None:
        logger.error("No models enabled! Check config.yaml")
        sys.exit(1)

    logger.info("All models loaded successfully.")

    # Warmup GPU (primera inferencia es mas lenta)
    logger.info("GPU warmup...")
    dummy = torch.randn(1, 3, 1024, 1024).to(device)
    if frcnn_model:
        with torch.no_grad():
            frcnn_model([dummy[0]])
    del dummy
    torch.cuda.empty_cache()
    logger.info("GPU warmup done.")

    # Ejecutar
    if args.test:
        run_test(config, frcnn_model, yolo_model, device, logger)
    else:
        run_worker(config, frcnn_model, yolo_model, device, logger)


if __name__ == "__main__":
    main()
