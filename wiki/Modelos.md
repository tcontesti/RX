# Modelos

[← Home](Home.md)

Fichas técnicas de los modelos entrenados y evaluados en el proyecto. Para cada uno: arquitectura, pretraining, hiperparámetros y métricas finales.

## Faster R-CNN ResNet-50 + FPN (preentrenado VinDr-CXR)

**Es el modelo individual de mayor peso en el ensemble.**

| Campo | Valor |
|---|---|
| Familia | Two-stage detector |
| Backbone | ResNet-50 + FPN |
| Cabezal | RoI pooling + clasificador binario (nodule / background) + regresión de cajas |
| Pretraining | COCO → VinDr-CXR 21 patologías → fine-tuning NODE21 |
| Resolución entrada | 1024 × 1024 |
| Anchor scales | (32, 64, 128, 256, 512) |
| Anchor ratios | (0.5, 1.0, 2.0) |
| Optimizer | SGD (lr = 5e-3, momentum = 0.9, weight decay = 5e-4) |
| Scheduler | Cosine annealing |
| Épocas | 30 |
| Batch size | 4 |
| Score threshold inferencia | 0.005 |
| NMS threshold | 0.2 |
| **NODE21 / CM / AUROC** | **0.9025 / 0.9146 / 0.9460** |
| Latencia | 55 ms / imagen (Grace Hopper GB10) |
| Implementación | `torchvision.models.detection.faster_rcnn` |
| Script | [`spark/scripts/train_frcnn_vindr.py`](../spark/scripts/train_frcnn_vindr.py) |

La clave de su rendimiento es el preentrenamiento sobre VinDr-CXR (21 patologías torácicas en 18 000 radiografías). Sin VinDr, la cifra cae ~5 puntos NODE21.

## YOLOv8s

| Campo | Valor |
|---|---|
| Familia | One-stage detector |
| Backbone | CSPDarknet53 ligero |
| Pretraining | COCO (Ultralytics default) |
| Resolución entrada | 1024 × 1024 |
| Optimizer | SGD (lr0 = 0.01, momentum = 0.937, weight decay = 5e-4) |
| Scheduler | Cosine LR (`cos_lr = True`) |
| Épocas | 100 |
| Batch size | 16 |
| Confidence threshold | 0.001 |
| NMS IoU | 0.45 |
| **NODE21 / CM / AUROC** | **0.9103 / 0.9283 / 0.9686** |
| Latencia | 25 ms / imagen |
| Implementación | `ultralytics.YOLO` |
| Script | [`spark/scripts/train_yolo.py`](../spark/scripts/train_yolo.py) |

YOLOv8s tiene la mejor AUROC individual del proyecto. Sin embargo, su precisión en cajas pequeñas es algo menor que la de Faster R-CNN, lo que motiva el ensemble.

## YOLO26s

| Campo | Valor |
|---|---|
| Familia | One-stage detector (generación posterior) |
| Backbone | CSPDarknet53 v2 |
| Pretraining | COCO |
| Resolución entrada | 1024 × 1024 |
| Hiperparámetros | Heredados del default de Ultralytics |
| **NODE21 / CM / AUROC** | **0.7929 / 0.8754 / 0.9557** |
| Latencia | 18 ms / imagen |
| Script | [`spark/scripts/train_yolo.py --model yolo26s`](../spark/scripts/train_yolo.py) |

YOLO26s no supera a YOLOv8s en este dataset, probablemente porque su regularización por defecto está calibrada para datasets más grandes que NODE21.

## Ensemble Weighted Box Fusion

| Campo | Valor |
|---|---|
| Modelos combinados | Faster R-CNN VinDr + YOLOv8s |
| Pesos | [0.90, 0.91] (proporcionales a NODE21 individual) |
| IoU threshold | 0.20 |
| Skip box threshold | 0.05 |
| **NODE21 / CM / AUROC** | **0.9391 / 0.9447 / 0.9683** |
| Latencia total | 81 ms / imagen |
| Implementación | `ensemble_boxes.weighted_boxes_fusion` |
| Script | [`spark/scripts/ensemble_wbf.py`](../spark/scripts/ensemble_wbf.py) |

Ganancia frente al mejor modelo individual: +2.6 puntos NODE21, +5.3 puntos Sens@0.25 FP. Es el modelo desplegado en producción en el prototipo asistencial.

## EfficientNet-B0 multicanal (clasificación binaria, TFG original)

Heredado del proyecto base UIB. **No forma parte del pipeline de producción**, pero se documenta por trazabilidad.

| Campo | Valor |
|---|---|
| Tarea | Clasificación binaria (nodule / no nodule) sobre crops |
| Backbone | EfficientNet-B0 |
| Input | 4 canales (RGB + máscara de pulmón) |
| Pretraining | ImageNet |
| Accuracy | ~ 0.88 (split nuestro, no comparable directamente con NODE21) |
| Notebook | [`original-project/PRACTICA_LINK_CONTESTI/SCRIPTS_MODELS_I_AVALUACIONS_MODELS/CLASSIFICACIO/node21_v1.2_cs.ipynb`](../original-project/PRACTICA_LINK_CONTESTI/SCRIPTS_MODELS_I_AVALUACIONS_MODELS/CLASSIFICACIO/node21_v1.2_cs.ipynb) |

Fue el modelo de referencia del TFG inicial, antes de pivotar a detección + ensemble.

## Comparativa rápida

| Modelo | NODE21 | AUROC | CM | Latencia | Tamaño |
|---|---|---|---|---|---|
| **Ensemble WBF** | **0.9391** | 0.9683 | **0.9447** | 81 ms | 250 MB |
| YOLOv8s | 0.9103 | **0.9686** | 0.9283 | 25 ms | 22 MB |
| FRCNN VinDr | 0.9025 | 0.9460 | 0.9146 | 55 ms | 165 MB |
| YOLO26s | 0.7929 | 0.9557 | 0.8754 | 18 ms | 22 MB |
| Behrendt et al. | — | — | 0.8390 | — | 21 detectores en cascada |

Para los hiperparámetros completos y los logs de entrenamiento ver [[Experimentos]] y [`docs/WEIGHTS_AND_CHECKPOINTS.md`](../docs/WEIGHTS_AND_CHECKPOINTS.md). Para la justificación arquitectónica del ensemble, ver el capítulo 4 de [[Memoria-academica]].
