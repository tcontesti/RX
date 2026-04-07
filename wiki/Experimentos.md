# Experimentos

[← Home](Home.md)

Mapa de los experimentos realizados, qué script o pipeline los reproduce y la métrica esperada. Todos los scripts viven en [`spark/scripts/`](../spark/scripts/) o en el módulo [`spark/pipeline/`](../spark/pipeline/).

## Tabla maestra de experimentos

| # | Experimento | Script | Métrica clave | Resultado |
|---|---|---|---|---|
| 1 | Faster R-CNN VinDr (modelo final) | `spark/scripts/train_frcnn_vindr.py` | NODE21 / CM / AUROC | 0.9025 / 0.9146 / 0.9460 |
| 2 | YOLOv8s individual | `spark/scripts/train_yolo.py --model yolov8s --imgsz 1024` | NODE21 / CM / AUROC | 0.9103 / 0.9283 / 0.9686 |
| 3 | YOLO26s individual | `spark/scripts/train_yolo.py --model yolo26s --imgsz 1024` | NODE21 / CM / AUROC | 0.7929 / 0.8754 / 0.9557 |
| 4 | **Ensemble WBF (FRCNN + YOLOv8s)** | `spark/scripts/ensemble_wbf.py` | NODE21 / CM / AUROC | **0.9391 / 0.9447 / 0.9683** |
| 5 | Replicación Behrendt et al. | `spark/scripts/train_frcnn_reproduce.py` | NODE21 limpio | 0.83 ± 0.02 |
| 6 | Ablation data leakage | `spark/scripts/generate_augmented_images.py` + `evaluate.py` | NODE21 sucio vs limpio | 0.9695 → 0.9025 |
| 7 | Ablation score threshold | `spark/scripts/evaluate.py --score-thresh {0.5, 0.1, 0.01, 0.005}` | NODE21 | 0.45 → 0.85 |

Las cifras son las honestas, posteriores a aplicar el split limpio (`StratifiedGroupKFold` agrupando por imagen original antes de aplicar augmentaciones offline). Para cada experimento, los checkpoints y configuraciones quedan documentados en [`docs/WEIGHTS_AND_CHECKPOINTS.md`](../docs/WEIGHTS_AND_CHECKPOINTS.md).

## Detalle por experimento

### 1 · Faster R-CNN preentrenado sobre VinDr-CXR

- **Backbone:** ResNet-50 + FPN.
- **Pretraining:** COCO → VinDr-CXR (21 patologías) → fine-tuning sobre NODE21.
- **Hiperparámetros clave:** lr = 5e-3, momentum = 0.9, weight decay = 5e-4, batch = 4, épocas = 30.
- **Resolución de entrada:** 1024 × 1024.
- **Score threshold de inferencia:** 0.005 (no 0.5 — ver experimento 7).

Es el modelo del que se extrae el peso de mayor influencia en el ensemble.

### 2 · YOLOv8s

- **Pretraining:** COCO (default Ultralytics).
- **Hiperparámetros clave:** imgsz = 1024, epochs = 100, batch = 16, lr0 = 0.01, cos_lr = True.
- **Confidence threshold inferencia:** 0.001 (NMS posterior).

YOLOv8s tiene la mejor AUROC individual (0.9686), comparable a la del ensemble.

### 3 · YOLO26s

- **Hiperparámetros:** ídem YOLOv8s.
- **Resultado:** la arquitectura más reciente no supera a YOLOv8s en este dataset — probablemente la regularización por defecto está calibrada para datasets más grandes.

### 4 · Ensemble Weighted Box Fusion

- **Modelos:** Faster R-CNN VinDr + YOLOv8s.
- **Pesos:** [0.90, 0.91] (proporcionales al NODE21 score individual).
- **IoU threshold:** 0.20.
- **Skip box threshold:** 0.05.

Resultado: +2.6 puntos NODE21 y +5.3 puntos Sens@0.25 FP sobre el mejor individual. El ensemble compensa que Faster R-CNN tiene mejor precisión en cajas pequeñas y YOLOv8s mejor en cajas medianas.

### 5 · Replicación de Behrendt et al.

Script `train_frcnn_reproduce.py` reproduce la configuración publicada por el grupo de Hamburg. La replicación honesta (sin las copias aumentadas filtradas en validación) obtiene NODE21 ≈ 0.83, valor coherente con el baseline limpio. Esto demuestra que la cifra publicada por Behrendt incorpora data leakage no documentado.

### 6 · Ablation del data leakage

Procedimiento:

1. Genera augmentaciones offline ignorando agrupación por imagen original (réplica del bug Behrendt).
2. Evalúa: obtiene NODE21 = 0.9695.
3. Aplica `StratifiedGroupKFold` agrupando por `image_uid` antes de aumentar.
4. Re-evalúa: obtiene NODE21 = 0.9025.

**Diferencia: +6.7 puntos absolutos de inflado falso.**

Es el hallazgo cuantitativo más importante de la auditoría. Documentado en [`docs/BEHRENDT_BUGS_ANALYSIS.md`](../docs/BEHRENDT_BUGS_ANALYSIS.md).

### 7 · Ablation del score threshold

| Score threshold | NODE21 score |
|---|---|
| 0.5 | 0.45 |
| 0.1 | 0.61 |
| 0.01 | 0.78 |
| 0.005 | 0.85 |

La métrica NODE21 ranquea por confidence; subir el threshold descarta cajas verdaderas con score bajo. La práctica habitual en literatura (threshold 0.5) penaliza injustamente al modelo. Threshold operativo del prototipo: 0.005 (inferencia) + 0.3 (visualización por defecto, ajustable por el radiólogo).

## Métricas usadas

- **NODE21 score** — métrica oficial del Challenge (CM): combina Sens@0.25 FP y AUROC ponderado.
- **Sens@0.25 FP** — sensibilidad a 0.25 falsos positivos por imagen (cribado realista).
- **AUROC** — comportamiento de clasificación binaria (caso vs control) al nivel de imagen.
- **FROC** — curva de sensibilidad vs falsos positivos por imagen.

## Resultados en una tabla

| # | Modelo | NODE21 | AUROC | CM | Latencia |
|---|---|---|---|---|---|
| 1 | Ensemble WBF | **0.9391** | 0.9683 | **0.9447** | 81 ms |
| 2 | YOLOv8s | 0.9103 | **0.9686** | 0.9283 | 25 ms |
| 3 | FRCNN VinDr | 0.9025 | 0.9460 | 0.9146 | 55 ms |
| 4 | YOLO26s | 0.7929 | 0.9557 | 0.8754 | 18 ms |
| — | Behrendt et al. (oficial) | — | — | 0.8390 | — |

Ver [[Modelos]] para las fichas de arquitectura completas de cada modelo, y [[Reproducibilidad]] para los comandos exactos de reproducción.
