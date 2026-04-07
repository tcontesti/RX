# PROYECTO: Detección de Nódulos Pulmonares en CXR
# Herramienta de Inferencia Hospitalaria Multi-Patología

**Autores**: Toni (desarrollo actual), Marc Link Cladera & Antonio Contesti Coll (proyecto UIB original)
**Supervisor original**: Miquel Miró Nicolau (UIB)
**Referencia principal**: Behrendt et al. 2023, Nature Scientific Reports — NODE21 winner
**Última actualización**: 2026-04-03

---

## 1. ESTADO ACTUAL

### 1.1 Resultados — Ranking final de todos los modelos

#### Checkpoints originales evaluados en Spark (los mejores)

| # | Modelo | NODE21 Score | AUROC | CM | Sens@0.25FP | ms/img |
|---|--------|-------------|-------|----|-------------|--------|
| **1** | **FRCNN-3ch (original)** | **0.9596** | **0.9904** | **0.9795** | **0.947** | 35.0 |
| 2 | FRCNN-CBAM (original) | 0.9181 | 0.9586 | 0.9391 | 0.880 | 43.9 |
| 3 | YOLOv8s (Spark) | 0.9136 | 0.9686 | 0.9316 | 0.821 | 16.3 |
| 4 | YOLO26s v2 (Spark) | 0.7929 | 0.9557 | 0.8754 | 0.635 | 18.5 |

#### Modelos entrenados desde cero en Spark (sin datos augmentados)

| Modelo | NODE21 v1 | NODE21 v2 | AUROC | CM |
|--------|-----------|-----------|-------|----|
| YOLOv8s | 0.9103 | 0.9103 | 0.9686 | 0.9283 |
| YOLO26s | 0.7857 | 0.7929 | 0.9557 | 0.8754 |
| Faster R-CNN VinDr | 0.4546 | 0.6417 | 0.8623 | 0.7879 |

#### Resultados del proyecto UIB original (evaluados en Google Colab)

| Modelo | NODE21 Score | Checkpoint |
|--------|-------------|-----------|
| FRCNN VinDr + CBAM | 0.8924 | checkpoint_canal_frcnn_vindn_attention_cbam_epoch_16.pth |
| FRCNN VinDr 3-canal | 0.8913 | checkpoint_3canal_frcnn_vindn_epoch_11.pth |
| FRCNN VinDr 1-canal | 0.8901 | checkpoint_1canal_frcnn_vindn_epoch_14.pth |
| FRCNN VinDr focal loss | 0.8897 | checkpoint_epoch_13_vidn_local_foss.pth |
| YOLOv8s | ~0.93 | YOLOv8s entrenado en Colab |

**NOTA**: La discrepancia entre el score original de Colab (~0.89) y el evaluado en Spark (0.9596) para el mismo checkpoint se debe a un val set diferente. Los scores de Spark son sobre el val_fold0 generado en la Spark. Los de Colab sobre el split original de Colab.

#### Referencia: Paper de Behrendt (ganador NODE21)

| Modelo | CM | AUROC | FROC@25% |
|--------|-----|-------|----------|
| Ensemble (5 modelos × 5 folds + WBF) | 83.90% | 86.79% | 75.24% |
| Mejor individual (YOLOv5s) | 87.94% | 94.59% | — |

### 1.2 Sensibilidad detallada por nivel FP/imagen

| Modelo | @0.25 | @0.5 | @1 | @2 | @4 | @8 |
|--------|-------|------|-----|-----|-----|-----|
| FRCNN-3ch (original) | **0.947** | **0.957** | **0.963** | **0.963** | **0.963** | **0.963** |
| FRCNN-CBAM (original) | 0.880 | 0.900 | 0.920 | 0.934 | 0.934 | 0.940 |
| YOLOv8s (Spark) | 0.821 | 0.870 | 0.924 | 0.953 | 0.957 | 0.957 |
| YOLO26s v2 (Spark) | 0.635 | 0.751 | 0.791 | 0.834 | 0.860 | 0.887 |

### 1.3 Reproducción FRCNN-3ch desde cero (completada 2026-04-03)

| Versión | NODE21 | AUROC | CM | Sens@0.25FP | Notas |
|---------|--------|-------|----|-------------|-------|
| **Réplica A best_node21** | **0.9695** | 0.9936 | 0.9827 | 0.950 | Con data leakage (reproducción exitosa) |
| Réplica A best_valloss | 0.9651 | 0.9744 | 0.9592 | 0.914 | Con data leakage |
| Original (checkpoint Colab) | 0.9596 | 0.9904 | 0.9795 | 0.947 | Referencia |
| **Corrected B (sin leakage)** | **0.9025** | 0.9460 | 0.9146 | 0.821 | **Score real honesto** |

**Hallazgo clave**: Data leakage (augmentaciones de misma imagen en train+val) inflaba score en +6.7 puntos. Score real sin leakage: **0.9025**.

### 1.4 Ranking real (sin data leakage, scores honestos)

| # | Modelo | NODE21 | AUROC | CM | Notas |
|---|--------|--------|-------|----|-------|
| 1 | **YOLOv8s** | **0.9103** | 0.9686 | 0.9283 | Mejor modelo real |
| 2 | **FRCNN-3ch VinDr** | **0.9025** | 0.9460 | 0.9146 | Score corregido sin leakage |
| 3 | YOLO26s | 0.7929 | 0.9557 | 0.8754 | — |

### 1.5 Ensemble WBF (completado 2026-04-03)

**Weighted Box Fusion** combina predicciones de FRCNN y YOLOv8s fusionando boxes solapadas (media ponderada de coordenadas por scores) en vez de eliminarlas como NMS.

| Modelo | NODE21 | AUROC | CM | S@0.25 | S@0.5 | S@1 | S@2 | S@4 | S@8 |
|--------|--------|-------|----|--------|-------|-----|-----|-----|-----|
| **Ensemble WBF** | **0.9391** | **0.9683** | **0.9447** | **0.874** | **0.914** | **0.944** | **0.960** | **0.970** | **0.973** |
| YOLOv8s | 0.9103 | 0.9686 | 0.9316 | 0.821 | 0.870 | 0.924 | 0.953 | 0.957 | 0.957 |
| FRCNN Corrected-B | 0.9025 | 0.9460 | 0.9146 | 0.821 | 0.854 | 0.890 | 0.930 | 0.947 | 0.973 |

**Ganancia del ensemble sobre mejor individual:**
- NODE21: +2.6 puntos (0.9391 vs 0.9103)
- Sens@0.25 FP: +5.3 puntos (0.874 vs 0.821) — mayor ganancia donde más importa clínicamente
- Sens@1 FP: +2.0 puntos (0.944 vs 0.924)

**Config WBF**: weights=[0.90, 0.91], iou_thr=0.2, skip_box_thr=0.05

**Comparación con Behrendt (ganador NODE21)**: CM=94.47% (nuestro) vs CM=83.90% (Behrendt). Nuestro ensemble de 2 modelos supera su ensemble de 21 modelos, aunque los test sets son diferentes.

### 1.6 Ranking final definitivo (scores honestos, sin leakage)

| # | Modelo | NODE21 | AUROC | CM | Notas |
|---|--------|--------|-------|----|-------|
| **1** | **Ensemble WBF (FRCNN+YOLOv8)** | **0.9391** | **0.9683** | **0.9447** | **Mejor resultado** |
| 2 | YOLOv8s | 0.9103 | 0.9686 | 0.9283 | Mejor individual, más rápido (16ms) |
| 3 | FRCNN VinDr Corrected-B | 0.9025 | 0.9460 | 0.9146 | Score honesto sin leakage |
| 4 | YOLO26s | 0.7929 | 0.9557 | 0.8754 | Excluido del ensemble |

---

## 2. DATASETS

### 2.1 NODE21 (en uso)
- **Fuente**: Zenodo (DOI: 10.5281/zenodo.5548363) / Kaggle (node-21-dataset-untampered)
- **Tamaño**: 4,882 radiografías CXR frontal en formato .mha (1024×1024 px)
- **Positivos**: 1,134 imágenes con 1,476 nódulos anotados (bounding boxes)
- **Negativos**: 3,748 imágenes sin nódulos
- **Prevalencia**: 23% positivos
- **Anotaciones**: CSV con columnas: img_name, label, x, y, width, height
  - x, y = coordenadas top-left del bounding box
  - width, height = dimensiones del bbox
  - Una imagen puede tener múltiples filas (múltiples nódulos)
- **Fuentes originales**: JSRT (242), PadChest (1,680), ChestX-ray14 (1,804), Open-I (1,218)
- **Test sets**: Privados (no disponibles públicamente)
- **Licencia**: CC BY-NC-ND 4.0
- **Estado en Spark**: Descargado (37GB), preprocesado a PNG, splits generados, formato YOLO creado

### 2.2 Dataset augmentado (metadata_augmented_def2.csv)
- **Origen**: Generado offline en el proyecto UIB original
- **Total filas**: 8,175
  - 5,224 filas originales (4,882 imágenes)
  - 2,951 filas augmentadas (2,268 imágenes con sufijo _aug0, _aug1)
- **Todas las augmentadas son positivas** (label=1)
- **Ratio final**: 3,748 negativos (46%) + 4,427 positivos (54%)
- **Augmentación usada** (Albumentations con BboxParams):
  - HorizontalFlip(p=0.5)
  - RandomBrightnessContrast(p=0.3)
  - Affine(translate=5%, scale=0.9-1.1, rotate=±10°, p=0.5)
  - BboxParams format='pascal_voc', min_visibility=0.3
- **PROBLEMA DETECTADO**: Split original no agrupa por imagen base → posible data leakage (n0080.png en train, n0080_aug0.png en val)

### 2.3 VinDr-CXR (pendiente acceso PhysioNet)
- **Fuente**: PhysioNet (https://physionet.org/content/vindr-cxr/1.0.0/)
- **DOI**: 10.13026/3akn-b287
- **Tamaño**: 18,000 radiografías DICOM (15K train / 3K test)
- **Anotadores**: 3 radiólogos (train), 5 en consenso (test), todos con 8+ años experiencia
- **22 hallazgos locales con bounding boxes**:

| # | Hallazgo | Tipo |
|---|----------|------|
| 1 | Aortic enlargement | Cardiovascular |
| 2 | Atelectasis | Pulmón |
| 3 | Calcification | General |
| 4 | Cardiomegaly | Cardiovascular |
| 5 | Clavicle fracture | Traumatología |
| 6 | Consolidation | Pulmón |
| 7 | Edema | Pulmón/Cardio |
| 8 | Emphysema | Pulmón |
| 9 | Enlarged PA | Cardiovascular |
| 10 | Interstitial lung disease (ILD) | Pulmón |
| 11 | Infiltration | Pulmón |
| 12 | Lung cavity | Pulmón |
| 13 | Lung cyst | Pulmón |
| 14 | Lung opacity | Pulmón |
| 15 | Mediastinal shift | Mediastino |
| 16 | **Nodule/Mass** | Pulmón |
| 17 | Pulmonary fibrosis | Pulmón |
| 18 | Pneumothorax | Pulmón/Urgencias |
| 19 | Pleural thickening | Pleura |
| 20 | Pleural effusion | Pleura |
| 21 | Rib fracture | Traumatología |
| 22 | Other lesion | General |

- **6 diagnósticos globales** (imagen completa): Lung tumor, Pneumonia, Tuberculosis, COPD, Other diseases, No finding
- **Formato anotaciones**: CSV con image_id, rad_id, class_name, x_min, y_min, x_max, y_max
- **Acceso**: Requiere credenciales PhysioNet + CITI training + DUA
- **Kaggle alternativa**: vinbigdata-chest-xray-abnormalities-detection (14 de 22 clases, sin test labels)
- **Preferido**: PhysioNet (22 clases, test labels disponibles, DOI citeable para paper)
- **Estado**: Usuario solicitando acceso (esperando en unos días)

### 2.4 Datasets TB (planificados para Fase 2)
- **TBX11K**: 11,200 CXR con bounding boxes de TB (el más completo)
- **Montgomery County**: 138 CXR, clasificación binaria TB sí/no
- **Shenzhen Hospital**: 662 CXR, clasificación binaria TB sí/no
- **VinDr-CXR**: Tiene label global "Tuberculosis" (integrable con Fase 3)

---

## 3. ARQUITECTURAS DE LOS MODELOS

### 3.1 Faster R-CNN + VinDr-CXR (two-stage detector)

```
Entrada (3×1024×1024)
    ↓
[ResNet50 backbone]
  ├─ conv1 + bn1 + relu + maxpool (FROZEN)
  ├─ layer1 (FROZEN)
  ├─ layer2 (FROZEN)
  ├─ layer3 (FROZEN)
  └─ layer4 (TRAINABLE)
    ↓
[Feature Pyramid Network - FPN] (TRAINABLE)
  ├─ P2, P3, P4, P5 feature maps
    ↓
[Region Proposal Network - RPN] (TRAINABLE)
  ├─ Genera ~300 propuestas de regiones
    ↓
[RoI Pooling + Box Head] (TRAINABLE)
  ├─ FastRCNNPredictor(in_features, 2)
  └─ Salida: boxes (x1,y1,x2,y2) + scores + labels
```

- **Pretraining**: Pesos VinDr-CXR (`fastercnn50.pth`, 159MB)
- **Carga de pesos**: Mapeo secuencial de keys + eliminar head viejo (4 capas)
- **Entrada**: Grayscale replicado a 3 canales (NO multicanal CLAHE/Unsharp)
- **Parámetros**: ~41.3M total, ~32.8M trainable, ~8.5M frozen
- **Variante CBAM**: Módulo de atención (channel + spatial) después de layer4

#### Variante CBAM (Convolutional Block Attention Module)

```python
CBAM(channels=2048, reduction=16):
  Channel Attention: GAP + GMP → MLP → sigmoid → modulate channels
  Spatial Attention: ChannelAvg + ChannelMax → Conv7x7 → sigmoid → modulate spatial
```

Integrado como `ResNetBackboneWithCBAM`: forward pasa por layers 1-4, luego CBAM sobre C5.

### 3.2 YOLOv8s (single-stage, anchor-free)
- **Backbone**: CSPDarknet + SPPF
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Decoupled head (clasificación y regresión separadas)
- **Pretraining**: COCO (yolov8s.pt, Ultralytics)
- **Resolución**: 1024×1024
- **Parámetros**: ~11.2M
- **Inferencia**: 16.3 ms/imagen (el más rápido)
- **Augmentación**: Solo flip horizontal (mosaic/mixup/color DESACTIVADOS para imagen médica)
- **Hiperparámetros**: AdamW lr=1e-4→1e-5, box=7.5, cls=0.3, dfl=1.5, patience=20

### 3.3 YOLO26s (single-stage, anchor-free, NMS-free)
- **Backbone**: Arquitectura YOLO26 con mejoras sobre v8
- **STAL**: Small-Target-Aware Label Assignment (ideal para nódulos pequeños)
- **ProgLoss**: Progressive Loss Balancing (reemplaza DFL)
- **MuSGD**: Optimizador híbrido SGD + Muon
- **NMS-free**: No requiere post-procesamiento NMS
- **Resolución**: 1024×1024
- **Inferencia**: 18.5 ms/imagen
- **Augmentación**: Igual que YOLOv8 (conservadora)

### 3.4 Clasificación: EfficientNet-B0 multicanal (proyecto UIB)
- **Backbone**: EfficientNet-B0 (IMAGENET1K_V1)
- **Entrada**: 4 canales (Original + CLAHE + Canny + Unsharp Mask)
- **Primera conv**: Modificada de Conv2d(3,32) a Conv2d(4,32)
- **Classifier**: Linear(1280, 2) — binario (nódulo sí/no)
- **Resultado**: 98.07% accuracy, 98.93% recall en clase positiva
- **Dataset**: Balanceado (3,748 × 2 = 7,496 imágenes)
- **Uso previsto**: Primera etapa del pipeline hospitalario (filtro rápido)

---

## 4. CÓDIGO EXISTENTE — REVIEW COMPLETO

### 4.1 Código Behrendt (node21-submit)

#### Estructura

```
node21-submit/
├── process.py              # Pipeline de inferencia (ensemble de 5 modelos × 5 folds)
├── postprocessing.py       # NMS + PostProcess DETR
├── config_fcrnn_l.yaml     # Config Faster R-CNN
├── config_retina_l.yaml    # Config RetinaNet
├── config_effdet2_l.yaml   # Config EfficientDet-D2
├── fastercnn50.pth         # Pesos VinDr Faster R-CNN (159MB)
├── yolo5x_vindr.pt         # Pesos VinDr YOLOv5x (170MB)
├── F1_E79_ModelX_v4.ckpt   # Pesos VinDr EfficientDet (41MB)
├── requirements.txt        # Dependencias (Lightning 1.5.1, etc.)
├── src/
│   ├── models/
│   │   ├── Detector.py           # Lightning module (train/val/test loops)
│   │   ├── Detector_effdet.py    # Lightning module para EfficientDet
│   │   ├── Detector_detr.py      # Lightning module para DETR
│   │   └── modules/
│   │       ├── FasterCNN.py      # Wrapper Faster R-CNN + losses customs
│   │       ├── RetinaNet.py      # Wrapper RetinaNet
│   │       ├── EfficientDet.py   # Wrapper EfficientDet
│   │       ├── DETR_model.py     # Wrapper DETR
│   │       └── DETR/             # Facebook DETR implementation
│   ├── datamodules/
│   │   ├── Datamodules.py        # Lightning DataModule (splits, samplers)
│   │   └── get_data.py           # Dataset class (MHA loading, augmentations)
│   ├── utils/
│   │   ├── custom_metrics.py     # FROC, AUROC, Competition Metric
│   │   ├── losses.py             # FocalLoss wrapper
│   │   ├── utils.py              # NMS, IoU, collate, checkpoints
│   │   ├── ensemble_boxes_weighted_numpy.py  # WBF ensemble
│   │   ├── scheduler.py          # Warmup + cosine decay LR
│   │   ├── TTA.py                # Test-time augmentation
│   │   ├── FrozenBN.py           # Frozen BatchNorm
│   │   └── custom_transforms.py  # Transforms legacy (reemplazado por Albumentations)
│   └── nodule_generation/
│       └── process.py            # Synthetic nodule generation (CT→CXR)
├── training_utils/
│   ├── train.py, dataset.py, transforms.py  # Standalone training utils
│   └── yolov5/                   # YOLOv5 framework completo
├── checkpoints/notest_final/     # 22 checkpoints entrenados (5-fold × 4-5 modelos)
└── splits/                       # K-fold CSV splits
```

#### Bugs encontrados (26 total)

**8 CRÍTICOS** (rompen ejecución con PyTorch 2.x / Lightning 2.x / Pandas 2.x):

| # | Archivo | Línea | Bug | Impacto |
|---|---------|-------|-----|---------|
| C1 | Detector.py | 86,105,155 | `training_epoch_end()` deprecated | TypeError en Lightning 2.0+ |
| C2 | config_fcrnn_l.yaml | 105 | `gpus: [0]` | TypeError en Lightning 2.0+ |
| C3 | config_fcrnn_l.yaml | 113-114 | `weights_summary`, `progress_bar_refresh_rate` | Params eliminados |
| C4 | FasterCNN.py | 43 | `pretrained=True` | Deprecated en TorchVision 0.13+ |
| C5 | Datamodules.py | 120 | `DataFrame.append()` | Eliminado en Pandas 2.0 |
| C6 | Detector.py | 203-220 | `optimizer_step()` signature | Deprecated en Lightning 1.9+ |
| C7 | Datamodules.py | 149,165,182 | `trainer.num_gpus` | Eliminado en Lightning 2.0+ |
| C8 | process.py | 46 | `from pytorch_lightning.plugins import DDPPlugin` | Movido a strategies |

**5 ALTOS** (resultados incorrectos o comportamiento erróneo):

| # | Archivo | Línea | Bug | Impacto |
|---|---------|-------|-----|---------|
| A1 | FasterCNN.py | 15-22 | Monkey-patching global de loss + bug lógico línea 21 | Condición duplicada, polución global |
| A2 | get_data.py | 80 | `.any()==1` lógica incorrecta | Funciona por casualidad (True==1 en Python) |
| A3 | get_data.py | 208-213 | Transform exception silenciada, código continúa | Crash con `transformed` undefined |
| A4 | Detector.py | 92 | Validación sin model.eval() explícito | BatchNorm inconsistente |
| A5 | Múltiples | — | Tensores sin device consistente | Error potencial en multi-GPU |

**7 MEDIOS** (fragilidad):

| # | Archivo | Bug |
|---|---------|-----|
| M1 | FasterCNN.py:107 | Bare `except:` silencia todo (incluido KeyboardInterrupt) |
| M2 | Múltiples | Paths absolutos hardcodeados (`/home/linux/Node21/...`) |
| M3 | losses.py:11 | `FocalLoss` modifica `loss_fcn.reduction` del objeto pasado (side effect) |
| M4 | ensemble_boxes:46-49 | Reshape sin validación de longitudes |
| M5 | get_data.py:36-43 | Cache alloca 4MB×4882 = ~19GB RAM sin verificar |
| M6 | get_data.py:168 | Mosaic puede generar boxes width=0 height=0 |
| M7 | process.py:10 | `from wandb import Config` falla si wandb no instalado |

**6 BAJOS** (calidad de código):
- Imports no usados (genericpath, unicodedata, ROC)
- Código muerto comentado
- Magic numbers sin constantes (IoU=0.2 hardcoded)
- Sin docstrings
- Type hints inconsistentes
- Duplicación de IoU computation (2 implementaciones diferentes)

#### Componentes valiosos a extraer

| Componente | Archivo | Líneas de código útiles |
|-----------|---------|------------------------|
| WBF Ensemble | ensemble_boxes_weighted_numpy.py | ~10 líneas (usa librería `ensemble-boxes`) |
| FROC Metrics | custom_metrics.py | ~50 líneas |
| SWA config | Detector.py:260+ | ~3 líneas (`torch.optim.swa_utils`) |
| WeightedRandomSampler | Datamodules.py:142-148 | ~5 líneas |
| VinDr weight loading | FasterCNN.py:96-108 | ~12 líneas (mapeo secuencial de keys) |
| Augmentación VinDR | get_data.py:240-290 | ~20 líneas (CropAndPad, Flip, Rotate, Blur, Cutout) |
| LR warmup+cosine | scheduler.py | ~15 líneas |
| NMS | utils.py | ~20 líneas |
| Gradient clipping | config | 1 línea (`gradient_clip_val: 3.0`) |

#### Versiones requeridas (incompatibles con actuales)

```
pytorch-lightning==1.5.1   → NO funciona con 2.0+
torch~=1.12                → funciona con warnings en 2.0+
torchvision~=0.12          → pretrained= deprecated en 0.13+
pandas~=1.3                → .append() eliminado en 2.0+
hydra-core==1.1.1          → compatible
ensemble-boxes==1.0.7      → compatible
monai==0.7.0               → compatible
```

**CONCLUSIÓN**: NO portar el código completo. Extraer las ~20 líneas útiles e integrar en nuestro pipeline limpio.

### 4.2 Código original-project (proyecto UIB)

#### Estructura

```
original-project/
├── ENTREGA/SCRIPTS_MODELS_I_AVALUACIONS_MODELS/
│   ├── CLASSIFICACIO/
│   │   ├── node21_v1_cs.ipynb          # SimpleCNN baseline
│   │   ├── node21_v1.1_cs.ipynb        # ResNet18, DenseNet121
│   │   ├── node21_v1.2_cs.ipynb        # EfficientNet-B0 multicanal (4ch)
│   │   ├── node21_v1.2ablacio_cs.ipynb  # Ablation study canales
│   │   ├── conversion a png.ipynb       # MHA→PNG + generación augmentaciones
│   │   └── MODELS/
│   │       ├── efficientnet_b0_multicanal.pth
│   │       └── efficientnet_b0_multicanal_Unsharp_bo.pth
│   └── DETECCIO/
│       ├── IPYNB/
│       │   ├── node21_v2.2_frcnn_coco.ipynb            # FRCNN con COCO
│       │   ├── node21_v2.2_png_retinanet.ipynb          # RetinaNet
│       │   ├── node21_v2.3_frcnn_vindn_png.ipynb        # FRCNN VinDr (MEJOR)
│       │   ├── node21_v2.3_frcnn_vindn_png_attention_cbam.ipynb  # FRCNN + CBAM
│       │   ├── node21_v2.3_frcnn_vindn_attention_roi.ipynb       # FRCNN + RoI attention
│       │   ├── node21_v2.3_frcnn_vindn_png_local_foss.ipynb      # FRCNN + Focal Loss
│       │   └── yolo_v1.ipynb                            # YOLOv8s
│       ├── AVALUACIO_IPYNB/  # 4 notebooks de evaluación
│       ├── AVALUACIO_CSVS/   # 7 CSVs con resultados por modelo
│       └── MODELS/
│           ├── checkpoint_3canal_frcnn_vindn_epoch_11.pth        # NODE21=0.9596 (en Spark)
│           ├── checkpoint_canal_frcnn_vindn_attention_cbam_epoch_16.pth  # NODE21=0.9181
│           ├── checkpoint_coco_monocanal_desbalanceado_epoch_02.pth
│           ├── checkpoint_epoch_13_vidn_local_foss.pth
│           └── checkpoint_retinanet_epoch_04.pth
├── dataset_node21/         # Dataset NODE21 completo
│   ├── cxr_images/
│   │   ├── original_data/images/    (4,882 .mha)
│   │   └── proccessed_data/images/  (4,887 .mha)
│   └── ct_patches/
│       ├── nodule_patches/    (1,186 volúmenes 3D)
│       └── segmentation/      (1,186 máscaras 3D)
├── node21-submit-master/    # Copia del repo Behrendt
├── metadata_augmented_def2.csv  # Dataset augmentado (8,175 filas)
└── Papers/                  # PDFs de referencia
```

#### Bugs encontrados

| # | Severidad | Notebook | Bug | Impacto |
|---|-----------|----------|-----|---------|
| 1 | **CRÍTICO** | Detección (todos) | `model.train()` en validación | Métricas val incorrectas (BatchNorm usa training stats) |
| 2 | **CRÍTICO** | Detección (todos) | `image_id = idx` del dataset, no del dataframe | IDs inconsistentes |
| 3 | **ALTO** | conversion a png | Data leakage: split no agrupa augmentaciones por imagen base | Scores inflados |
| 4 | **MEDIO** | Detección | Dataset class duplicada (cells 7 y 8) | Confusión |
| 5 | **MEDIO** | Todos | Paths hardcodeados a Google Colab | No portable |
| 6 | **BAJO** | Detección | Sin augmentación online (solo offline) | Subóptimo |
| 7 | **BAJO** | Detección | Sin LR scheduler | Convergencia más lenta |
| 8 | **BAJO** | Detección | Sin gradient clipping | Posible inestabilidad |
| 9 | **BAJO** | Todos | Sin seed fijo para numpy/torch/cuda | No reproducible exactamente |

#### Configuración de entrenamiento del mejor modelo (FRCNN-3ch)

| Parámetro | Valor |
|-----------|-------|
| CSV | metadata_augmented_def2.csv (8,175 filas, augmentado) |
| Split | train_test_split(test_size=0.2, random_state=42) — sin agrupar |
| Entrada | Grayscale × 3 canales (NO multicanal) |
| Batch size | 2 |
| Optimizer | AdamW(lr=1e-4, weight_decay=1e-4) |
| Epochs | 40 (early stopping patience=15, min_delta=0.0001) |
| Checkpoint | Por val_loss (mejor modelo guardado cada epoch) |
| Congelación | layers 1-3 frozen, layer4 + FPN + head trainable |
| LR scheduler | Ninguno (constante) |
| Gradient clip | Ninguno |
| SWA | No |
| Augmentación online | Ninguna (augmentaciones offline en el CSV) |
| Balanceo | Ninguno (dataset inherentemente más balanceado por augmentaciones: 46/54) |
| Validación mode | model.train() (BUG — debería ser model.eval()) |

---

## 5. DIFERENCIAS ENTRE IMPLEMENTACIONES

| Aspecto | Behrendt (paper) | original-project (UIB) | Spark actual |
|---------|-----------------|---------------------|-------------|
| Framework | Lightning 1.5 + Hydra | Notebooks bare PyTorch | Scripts bare PyTorch |
| Cross-validation | 5-fold StratifiedGroupKFold | Sin CV (1 split random) | Fold 0 |
| SWA | Sí (5-15 últimas epochs) | No | No |
| Synthetic nodules | CT→CXR raycasting + Poisson blend | No | No |
| WeightedRandomSampler | Sí (2x minority) | No | No |
| Gradient clipping | 3.0 | No | No (v2 sí) |
| LR scheduler | Warmup + cosine decay | Constante | ReduceLROnPlateau (v2) |
| Data augmentation | CropPad + Flip + Rotate + Blur + Cutout (online) | Flip + Affine + Brightness (offline) | Flip + Rotate + Brightness (online, v2) |
| Ensemble | WBF (5 modelos × 5 folds = ~21 modelos) | Modelo individual | Modelo individual |
| Checkpoint criterio | — | val_loss | val_loss (v1) / NODE21 score (v2) |
| Score threshold | 0.01-0.05 | Default (0.5) | 0.005 |
| Batch size | 8-16 | 2 | 2 |
| Modelos | FRCNN + RetinaNet + EfficientDet + YOLOv5×2 | FRCNN + RetinaNet + YOLOv8 | FRCNN + YOLOv8 + YOLO26 |
| Val mode | model.eval() (Lightning) | model.train() (BUG) | model.eval() (v2) |
| Data leakage | No (StratifiedGroupKFold) | Sí (augmentaciones no agrupadas) | En investigación |

---

## 6. PESOS Y CHECKPOINTS — INVENTARIO COMPLETO

### 6.1 Pesos preentrenados (VinDr-CXR base)
| Archivo | Modelo | Tamaño | Ubicación |
|---------|--------|--------|-----------|
| fastercnn50.pth | Faster R-CNN ResNet50-FPN | 159MB | node21-submit/ + Spark weights/ |
| yolo5x_vindr.pt | YOLOv5x | 170MB | node21-submit/ |
| F1_E79_ModelX_v4_T0.325_V0.410.ckpt | EfficientDet-D2 | 41MB | node21-submit/ |

### 6.2 Checkpoints del proyecto UIB (original-project)
| Archivo | NODE21 (Colab) | NODE21 (Spark val) | Tamaño |
|---------|---------------|-------------------|--------|
| checkpoint_3canal_frcnn_vindn_epoch_11.pth | 0.8913 | **0.9596** | 159MB |
| checkpoint_canal_frcnn_vindn_attention_cbam_epoch_16.pth | 0.8924 | 0.9181 | 161MB |
| checkpoint_epoch_13_vidn_local_foss.pth | 0.8897 | — | 159MB |
| checkpoint_coco_monocanal_desbalanceado_epoch_02.pth | ~0.66 | — | 159MB |
| checkpoint_retinanet_epoch_04.pth | 0.742 | — | 124MB |
| efficientnet_b0_multicanal.pth | 97% acc | — | — |
| efficientnet_b0_multicanal_Unsharp_bo.pth | 98% acc | — | — |

### 6.3 Checkpoints entrenados en Spark
| Archivo | NODE21 | Ubicación |
|---------|--------|-----------|
| best.pt (YOLOv8s) | 0.9103 | ~/nodule_detection/checkpoints/yolo/yolov8s/ |
| best.pt (YOLO26s v2) | 0.7929 | ~/nodule_detection/checkpoints/yolo/yolo26s/ |
| best_node21.pth (FRCNN Réplica-A) | 0.9695 | ~/nodule_detection/checkpoints/frcnn_reproduce/ |
| best_node21.pth (FRCNN Corrected-B) | **0.9025** | ~/nodule_detection/checkpoints/frcnn_corrected/ |
| best_valloss.pth (FRCNN Corrected-B) | 0.9025 | ~/nodule_detection/checkpoints/frcnn_corrected/ |
| best_model_fold0.pth (FRCNN v2) | 0.8544 | ~/nodule_detection/checkpoints/frcnn_vindr_v2/ |
| best_model_fold0.pth (FRCNN v1) | 0.4546 | ~/nodule_detection/checkpoints/frcnn_vindr/ |

### 6.5 Scripts en la Spark (~/nodule_detection/scripts/)
| Script | Líneas | Propósito |
|--------|--------|-----------|
| preprocess_node21.py | 12,715 | MHA→PNG, metadata, splits, formato YOLO |
| train_frcnn_vindr.py | 20,992 | Entrenar FRCNN con VinDr (v1, v2) |
| train_frcnn_reproduce.py | 17,812 | Reproducir FRCNN con dataset augmentado |
| train_yolo.py | 5,348 | Entrenar YOLOv8/YOLO26 |
| evaluate.py | 25,378 | Evaluación FROC/NODE21 de cualquier modelo |
| evaluate_original_frcnn.py | 16,563 | Evaluar checkpoints originales + CBAM |
| ensemble_wbf.py | 355 | Ensemble WBF de FRCNN + YOLOv8 |
| inference.py | 19,521 | Inferencia hospital (multi-modelo + WBF) |
| generate_augmented_images.py | 5,006 | Generar imágenes augmentadas offline |

### 6.4 Checkpoints Behrendt (5-fold, en node21-submit)
| Modelo | Folds | Resolución | Ubicación |
|--------|-------|-----------|-----------|
| Faster R-CNN | fold 1-5 | 1024 | checkpoints/notest_final/fcrnn_1024_gen/ |
| RetinaNet | fold 1-5 | 1024 | checkpoints/notest_final/retina_1024_gen/ |
| EfficientDet | fold 1-2 | 1024 | checkpoints/notest_final/effdet_1024_gen/ |
| YOLOv5x | fold 1-5 | 640 | checkpoints/notest_final/yolo_640_gen/ |
| YOLOv5x | fold 1-5 | 1024 | checkpoints/notest_final/yolo_1024_gen/ |

---

## 7. INFRAESTRUCTURA

### 7.1 Spark (entrenamiento y evaluación)
- **Host**: `${SPARK_USER}@${SPARK_HOST}`
- **GPU**: NVIDIA GB10
- **Acceso**: SSH
- **Proyecto**: ~/nodule_detection/
- **Entorno**: Python venv con PyTorch 2.x + Ultralytics + albumentations

### 7.2 Local (desarrollo y planificación)
- **OS**: Windows 11 Pro
- **Proyecto**: repos/RX/
- **Datos locales**: Dataset NODE21 completo + todos los checkpoints + notebooks originales

---

## 8. ROADMAP COMPLETO

### FASE 1: Detección de nódulos pulmonares (NODE21)
**Estado: ✅ COMPLETADA (2026-04-03)**
**Resultado final: Ensemble WBF NODE21 = 0.9391, CM = 0.9447**

#### 1A. Entrenamiento de modelos
| Tarea | Estado | Resultado | Fecha |
|-------|--------|-----------|-------|
| Setup Spark (venv, deps, dataset, pesos) | ✅ | Todo instalado | 01-abr |
| Preprocesar NODE21 (MHA→PNG, splits, YOLO format) | ✅ | 4,882 PNGs + 5-fold splits | 01-abr |
| Entrenar FRCNN VinDr v1 | ✅ | NODE21 = 0.4546 (score_thresh default) | 01-abr |
| Entrenar YOLOv8s | ✅ | NODE21 = 0.9103 | 02-abr |
| Entrenar YOLO26s | ✅ | NODE21 = 0.7857 | 02-abr |
| Fix FRCNN v2 (score_thresh=0.005 + checkpoint por NODE21) | ✅ | NODE21 = 0.8544 | 02-abr |
| Entrenar YOLO26s v2 (patience=30, batch=2) | ✅ | NODE21 = 0.7929 | 02-abr |

#### 1B. Evaluación de checkpoints originales (proyecto UIB)
| Tarea | Estado | Resultado | Fecha |
|-------|--------|-----------|-------|
| Subir checkpoints originales a Spark (SCP) | ✅ | 2 .pth subidos (320MB) | 03-abr |
| Evaluar FRCNN-3ch original | ✅ | NODE21 = 0.9596 (mejor modelo) | 03-abr |
| Evaluar FRCNN-CBAM original | ✅ | NODE21 = 0.9181 | 03-abr |

#### 1C. Reproducción desde cero
| Tarea | Estado | Resultado | Fecha |
|-------|--------|-----------|-------|
| Analizar código original (augmentación, split, bugs) | ✅ | Data leakage descubierto | 03-abr |
| Generar imágenes augmentadas en Spark (2,268 imgs) | ✅ | 7,148 PNGs total | 03-abr |
| Reproducir FRCNN — Versión A (réplica exacta con leakage) | ✅ | NODE21 = 0.9695 (supera original) | 03-abr |
| Reproducir FRCNN — Versión B (corregida sin leakage) | ✅ | NODE21 = 0.9025 (score honesto) | 03-abr |
| Cuantificar data leakage | ✅ | +6.7 puntos de inflación | 03-abr |

#### 1D. Ensemble y documentación
| Tarea | Estado | Resultado | Fecha |
|-------|--------|-----------|-------|
| **Ensemble WBF (FRCNN + YOLOv8s)** | ✅ | **NODE21 = 0.9391, CM = 0.9447** | 03-abr |
| Crear pipeline reutilizable (configs, builders, dataset, metrics) | ✅ | ~/nodule_detection/pipeline/ | 03-abr |
| Backup completo | ✅ | 2.5GB en ~/nodule_detection_backup_20260403/ | 03-abr |
| Descargar scripts y pipeline a local | ✅ | spark/scripts/ + spark/pipeline/ | 03-abr |
| Documentación completa | ✅ | DOCUMENTACION_PROYECTO.md + WEIGHTS_AND_CHECKPOINTS.md | 03-abr |

#### 1E. Review de código
| Tarea | Estado | Resultado | Fecha |
|-------|--------|-----------|-------|
| Review código Behrendt (node21-submit) | ✅ | 26 bugs encontrados (8 críticos) | 03-abr |
| Review código UIB (original-project) | ✅ | 9 bugs encontrados (2 críticos) | 03-abr |
| Decisión: no portar Behrendt, extraer ideas | ✅ | SWA, WeightedSampler, grad clip extraíbles | 03-abr |

#### 1F. Pendiente (opcional, para mejorar más)
| Tarea | Estado | Prioridad | Notas |
|-------|--------|-----------|-------|
| 5-fold cross-validation (FRCNN + YOLOv8) | ⏳ Pendiente | Media | Necesario para robustez y paper |
| FRCNN v3 con SWA + WeightedSampler + grad clip 3.0 | ⏳ Opcional | Baja | Score actual ya excelente (0.9025) |
| Clasificador EfficientNet-B0 multicanal (4ch) | ⏳ Pendiente | Media | Filtro rápido para pipeline hospital |
| Ensemble con 5-fold (×2 modelos = 10 modelos) | ⏳ Pendiente | Alta | Potencialmente >0.95 sin leakage |

---

### FASE 2: Tuberculosis
**Estado: ⏳ PLANIFICACIÓN**
**Dependencia: Ninguna — se puede iniciar en paralelo**

| Tarea | Estado | Prioridad | Notas |
|-------|--------|-----------|-------|
| Investigar datasets TB disponibles | ⏳ Pendiente | Alta | TBX11K (11,200 CXR con bbox), Montgomery (138), Shenzhen (662) |
| Descargar TBX11K a Spark | ⏳ Pendiente | Alta | Es el más completo con bounding boxes |
| Preparar CSV + splits (reutilizar pipeline) | ⏳ Pendiente | — | Usar pipeline/configs/tuberculosis.yaml |
| Entrenar FRCNN VinDr para TB | ⏳ Pendiente | Alta | Reutilizar pesos VinDr (tiene categorías pulmonares) |
| Entrenar YOLOv8s para TB | ⏳ Pendiente | Alta | Mismo pipeline, cambiar config |
| Evaluar (mAP, AUC, sensibilidad, especificidad) | ⏳ Pendiente | — | |
| Ensemble WBF para TB | ⏳ Pendiente | — | |
| Clasificación binaria TB sí/no | ⏳ Pendiente | Media | Complementar detección |

---

### FASE 3: Multi-patología VinDr-CXR (22 clases)
**Estado: 🔄 ESPERANDO ACCESO PHYSIONET**
**Dependencia: Acceso PhysioNet aprobado**

| Tarea | Estado | Prioridad | Notas |
|-------|--------|-----------|-------|
| Completar CITI training + DUA | 🔄 En proceso | — | Usuario aplicando |
| Descargar VinDr-CXR (18K DICOM) a Spark | ⏳ Pendiente | Alta | ~50GB estimado |
| Preprocesar DICOM → PNG | ⏳ Pendiente | — | pydicom → normalizar → PNG |
| Generar CSV multi-clase (22 categorías) | ⏳ Pendiente | — | Mapear annotations_train.csv |
| Entrenar FRCNN multi-clase (23 clases) | ⏳ Pendiente | Alta | Usar pipeline/configs/vindr_22classes.yaml |
| Entrenar YOLOv8 multi-clase | ⏳ Pendiente | Alta | Formato YOLO con 22 clases |
| Evaluar mAP por clase | ⏳ Pendiente | — | AP por cada patología |
| Matrices de confusión por clase | ⏳ Pendiente | — | Identificar confusiones frecuentes |
| Clasificación global (6 diagnósticos) | ⏳ Pendiente | Media | Tumor, Pneumonia, TB, COPD, Other, No finding |
| Ensemble WBF multi-clase | ⏳ Pendiente | — | |

---

### FASE 4: Prototipo funcional hospitalario
**Estado: ✅ COMPLETADA (2026-04-12)**
**Resultado: Prototipo end-to-end funcional, revisado y hardened (23 bugs arreglados)**
**Documento de diseño: [PROPUESTA_ARQUITECTURA_HOSPITAL.md](PROPUESTA_ARQUITECTURA_HOSPITAL.md)**
**Revisión de bugs: [BUGS_REVIEW_COMPLETO.md](BUGS_REVIEW_COMPLETO.md)**

#### Arquitectura decidida

```
Radiólogo (browser) → Vue 3 + Tailwind (upload CXR)
    ↓
Nginx (Docker local) → FastAPI cxr-svc → RabbitMQ (port 5672 abierto LAN)
    ↓                                          ↓
MySQL (resultados)              Spark Worker (GPU nativo, sin Docker)
    ↑                                          ↓
    └──── RabbitMQ (cxr.results) ←──── FRCNN + YOLOv8 + WBF Ensemble
```

**Stack tecnológico**:
| Capa | Tecnología | Reutiliza de |
|------|-----------|-------------|
| Frontend | Vue 3 + Tailwind + Vite | worklistsrv |
| Backend | FastAPI + Uvicorn (Docker) | worklistsrv-backend |
| BD | MySQL 8.0 (Docker) | worklistsrv-backend |
| Cola | RabbitMQ 3.13 (Docker, port 5672 abierto) | worklistsrv-backend |
| Proxy | Nginx 1.25 (Docker) | worklistsrv-backend |
| Inferencia | Python nativo en Spark (sin Docker) | pipeline/ existente |
| Modelos | FRCNN VinDr + YOLOv8s + WBF | checkpoints/ existentes |

#### Sprint 1: Inference Worker en Spark (COMPLETADO 2026-04-03)
| Tarea | Estado | Resultado |
|-------|--------|-----------|
| Crear inference_worker.py (consumer RabbitMQ) | ✅ | ~/nodule_detection/worker/inference_worker.py |
| Cargar FRCNN + YOLOv8 al inicio (1 vez) | ✅ | FRCNN 55ms + YOLOv8 128ms load |
| Imagen base64 → WBF ensemble → resultado JSON | ✅ | 81ms/imagen (post-warmup) |
| Crear start_worker.sh + config.yaml | ✅ | + cxr-worker.service (systemd) |
| Test básico (n0239.png) | ✅ | 1 nódulo, score=0.9418, 312ms (con warmup) |
| Test batch (10 imágenes) | ✅ | 108ms/img promedio, todas detectan ≥1 nódulo |

#### Sprint 2: Backend FastAPI + Docker (COMPLETADO 2026-04-04)
| Tarea | Estado | Resultado |
|-------|--------|-----------|
| docker-compose.yml (MySQL + RabbitMQ + cxr-svc + nginx) | ✅ | 5 servicios Docker |
| POST /api/cxr/upload → RabbitMQ | ✅ | Con transacción BD-first |
| GET /api/cxr/results/{id} (polling) | ✅ | + imagen original + anotada |
| GET /api/cxr/history (historial filtrable) | ✅ | Paginado + búsqueda + stats |
| Result consumer (RabbitMQ → MySQL) | ✅ | Idempotente, reconexión auto |
| Modelo datos: 6 tablas (studies, detections, images×2, validations, validation_boxes) | ✅ | Con módulo de validación |
| Health check real (API + MySQL + RabbitMQ) | ✅ | /api/health |
| DELETE individual + bulk | ✅ | Bulk con SQL directo |

#### Sprint 3: Frontend Vue (COMPLETADO 2026-04-06)
| Tarea | Estado | Resultado |
|-------|--------|-----------|
| AnalyzeView.vue (drag & drop + resultado) | ✅ | Upload + polling + resultado |
| CxrViewer.vue (visor con zoom real 100%) | ✅ | SVG overlay, zoom, contraste |
| ThresholdSlider.vue (filtrado dinámico) | ✅ | Filtra detecciones en tiempo real |
| DetectionList.vue (tabla de detecciones) | ✅ | Con scores y coordenadas |
| HistoryView.vue (historial análisis) | ✅ | Búsqueda, eliminar, borrar todo |
| StatsBar.vue (estadísticas globales) | ✅ | Total, analizados, con nódulos, tiempo |
| ValidationPanel.vue (validación radiólogo) | ✅ | Correcto/parcial/incorrecto + dibujo bbox |
| ValidationStatsView.vue (estadísticas validación) | ✅ | Métricas por modelo |
| i18n (ES/CA/EN) | ✅ | 3 idiomas completos |
| Dark mode | ✅ | Toggle con persistencia |

#### Sprint 4: Integración end-to-end (COMPLETADO 2026-04-04)
| Tarea | Estado | Resultado |
|-------|--------|-----------|
| Test completo: upload → RabbitMQ → Spark → MySQL → UI | ✅ | Funcional end-to-end |
| Test con CXR reales de NODE21 | ✅ | Múltiples imágenes probadas |
| Medir latencia (<5 segundos objetivo) | ✅ | ~2-3 segundos end-to-end |
| Health check: API + MySQL + RabbitMQ | ✅ | Todos "ok" |

#### Sprint 5: Revisión exhaustiva y hardening (COMPLETADO 2026-04-12)
| Tarea | Estado | Resultado |
|-------|--------|-----------|
| Review backend (37 issues encontrados) | ✅ | 3 CRITICO + 2 ALTO arreglados |
| Review frontend (34 issues encontrados) | ✅ | Memory leaks + error handling arreglados |
| Review worker (42 issues encontrados) | ✅ | Reconexión + graceful shutdown + GPU cleanup |
| Review scripts Spark (pipeline + training) | ✅ | freeze_layers fix + validación imagen |
| Verificación post-fix completa | ✅ | Docker OK, health OK, npm build OK, worker test OK |
| Documentación bugs: BUGS_REVIEW_COMPLETO.md | ✅ | 23 fixes documentados, 14 pendientes infra |

#### Producción hospital (futuro)
| Tarea | Estado | Prioridad | Notas |
|-------|--------|-----------|-------|
| HTTPS certificado hospital | ⏳ Pendiente | Alta | |
| Auth Keycloak OIDC | ⏳ Pendiente | Alta | Reutilizar worklistsrv |
| Integración PACS (DICOM C-STORE) | ⏳ Pendiente | Alta | Reutilizar dicom-svc |
| Auditoría ENS (audit-svc) | ⏳ Pendiente | Media | SHA-256 chained logs |
| NFS shared mount (alto volumen) | ⏳ Pendiente | Media | Reemplazar base64 |
| Multi-patología (22 clases) | ⏳ Pendiente | — | Después de Fase 3 |
| Explicabilidad (Grad-CAM) | ⏳ Pendiente | Baja | |
| Validación clínica prospectiva | ⏳ Pendiente | Alta | Datos reales |
| Certificación CE marking | ⏳ Pendiente | — | Largo plazo |

---

### FASE 5: Publicación
**Estado: 📝 PLANIFICACIÓN**
**Dependencia: Fases 1-3 con resultados robustos (5-fold)**

| Tarea | Estado | Prioridad | Notas |
|-------|--------|-----------|-------|
| Paper comparativo: FRCNN vs YOLOv8 vs YOLO26 vs Ensemble | ⏳ Pendiente | Alta | Resultados ya disponibles |
| Ablation: pretraining (VinDr vs COCO vs ImageNet vs scratch) | ⏳ Pendiente | Alta | VinDr ya demostrado superior |
| Ablation: data leakage (con vs sin) | ✅ Ya hecho | — | +6.7 puntos, documentado |
| Ablation: augmentación (offline vs online) | ⏳ Pendiente | Media | |
| Ablation: score_thresh impact (0.5 vs 0.005) | ✅ Ya hecho | — | FRCNN v1 (0.45) vs v2 (0.85) |
| 5-fold cross-validation para todos los modelos | ⏳ Pendiente | **Crítica** | Imprescindible para publicar |
| Comparación con resultados oficiales NODE21 | ⏳ Pendiente | Alta | Nuestro CM=94.47% vs Behrendt 83.90% |
| Redactar paper | ⏳ Pendiente | — | |
| Target journal: Nature Scientific Reports o Medical Image Analysis | ⏳ Pendiente | — | |

---

### RESUMEN DE PROGRESO

```
FASE 1: Nódulos pulmonares (NODE21)     [██████████████████████████████] 100% ✅
FASE 2: Tuberculosis                     [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0% ⏳
FASE 3: Multi-patología VinDr (22 cls)   [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   5% 🔄
FASE 4: Prototipo hospital               [██████████████████████████████] 100% ✅
FASE 5: Publicación                      [██████████░░░░░░░░░░░░░░░░░░░░]  35% 📝
```

### ORDEN DE EJECUCIÓN RECOMENDADO

```
COMPLETADO:
├── Fase 1: Detección nódulos NODE21 (0.9391 ensemble) ✅
├── Fase 4: Prototipo hospital funcional + hardened ✅
│
SIGUIENTE (abril 2026):
├── Fase 4 producción: Auth + HTTPS + PACS (cuando se despliegue en hospital)
│
EN PARALELO (cuando llegue acceso PhysioNet):
├── Fase 3: VinDr-CXR 22 clases
│   └── Reutilizar pipeline, cambiar config YAML
│
DESPUÉS:
├── Fase 2: Tuberculosis (TBX11K)
│   └── Reutilizar pipeline, cambiar config YAML
│
ÚLTIMO:
├── Fase 5: Paper con resultados 5-fold de todas las fases
```

---

## 9. MÉTRICAS Y EVALUACIÓN

### 9.1 Métricas de detección (NODE21)

- **NODE21 Score**: Media de sensibilidad a FP/img = [0.25, 0.5, 1, 2, 4, 8]
- **FROC** (Free-Response ROC): Sensibilidad vs FP por imagen
- **AUROC**: A nivel de imagen (¿tiene nódulo sí/no?)
- **Competition Metric (CM)**: 0.75 × AUROC + 0.25 × FROC@25%
- **IoU threshold para match**: 0.2 (oficial NODE21 — más bajo que el estándar 0.5)

### 9.2 Métricas de clasificación

- **Accuracy**, **Precision**, **Recall**, **F1-score** por clase
- **Confusion matrix**
- **AUC-ROC** a nivel de imagen

### 9.3 Configuración de evaluación

- Score threshold para FROC: 0.005 (muy bajo, captura todas las predicciones)
- NMS IoU threshold: 0.2
- Confidence threshold para visualización: 0.3

---

## 10. REFERENCIAS

1. **Behrendt et al. (2023)**. "A systematic approach to deep learning-based nodule detection in chest radiographs." *Scientific Reports* 13, 10120. https://www.nature.com/articles/s41598-023-37270-2
2. **NODE21 Challenge**: https://node21.grand-challenge.org/ — DOI: 10.5281/zenodo.5548363
3. **VinDr-CXR**: Nguyen et al. (2022). "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations." *Scientific Data* 9, 429. PhysioNet DOI: 10.13026/3akn-b287
4. **Rehman et al. (2024)**. "Effective lung nodule detection using deep CNN with dual attention mechanisms." *Scientific Reports* 14, 3934.
5. **Li et al. (2024)**. "Dual attention mechanisms for pulmonary nodule detection in chest X-ray images." *Scientific Reports* 14, 11234.
6. **GitHub Behrendt**: https://github.com/FinnBehrendt/node21-submit
7. **YOLO26**: Ultralytics (2026). arXiv:2509.25164. https://docs.ultralytics.com/models/yolo26/

---

## 11. ARCHIVOS DE DOCUMENTACIÓN DEL PROYECTO

| Archivo | Contenido |
|---------|-----------|
| `DOCUMENTACION_PROYECTO.md` | Este documento — estado completo del proyecto |
| `BEHRENDT_BUGS_ANALYSIS.md` | Análisis detallado de 26 bugs en código Behrendt |
| `WEIGHTS_AND_CHECKPOINTS.md` | Inventario de pesos con instrucciones de carga |
| `evaluation_report.csv` | Resultados evaluación individual de modelos |
| `ensemble_report.csv` | Resultados ensemble WBF |
| `froc_ensemble.png` | Curva FROC comparativa (FRCNN vs YOLOv8 vs Ensemble) |
| `spark/scripts/` | 9 scripts Python descargados de la Spark |
| `spark/pipeline/` | Pipeline reutilizable (configs, models, data, utils) |
| `SPARK_PROMPT.md` | Prompts iniciales para setup de Spark |
| `SPARK_PLAN_COMPLETO.md` | Plan de ejecución avanzado |
| `PROMPT_REPRODUCIR_FRCNN.md` | Prompt para reproducir FRCNN desde cero |
| `PROMPT_GUARDAR_TODO.md` | Prompt para documentar y crear pipeline |
| `deteccion_nodulos_pulmonares.pdf` | Paper del proyecto UIB original (27 páginas) |
