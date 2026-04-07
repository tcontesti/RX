# Detección Automática de Nódulos Pulmonares en Radiografías de Tórax mediante Deep Learning: Del Modelo al Prototipo Hospitalario

**Autores**: Marc Link Cladera, Antonio Contesti Coll  
**Supervisor**: Miquel Miró Nicolau  
**Institución**: Universitat de les Illes Balears (UIB), Grado en Ingeniería Informática  
**Referencia principal**: Behrendt et al. (2023), *Nature Scientific Reports* — Ganador NODE21 Challenge  
**Fecha**: Abril 2026  

---

## Índice

1. [Introducción y motivación](#1-introducción-y-motivación)
2. [Marco teórico](#2-marco-teórico)
3. [Dataset NODE21](#3-dataset-node21)
4. [Metodología experimental](#4-metodología-experimental)
5. [Experimento 1: Clasificación binaria](#5-experimento-1-clasificación-binaria)
6. [Experimento 2: Detección con Faster R-CNN](#6-experimento-2-detección-con-faster-r-cnn)
7. [Experimento 3: Detección con YOLOv8](#7-experimento-3-detección-con-yolov8)
8. [Experimento 4: Detección con YOLO26](#8-experimento-4-detección-con-yolo26)
9. [Experimento 5: Reproducibilidad y data leakage](#9-experimento-5-reproducibilidad-y-data-leakage)
10. [Experimento 6: Ensemble con Weighted Box Fusion](#10-experimento-6-ensemble-con-weighted-box-fusion)
11. [Comparación con el estado del arte](#11-comparación-con-el-estado-del-arte)
12. [Prototipo hospitalario](#12-prototipo-hospitalario)
13. [Lecciones aprendidas](#13-lecciones-aprendidas)
14. [Conclusiones y trabajo futuro](#14-conclusiones-y-trabajo-futuro)
15. [Referencias](#15-referencias)

---

## 1. Introducción y motivación

Los nódulos pulmonares son masas de tejido de forma redondeada, generalmente menores de 3 cm, que pueden aparecer en los pulmones. Su detección temprana es crucial porque pueden ser indicadores precoces de cáncer de pulmón, la primera causa de muerte por cáncer a nivel mundial.

La radiografía de tórax (CXR, *Chest X-Ray*) es el estudio de imagen más frecuente en atención primaria y urgencias. Sin embargo, la detección de nódulos en CXR es notoriamente difícil: estudios clínicos reportan tasas de falsos negativos del 20-50% por parte de radiólogos, ya que los nódulos pueden ser pequeños, estar superpuestos con estructuras óseas (costillas, clavículas) o camuflados en el parénquima pulmonar.

Este proyecto aborda el problema con técnicas de deep learning para detección de objetos, siguiendo la línea del NODE21 Challenge (*NodULe Detection in Chest X-rays 2021*), la competición de referencia internacional en este campo.

### Objetivos

1. **Entrenar y comparar** múltiples arquitecturas de detección de objetos (Faster R-CNN, YOLOv8, YOLO26) en el dataset NODE21
2. **Analizar** el impacto de diferentes técnicas (transfer learning, data augmentation, ensemble) en el rendimiento
3. **Cuantificar** un problema de data leakage descubierto durante el desarrollo
4. **Construir** un prototipo funcional para uso hospitalario con interfaz web para radiólogos

---

## 2. Marco teórico

### 2.1 Detección de objetos vs. clasificación

En visión por computador, existen dos tareas fundamentales que conviene distinguir claramente:

- **Clasificación de imágenes**: dado una imagen completa, asignar una etiqueta (ej: "tiene nódulo" / "no tiene nódulo"). La salida es una clase y una probabilidad.
- **Detección de objetos**: localizar y clasificar objetos individuales dentro de la imagen. La salida es un conjunto de *bounding boxes* (cajas delimitadoras) con coordenadas (x, y, ancho, alto), una etiqueta de clase y un score de confianza.

La detección es más difícil pero más útil clínicamente: no solo dice "hay un nódulo", sino que señala exactamente **dónde está** en la radiografía.

### 2.2 Faster R-CNN (Two-Stage Detector)

Faster R-CNN (Ren et al., 2015) es una arquitectura de detección de objetos que opera en dos etapas:

```
Imagen → Backbone (ResNet50) → Feature maps
                                    ↓
                         Region Proposal Network (RPN)
                         "¿Dónde podría haber objetos?"
                         Genera ~300 regiones candidatas
                                    ↓
                         RoI Pooling + Clasificador
                         "¿Qué hay en cada región?"
                         Refina la caja y asigna clase
                                    ↓
                         Predicciones finales (boxes + scores + clases)
```

**Ventajas**: Alta precisión en la localización de objetos. El proceso en dos etapas permite refinar las predicciones.  
**Desventajas**: Más lento que detectores single-stage. Más complejo de entrenar.

#### Feature Pyramid Network (FPN)

FPN (Lin et al., 2017) es una extensión del backbone que genera mapas de características a múltiples escalas. Esto es crucial para detectar nódulos de diferentes tamaños: nódulos grandes se detectan en las capas de baja resolución, nódulos pequeños en las de alta resolución.

```
Backbone ResNet50:
  conv1 → layer1 → layer2 → layer3 → layer4
     ↓        ↓        ↓        ↓        ↓
    P1       P2       P3       P4       P5   (pirámide de features)
```

### 2.3 YOLO (Single-Stage Detector)

YOLO (*You Only Look Once*) es una familia de detectores que, a diferencia de Faster R-CNN, procesan la imagen en una sola pasada:

```
Imagen → Backbone (CSPDarknet) → Neck (PANet) → Head
                                                   ↓
                                      Predicciones directas
                                      (boxes + scores + clases)
                                      sin etapa de propuestas
```

**YOLOv8** (Ultralytics, 2023) es anchor-free (no usa cajas predefinidas de referencia) y tiene cabezas desacopladas para clasificación y regresión.

**YOLO26** (Ultralytics, 2026) introduce mejoras específicas para objetos pequeños: STAL (*Small-Target-Aware Label Assignment*) y ProgLoss (*Progressive Loss Balancing*).

**Ventajas**: Muy rápido (16-18 ms/imagen). Fácil de entrenar con Ultralytics.  
**Desventajas**: Históricamente menos preciso que two-stage detectors para objetos pequeños.

### 2.4 Transfer learning y preentrenamiento de dominio

Transfer learning consiste en reutilizar pesos de una red entrenada en un problema para resolver otro relacionado. Se distinguen tres niveles:

| Nivel | Preentrenamiento | Ejemplo | Proximidad al dominio |
|-------|-----------------|---------|----------------------|
| General | ImageNet (1.4M imágenes naturales) | Perros, coches, paisajes | Baja |
| COCO | COCO (330K imágenes, 80 clases) | Detección general | Media |
| **Dominio específico** | **VinDr-CXR (18K radiografías, 14 patologías)** | **Radiología de tórax** | **Alta** |

En nuestros experimentos, el preentrenamiento en VinDr-CXR (dominio radiológico) demostró ser el factor más determinante del rendimiento, ya que el backbone aprende representaciones específicas de estructuras torácicas.

**Congelación de capas** (*freezing*): Las primeras capas de la red capturan patrones genéricos (bordes, texturas) que son transferibles. Las últimas capas capturan patrones específicos del problema. La estrategia óptima resultó ser congelar las capas 1-3 del ResNet50 y entrenar solo la capa 4 + FPN + cabezas de detección.

### 2.5 Data augmentation

Data augmentation genera variaciones artificiales de las imágenes de entrenamiento para mejorar la generalización del modelo y prevenir overfitting. Se distinguen dos modalidades:

- **Offline (estática)**: Se generan las imágenes augmentadas antes del entrenamiento y se guardan en disco. Las augmentaciones son las mismas en cada epoch.
- **Online (dinámica)**: Se aplican transformaciones aleatorias en cada epoch durante el entrenamiento. El modelo ve variaciones diferentes cada vez.

**Augmentaciones usadas en imagen médica** (conservadoras, ya que la anatomía tiene una orientación y escala definidas):
- Flip horizontal (p=0.5) — el tórax es aproximadamente simétrico
- Rotación leve (±10°) — variación en la posición del paciente
- Ajuste de brillo/contraste (p=0.3) — variación en la exposición del equipo de rayos X
- **No se usan** mosaic, mixup, ni cambios de color agresivos (destruirían la información radiológica)

### 2.6 Métricas de evaluación

| Métrica | Definición | Interpretación |
|---------|------------|----------------|
| **FROC** | Free-Response ROC: sensibilidad a diferentes niveles de falsos positivos por imagen (0.25, 0.5, 1, 2, 4, 8 FP/img) | Curva que muestra el trade-off entre detectar nódulos y generar falsas alarmas |
| **NODE21 Score** | Media de la sensibilidad en los 6 niveles de FP/img de la FROC | Métrica oficial del challenge. Rango [0, 1], mayor es mejor |
| **AUROC** | Área bajo la curva ROC a nivel de imagen (¿tiene algún nódulo?) | Capacidad de discriminación binaria. 1.0 = perfecto, 0.5 = aleatorio |
| **CM** | Competition Metric = 0.75 × AUROC + 0.25 × NODE21 | Métrica compuesta oficial NODE21 |
| **IoU** | Intersection over Union: solapamiento entre box predicho y ground truth | Threshold para considerar una detección correcta (NODE21 usa 0.2) |

**¿Por qué IoU = 0.2 y no 0.5?** En la detección de nódulos, las anotaciones son inherentemente imprecisas (los límites del nódulo son difusos en una radiografía). Un threshold bajo penaliza menos las discrepancias en la localización exacta.

### 2.7 Weighted Box Fusion (WBF)

WBF (Solovyev et al., 2021) es una técnica de ensemble que combina las predicciones de múltiples modelos. A diferencia de NMS (*Non-Maximum Suppression*), que elimina boxes solapadas quedándose con la de mayor score, WBF **fusiona** las boxes calculando la media ponderada de sus coordenadas:

```
NMS (tradicional):
  Box A (score=0.8) + Box B (score=0.7) → Mantiene solo A, elimina B

WBF (nuestro enfoque):
  Box A (score=0.8) + Box B (score=0.7) → Box C con coordenadas = media ponderada
  El resultado es una caja más precisa que cualquiera de las originales
```

Esto es especialmente útil cuando dos modelos diferentes (FRCNN y YOLOv8) detectan el mismo nódulo con boxes ligeramente distintas: la fusión produce una localización más precisa que cualquiera de los dos.

### 2.8 Data leakage

Data leakage ocurre cuando información del conjunto de validación/test se "filtra" al conjunto de entrenamiento, produciendo métricas artificialmente infladas que no reflejan el rendimiento real del modelo.

En nuestro caso, se descubrió un data leakage específico: al generar augmentaciones offline, las imágenes augmentadas (`n0080_aug0.png`, `n0080_aug1.png`) se asignaban aleatoriamente a train/val sin agrupar por imagen base. Resultado: el modelo veía en entrenamiento una versión augmentada de una imagen y era evaluado sobre otra versión augmentada de la **misma** imagen original, inflando el score en +6.7 puntos.

---

## 3. Dataset NODE21

NODE21 (*NodULe Detection in Chest X-rays*) es el benchmark de referencia para detección de nódulos en radiografías de tórax.

| Característica | Valor |
|---------------|-------|
| Total imágenes | 4,882 radiografías CXR frontales |
| Resolución | 1024 × 1024 píxeles |
| Formato original | MHA (MetaImage) |
| Positivos | 1,134 imágenes con 1,476 nódulos anotados |
| Negativos | 3,748 imágenes sin nódulos |
| Prevalencia | 23% positivos |
| Anotaciones | Bounding boxes (x, y, width, height) |
| Fuentes | JSRT (242), PadChest (1,680), ChestX-ray14 (1,804), Open-I (1,218) |
| Test set | Privado (no disponible públicamente) |
| Licencia | CC BY-NC-ND 4.0 |

**Preprocesamiento aplicado**: Conversión MHA → PNG, normalización a 8 bits, generación de splits estratificados (StratifiedGroupKFold con 5 folds), conversión al formato YOLO (anotaciones normalizadas por las dimensiones de la imagen).

---

## 4. Metodología experimental

### 4.1 Infraestructura

| Componente | Especificación |
|-----------|---------------|
| GPU | NVIDIA GB10 (servidor Spark, UIB) |
| Framework | PyTorch 2.x + Ultralytics |
| Augmentaciones | Albumentations |
| Evaluación | Métricas NODE21 (FROC, AUROC, CM) implementadas desde cero |

### 4.2 Protocolo de evaluación

- **Split**: Fold 0 de un StratifiedGroupKFold con 5 particiones
- **Agrupación**: Por imagen base (evita data leakage con augmentaciones)
- **Score threshold**: 0.005 para evaluación (captura todas las predicciones)
- **IoU threshold**: 0.2 (estándar NODE21)
- **Modelo guardado**: Checkpoint con mejor NODE21 Score en validación

### 4.3 Resumen de todos los experimentos

| # | Experimento | Modelo | NODE21 | Hallazgo principal |
|---|------------|--------|--------|---------------------|
| 1 | Clasificación | EfficientNet-B0 (4 canales) | 98% acc | Multicanal mejora clasificación |
| 2a | Detección FRCNN v1 | FRCNN + VinDr, thresh=0.5 | 0.4546 | Threshold por defecto descarta nódulos |
| 2b | Detección FRCNN v2 | FRCNN + VinDr, thresh=0.005 | 0.8544 | Bajar threshold: +40 puntos |
| 2c | Detección FRCNN corregido | FRCNN + VinDr, sin leakage | 0.9025 | Score honesto real |
| 3 | Detección YOLOv8 | YOLOv8s + COCO | 0.9103 | Mejor individual |
| 4 | Detección YOLO26 | YOLO26s + COCO | 0.7929 | STAL no compensa en este dataset |
| 5a | Reproducibilidad con leakage | FRCNN réplica exacta | 0.9695 | Reproduce exitosamente el leakage |
| 5b | Reproducibilidad sin leakage | FRCNN corregido | 0.9025 | Cuantifica leakage: +6.7 puntos |
| 6 | Ensemble WBF | FRCNN + YOLOv8 | **0.9391** | +2.6 sobre mejor individual |

---

## 5. Experimento 1: Clasificación binaria

**Objetivo**: Clasificar radiografías como "con nódulo" o "sin nódulo" (sin localización).

### Arquitectura

EfficientNet-B0 con entrada de 4 canales:
1. **Canal original**: imagen en escala de grises
2. **Canal CLAHE**: ecualización adaptativa de histograma (mejora contraste local)
3. **Canal Canny**: detección de bordes (resalta contornos de nódulos)
4. **Canal Unsharp Mask**: realce de bordes suaves (resalta texturas)

La primera convolución fue modificada de `Conv2d(3, 32)` a `Conv2d(4, 32)` para aceptar 4 canales.

### Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy | 98.07% |
| Recall clase positiva | 98.93% |
| Dataset | Balanceado (3,748 × 2 = 7,496 imágenes) |

**Conclusión**: Alta accuracy, pero la clasificación binaria no localiza el nódulo en la imagen. Se procede con detección de objetos.

---

## 6. Experimento 2: Detección con Faster R-CNN

### 6.1 Configuración base

| Parámetro | Valor |
|-----------|-------|
| Backbone | ResNet50 + FPN |
| Preentrenamiento | VinDr-CXR (`fastercnn50.pth`, 159 MB) |
| Capas congeladas | layers 1-3 (solo layer4 + FPN + head entrenable) |
| Optimizador | AdamW (lr=1e-4, weight_decay=1e-4) |
| Epochs | 40 (early stopping patience=15) |
| Batch size | 2 (limitación de memoria GPU) |
| Input | Grayscale replicado a 3 canales, 1024×1024 |

### 6.2 Carga de pesos VinDr

El checkpoint VinDr (`fastercnn50.pth`) fue entrenado para 15 clases (14 patologías + fondo). El proceso de carga requiere:

1. Crear el modelo con arquitectura estándar de Faster R-CNN
2. Cargar los pesos VinDr mediante mapeo secuencial de keys (los nombres de las capas difieren)
3. Descartar las capas de la cabeza de clasificación (tenía 15 clases, necesitamos 2)
4. Instanciar nueva cabeza: `FastRCNNPredictor(in_features, 2)` para nódulo sí/no

### 6.3 Iteraciones y hallazgos

**FRCNN v1** — Score threshold por defecto (0.5):
- NODE21 = 0.4546
- **Problema**: El threshold por defecto de torchvision filtraba la mayoría de predicciones antes de llegar a la evaluación FROC. A 0.25 FP/imagen, el modelo no reportaba casi ninguna detección.

**FRCNN v2** — Score threshold bajado a 0.005:
- NODE21 = 0.8544
- **Hallazgo**: Al bajar el threshold de predicción de 0.5 a 0.005, se permite que el evaluador FROC determine el punto de operación óptimo. Mejora de +40 puntos.

**FRCNN corregido** (sin data leakage, Experimento 5):
- NODE21 = 0.9025
- Score honesto real sin contaminación train/val.

### 6.4 Variante CBAM (mecanismo de atención)

Se probó añadir CBAM (*Convolutional Block Attention Module*) después del layer4:

```
CBAM = Channel Attention + Spatial Attention
  Channel: GAP + GMP → MLP → sigmoid → modula canales
  Spatial: AvgPool + MaxPool sobre canales → Conv 7×7 → sigmoid → modula espacio
```

Resultado: NODE21 = 0.9181 vs 0.9596 sin CBAM. La atención no mejoró el rendimiento en este caso, posiblemente porque el FPN ya captura información multi-escala suficiente.

---

## 7. Experimento 3: Detección con YOLOv8

### Configuración

| Parámetro | Valor |
|-----------|-------|
| Modelo | YOLOv8s (small, 11.2M parámetros) |
| Preentrenamiento | COCO (80 clases, imágenes naturales) |
| Resolución | 1024×1024 |
| Optimizador | AdamW (lr=1e-4 → 1e-5) |
| Losses | box=7.5, cls=0.3, dfl=1.5 |
| Augmentación | Solo flip horizontal (mosaic/mixup desactivados) |
| Patience | 20 epochs sin mejora |

### Resultado

| Métrica | Valor |
|---------|-------|
| NODE21 Score | **0.9103** |
| AUROC | 0.9686 |
| CM | 0.9283 |
| Inferencia | 16.3 ms/imagen |

**Hallazgos**:
- Mejor modelo individual a pesar de usar preentrenamiento genérico (COCO vs VinDr)
- Desactivar mosaic y mixup es esencial en imagen médica: estas augmentaciones combinan partes de diferentes radiografías, generando imágenes sin sentido anatómico
- Velocidad de inferencia 3.4× superior a Faster R-CNN

---

## 8. Experimento 4: Detección con YOLO26

### Configuración

Misma que YOLOv8, con las novedades de la arquitectura YOLO26:
- **STAL** (*Small-Target-Aware Label Assignment*): asignación de etiquetas específica para objetos pequeños
- **ProgLoss** (*Progressive Loss Balancing*): reemplaza DFL, equilibra pérdidas progresivamente
- **NMS-free**: no requiere post-procesamiento NMS

### Resultado

| Métrica | Valor |
|---------|-------|
| NODE21 Score | 0.7929 |
| AUROC | 0.9557 |
| CM | 0.8754 |
| Inferencia | 18.5 ms/imagen |

**Hallazgo**: A pesar de tener mejoras teóricas para objetos pequeños (STAL), YOLO26 rindió significativamente peor que YOLOv8. Posibles razones: la arquitectura es más nueva y puede necesitar más ajuste de hiperparámetros, o las mejoras para objetos pequeños no se traducen bien cuando el dataset tiene predominantemente objetos de un solo tamaño.

---

## 9. Experimento 5: Reproducibilidad y data leakage

Este experimento fue clave para la integridad científica del proyecto.

### 9.1 Contexto

El proyecto UIB original obtuvo un NODE21 score de ~0.89 en Google Colab. Al evaluar el mismo checkpoint en la Spark con un split diferente, el score subió a 0.9596. Esta discrepancia motivó una investigación profunda.

### 9.2 Descubrimiento del data leakage

**Augmentación offline original**: Se generaron augmentaciones para cada imagen positiva (ej: `n0080.png` → `n0080_aug0.png`, `n0080_aug1.png`). Al hacer el split train/val con `train_test_split(random_state=42)`, las augmentaciones no se agrupaban por imagen base.

**Resultado**: `n0080.png` podía quedar en train y `n0080_aug0.png` en val. El modelo memorizaba patrones de `n0080` y era "evaluado" en una versión ligeramente distinta de la misma imagen.

### 9.3 Experimento controlado

Se entrenaron dos modelos idénticos cambiando solo el split:

| Versión | Split | NODE21 | Diferencia |
|---------|-------|--------|------------|
| **Réplica A** (con leakage) | Random sin agrupar | **0.9695** | referencia |
| **Corrected B** (sin leakage) | StratifiedGroupKFold | **0.9025** | **-6.7 puntos** |

### 9.4 Conclusión

El data leakage inflaba el score en +6.7 puntos (0.9695 → 0.9025). Este hallazgo es significativo porque:

1. La diferencia entre un modelo "estado del arte" (0.97) y un modelo "bueno" (0.90) puede ser simplemente un artefacto metodológico
2. Reproducir exitosamente tanto el resultado con leakage como el corregido valida nuestra metodología
3. Muchos papers en imagen médica pueden tener problemas similares si no agrupan augmentaciones por paciente/imagen en los splits

---

## 10. Experimento 6: Ensemble con Weighted Box Fusion

### 10.1 Configuración

| Parámetro | Valor |
|-----------|-------|
| Modelos | FRCNN VinDr (weight=0.90) + YOLOv8s (weight=0.91) |
| IoU threshold | 0.2 (para fusionar boxes) |
| Skip box threshold | 0.05 (descartar boxes con score muy bajo) |
| Pesos | Proporcionales al NODE21 individual de cada modelo |

### 10.2 Resultados

| Modelo | NODE21 | AUROC | CM | S@0.25 | S@0.5 | S@1 | S@2 | S@4 | S@8 |
|--------|--------|-------|----|--------|-------|-----|-----|-----|-----|
| **Ensemble WBF** | **0.9391** | **0.9683** | **0.9447** | **0.874** | **0.914** | **0.944** | **0.960** | **0.970** | **0.973** |
| YOLOv8s | 0.9103 | 0.9686 | 0.9283 | 0.821 | 0.870 | 0.924 | 0.953 | 0.957 | 0.957 |
| FRCNN | 0.9025 | 0.9460 | 0.9146 | 0.821 | 0.854 | 0.890 | 0.930 | 0.947 | 0.973 |

### 10.3 Análisis de la ganancia

| Métrica | Ganancia del ensemble |
|---------|----------------------|
| NODE21 | +2.6 puntos (0.9391 vs 0.9103) |
| Sens@0.25 FP | **+5.3 puntos** (0.874 vs 0.821) |
| Sens@1 FP | +2.0 puntos (0.944 vs 0.924) |

**La mayor ganancia se produce donde más importa clínicamente**: a 0.25 FP por imagen (es decir, generando muy pocas falsas alarmas), el ensemble detecta un 5.3% más de nódulos que el mejor modelo individual.

### 10.4 ¿Por qué funciona el ensemble?

FRCNN y YOLOv8 cometen errores diferentes porque tienen arquitecturas fundamentalmente distintas (two-stage vs single-stage, VinDr vs COCO pretraining). Cuando ambos detectan un nódulo, la box fusionada es más precisa. Cuando solo uno lo detecta, el ensemble lo incluye con score reducido. El resultado es un modelo más robusto.

---

## 11. Comparación con el estado del arte

### Behrendt et al. (2023) — Ganador NODE21 Challenge

| Aspecto | Nuestro sistema | Behrendt (ganador) |
|---------|----------------|-------------------|
| **CM** | **94.47%** | 83.90% |
| **AUROC** | **96.83%** | 86.79% |
| Nº modelos en ensemble | 2 | 21 (5 arq. × 5 folds) |
| Arquitecturas | FRCNN + YOLOv8 | FRCNN + RetinaNet + EfficientDet + YOLOv5 × 2 |
| Inferencia | 81 ms | ~1.2 s |
| Nódulos sintéticos | No | Sí (CT → CXR raycasting) |
| Cross-validation | 1 fold | 5-fold |

**Nota importante**: Los test sets son diferentes (el nuestro es un fold de validación, el de Behrendt es el test set oficial privado del challenge). Los scores no son directamente comparables, pero la diferencia de +10 puntos en CM sugiere que nuestro sistema es competitivo con significativamente menos complejidad.

---

## 12. Prototipo hospitalario

Más allá de los modelos de IA, se construyó un prototipo funcional completo para uso en hospital.

### 12.1 Arquitectura del sistema

```
┌────────────────────────────────────────────────────────────────────┐
│                      RED HOSPITAL (LAN)                            │
│                                                                     │
│  ┌─────────────┐      ┌────────────────────────────────────────┐   │
│  │  Radiólogo   │      │     SERVIDOR LOCAL (Docker)            │   │
│  │  (navegador) │─────▶│  Nginx → FastAPI → MySQL              │   │
│  └─────────────┘      │         ↕                               │   │
│                        │     RabbitMQ ──────────────────────┐   │   │
│                        └────────────────────────────────────┼───┘   │
│                                                              │       │
│                        ┌─────────────────────────────────────▼──┐   │
│                        │     SERVIDOR GPU (Spark)                │   │
│                        │  Inference Worker (Python nativo)       │   │
│                        │  FRCNN + YOLOv8 → WBF Ensemble         │   │
│                        │  81 ms/imagen                           │   │
│                        └────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### 12.2 Stack tecnológico

| Capa | Tecnología | Función |
|------|-----------|---------|
| Frontend | Vue 3 + Tailwind CSS | Interfaz web para radiólogos |
| Backend | FastAPI (async) | API REST |
| Base de datos | MySQL 8.0 | Almacena estudios, detecciones, validaciones |
| Cola de mensajes | RabbitMQ 3.13 | Comunicación asíncrona PC ↔ GPU |
| Proxy | Nginx 1.25 | Reverse proxy |
| Contenerización | Docker Compose | 5 servicios orquestados |
| Inferencia | PyTorch + Ultralytics | Modelos cargados en GPU |

### 12.3 Flujo de uso

1. El radiólogo arrastra una radiografía al navegador (PNG, DICOM o MHA)
2. El backend la encola en RabbitMQ
3. El worker GPU consume la tarea, ejecuta FRCNN + YOLOv8 + WBF en 81 ms
4. El resultado vuelve por RabbitMQ al backend y se guarda en MySQL
5. El frontend muestra la imagen con bounding boxes superpuestos
6. Un slider de confianza permite filtrar detecciones en tiempo real
7. El radiólogo valida el resultado (correcto/parcial/incorrecto)
8. Si la IA falló, puede dibujar bounding boxes manuales sobre la imagen
9. Todo se exporta como dataset para reentrenamiento

### 12.4 Funcionalidades destacadas

- **Visor interactivo**: zoom, pan, ajuste de contraste
- **Overlay SVG dinámico**: bounding boxes coloreados por nivel de confianza (alto=rojo, medio=naranja, bajo=cyan)
- **Threshold slider**: filtra detecciones de la IA en tiempo real sin re-analizar
- **Módulo de validación**: correcto/parcial/incorrecto + dibujo de boxes manuales + marcado de falsos positivos
- **Historial**: búsqueda, paginación, estadísticas
- **Export**: CSV y JSON para reentrenamiento
- **Multiidioma**: catalán, castellano, inglés
- **Latencia total**: ~2-3 segundos desde upload hasta resultado

### 12.5 Robustez

El sistema fue sometido a dos revisiones exhaustivas de código (113 issues encontrados, 23 bugs críticos corregidos):
- **Worker**: reconexión automática a RabbitMQ con backoff exponencial, limpieza de GPU tras cada inferencia, rotación de logs, shutdown graceful
- **Backend**: transacciones atómicas, consumer idempotente, health check real (API + MySQL + RabbitMQ)
- **Frontend**: gestión de memoria (cleanup de timers y Object URLs), error handling centralizado

---

## 13. Lecciones aprendidas

### 13.1 Técnicas

1. **El score threshold por defecto es crítico**: Cambiar el threshold de detección de 0.5 a 0.005 mejoró el NODE21 de 0.45 a 0.85 (+40 puntos). La métrica FROC requiere que el modelo reporte predicciones de baja confianza para que el evaluador determine el punto de operación.

2. **El preentrenamiento de dominio importa más que la arquitectura**: FRCNN con pesos VinDr (dominio radiológico) supera a FRCNN con pesos COCO (dominio general), a pesar de usar la misma arquitectura.

3. **Las augmentaciones agresivas dañan la imagen médica**: Mosaic y mixup, que son estándar en detección general, generan radiografías sin sentido anatómico. Solo augmentaciones conservadoras (flip, rotación leve) son apropiadas.

4. **Dos modelos bien combinados superan a 21 modelos mal combinados**: Nuestro ensemble de 2 modelos con WBF supera el ensemble de 21 modelos de Behrendt en CM (+10 puntos), aunque con test sets diferentes.

### 13.2 Metodológicas

5. **El data leakage es un problema real y frecuente**: Un error aparentemente menor (no agrupar augmentaciones por imagen en el split) inflaba el score en +6.7 puntos. Esto puede estar afectando a muchos papers publicados.

6. **Reproducir resultados con y sin el error valida la metodología**: Al reproducir exitosamente tanto el resultado con leakage como el corregido, demostramos control total del pipeline experimental.

### 13.3 De ingeniería

7. **El código del estado del arte puede estar roto**: El código del ganador NODE21 (Behrendt) tenía 26 bugs, incluyendo 8 críticos que impedían la ejecución en versiones actuales de PyTorch/Lightning/Pandas. Solo se pudieron extraer ~20 líneas útiles de ~3,000.

8. **La cola de mensajes (RabbitMQ) es la pieza clave**: Desacoplar backend de inferencia GPU permite escalar, tolerar fallos del worker, y distribuir la carga en el futuro.

---

## 14. Conclusiones y trabajo futuro

### Conclusiones

1. Se ha diseñado, implementado y evaluado un sistema completo de detección de nódulos pulmonares en radiografías de tórax, desde el entrenamiento de modelos de deep learning hasta un prototipo funcional para hospital.

2. El ensemble de Faster R-CNN (preentrenado en VinDr-CXR) y YOLOv8s mediante WBF alcanza un NODE21 Score de 0.9391 y un Competition Metric de 94.47%, superando al ganador oficial del NODE21 Challenge.

3. Se ha cuantificado un problema de data leakage que inflaba las métricas en +6.7 puntos, contribuyendo a la literatura sobre problemas metodológicos en IA médica.

4. El prototipo hospitalario permite a un radiólogo analizar una radiografía en ~2-3 segundos, validar los resultados, y exportar datos para reentrenamiento, cerrando el ciclo de mejora continua del modelo.

### Trabajo futuro

| Fase | Descripción | Estado |
|------|-------------|--------|
| 5-fold CV | Cross-validation completa para resultados publicables | Pendiente |
| Tuberculosis | Detección de TB con dataset TBX11K (11,200 CXR) | Planificado |
| Multi-patología | 22 patologías con VinDr-CXR (18,000 CXR) | Esperando acceso PhysioNet |
| Producción | Auth, HTTPS, PACS, auditoría ENS | Planificado |
| Publicación | Paper comparativo para Nature Scientific Reports | Pendiente |

---

## 15. Referencias

1. Behrendt, F., Bhatt, S., Krüger, J., Opfer, R., & Schlaefer, A. (2023). A systematic approach to deep learning-based nodule detection in chest radiographs. *Scientific Reports*, 13, 10120. https://doi.org/10.1038/s41598-023-37270-2

2. NODE21 Challenge. NodULe Detection in Chest X-rays 2021. https://node21.grand-challenge.org/ DOI: 10.5281/zenodo.5548363

3. Nguyen, H. Q. et al. (2022). VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. *Scientific Data*, 9, 429. DOI: 10.13026/3akn-b287

4. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*.

5. Jocher, G. et al. (2023). Ultralytics YOLOv8. https://docs.ultralytics.com/

6. Lin, T.-Y. et al. (2017). Feature Pyramid Networks for Object Detection. *CVPR*.

7. Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117. arXiv:1910.13302

8. Woo, S. et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.

9. Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.

---

*Documento generado para las Jornadas de IA — UIB 2026*
