# Dataset NODE21

[← Home](Home.md)

NODE21 es el dataset y benchmark oficial del NODE21 Challenge, publicado en *IEEE Transactions on Medical Imaging* (2024). Este proyecto lo usa como dataset principal de entrenamiento y evaluación.

## Composición

| Característica | Valor |
|---|---|
| Total de radiografías | 4 882 |
| Radiografías con nódulo | 2 091 |
| Radiografías control (sin nódulo) | 2 791 |
| Total de nódulos anotados | 3 049 |
| Resolución original | Variable (típicamente 2 000 × 2 000 a 3 000 × 3 000 px) |
| Formato original | MetaImage (`.mha`) |
| Anotación | Bounding box por nódulo (x, y, w, h) |

## Fuentes

NODE21 se construye a partir de cuatro datasets públicos previamente curados, con un proceso de armonización y reanotación:

| Fuente | Aporte |
|---|---|
| JSRT | 247 radiografías japonesas con anotaciones originales |
| PadChest | Selección manual de casos con nódulo |
| ChestX-ray14 (NIH) | Selección filtrada y reanotada |
| Open-I (NLM) | Selección filtrada y reanotada |

La anotación final es uniforme: bounding box por nódulo, independientemente del dataset de origen.

## Acceso

El dataset está disponible mediante registro en <https://node21.grand-challenge.org/>. Licencia CC BY-NC-ND 4.0: no se redistribuye en este repositorio.

## Splits

El protocolo de evaluación oficial divide en train / public-test / private-test. Para los experimentos internos, usamos cross-validation estratificado:

| Split | Imágenes | Uso |
|---|---|---|
| Train | 70 % | Entrenamiento de los detectores |
| Val | 15 % | Selección de hiperparámetros y early stopping |
| Test | 15 % | Cifras finales reportadas en la memoria |

Los tres splits respetan las dos restricciones siguientes simultáneamente:

1. **Estratificación por presencia de nódulo** — el ratio caso/control se mantiene constante en cada split.
2. **Agrupación por imagen original** — todas las augmentaciones offline de una misma radiografía caen en el mismo split.

La función que aplica ambas a la vez es `sklearn.model_selection.StratifiedGroupKFold`, parametrizada con `groups = image_uid` y `y = has_nodule`.

## Preprocesado

```
MHA → PNG → normalización 8-bit → resize 1024×1024
```

Pasos detallados:

1. **Lectura MHA** con SimpleITK.
2. **Conversión a PNG mono-canal** preservando las dimensiones originales.
3. **Normalización 8-bit por percentil** (1 % y 99 % de la histograma) para mitigar diferencias de calibración entre fuentes.
4. **Resize a 1024 × 1024** para entrada uniforme a los detectores.
5. **Reescalado de las bounding boxes** al sistema de coordenadas redimensionado.

Implementación en [`spark/scripts/preprocess_node21.py`](../spark/scripts/preprocess_node21.py).

## Protocolo anti-data-leakage

El bug crítico de la implementación de referencia (Behrendt et al.) era aplicar augmentaciones offline ANTES del split:

```python
# Bug Behrendt: genera 10 copias aumentadas por imagen y luego divide
all_images = original_images + augmented_copies      # ← mezcla original + aug
train, val = train_test_split(all_images, test_size=0.2, stratify=labels)
```

Resultado: copias aumentadas de la misma imagen original aparecen tanto en train como en val. El modelo aprende a reconocer el original visto en train y "predice bien" sobre el augment visto en val. Inflado: +6.7 puntos NODE21 absolutos.

Protocolo limpio:

```python
# Limpio: agrupa por imagen original, divide, y SÓLO LUEGO aumenta train
splitter = StratifiedGroupKFold(n_splits=5)
for train_idx, val_idx in splitter.split(X=original_images,
                                         y=has_nodule,
                                         groups=image_uid):
    train_originals = original_images[train_idx]
    val_originals   = original_images[val_idx]
    train_augmented = generate_augmentations(train_originals)   # SÓLO train
```

La diferencia cuantificada (NODE21 = 0.9695 → 0.9025) se documenta en [`docs/BEHRENDT_BUGS_ANALYSIS.md`](../docs/BEHRENDT_BUGS_ANALYSIS.md).

## Datasets complementarios

### VinDr-CXR (preentrenamiento)

| Campo | Valor |
|---|---|
| Imágenes | 18 000 |
| Patologías anotadas | 21 (consolidación, atelectasia, nódulo, masa, etc.) |
| Formato | DICOM |
| Acceso | <https://physionet.org/content/vindr-cxr/> (restricción PhysioNet) |
| Uso en este trabajo | Preentrenamiento del backbone de Faster R-CNN |

VinDr es el factor que más influye en el rendimiento del Faster R-CNN: sin él, la cifra cae ~5 puntos NODE21.

### Datasets futuros (fases 2 y 3 del roadmap)

| Dataset | Fase | Tarea | Estado |
|---|---|---|---|
| TBX11K | Fase 2 — Tuberculosis | Detección + clasificación de TB | Pendiente |
| VinDr-CXR multilabel | Fase 3 — 22 patologías | Clasificación multietiqueta | Descarga en curso |

## Recursos en este repositorio

- Preprocesado: [`spark/scripts/preprocess_node21.py`](../spark/scripts/preprocess_node21.py)
- Augmentaciones offline: [`spark/scripts/generate_augmented_images.py`](../spark/scripts/generate_augmented_images.py)
- Pipeline modular: [`spark/pipeline/data/`](../spark/pipeline/data/)
- Splits StratifiedGroupKFold: integrados en la pipeline (módulo `data/splits.py`)
- Discusión completa: capítulo 3 de [[Memoria-academica]]
