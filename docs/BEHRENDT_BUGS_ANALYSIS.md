# ANÁLISIS COMPLETO DE BUGS — Código Behrendt (node21-submit)
# Revisión exhaustiva para reproducción en Spark

---

## RESUMEN EJECUTIVO

| Severidad | Cantidad | Impacto |
|-----------|----------|---------|
| CRÍTICO (rompe ejecución) | 8 | No funciona con PyTorch 2.x / Lightning 2.x / Pandas 2.x |
| ALTO (resultados incorrectos) | 5 | Métricas erróneas o comportamiento inesperado |
| MEDIO (fragilidad) | 7 | Funciona pero es propenso a fallos silenciosos |
| BAJO (calidad de código) | 6 | Mantenibilidad, imports no usados |

**Veredicto: NO vale la pena portar el código Behrendt completo.** Es más eficiente extraer las ideas clave (SWA, WeightedRandomSampler, augmentación, ensemble WBF) e implementarlas en nuestro pipeline limpio.

---

## BUGS CRÍTICOS (rompen ejecución con versiones modernas)

### BUG C1: Lightning epoch_end deprecated
**Archivo**: `src/models/Detector.py` líneas 86, 105, 155
```python
def training_epoch_end(self, outputs) -> None:    # DEPRECATED
def validation_epoch_end(self, outputs) -> None:   # DEPRECATED  
def test_epoch_end(self, outputs) -> None:          # DEPRECATED
```
**Error**: `TypeError: on_train_epoch_end() takes 1 positional argument but 2 were given`
**Fix**: Renombrar a `on_train_epoch_end(self)`, `on_validation_epoch_end(self)`, etc. Eliminar parámetro `outputs`.

### BUG C2: Lightning Trainer `gpus` parameter
**Archivo**: `config_fcrnn_l.yaml` línea 105
```yaml
gpus: [0]
```
**Error**: `TypeError: Trainer.__init__() got unexpected keyword argument 'gpus'`
**Fix**: 
```yaml
accelerator: gpu
devices: [0]
```

### BUG C3: Lightning `weights_summary` y `progress_bar_refresh_rate`
**Archivo**: `config_fcrnn_l.yaml` líneas 113-114
```yaml
weights_summary: null
progress_bar_refresh_rate: 25
```
**Error**: Parámetros eliminados en Lightning 2.0
**Fix**: Eliminar `weights_summary`, cambiar `progress_bar_refresh_rate` a `log_every_n_steps: 25`

### BUG C4: TorchVision `pretrained` deprecated
**Archivo**: `src/models/modules/FasterCNN.py` líneas 43-47
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained = cfg.pretrained,
    pretrained_backbone = cfg.pretrained_backbone,
    ...
)
```
**Error**: `FutureWarning: 'pretrained' deprecated, use 'weights' instead` → Error en TorchVision 0.15+
**Fix**:
```python
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if cfg.pretrained else None
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, ...)
```

### BUG C5: Pandas DataFrame.append() eliminado
**Archivo**: `src/datamodules/Datamodules.py` línea 120
```python
self.train = get_data.DataSet(self.csv['train'].append(self.csv['val']), self.cfg)
```
**Error**: `AttributeError: 'DataFrame' object has no attribute 'append'`
**Fix**: `pd.concat([self.csv['train'], self.csv['val']], ignore_index=True)`

### BUG C6: Lightning optimizer_step signature deprecated
**Archivo**: `src/models/Detector.py` líneas 203-220
```python
def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu=False, using_native_amp=False,
                   using_lbfgs=False):
```
**Error**: Signature cambiada en Lightning 1.9+, eliminada en 2.0+
**Fix**: Eliminar método entero, usar `LambdaLR` scheduler con warmup

### BUG C7: trainer.num_gpus deprecated
**Archivo**: `src/datamodules/Datamodules.py` líneas 149, 165, 182
```python
num_workers=self.cfg.num_workers * int(self.trainer.num_gpus)
```
**Error**: `AttributeError: 'Trainer' has no attribute 'num_gpus'`
**Fix**: `self.trainer.num_devices` o `len(self.trainer.device_ids)`

### BUG C8: DDPPlugin deprecated
**Archivo**: `process.py` línea 46
```python
from pytorch_lightning.plugins import DDPPlugin
```
**Error**: `ImportError: cannot import name 'DDPPlugin'`
**Fix**: `from pytorch_lightning.strategies import DDPStrategy`

---

## BUGS ALTOS (resultados incorrectos o comportamiento erróneo)

### BUG A1: Monkey-patching global de loss function
**Archivo**: `src/models/modules/FasterCNN.py` líneas 15-22
```python
if cfg.get('label_smoothing',False):
    torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_ls
elif cfg.get('class_weighting',False):
    torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_cw
elif cfg.get('focal_loss',False):
    torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_focal
elif cfg.get('class_weighting',False) and cfg.get('class_weighting',False):  # BUG LÓGICO
    torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_both
```
**Problemas**:
1. **Bug lógico línea 21**: `class_weighting and class_weighting` es siempre True si class_weighting=True, pero NUNCA se alcanza porque la línea 17 ya lo captura
2. **Polución global**: Parchea una función interna de TorchVision que persiste para TODOS los modelos
3. **Fragilidad**: Cualquier actualización de TorchVision puede cambiar la API interna
4. **El comentario del autor**: `# this is so bad...` — el propio autor sabe que es mala práctica

### BUG A2: Lógica incorrecta en .any()==1
**Archivo**: `src/datamodules/get_data.py` línea 80
```python
if nodule_data['label'].any()==1:
```
**Problema**: `.any()` retorna `bool` (True/False), no un valor numérico. `True == 1` es True en Python, así que FUNCIONA por casualidad, pero es conceptualmente incorrecto.
**Fix**: `if (nodule_data['label'] == 1).any():`

### BUG A3: Transform exception silenciada
**Archivo**: `src/datamodules/get_data.py` líneas 208-213
```python
try:
    transformed = self.transform(image=img, bboxes=boxes, class_labels=lab)
except:
    print('error in transformation')
# CONTINÚA sin return/continue → transformed no definida → crash
img = transformed['image']  # ← AttributeError si except fue ejecutado
```
**Fix**: Añadir `return self._empty_target(idx)` en el except, o al menos `continue`

### BUG A4: Validación en training mode
**Archivo**: `src/models/Detector.py` línea 92 (validation_step)
```python
def validation_step(self, batch, batch_idx):
    images, targets = batch
    # NO hay model.eval() explícito
    # Lightning puede o no poner en eval mode dependiendo de la versión
```
**Problema**: BatchNorm usa estadísticas de training en vez de running stats → métricas de validación inconsistentes.

### BUG A5: Score tensor sin device consistente
**Archivo**: `src/models/Detector.py` línea 100 y `src/utils/custom_metrics.py` línea 74
```python
# Detector.py:100
sample['scores'] = torch.tensor([0.0], device=self.device)

# custom_metrics.py:74  
mat[i,j] = torch.tensor(scores[i])  # SIN device
```
**Problema**: Tensores en diferentes devices → posible error en multi-GPU

---

## BUGS MEDIOS (fragilidad, pueden causar fallos silenciosos)

### BUG M1: Bare except silencia TODO
**Archivo**: `src/models/modules/FasterCNN.py` línea 107
```python
except:
    print('loading of ckpt failed. This is ok in Evaluation')
```
**Problema**: Captura KeyboardInterrupt, SystemExit, MemoryError. Si los pesos no cargan, el entrenamiento continúa con pesos aleatorios SIN aviso claro.

### BUG M2: Paths absolutos hardcodeados
**Archivos**: Múltiples
- `FasterCNN.py:48` → `/home/linux/Node21/pretrained/FCRNN/epoch-8_step-20105_loss-4.17.ckpt`
- `process.py:78-79` → Paths hardcodeados de input/output
- `nodule_generation/process.py:42,51` → Paths a CSV y datos

### BUG M3: FocalLoss modifica loss_fcn pasada
**Archivo**: `src/utils/losses.py` línea 11
```python
self.loss_fcn.reduction = 'none'  # Modifica el objeto original
```
**Problema**: Side effect — si otro código reutiliza esa loss_fcn, su reduction ya cambió.

### BUG M4: Ensemble boxes reshape sin validación
**Archivo**: `src/utils/ensemble_boxes_weighted_numpy.py` líneas 46-49
```python
boxes_pred = np.reshape(boxes_pred, [-1, len(preds)]).T
scores = np.reshape(scores, [-1, len(preds)]).T
```
**Problema**: Si los modelos producen diferente número de predicciones, reshape falla silenciosamente.

### BUG M5: Cache memory allocation excesiva
**Archivo**: `src/datamodules/get_data.py` líneas 36-43
```python
shared_array_base = mp.Array(ctypes.c_float, len(self.labels) * 1024 * 1024)
```
**Problema**: Aloca 4MB por imagen × 4882 imágenes = ~19GB de RAM compartida. OOM en máquinas con poca RAM.

### BUG M6: Mosaic augmentation puede generar boxes inválidas
**Archivo**: `src/datamodules/get_data.py` línea 168
```python
if len(boxes) > 0:
    boxes = np.array(boxes)
    boxes = np.clip(boxes, 0, self.new_shape)
```
**Problema**: `clip` puede generar boxes con width=0 o height=0 si el nódulo queda fuera del crop del mosaic.

### BUG M7: Import no usado pero con side effect potencial
**Archivo**: `process.py` línea 10
```python
from wandb import Config  # Importa wandb aunque no se use en inferencia
```
**Problema**: Si wandb no está instalado, la inferencia falla aunque no necesite wandb.

---

## BUGS BAJOS (calidad de código, mantenibilidad)

### BUG L1: Imports no usados
- `process.py:1` → `from genericpath import exists`
- `ensemble_boxes_weighted_numpy.py:1` → `from unicodedata import name`
- `custom_metrics.py:4` → `from torchmetrics import ROC` (importado pero no siempre usado)

### BUG L2: Código muerto / comentado
- `process.py:25-28` → imports comentados con `#`
- `Datamodules.py:115,129` → código `.append()` comentado

### BUG L3: Magic numbers sin constantes
- `custom_metrics.py:57` → `thresh = 0.2` hardcodeado (IoU threshold)
- `utils.py` NMS → `lambda_nms = 0.2` como parámetro default pero no configurable desde YAML

### BUG L4: Docstrings ausentes en funciones públicas
Ninguna función tiene docstring. Las funciones de `custom_metrics.py` son especialmente difíciles de entender sin documentación.

### BUG L5: Type hints inconsistentes
Algunas funciones tienen type hints (`process.py`), la mayoría no.

### BUG L6: Duplicación de IoU computation
- `utils.py:intersection_over_union()` — implementación manual
- `custom_metrics.py:calc_iou()` — usa `torchvision.ops.box_iou`
Dos implementaciones diferentes de la misma función.

---

## COMPONENTES VALIOSOS PARA EXTRAER (no reproducir todo)

En vez de portar todo el código, extraer estas ideas:

| Componente | Archivo | Qué extraer |
|-----------|---------|-------------|
| **WBF Ensemble** | `ensemble_boxes_weighted_numpy.py` | Función `ensemble_boxes()` — usa librería `ensemble-boxes` |
| **FROC Metrics** | `custom_metrics.py` | `compute_froc_curve_data()`, `compute_froc_score()` |
| **SWA** | `Detector.py` líneas 260+ | Configuración de SWA epochs |
| **WeightedRandomSampler** | `Datamodules.py` líneas 142-148 | Balance de clases para sampling |
| **VinDr weight loading** | `FasterCNN.py` líneas 96-108 | Mapeo secuencial de keys |
| **Augmentación VinDR** | `get_data.py` líneas 240-290 | CropAndPad, BrightnessContrast, Flip, Rotate, Blur, Cutout |
| **LR Scheduler** | `scheduler.py` | Warmup + cosine decay |
| **NMS** | `utils.py` | `get_NonMaxSup_boxes()` |
| **Nodule generation** | `nodule_generation/process.py` | CT→CXR raycasting con Poisson blending |

---

## VERSIONES REQUERIDAS POR EL CÓDIGO ORIGINAL

```
pytorch-lightning==1.5.1  (NO funciona con 2.0+)
torch==1.12.x             (funciona con 2.0 pero con warnings)
torchvision==0.12.x       (pretrained= deprecated en 0.13+)
pandas==1.3.x             (.append() eliminado en 2.0)
effdet                    (sin versión fijada)
timm                      (sin versión fijada)
hydra-core==1.1.1         (funciona con versiones nuevas)
monai==0.7.0              (sin problemas de compatibilidad)
ensemble-boxes==1.0.7     (funciona)
```

---

## RECOMENDACIÓN FINAL

**NO portar el código Behrendt completo.** Razones:
1. 8 bugs críticos que rompen ejecución con PyTorch 2.x
2. Dependencia fuerte de Lightning 1.5 + Hydra → difícil de portar
3. Monkey-patching de TorchVision internals → frágil
4. El pipeline ya funciona con nuestro código limpio en Spark

**SÍ extraer:**
1. SWA → `torch.optim.swa_utils` (3 líneas de código)
2. WeightedRandomSampler → `torch.utils.data.WeightedRandomSampler` (5 líneas)
3. Augmentación Albumentations → ya la tenemos
4. WBF Ensemble → `from ensemble_boxes import weighted_boxes_fusion` (10 líneas)
5. FROC metrics → ya implementadas en nuestro evaluate.py
6. Gradient clipping → `clip_grad_norm_(model.parameters(), 3.0)` (1 línea)

Total: ~20 líneas de código útiles vs ~3000 líneas con 26 bugs.
