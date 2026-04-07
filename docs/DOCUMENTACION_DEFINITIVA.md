# CXR NODULE DETECTION — Documentación Definitiva v1.0
# Sistema de detección de nódulos pulmonares en radiografías de tórax
# Prototipo funcional para hospital

**Versión**: 1.0.0
**Fecha cierre**: 2026-04-04
**Autores**: Toni Contesti (desarrollo), Marc Link Cladera & Antonio Contesti Coll (proyecto base UIB)
**Supervisor original**: Miquel Miró Nicolau (UIB)

---

## ÍNDICE

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Resultados científicos](#2-resultados-científicos)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Modelos de IA](#4-modelos-de-ia)
5. [Backend (FastAPI + Docker)](#5-backend)
6. [Frontend (Vue 3 + Tailwind)](#6-frontend)
7. [Inference Worker (Spark GPU)](#7-inference-worker)
8. [Módulo de validación y anotación](#8-módulo-de-validación)
9. [Despliegue y operación](#9-despliegue)
10. [Bugs conocidos y limitaciones](#10-bugs-conocidos)
11. [Roadmap futuro](#11-roadmap)
12. [Inventario de archivos](#12-inventario)
13. [Referencias](#13-referencias)
14. [Apéndices](#14-apéndices)

---

## 1. RESUMEN EJECUTIVO

### Qué es
Sistema end-to-end de detección automática de nódulos pulmonares en radiografías de tórax (CXR), con interfaz web para radiólogos, inferencia GPU en tiempo real, y módulo de validación para estudios prospectivos.

### Qué hace
1. El radiólogo sube una radiografía de tórax (PNG, DICOM, MHA)
2. La IA analiza la imagen con 2 modelos (Faster R-CNN + YOLOv8) fusionados con WBF
3. En ~2 segundos muestra las detecciones con bounding boxes sobre la imagen
4. El radiólogo valida el resultado (correcto/parcial/incorrecto)
5. Si la IA falló, dibuja bounding boxes manualmente para corregir
6. Todo se guarda para estudios prospectivos y reentrenamiento de modelos
7. Se puede exportar el dataset validado en CSV/JSON

### Resultados clave
- **NODE21 Score: 0.9391** (Ensemble WBF, scores honestos sin data leakage)
- **Inferencia: 81ms/imagen** en GPU
- **Latencia end-to-end: ~2-3 segundos** (incluyendo upload, red, inferencia, resultado)
- Supera al ganador oficial NODE21 (Behrendt et al., CM=83.90%) con CM=94.47%

### Stack tecnológico
| Capa | Tecnología |
|------|-----------|
| Frontend | Vue 3.5 + Tailwind CSS 4.2 + Vite 7 |
| Backend | FastAPI 0.115 + SQLAlchemy async + Docker Compose |
| Base de datos | MySQL 8.0 |
| Cola de mensajes | RabbitMQ 3.13 |
| Proxy | Nginx 1.25 |
| Inferencia | PyTorch 2.x + Ultralytics (YOLOv8) |
| GPU | NVIDIA GB10 (130GB VRAM) |

---

## 2. RESULTADOS CIENTÍFICOS

### 2.1 Dataset NODE21
- 4,882 radiografías CXR (1024×1024 px)
- 1,134 con nódulos (1,476 anotaciones bounding box)
- 3,748 sin nódulos (prevalencia: 23%)
- Fuentes: JSRT, PadChest, ChestX-ray14, Open-I

### 2.2 Métricas
| Métrica | Descripción |
|---------|-------------|
| NODE21 Score | Media de sensibilidad a FP/img = [0.25, 0.5, 1, 2, 4, 8] |
| AUROC | Área bajo curva ROC a nivel de imagen |
| CM | Competition Metric = 0.75 × AUROC + 0.25 × NODE21 |
| IoU threshold | 0.2 (oficial NODE21) |

### 2.3 Ranking final (scores honestos, sin data leakage)

| # | Modelo | NODE21 | AUROC | CM | S@0.25FP |
|---|--------|--------|-------|----|----------|
| **1** | **Ensemble WBF (FRCNN+YOLOv8)** | **0.9391** | **0.9683** | **0.9447** | **0.874** |
| 2 | YOLOv8s individual | 0.9103 | 0.9686 | 0.9283 | 0.821 |
| 3 | FRCNN VinDr Corrected-B | 0.9025 | 0.9460 | 0.9146 | 0.821 |
| 4 | YOLO26s | 0.7929 | 0.9557 | 0.8754 | 0.635 |

### 2.4 Sensibilidad por nivel FP/imagen

| Modelo | @0.25 | @0.5 | @1 | @2 | @4 | @8 |
|--------|-------|------|-----|-----|-----|-----|
| Ensemble WBF | **0.874** | **0.914** | **0.944** | **0.960** | **0.970** | **0.973** |
| YOLOv8s | 0.821 | 0.870 | 0.924 | 0.953 | 0.957 | 0.957 |
| FRCNN VinDr | 0.821 | 0.854 | 0.890 | 0.930 | 0.947 | 0.973 |

### 2.5 Hallazgos científicos

1. **Data leakage cuantificado**: Augmentaciones offline de la misma imagen en train/val inflaban NODE21 en +6.7 puntos (0.9695 vs 0.9025). Corregido con StratifiedGroupKFold agrupando por imagen base.

2. **Score threshold crítico**: El threshold por defecto de Faster R-CNN (0.5) filtraba la mayoría de predicciones. Bajarlo a 0.005 mejoró NODE21 de 0.4546 a 0.8544.

3. **VinDr pretraining esencial**: Preentrenar el backbone en dominio radiológico (VinDr-CXR, 14 patologías) es el factor más determinante del rendimiento.

4. **WBF > individual**: El ensemble gana +2.6 NODE21 y +5.3 puntos de sensibilidad a 0.25 FP/img vs el mejor modelo individual.

5. **Código Behrendt inutilizable**: 26 bugs encontrados en el código del ganador NODE21 (incompatible con PyTorch 2.x / Lightning 2.x). Solo se extrajeron las ideas clave (~20 líneas útiles de ~3000).

### 2.6 Comparación con referencia

| | Nuestro ensemble | Behrendt (ganador NODE21) |
|---|---|---|
| CM | **94.47%** | 83.90% |
| Modelos | 2 | 21 (5 arquitecturas × 5 folds) |
| Inferencia | 81ms | ~1.2s |

---

## 3. ARQUITECTURA DEL SISTEMA

```
┌──────────────────────────────────────────────────────────────┐
│                    RED HOSPITAL (LAN)                         │
│                                                               │
│  Radiólogo → Vue 3 Frontend (http://localhost:5177)          │
│                    ↓                                          │
│  Docker local:                                                │
│  ┌─ Nginx (:8090)                                            │
│  ├─ FastAPI cxr-svc (:9020)                                  │
│  │    POST /api/cxr/upload → RabbitMQ                        │
│  │    GET  /api/cxr/results/{uid}                            │
│  │    POST /api/cxr/results/{uid}/validate                   │
│  │    GET  /api/cxr/validation/export                        │
│  ├─ MySQL (:3306 internal)                                   │
│  ├─ RabbitMQ (:5672 LAN, :15674 mgmt)                       │
│  └─ Result Consumer (RabbitMQ → MySQL)                       │
│                    ↓ AMQP                                     │
│  Spark GPU (nativo, sin Docker):                             │
│  └─ Inference Worker                                         │
│       ├─ FRCNN VinDr (NODE21=0.9025, ~55ms)                 │
│       ├─ YOLOv8s (NODE21=0.9103, ~25ms)                     │
│       └─ WBF Ensemble (NODE21=0.9391, ~81ms total)          │
└──────────────────────────────────────────────────────────────┘
```

### Flujo de datos completo

```
1. Upload CXR (PNG/DICOM) → POST /api/cxr/upload
2. Backend: valida → MySQL (status:queued) → imagen original → RabbitMQ (base64)
3. Spark Worker: consume → decode → FRCNN + YOLOv8 → WBF → publica resultado
4. Result Consumer: RabbitMQ → MySQL (status:completed, detecciones, imagen anotada)
5. Frontend (polling 1.5s): GET /results/{uid} → SVG overlay + slider threshold
6. Radiólogo valida: correcto/parcial/incorrecto + dibujar boxes manuales
7. POST /validate → MySQL (validación + anotaciones)
8. GET /validation/export → CSV/JSON para reentrenamiento
```

---

## 4. MODELOS DE IA

### 4.1 Faster R-CNN + VinDr-CXR
- Backbone: ResNet50 + FPN, preentrenado en VinDr-CXR (14 patologías)
- Congelación: layers 1-3 frozen, layer4 + FPN + head trainable
- Entrada: grayscale × 3 canales, 1024×1024
- Parámetros: 41.3M total, 32.8M trainable
- NODE21: 0.9025 | Inferencia: ~55ms

### 4.2 YOLOv8s
- Anchor-free, preentrenado en COCO
- Resolución: 1024×1024
- Augmentación: solo flip horizontal (sin mosaic/mixup/color)
- NODE21: 0.9103 | Inferencia: ~25ms

### 4.3 Ensemble WBF
- Weighted Box Fusion: fusiona boxes por media ponderada de coordenadas
- Weights: [0.90, 0.91] (proporcional a NODE21 individual)
- IoU threshold: 0.2, skip_box_thr: 0.05
- NODE21: **0.9391** | Inferencia total: ~81ms

---

## 5. BACKEND (FastAPI + Docker)

### 5.1 Servicios Docker

| Servicio | Imagen | Puerto | Función |
|----------|--------|--------|---------|
| cxr-mysql | mysql:8.0 | 3306 (int) | Base de datos |
| cxr-rabbitmq | rabbitmq:3.13 | 5672, 15674 | Cola de mensajes |
| cxr-backend | python:3.12-slim | 9020 | API REST |
| cxr-result-consumer | python:3.12-slim | — | RabbitMQ → MySQL |
| cxr-nginx | nginx:1.25 | 8090 | Reverse proxy |

### 5.2 API Endpoints

| Método | Endpoint | Función |
|--------|----------|---------|
| POST | `/api/cxr/upload` | Subir CXR para análisis |
| GET | `/api/cxr/results/{uid}` | Resultado con detecciones |
| GET | `/api/cxr/results/{uid}/original` | Imagen original PNG |
| GET | `/api/cxr/results/{uid}/image` | Imagen anotada PNG |
| DELETE | `/api/cxr/results/{uid}` | Eliminar estudio |
| DELETE | `/api/cxr/all` | Eliminar todos |
| GET | `/api/cxr/history` | Historial paginado + búsqueda |
| GET | `/api/cxr/stats` | Estadísticas generales |
| POST | `/api/cxr/results/{uid}/validate` | Validar resultado + anotaciones |
| GET | `/api/cxr/results/{uid}/validation` | Obtener validación |
| GET | `/api/cxr/validation/stats` | Estadísticas de validación |
| GET | `/api/cxr/validation/export` | Exportar dataset CSV/JSON |
| GET | `/api/health` | Estado API + MySQL + RabbitMQ |

### 5.3 Base de datos (6 tablas)

```
cxr_studies             — Estudio principal (uid, status, detecciones, timestamps)
cxr_detections          — Detecciones IA (bounding box + score + modelo)
cxr_original_images     — Imagen original subida (LONGBLOB)
cxr_annotated_images    — Imagen con boxes dibujados (LONGBLOB)
cxr_validations         — Validación del radiólogo (correct/partial/incorrect)
cxr_manual_annotations  — Bounding boxes dibujados manualmente (missed/false_positive)
```

### 5.4 Características de producción
- Connection pool: pool_size=10, pool_recycle=300, pool_pre_ping=True
- Graceful shutdown: lifespan con engine.dispose()
- CORS restrictivo: solo métodos necesarios
- Contenedores non-root, read-only, no-new-privileges
- Healthcheck en todos los servicios

---

## 6. FRONTEND (Vue 3 + Tailwind CSS)

### 6.1 Vistas

| Vista | Ruta | Función |
|-------|------|---------|
| AnalyzeView | `/` | Upload + análisis + resultado + validación |
| HistoryView | `/history` | Historial paginado con búsqueda + detalle |
| ValidationStatsView | `/validation` | Estadísticas + export dataset |

### 6.2 Componentes

| Componente | Función |
|-----------|---------|
| **CxrViewer** | Visor de imagen con zoom/pan + SVG overlay (detecciones + anotaciones manuales) + modo edición (dibujar boxes, marcar FP) |
| **ValidationPanel** | Workflow de validación: botones correcto/parcial/incorrecto, modo edición, notas por box, radiólogo, guardar/editar |
| **ThresholdSlider** | Slider 5%-99% que filtra detecciones en tiempo real |
| **DetectionList** | Lista de nódulos con score, riesgo, coordenadas |
| **DropZone** | Drag & drop de imágenes |
| **StudyStatus** | Badge de estado (procesando/completado/error) |
| **StatsBar** | Contadores (total, analizados, con nódulos, tiempo medio) |

### 6.3 Funcionalidades clave

- **Visor interactivo**: zoom rueda + pan arrastrar + botones Fit/1:1 + porcentaje real
- **SVG overlay dinámico**: boxes filtrados por threshold, colores por riesgo (rojo/naranja/cyan)
- **Modo edición**: dibujar boxes manuales (verde), marcar FP (click → gris tachado)
- **Nota por anotación**: cada box manual tiene campo de texto
- **Validación con bloqueo**: después de guardar se bloquea, botón "Editar" para modificar
- **Historial**: paginación, búsqueda por Enter, iconos de validación (✓/⚠/✗/⏱)
- **Memory management**: revokeObjectURL, cleanup polling en onUnmounted
- **Export dataset**: CSV/JSON desde vista de estadísticas

---

## 7. INFERENCE WORKER (Spark GPU)

### 7.1 Características
- Modelos cargados 1 vez al inicio (~2s)
- Inferencia: ~81ms/imagen (post-warmup)
- Reconexión RabbitMQ con exponential backoff (5s → 60s)
- Limpieza GPU (torch.cuda.empty_cache()) tras cada task
- Log rotation: 10MB × 5 archivos
- Graceful shutdown con flag _shutdown
- Config YAML externalizada

### 7.2 Rendimiento

| Métrica | Valor |
|---------|-------|
| FRCNN inferencia | ~55ms |
| YOLOv8 inferencia | ~25ms |
| WBF ensemble | ~1ms |
| Decode + preprocess | ~20ms |
| **Total por imagen** | **~81ms** |
| **Latencia end-to-end** | **~2-3s** |

---

## 8. MÓDULO DE VALIDACIÓN Y ANOTACIÓN

### 8.1 Flujo

```
1. IA analiza imagen → muestra detecciones
2. Radiólogo pulsa Correcto / Parcial / Incorrecto
3. Si parcial/incorrecto → modo edición activado sobre la imagen grande
4. Radiólogo dibuja boxes manuales (nódulos no detectados) → verde
5. Radiólogo marca detecciones IA como falso positivo (click) → gris tachado
6. Añade nota por cada anotación + notas generales + nombre radiólogo
7. Guarda validación → se bloquea (botón "Editar" para modificar)
8. Todo visible en historial con iconos de validación
9. Export CSV/JSON con detecciones IA + correcciones manuales
```

### 8.2 Datos exportados (para reentrenamiento)

| Campo | Descripción |
|-------|-------------|
| study_uid | Identificador del estudio |
| patient_id | ID paciente |
| validation_result | correct/incorrect/partial/pending |
| validated_by | Nombre del radiólogo |
| source | ai / radiologist / none |
| x, y, width, height | Bounding box |
| score | Confianza (1.0 para anotaciones manuales) |
| label | Tipo de hallazgo |
| type | detection / missed / false_positive / negative |

---

## 9. DESPLIEGUE Y OPERACIÓN

### 9.1 Arrancar el sistema

**Docker (local):**
```bash
cd /path/to/cxr-detection
docker compose up -d
```

**Frontend (desarrollo):**
```bash
cd /path/to/cxr-frontend
npm run dev
```

**Spark Worker:**
```bash
ssh ${SPARK_USER}@${SPARK_HOST}
cd ~/nodule_detection
./worker/start_worker.sh
```

### 9.2 URLs

| Servicio | URL |
|----------|-----|
| Frontend | http://localhost:5177 |
| Backend API | http://localhost:9020 |
| API Docs (Swagger) | http://localhost:9020/docs |
| Health check | http://localhost:9020/api/health |
| RabbitMQ Management | http://localhost:15674 |

### 9.3 Monitorización

| Qué | Cómo |
|-----|------|
| Backend health | `curl http://localhost:9020/api/health` |
| Worker logs | `tail -f ~/nodule_detection/worker/worker.log` |
| GPU | `nvidia-smi` en Spark |
| Cola pendiente | RabbitMQ UI → Queues → cxr.inference |
| Docker logs | `docker logs cxr-backend --tail 20` |

---

## 10. BUGS CONOCIDOS Y LIMITACIONES

### 10.1 Funcionales (aceptados para prototipo)
- Sin autenticación (cualquiera puede acceder)
- Sin HTTPS (datos en texto plano)
- Sin integración PACS (upload manual)
- Sin auditoría ENS
- Frontend en modo desarrollo (Vite, no build producción)

### 10.2 Pendientes para producción
- Auth Keycloak OIDC
- HTTPS con certificado hospital
- Integración DICOM C-STORE desde PACS
- Audit trail SHA-256
- Systemd service para worker auto-inicio
- Build producción frontend + nginx sirve dist/

### 10.3 Review de bugs realizado
- 113 issues identificados (37 backend + 34 frontend + 42 worker)
- 11 fixes críticos aplicados
- Documentado en BUGS_REVIEW_COMPLETO.md

---

## 11. ROADMAP FUTURO

```
✅ FASE 1: Detección nódulos NODE21          100%
✅ FASE 4: Prototipo hospital                 85%
  ├─ Sprint 1: Worker Spark GPU               ✅
  ├─ Sprint 2: Backend FastAPI + Docker        ✅
  ├─ Sprint 3: Frontend Vue + Tailwind         ✅
  ├─ Sprint 4: Integración end-to-end          ✅
  ├─ Módulo validación/anotación               ✅
  └─ Auth + HTTPS + PACS                       ⏳

⏳ FASE 2: Tuberculosis (TBX11K)               0%
🔄 FASE 3: Multi-patología VinDr 22 clases     5% (esperando PhysioNet)
📝 FASE 5: Publicación paper                   10%
```

### Próximos pasos inmediatos
1. Auth Keycloak + HTTPS → producción hospital
2. Acceso PhysioNet → entrenar 22 patologías
3. 5-fold cross-validation → resultados robustos para paper
4. Paper comparativo → Nature Scientific Reports

---

## 12. INVENTARIO DE ARCHIVOS

### 12.1 Proyectos

| Proyecto | Ubicación | Función |
|----------|-----------|---------|
| cxr-detection | repos/cxr-detection/ | Backend Docker |
| cxr-frontend | repos/cxr-frontend/ | Frontend Vue |
| nodule_detection | ~/nodule_detection/ (Spark) | Worker + modelos + pipeline |
| RX | repos/RX/ | Documentación + proyecto original |

### 12.2 Documentación

| Archivo | Contenido |
|---------|-----------|
| DOCUMENTACION_DEFINITIVA.md | **Este documento** |
| DOCUMENTACION_PROYECTO.md | Roadmap y resultados detallados |
| DOCUMENTACION_TECNICA.md | Arquitectura y API |
| PROPUESTA_ARQUITECTURA_HOSPITAL.md | Diseño de arquitectura |
| BEHRENDT_BUGS_ANALYSIS.md | 26 bugs código referencia |
| BUGS_REVIEW_COMPLETO.md | 113 issues del prototipo |
| WEIGHTS_AND_CHECKPOINTS.md | Inventario de pesos |

### 12.3 Pesos y checkpoints

| Archivo | NODE21 | Ubicación |
|---------|--------|-----------|
| fastercnn50.pth | — (pretrained) | Spark weights/ |
| best_node21.pth (FRCNN) | 0.9025 | Spark checkpoints/frcnn_corrected/ |
| best.pt (YOLOv8s) | 0.9103 | Spark checkpoints/yolo/yolov8s/ |

---

## 13. REFERENCIAS

1. Behrendt et al. (2023). "A systematic approach to deep learning-based nodule detection in chest radiographs." *Scientific Reports* 13, 10120.
2. NODE21 Challenge. https://node21.grand-challenge.org/
3. VinDr-CXR: Nguyen et al. (2022). *Scientific Data* 9, 429.
4. Solovyev et al. (2021). "Weighted boxes fusion." arXiv:1910.13302.
5. Ultralytics YOLOv8. https://docs.ultralytics.com/

---

## 14. APÉNDICES

### Apéndice A: Configuración Worker (config.yaml)

```yaml
rabbitmq:
  host: "${RABBITMQ_HOST}"
  port: 5672
  user: "cxr_worker"
  password: "${RABBITMQ_PASS}"
  queue_input: "cxr.inference"
  queue_output: "cxr.results"

models:
  frcnn:
    enabled: true
    weights: "checkpoints/frcnn_corrected/best_node21.pth"
    score_thresh: 0.005
  yolov8:
    enabled: true
    weights: "checkpoints/yolo/yolov8s/best.pt"
    conf_thresh: 0.001
    imgsz: 1024

ensemble:
  enabled: true
  weights: [0.90, 0.91]
  iou_thr: 0.2
  skip_box_thr: 0.05
```

### Apéndice B: Variables de entorno (.env)

```env
MYSQL_ROOT_PASSWORD=CHANGE_ME_MYSQL_ROOT_PASS
MYSQL_USER=cxr_app
MYSQL_PASSWORD=CHANGE_ME_MYSQL_PASS
RABBITMQ_USER=cxr_worker
RABBITMQ_PASS=CHANGE_ME_RABBITMQ_PASS
ENVIRONMENT=development
LOG_LEVEL=DEBUG
CORS_ORIGINS=http://localhost:5175,http://localhost:5176,http://localhost:8090
MAX_UPLOAD_SIZE_MB=50
```

### Apéndice C: Formato mensajes RabbitMQ

**cxr.inference (Backend → Spark):**
```json
{"study_id": "CXR-A1B2C3D4E5F6", "image_data": "<base64>", "format": "png"}
```

**cxr.results (Spark → Backend):**
```json
{
  "study_id": "CXR-A1B2C3D4E5F6",
  "status": "completed",
  "num_detections": 2,
  "detections": [{"x1":205,"y1":353,"x2":257,"y2":409,"score":0.827,"label":"nodule"}],
  "model_details": {"frcnn":{"num_detections":37,"time_ms":55},"yolov8":{"num_detections":6,"time_ms":25}},
  "inference_time_ms": 82,
  "annotated_image_base64": "<base64>"
}
```

### Apéndice D: Infraestructura

| Componente | Detalle |
|-----------|---------|
| Servidor local | Windows 11 Pro, Docker Desktop |
| Spark GPU | NVIDIA GB10, 130GB VRAM, Ubuntu |
| SSH Spark | ${SPARK_USER}@${SPARK_HOST} |
| SSH Key | almacenada localmente fuera del repositorio |
| Red | LAN interna del centro |
