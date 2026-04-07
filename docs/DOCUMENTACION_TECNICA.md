# DOCUMENTACIÓN TÉCNICA — Sistema CXR Nodule Detection
# Herramienta de detección de nódulos pulmonares en radiografías de tórax

**Versión**: 1.0.0
**Fecha**: 2026-04-04
**Autores**: Toni (desarrollo), Marc Link Cladera & Antonio Contesti Coll (proyecto base UIB)

---

## 1. ARQUITECTURA DEL SISTEMA

### 1.1 Diagrama general

```
┌──────────────────────────────────────────────────────────────────┐
│                    RED HOSPITAL (LAN)                             │
│                                                                   │
│  ┌─────────────┐    ┌────────────────────────────────────────┐   │
│  │  Radiólogo   │    │     SERVIDOR LOCAL (Docker)            │   │
│  │  (browser)   │───▶│                                        │   │
│  └─────────────┘    │  ┌────────┐  ┌───────────────────────┐│   │
│                      │  │ Nginx  │  │ Frontend Vue 3        ││   │
│                      │  │ :8090  │  │ + Tailwind CSS        ││   │
│                      │  └───┬────┘  │ http://localhost:5177 ││   │
│                      │      │       └───────────────────────┘│   │
│                      │  ┌───┴────────────────┐               │   │
│                      │  │ FastAPI (cxr-svc)   │               │   │
│                      │  │ :9020               │               │   │
│                      │  │ POST /api/cxr/upload│               │   │
│                      │  │ GET  /api/cxr/results│              │   │
│                      │  └───┬────────────────┘               │   │
│                      │      │                                 │   │
│                      │  ┌───┴──────┐  ┌──────────────────┐   │   │
│                      │  │ MySQL    │  │ RabbitMQ         │   │   │
│                      │  │ :3306    │  │ :5672 (AMQP)     │───┼───┐
│                      │  │ int only │  │ :15674 (mgmt)    │   │   │
│                      │  └──────────┘  └──────────────────┘   │   │
│                      │                                        │   │
│                      │  ┌──────────────────────┐              │   │
│                      │  │ Result Consumer       │              │   │
│                      │  │ (RabbitMQ → MySQL)    │              │   │
│                      │  └──────────────────────┘              │   │
│                      └────────────────────────────────────────┘   │
│                                                                   │
│                      ┌────────────────────────────────────────┐   │
│                      │        SPARK (GPU Server)              │◀──┘
│                      │                                        │
│                      │  ┌──────────────────────────────────┐  │
│                      │  │ Inference Worker (Python nativo)  │  │
│                      │  │                                   │  │
│                      │  │ Cola RabbitMQ cxr.inference ──┐   │  │
│                      │  │                                │   │  │
│                      │  │ FRCNN VinDr ─┐                │   │  │
│                      │  │              ├─▶ WBF Ensemble  │   │  │
│                      │  │ YOLOv8s ─────┘                │   │  │
│                      │  │                                │   │  │
│                      │  │ ──▶ Cola RabbitMQ cxr.results  │   │  │
│                      │  └──────────────────────────────────┘  │
│                      │  GPU: NVIDIA GB10 (130GB VRAM)         │
│                      └────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Flujo de datos

```
1. Radiólogo arrastra CXR (PNG/DICOM) al navegador
   ↓
2. Frontend Vue → POST /api/cxr/upload (multipart form)
   ↓
3. Backend FastAPI:
   a. Valida formato y tamaño
   b. Genera study_uid (CXR-XXXXXXXXXXXX)
   c. Guarda imagen original en MySQL (LONGBLOB)
   d. Publica tarea en RabbitMQ cola cxr.inference
      Mensaje: {study_id, image_data (base64), format}
   e. Responde: {study_uid, status: "processing"}
   ↓
4. Spark Worker (GPU):
   a. Consume tarea de cxr.inference
   b. Decodifica imagen base64 → numpy array
   c. Preprocesa: resize 1024×1024, normaliza [0,1]
   d. Inferencia FRCNN (~55ms): grayscale×3 → boxes+scores
   e. Inferencia YOLOv8 (~25ms): BGR → boxes+scores
   f. Ensemble WBF: fusión de boxes ponderada
   g. Publica resultado en cxr.results
      Resultado: {study_id, status, detections, inference_time_ms, annotated_image_base64}
   h. Limpia GPU cache
   ↓
5. Result Consumer (Docker):
   a. Consume resultado de cxr.results
   b. Actualiza estudio en MySQL (status, detecciones, imagen anotada)
   ↓
6. Frontend (polling cada 1.5s):
   a. GET /api/cxr/results/{study_uid}
   b. Cuando status="completed": muestra imagen + overlay SVG
   c. Slider de threshold filtra detecciones en tiempo real
   ↓
7. Radiólogo revisa y valida resultado

Latencia total: ~2-3 segundos (upload + inferencia + resultado)
```

### 1.3 Componentes

| Componente | Tecnología | Ubicación | Puerto | Función |
|-----------|-----------|-----------|--------|---------|
| Frontend | Vue 3.5 + Tailwind 4.2 + Vite 7 | cxr-frontend/ (local) | 5177 | UI para radiólogos |
| Backend API | FastAPI 0.115 + Uvicorn | Docker (cxr-svc) | 9020 | REST API |
| Result Consumer | Python async (aio-pika) | Docker (result-consumer) | — | RabbitMQ → MySQL |
| MySQL | MySQL 8.0 | Docker (cxr-mysql) | 3306 (int) | Almacenamiento |
| RabbitMQ | RabbitMQ 3.13 | Docker (cxr-rabbitmq) | 5672 / 15674 | Cola de mensajes |
| Nginx | Nginx 1.25 | Docker (cxr-nginx) | 8090 | Reverse proxy |
| Inference Worker | Python + PyTorch + Ultralytics | Spark (nativo) | — | Inferencia GPU |

---

## 2. MODELOS DE IA

### 2.1 Faster R-CNN con VinDr-CXR

| Parámetro | Valor |
|-----------|-------|
| Backbone | ResNet50 + Feature Pyramid Network (FPN) |
| Pretraining | VinDr-CXR (14 patologías radiológicas) |
| Clases | 2 (fondo + nódulo) |
| Entrada | 3 canales (grayscale replicado) |
| Congelación | Layers 1-3 frozen, layer4 + FPN trainable |
| Score threshold | 0.005 (evaluación) / 0.01 (producción) |
| NMS threshold | 0.2 |
| Parámetros | 41.3M total, 32.8M trainable |
| Inferencia | ~55ms/imagen |
| NODE21 Score | 0.9025 (sin data leakage) |

**Carga de pesos VinDr:**
```python
# 1. Crear modelo vacío
model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
# 2. Cargar pesos VinDr con mapeo secuencial de keys
vindr_state = torch.load("fastercnn50.pth")
# 3. Eliminar head viejo (VinDr tenía 15 clases)
# 4. Nuevo head: FastRCNNPredictor(in_features, 2)
# 5. Congelar layers 1-3, entrenar layer4 + FPN
```

### 2.2 YOLOv8s

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | YOLOv8 Small (anchor-free) |
| Pretraining | COCO |
| Resolución | 1024×1024 |
| Inferencia | ~25ms/imagen |
| NODE21 Score | 0.9103 |
| Augmentación | Solo flip horizontal (sin mosaic/mixup/color) |

### 2.3 Ensemble WBF (Weighted Box Fusion)

| Parámetro | Valor |
|-----------|-------|
| Modelos | FRCNN (weight=0.90) + YOLOv8 (weight=0.91) |
| IoU threshold | 0.2 |
| Skip box threshold | 0.05 |
| NODE21 Score | **0.9391** |
| Método | Media ponderada de coordenadas por scores |

**Cómo funciona WBF:**
A diferencia de NMS (que elimina boxes solapadas), WBF **fusiona** boxes de múltiples modelos calculando la media ponderada de sus coordenadas, usando los confidence scores como pesos. Resultado: boxes más precisas que cualquiera de los modelos individuales.

---

## 3. BACKEND (FastAPI + Docker)

### 3.1 Estructura

```
cxr-detection/
├── docker-compose.yml          # 5 servicios
├── .env                        # Credenciales (dev)
├── mysql/init/01_create_db.sql # Schema MySQL
├── rabbitmq/rabbitmq.conf      # Config RabbitMQ
├── nginx/nginx.conf            # Reverse proxy
├── shared/messaging.py         # Helpers RabbitMQ
└── services/cxr-svc/
    ├── Dockerfile              # Python 3.12 slim, non-root
    ├── requirements.txt        # 10 dependencias
    └── app/
        ├── main.py             # FastAPI app + lifespan
        ├── config.py           # Settings desde env vars
        ├── database.py         # SQLAlchemy async + pool
        ├── models/models.py    # 4 tablas: studies, detections, images×2
        ├── schemas/schemas.py  # Pydantic models
        ├── routers/
        │   ├── upload.py       # POST /api/cxr/upload
        │   ├── results.py      # GET results + DELETE + original image
        │   ├── history.py      # GET history + stats + search
        │   └── health.py       # GET health (API + MySQL + RabbitMQ)
        └── consumers/
            └── result_consumer.py  # RabbitMQ cxr.results → MySQL
```

### 3.2 API Endpoints

| Método | Endpoint | Función |
|--------|----------|---------|
| POST | `/api/cxr/upload` | Subir CXR (multipart), encolar en RabbitMQ |
| GET | `/api/cxr/results/{uid}` | Obtener resultado + detecciones |
| GET | `/api/cxr/results/{uid}/original` | Imagen original (PNG) |
| GET | `/api/cxr/results/{uid}/image` | Imagen anotada (PNG) |
| DELETE | `/api/cxr/results/{uid}` | Eliminar estudio |
| DELETE | `/api/cxr/all` | Eliminar todos los estudios |
| GET | `/api/cxr/history` | Historial (paginado, filtrable, buscable) |
| GET | `/api/cxr/stats` | Estadísticas (total, analizados, con nódulos, tiempo medio) |
| GET | `/api/health` | Estado de API + MySQL + RabbitMQ |

### 3.3 Modelo de datos MySQL

```sql
cxr_studies          -- Estudio principal
├── id, study_uid    -- Identificador único (CXR-XXXXXXXXXXXX)
├── patient_id       -- ID paciente (opcional)
├── status           -- queued → processing → completed/error
├── num_detections   -- Nódulos detectados
├── inference_time_ms-- Tiempo de inferencia
├── created_at       -- Fecha/hora upload
└── completed_at     -- Fecha/hora resultado

cxr_detections       -- Detecciones individuales
├── study_id (FK)    -- Referencia al estudio
├── x1, y1, x2, y2  -- Bounding box (píxeles)
├── score            -- Confidence (0-1)
├── label            -- "nodule"
└── model_source     -- "ensemble"

cxr_original_images  -- Imagen CXR original
├── study_id (FK)    -- 1:1 con estudio
└── image_data       -- LONGBLOB

cxr_annotated_images -- Imagen con boxes dibujados
├── study_id (FK)    -- 1:1 con estudio
└── image_data       -- LONGBLOB
```

### 3.4 Docker Compose

| Servicio | Imagen | Puertos | Red |
|----------|--------|---------|-----|
| cxr-mysql | mysql:8.0 | — (int only) | backend_net |
| cxr-rabbitmq | rabbitmq:3.13-management | 5672, 15674 | frontend + backend |
| cxr-backend | python:3.12-slim | 9020 | frontend + backend |
| cxr-result-consumer | python:3.12-slim | — | backend_net |
| cxr-nginx | nginx:1.25-alpine | 8090 | frontend_net |

### 3.5 Seguridad aplicada

- Contenedores read-only con tmpfs
- Usuario non-root (appuser UID 1000)
- `no-new-privileges:true`
- MySQL en red interna (no accesible desde fuera)
- CORS restrictivo (solo métodos necesarios)
- Healthchecks en todos los servicios
- Connection pool con pre_ping y recycle

---

## 4. FRONTEND (Vue 3 + Tailwind CSS)

### 4.1 Estructura

```
cxr-frontend/
├── package.json                # Vue 3.5, Tailwind 4.2, Vite 7
├── vite.config.js              # Proxy /api → :9020
└── src/
    ├── App.vue                 # Header + router-view
    ├── router/index.js         # / (Analizar) + /history
    ├── lib/api.js              # API client con error handling
    ├── composables/
    │   ├── useUpload.js        # Upload + polling + cleanup
    │   └── useDetections.js    # Filtrado por threshold
    ├── components/
    │   ├── DropZone.vue        # Drag & drop
    │   ├── CxrViewer.vue       # Visor con zoom/pan + SVG overlay
    │   ├── ThresholdSlider.vue # Slider 5%-99%
    │   ├── DetectionList.vue   # Lista detecciones con riesgo
    │   ├── StudyStatus.vue     # Badge estado
    │   └── StatsBar.vue        # Contadores
    └── views/
        ├── AnalyzeView.vue     # Upload + resultado
        └── HistoryView.vue     # Historial con búsqueda
```

### 4.2 Funcionalidades

| Feature | Implementación |
|---------|---------------|
| Upload | Drag & drop o click, validación formato + tamaño (50MB max) |
| Visor CXR | Zoom rueda ratón + pan arrastrar + botones +/-/Fit/1:1 |
| Bounding boxes | SVG overlay dinámico, colores por riesgo (rojo/naranja/cyan) |
| Threshold slider | Filtra detecciones en tiempo real (5%-99%) |
| Detecciones | Lista con score, riesgo (Alto/Medio/Bajo), coordenadas, tamaño |
| Polling | Cada 1.5s, max 10 errores, cleanup en unmount |
| Historial | Búsqueda (Enter), fecha/hora, slider por estudio, eliminar |
| Stats | Total, analizados, con nódulos, tiempo medio inferencia |
| Zoom real | 100% = 1px imagen = 1px pantalla |

### 4.3 Gestión de memoria

- `URL.revokeObjectURL()` al cambiar imagen o desmontar
- Polling timer limpiado en `onUnmounted()`
- Max 10 errores de polling antes de parar
- No hay memory leaks conocidos

---

## 5. INFERENCE WORKER (Spark GPU)

### 5.1 Estructura

```
~/nodule_detection/worker/
├── inference_worker.py     # Worker principal (~500 líneas)
├── config.yaml             # Configuración RabbitMQ + modelos
├── start_worker.sh         # Script de inicio con checks
├── cxr-worker.service      # Systemd unit (para auto-inicio)
└── worker.log              # Log con rotación (10MB × 5)
```

### 5.2 Características de producción

| Feature | Implementación |
|---------|---------------|
| Reconexión RabbitMQ | Exponential backoff (5s → 60s max), bucle infinito |
| Limpieza GPU | `torch.cuda.empty_cache()` tras cada task |
| Log rotation | RotatingFileHandler 10MB × 5 archivos |
| Graceful shutdown | Flag `_shutdown`, limpieza modelos + GPU |
| Error handling | JSON malformado → drop, error de inferencia → error result |
| Config path | CWD primero, luego relativo al script |
| GPU check | Verificación CUDA en start_worker.sh |
| Prefetch | 1 task a la vez (GPU exclusiva) |

### 5.3 Rendimiento

| Métrica | Valor |
|---------|-------|
| Carga de modelos | ~2s (una vez al inicio) |
| Warmup GPU | ~0.5s |
| FRCNN inferencia | ~55ms |
| YOLOv8 inferencia | ~25ms |
| WBF ensemble | ~1ms |
| **Total por imagen** | **~81ms** (post-warmup) |
| Decode base64 + preprocess | ~20ms |
| **Latencia end-to-end** | **~2-3s** (incluyendo red) |

---

## 6. RESULTADOS CIENTÍFICOS

### 6.1 Dataset NODE21

- 4,882 radiografías CXR (1024×1024 px)
- 1,134 con nódulos (1,476 anotaciones bbox)
- 3,748 sin nódulos (23% prevalencia)
- Fuentes: JSRT, PadChest, ChestX-ray14, Open-I

### 6.2 Métricas

| Métrica | Descripción |
|---------|-------------|
| NODE21 Score | Media de sensibilidad a FP/img = [0.25, 0.5, 1, 2, 4, 8] |
| AUROC | Área bajo curva ROC a nivel de imagen |
| CM | Competition Metric = 0.75 × AUROC + 0.25 × NODE21 |
| IoU threshold | 0.2 (oficial NODE21) |

### 6.3 Resultados (scores honestos, sin data leakage)

| # | Modelo | NODE21 | AUROC | CM | S@0.25FP |
|---|--------|--------|-------|----|----------|
| **1** | **Ensemble WBF** | **0.9391** | **0.9683** | **0.9447** | **0.874** |
| 2 | YOLOv8s | 0.9103 | 0.9686 | 0.9283 | 0.821 |
| 3 | FRCNN VinDr | 0.9025 | 0.9460 | 0.9146 | 0.821 |

### 6.4 Hallazgos clave

1. **Data leakage**: Descubierto que augmentaciones offline de la misma imagen en train y val inflaban scores en +6.7 puntos. Corregido con StratifiedGroupKFold agrupando por imagen base.

2. **Score threshold**: El threshold por defecto de Faster R-CNN (0.5) es demasiado alto para evaluación FROC. Bajarlo a 0.005 mejoró NODE21 de 0.45 a 0.85.

3. **VinDr pretraining**: Preentrenar en dominio radiológico (VinDr-CXR) es clave. COCO pretraining rinde significativamente peor.

4. **WBF vs individual**: El ensemble WBF gana +2.6 puntos NODE21 sobre el mejor individual, especialmente en bajo FP (clínicamente relevante).

### 6.5 Comparación con referencia

| Resultado | NODE21 | CM | Notas |
|-----------|--------|----|-------|
| **Nuestro ensemble** | **0.9391** | **0.9447** | 2 modelos |
| Behrendt (ganador NODE21) | — | 0.8390 | 21 modelos, 5-fold |

---

## 7. DESPLIEGUE

### 7.1 Requisitos

**Servidor local (Docker):**
- Docker Desktop con Docker Compose
- 4GB RAM mínimo (8GB recomendado)
- 10GB disco (MySQL + imágenes)
- Red LAN con visibilidad a la Spark

**Spark (GPU):**
- NVIDIA GPU con CUDA (GB10 con 130GB VRAM actual)
- Python 3.11+ con venv
- PyTorch 2.x + Ultralytics + pika
- Acceso a RabbitMQ del servidor local por LAN (port 5672)

### 7.2 Cómo arrancar

**Servidor Docker (local):**
```bash
cd /path/to/cxr-detection
docker compose up -d
# Verificar: curl http://localhost:9020/api/health
```

**Frontend (desarrollo):**
```bash
cd /path/to/cxr-frontend
npm run dev
# Abrir: http://localhost:5177
```

**Spark Worker:**
```bash
ssh ${SPARK_USER}@${SPARK_HOST}
cd ~/nodule_detection
./worker/start_worker.sh
# O en background:
nohup ./worker/start_worker.sh > worker/worker_live.log 2>&1 &
```

### 7.3 Monitorización

| Qué | Cómo |
|-----|------|
| Backend health | `GET http://localhost:9020/api/health` |
| RabbitMQ UI | http://localhost:15674 (cxr_worker / `$RABBITMQ_PASS`) |
| Worker logs | `tail -f ~/nodule_detection/worker/worker_live.log` |
| GPU | `nvidia-smi` en la Spark |
| Cola pendiente | RabbitMQ UI → Queues → cxr.inference (messages ready) |
| Docker logs | `docker logs cxr-backend --tail 20` |

---

## 8. ARCHIVOS DEL PROYECTO

### 8.1 Documentación

| Archivo | Contenido |
|---------|-----------|
| DOCUMENTACION_PROYECTO.md | Estado del proyecto, resultados, roadmap |
| DOCUMENTACION_TECNICA.md | Este documento — arquitectura y API |
| BEHRENDT_BUGS_ANALYSIS.md | 26 bugs en código de referencia |
| BUGS_REVIEW_COMPLETO.md | 113 issues del prototipo revisados |
| PROPUESTA_ARQUITECTURA_HOSPITAL.md | Diseño de la arquitectura |
| WEIGHTS_AND_CHECKPOINTS.md | Inventario de pesos con instrucciones carga |

### 8.2 Código

| Directorio | Contenido |
|-----------|-----------|
| cxr-detection/ | Backend FastAPI + Docker Compose |
| cxr-frontend/ | Frontend Vue 3 + Tailwind |
| spark/worker/ | Worker de inferencia (copia local) |
| spark/scripts/ | Scripts de entrenamiento (copia local) |
| spark/pipeline/ | Pipeline reutilizable (copia local) |

### 8.3 Spark

| Directorio | Contenido |
|-----------|-----------|
| ~/nodule_detection/worker/ | Inference worker + config |
| ~/nodule_detection/checkpoints/ | Modelos entrenados |
| ~/nodule_detection/weights/ | Pesos VinDr pretrained |
| ~/nodule_detection/pipeline/ | Pipeline reutilizable |
| ~/nodule_detection/scripts/ | Scripts de entrenamiento |
| ~/nodule_detection/data/ | Dataset NODE21 + PNGs |

---

## APÉNDICES

### Apéndice A: Configuración del Worker (config.yaml)

```yaml
rabbitmq:
  host: "${RABBITMQ_HOST}"
  port: 5672
  user: "cxr_worker"
  password: "${RABBITMQ_PASS}"
  queue_input: "cxr.inference"
  queue_output: "cxr.results"
  prefetch_count: 1
  heartbeat: 600

models:
  frcnn:
    enabled: true
    weights: "checkpoints/frcnn_corrected/best_node21.pth"
    score_thresh: 0.005
    nms_thresh: 0.2
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

inference:
  device: "cuda:0"
  input_size: 1024
  output_confidence_thresh: 0.01
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

### Apéndice C: Formato de mensajes RabbitMQ

**Cola cxr.inference (Backend → Spark):**
```json
{
  "study_id": "CXR-A0A9D41ACD46",
  "image_data": "<base64 encoded PNG>",
  "format": "png",
  "timestamp": "2026-04-04T08:00:00Z"
}
```

**Cola cxr.results (Spark → Backend):**
```json
{
  "study_id": "CXR-A0A9D41ACD46",
  "status": "completed",
  "num_detections": 2,
  "detections": [
    {"x1": 205.0, "y1": 353.5, "x2": 256.6, "y2": 408.9, "score": 0.827, "label": "nodule"},
    {"x1": 650.2, "y1": 420.1, "x2": 710.8, "y2": 475.3, "score": 0.412, "label": "nodule"}
  ],
  "model_details": {
    "frcnn": {"num_detections": 37, "time_ms": 54.7},
    "yolov8": {"num_detections": 6, "time_ms": 25.3}
  },
  "inference_time_ms": 82.0,
  "ensemble": true,
  "confidence_threshold": 0.01,
  "annotated_image_base64": "<base64 encoded PNG with boxes>"
}
```

### Apéndice D: Referencias

1. Behrendt et al. (2023). Nature Scientific Reports 13, 10120.
2. NODE21 Challenge: https://node21.grand-challenge.org/
3. VinDr-CXR: Nguyen et al. (2022). Scientific Data 9, 429.
4. YOLO26: Ultralytics (2026). arXiv:2509.25164.
5. Weighted Box Fusion: Solovyev et al. (2021). arXiv:1910.13302.
