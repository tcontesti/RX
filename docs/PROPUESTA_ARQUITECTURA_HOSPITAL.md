# PROPUESTA DE ARQUITECTURA — Herramienta de Detección CXR para Hospital
# Prototipo funcional para pruebas en real

---

## 1. VISIÓN GENERAL

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RED HOSPITAL (LAN)                          │
│                                                                    │
│  ┌──────────────┐     ┌─────────────────────────────────────┐     │
│  │  Modalidad   │     │   SERVIDOR LOCAL (Docker)            │     │
│  │  RX / PACS   │     │                                     │     │
│  │              │────▶│  ┌─────────┐  ┌──────────────────┐  │     │
│  └──────────────┘     │  │  Nginx  │  │  Frontend Vue 3  │  │     │
│                       │  │  proxy  │  │  + Tailwind CSS   │  │     │
│  ┌──────────────┐     │  └────┬────┘  └──────────────────┘  │     │
│  │  Radiólogo   │     │       │                              │     │
│  │  (browser)   │────▶│  ┌────┴────────────────────────┐    │     │
│  └──────────────┘     │  │   Backend FastAPI (cxr-svc)  │    │     │
│                       │  │   - Upload CXR               │    │     │
│                       │  │   - Gestión pacientes        │    │     │
│                       │  │   - Resultados               │    │     │
│                       │  │   - Cola RabbitMQ ──────────────────┐  │
│                       │  └────┬─────────────────────────┘ │  │  │
│                       │       │                            │  │  │
│                       │  ┌────┴──────┐  ┌──────────┐      │  │  │
│                       │  │  MySQL    │  │ RabbitMQ │      │  │  │
│                       │  │  results  │  │  cola    │      │  │  │
│                       │  └───────────┘  └──────────┘      │  │  │
│                       └─────────────────────────────────────┘  │  │
│                                                                │  │
│                       ┌────────────────────────────────────────┘  │
│                       │                                           │
│                       ▼                                           │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              SPARK (GPU Server)                       │        │
│  │                                                       │        │
│  │  ┌──────────────────────────────────────────────┐    │        │
│  │  │   Inference Worker (Python, NO Docker)        │    │        │
│  │  │                                               │    │        │
│  │  │   - Consume de RabbitMQ                       │    │        │
│  │  │   - Carga modelos en GPU (1 vez al inicio)    │    │        │
│  │  │   - FRCNN VinDr + YOLOv8s → WBF Ensemble     │    │        │
│  │  │   - Publica resultados a RabbitMQ             │    │        │
│  │  │   - GPU: NVIDIA GB10                          │    │        │
│  │  │                                               │    │        │
│  │  │   Modelos cargados:                           │    │        │
│  │  │   ├── FRCNN Corrected-B (NODE21=0.9025)      │    │        │
│  │  │   ├── YOLOv8s (NODE21=0.9103)                │    │        │
│  │  │   └── WBF Ensemble (NODE21=0.9391)           │    │        │
│  │  └──────────────────────────────────────────────┘    │        │
│  └──────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. POR QUÉ ESTA ARQUITECTURA

### ¿Por qué NO Docker en la Spark?

Docker añade overhead de GPU (NVIDIA Container Toolkit, driver mapping) y complica el acceso directo a los modelos/pesos que ya están instalados. Un worker Python nativo en la Spark es:
- Más simple de mantener
- Acceso directo a GPU sin overhead
- Los modelos ya están cargados en ~/nodule_detection/
- Más rápido (~16ms/img vs ~25ms con Docker GPU passthrough)

### ¿Por qué RabbitMQ en vez de HTTP directo?

1. **Desacoplamiento**: El backend no necesita esperar la respuesta de la Spark (async)
2. **Cola de trabajo**: Si llegan 10 CXR a la vez, se encolan y procesan una a una (la GPU solo puede hacer una a la vez)
3. **Resiliencia**: Si la Spark se reinicia, las tareas no se pierden (mensajes persistentes)
4. **Ya lo usas**: Tu infraestructura Docker ya tiene RabbitMQ en worklistsrv-backend
5. **Escalabilidad**: En futuro, puedes añadir otra Spark y ambas consumen de la misma cola

### ¿Por qué MySQL para resultados?

- Consistente con tu stack actual (worklistsrv usa MySQL)
- Almacena resultados de detección vinculados a pacientes/estudios
- Permite queries históricas ("¿cuántos nódulos detectados este mes?")
- Auditoría (ENS compliance, como ya tienes en worklistsrv)

---

## 3. COMPONENTES DETALLADOS

### 3.1 Frontend (Vue 3 + Tailwind) — Servidor local Docker

Reutilizar la estructura de worklistsrv:

```
cxr-frontend/
├── src/
│   ├── views/
│   │   ├── UploadView.vue        # Subir CXR (drag & drop, DICOM/PNG)
│   │   ├── ResultsView.vue       # Ver resultados con imagen anotada
│   │   ├── HistoryView.vue       # Historial de análisis
│   │   └── DashboardView.vue     # Estadísticas (nódulos/día, sensibilidad)
│   ├── components/
│   │   ├── CxrViewer.vue         # Visor de radiografías con zoom/contraste
│   │   ├── DetectionOverlay.vue  # Overlay de bounding boxes sobre imagen
│   │   ├── ResultCard.vue        # Tarjeta con resultado por detección
│   │   └── StatusBadge.vue       # Estado: procesando/completado/error
│   ├── composables/
│   │   ├── useUpload.js          # Lógica de upload
│   │   └── useResults.js         # Polling/WebSocket de resultados
│   └── lib/
│       └── api.js                # API client (reutilizar de worklistsrv)
```

**Funcionalidad clave**:
- Drag & drop de CXR (DICOM o PNG)
- Visualización del resultado con boxes superpuestos
- Ajuste de contraste/brillo en el visor (windowing DICOM)
- Score de confianza por detección
- Indicador de qué modelo detectó qué (FRCNN/YOLO/ambos)
- Historial por paciente

### 3.2 Backend FastAPI (cxr-svc) — Servidor local Docker

```
cxr-backend/
├── docker-compose.yml
├── services/
│   └── cxr-svc/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── main.py
│           ├── config.py
│           ├── routers/
│           │   ├── upload.py        # POST /api/cxr/upload
│           │   ├── results.py       # GET /api/cxr/results/{id}
│           │   ├── history.py       # GET /api/cxr/history
│           │   └── health.py        # GET /api/health
│           ├── models/
│           │   └── models.py        # SQLAlchemy: CxrStudy, Detection
│           ├── services/
│           │   ├── queue.py         # RabbitMQ producer
│           │   └── storage.py       # Guardar imágenes
│           └── schemas/
│               └── schemas.py       # Pydantic models
├── mysql/
│   └── init/
│       └── 01_create_db.sql
├── rabbitmq/
│   └── rabbitmq.conf
└── nginx/
    └── nginx.conf
```

**API Endpoints**:

```
POST   /api/cxr/upload              # Subir CXR → encola en RabbitMQ → devuelve study_id
GET    /api/cxr/results/{study_id}  # Obtener resultado (polling o WebSocket)
GET    /api/cxr/history             # Historial con filtros y paginación
GET    /api/cxr/stats               # Estadísticas (nódulos/día, modelo performance)
GET    /api/health                  # Health check (incluye estado de Spark)
WS     /api/cxr/ws/{study_id}       # WebSocket para resultado en tiempo real
```

**Flujo de upload**:
```python
@router.post("/upload")
async def upload_cxr(file: UploadFile, patient_id: str = None):
    # 1. Guardar imagen en disco
    path = save_image(file)
    
    # 2. Crear registro en MySQL (estado: "processing")
    study = CxrStudy(patient_id=patient_id, image_path=path, status="processing")
    db.add(study)
    
    # 3. Publicar tarea en RabbitMQ
    await publish_task(queue="cxr.inference", body={
        "study_id": study.id,
        "image_path": path,  # ruta accesible por la Spark (NFS/shared mount)
    })
    
    return {"study_id": study.id, "status": "processing"}
```

**Modelo de datos MySQL**:
```sql
CREATE TABLE cxr_studies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(50),
    image_path VARCHAR(500) NOT NULL,
    status ENUM('processing', 'completed', 'error') DEFAULT 'processing',
    node21_score FLOAT,
    num_detections INT DEFAULT 0,
    inference_time_ms FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

CREATE TABLE cxr_detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    study_id INT NOT NULL,
    x1 FLOAT, y1 FLOAT, x2 FLOAT, y2 FLOAT,
    score FLOAT,
    model_source VARCHAR(20),  -- 'frcnn', 'yolov8', 'ensemble'
    label VARCHAR(50) DEFAULT 'nodule',
    FOREIGN KEY (study_id) REFERENCES cxr_studies(id)
);
```

### 3.3 Inference Worker (Spark) — SIN Docker

```
~/nodule_detection/
├── worker/
│   ├── inference_worker.py    # Worker principal
│   ├── config.yaml            # Configuración
│   └── start_worker.sh        # Script de inicio
├── pipeline/                  # Ya existe (builders, dataset, metrics, ensemble)
├── checkpoints/               # Ya existe (todos los pesos)
└── weights/                   # Ya existe (VinDr pretrained)
```

**inference_worker.py** — Worker que consume de RabbitMQ:

```python
"""
Inference Worker — Corre en la Spark con GPU.
Consume tareas de RabbitMQ, ejecuta inferencia, publica resultados.

Uso: python inference_worker.py --config config.yaml
"""

import pika
import json
import torch
import time
import logging
from pipeline.models.frcnn_builder import build_frcnn
from pipeline.utils.ensemble import ensemble_wbf
from ultralytics import YOLO

# ==========================================
# 1. CARGAR MODELOS (una sola vez al inicio)
# ==========================================
device = torch.device("cuda:0")

# FRCNN
frcnn_model = build_frcnn(config, device)
frcnn_model.load_state_dict(torch.load("checkpoints/frcnn_corrected/best_node21.pth"))
frcnn_model.eval()
frcnn_model.roi_heads.score_thresh = 0.005

# YOLOv8
yolo_model = YOLO("checkpoints/yolo/yolov8s/best.pt")

print("Models loaded. Waiting for tasks...")

# ==========================================
# 2. CONECTAR A RABBITMQ
# ==========================================
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='servidor-local.hospital.lan',  # IP del servidor Docker
        port=5672,
        credentials=pika.PlainCredentials('cxr_worker', 'password')
    )
)
channel = connection.channel()
channel.queue_declare(queue='cxr.inference', durable=True)
channel.queue_declare(queue='cxr.results', durable=True)

# ==========================================
# 3. PROCESAR TAREAS
# ==========================================
def process_task(ch, method, properties, body):
    task = json.loads(body)
    study_id = task['study_id']
    image_path = task['image_path']
    
    try:
        start = time.time()
        
        # Cargar imagen
        img = load_and_preprocess(image_path)
        
        # Inferencia FRCNN
        frcnn_preds = infer_frcnn(frcnn_model, img, device)
        
        # Inferencia YOLO
        yolo_preds = infer_yolo(yolo_model, image_path)
        
        # Ensemble WBF
        ensemble_result = ensemble_wbf(
            [frcnn_preds, yolo_preds],
            weights=[0.90, 0.91],
            iou_thr=0.2
        )
        
        elapsed = (time.time() - start) * 1000
        
        # Publicar resultado
        result = {
            'study_id': study_id,
            'status': 'completed',
            'detections': format_detections(ensemble_result),
            'num_detections': len(ensemble_result['boxes']),
            'inference_time_ms': elapsed,
            'node21_score': calculate_confidence(ensemble_result)
        }
        
        channel.basic_publish(
            exchange='',
            routing_key='cxr.results',
            body=json.dumps(result),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        # Error → publicar error
        channel.basic_publish(
            exchange='',
            routing_key='cxr.results',
            body=json.dumps({
                'study_id': study_id,
                'status': 'error',
                'error': str(e)
            })
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)  # 1 tarea a la vez (GPU)
channel.basic_consume(queue='cxr.inference', on_message_callback=process_task)
channel.start_consuming()
```

### 3.4 Docker Compose (servidor local)

```yaml
version: "3.8"

services:
  # === INFRAESTRUCTURA ===
  mysql:
    image: mysql:8.0
    container_name: cxr-mysql
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: cxr_detection
      MYSQL_USER: cxr_app
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./mysql/init:/docker-entrypoint-initdb.d
    networks:
      - backend_net
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  rabbitmq:
    image: rabbitmq:3.13-management
    container_name: cxr-rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER}
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASS}
    ports:
      - "5672:5672"      # AMQP — abierto para que la Spark conecte
      - "15672:15672"    # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - backend_net
      - frontend_net     # Accesible desde la Spark

  # === BACKEND ===
  cxr-svc:
    build:
      context: .
      dockerfile: services/cxr-svc/Dockerfile
    container_name: cxr-backend
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_DATABASE=cxr_detection
      - MYSQL_USER=cxr_app
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - RABBITMQ_HOST=rabbitmq
      - RABBITMQ_PORT=5672
      - RABBITMQ_USER=${RABBITMQ_USER}
      - RABBITMQ_PASS=${RABBITMQ_PASS}
      - SHARED_STORAGE=/shared/images
    volumes:
      - shared_images:/shared/images    # Compartido con Spark via NFS
    ports:
      - "9020:8000"
    depends_on:
      mysql:
        condition: service_healthy
      rabbitmq:
        condition: service_started
    networks:
      - backend_net
      - frontend_net

  # === RESULT CONSUMER ===
  result-consumer:
    build:
      context: .
      dockerfile: services/cxr-svc/Dockerfile
    container_name: cxr-result-consumer
    command: ["python", "-m", "app.consumers.result_consumer"]
    environment:
      - MYSQL_HOST=mysql
      - RABBITMQ_HOST=rabbitmq
      # ... (mismas vars que cxr-svc)
    depends_on:
      - mysql
      - rabbitmq
    networks:
      - backend_net

  # === FRONTEND ===
  nginx:
    image: nginx:1.25-alpine
    container_name: cxr-nginx
    ports:
      - "8080:80"
      - "8443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend/dist:/usr/share/nginx/html:ro    # Build estático de Vue
    depends_on:
      - cxr-svc
    networks:
      - frontend_net

volumes:
  mysql_data:
  rabbitmq_data:
  shared_images:    # Mount NFS para compartir con Spark

networks:
  frontend_net:
    driver: bridge
  backend_net:
    driver: bridge
    internal: true
```

---

## 4. COMUNICACIÓN SERVIDOR LOCAL ↔ SPARK

### Opción A: RabbitMQ directo (RECOMENDADA)

```
Servidor Docker (RabbitMQ port 5672 abierto en LAN)
        ↕
Spark Worker (conecta a RabbitMQ del servidor)
```

- La Spark consume de `cxr.inference` y publica en `cxr.results`
- El servidor tiene un consumer (`result-consumer`) que lee `cxr.results` y actualiza MySQL
- **Ventaja**: Simple, robusto, ya lo conoces
- **Requisito**: La Spark necesita ver el puerto 5672 del servidor (misma LAN)

### Opción B: Imágenes compartidas

Para que la Spark acceda a las imágenes subidas al servidor:

**Opción B1: NFS mount** (más simple)
```bash
# En el servidor Docker: exportar /shared/images vía NFS
# En la Spark: mount servidor:/shared/images ~/nodule_detection/inference_input/
```

**Opción B2: Base64 en el mensaje RabbitMQ** (sin NFS)
```python
# El backend codifica la imagen en base64 y la mete en el mensaje
task = {
    "study_id": 123,
    "image_base64": base64.b64encode(image_bytes).decode(),
    "format": "png"
}
```
- **Ventaja**: No necesita NFS/shared mount
- **Desventaja**: Mensajes grandes (~1-2MB por CXR)
- **RabbitMQ soporta** hasta 128MB por mensaje, así que funciona

**RECOMENDACIÓN**: Base64 en el mensaje para el prototipo (más simple, sin NFS). NFS para producción con volumen alto.

---

## 5. FLUJO COMPLETO DE UN CASO

```
1. Radiólogo sube CXR en el navegador (drag & drop)
        ↓
2. Frontend → POST /api/cxr/upload (imagen + patient_id)
        ↓
3. Backend (cxr-svc):
   a. Guarda imagen en disco
   b. Crea registro MySQL (status: "processing")
   c. Publica tarea en RabbitMQ (cola: cxr.inference)
   d. Responde: { study_id: 42, status: "processing" }
        ↓
4. Frontend muestra spinner + polling GET /api/cxr/results/42
        ↓
5. Spark Worker:
   a. Consume tarea de RabbitMQ
   b. Decodifica imagen (base64 o lee de NFS)
   c. Inferencia FRCNN (~35ms) + YOLOv8 (~16ms)
   d. Ensemble WBF
   e. Publica resultado en RabbitMQ (cola: cxr.results)
        ↓
6. Result Consumer (servidor Docker):
   a. Lee resultado de RabbitMQ
   b. Actualiza MySQL (status: "completed", detecciones, scores)
   c. Genera imagen anotada con boxes
        ↓
7. Frontend (siguiente polling):
   a. GET /api/cxr/results/42 → status: "completed"
   b. Muestra imagen con detecciones superpuestas
   c. Muestra scores, modelo fuente, confianza
        ↓
8. Radiólogo revisa y valida

Tiempo total: ~2-5 segundos (incluyendo red + processing)
```

---

## 6. PARA PRODUCCIÓN (HOSPITAL REAL)

### Misma arquitectura, cambios mínimos:

| Componente | Prototipo | Producción |
|-----------|-----------|------------|
| Frontend | localhost:8080 | HTTPS con certificado hospital |
| Backend | Docker local | Docker en servidor hospital |
| RabbitMQ | Docker local | Docker en servidor hospital |
| MySQL | Docker local | Docker o MySQL hospital existente |
| Spark | Spark dev (laboratorio) | **Spark dedicada para producción** |
| Auth | Sin auth (prototipo) | Keycloak OIDC (como worklistsrv) |
| Imágenes | Base64 en RabbitMQ | NFS mount compartido |
| Auditoría | Sin auditoría | audit-svc (reutilizar worklistsrv) |
| DICOM | Upload manual | Integración directa con PACS |

### Integración PACS futura:
```
PACS → DICOM C-STORE → dicom-svc → cxr-svc → RabbitMQ → Spark
```
Reutilizar dicom-svc de worklistsrv para recibir imágenes directamente del PACS.

---

## 7. TECNOLOGÍAS FINALES

| Capa | Tecnología | Motivo |
|------|-----------|--------|
| **Frontend** | Vue 3 + Tailwind CSS | Ya lo usas en worklistsrv, reutilizable |
| **Backend API** | FastAPI + Uvicorn | Async, rápido, ya lo conoces |
| **Base de datos** | MySQL 8.0 | Consistente con tu stack |
| **Cola de mensajes** | RabbitMQ 3.13 | Ya lo usas, perfecto para desacoplar GPU |
| **Proxy** | Nginx | Ya lo tienes configurado |
| **Inferencia GPU** | Python nativo en Spark | Sin Docker = máximo rendimiento GPU |
| **Modelos** | PyTorch + Ultralytics | Ya entrenados y validados |
| **Ensemble** | WBF (ensemble-boxes) | NODE21 = 0.9391 |
| **Contenedores** | Docker Compose | Solo para servidor local (no Spark) |
| **Comunicación** | AMQP (RabbitMQ) | Async, resiliente, cola de trabajo |

---

## 8. IMPLEMENTACIÓN — ORDEN DE DESARROLLO

### Sprint 1: Inference Worker en Spark (1-2 días)
- Crear `inference_worker.py` con conexión RabbitMQ
- Probar con mensajes manuales
- Verificar que carga modelos 1 vez y procesa múltiples imágenes

### Sprint 2: Backend FastAPI (2-3 días)
- Docker compose con MySQL + RabbitMQ
- Endpoint POST /upload + consumer de resultados
- Modelo de datos básico

### Sprint 3: Frontend Vue (2-3 días)
- Upload view con drag & drop
- Results view con imagen anotada + boxes
- Polling de resultados

### Sprint 4: Integración y testing (1-2 días)
- End-to-end: upload → cola → Spark → resultado → visualización
- Test con CXR reales del dataset NODE21
- Medir latencia total

**Total prototipo funcional: ~7-10 días**

---

## 9. COMPARACIÓN CON ALTERNATIVAS DESCARTADAS

| Alternativa | Pros | Contras | Decisión |
|-------------|------|---------|----------|
| HTTP directo Backend→Spark | Simple | Bloquea si Spark lenta, sin cola, sin resiliencia | ❌ |
| Docker GPU en Spark | Encapsulado | Overhead GPU, complejidad NVIDIA Container Toolkit | ❌ |
| Todo en Spark (frontend+backend+GPU) | Un solo servidor | Mezcla roles, difícil de escalar, GPU compartida | ❌ |
| gRPC en vez de RabbitMQ | Más rápido para 1:1 | Sin cola, sin persistencia, más complejo | ❌ |
| Kafka en vez de RabbitMQ | Mejor para alto volumen | Overkill para hospital (~50 CXR/día), ya usas RabbitMQ | ❌ |
| **RabbitMQ + Worker nativo** | **Robusto, desacoplado, ya conocido, cola de trabajo** | **Requiere RabbitMQ accesible en LAN** | **✅** |
