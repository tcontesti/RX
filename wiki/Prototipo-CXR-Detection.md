# Prototipo CXR Detection

[← Home](Home.md)

Aplicación web operativa que pone el ensemble Faster R-CNN + YOLOv8 + WBF al alcance del radiólogo en el navegador. Cuatro saltos asíncronos entre la radiografía y la sugerencia de bounding box.

## Visión general

```
Radiólogo (navegador)
       │  PNG / DICOM / MHA (drag & drop)
       ▼
Frontend Vue 3 + Tailwind 4 + Vite 7
       │  REST async
       ▼
Backend FastAPI 0.115 + SQLAlchemy
       │  AMQP   ·  cxr.inference
       ▼
RabbitMQ 3.13 (Docker)
       │  AMQP   ·  cxr.results
       ▼
Worker GPU sobre NVIDIA Grace Hopper GB10
   PyTorch 2.x + Ultralytics
   Faster R-CNN + YOLOv8s + Weighted Box Fusion
```

Tiempo extremo a extremo: 2–3 segundos por radiografía (subida + inferencia + render).

## Repositorios

El prototipo está repartido entre tres repositorios:

- **Backend:** <https://github.com/tcontesti/cxr-detection> — FastAPI + Docker Compose + MySQL + RabbitMQ + Nginx.
- **Frontend:** <https://github.com/tcontesti/cxr-frontend> — Vue 3.5 + Tailwind CSS 4.2 + Vite 7. Trilingüe (ES / CA / EN) y con dark mode.
- **Worker GPU:** [`spark/worker/`](../spark/worker/) en este mismo repositorio.

## Stack tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| Frontend | Vue + Tailwind + Vite | 3.5 / 4.2 / 7 |
| Backend | FastAPI + SQLAlchemy async + Pydantic | 0.115 / 2.x / 2.x |
| Base de datos | MySQL | 8.0 |
| Mensajería | RabbitMQ | 3.13 |
| Proxy | Nginx | 1.25 |
| Worker | PyTorch + Ultralytics + ensemble-boxes | 2.x / 8.x / 1.0 |
| Hardware GPU | NVIDIA DGX Spark · Grace Hopper GB10 | aarch64 / CUDA 13 |
| Orquestación | Docker Compose | 2.x |

Detalles completos de versiones en [`docs/DOCUMENTACION_TECNICA.md`](../docs/DOCUMENTACION_TECNICA.md).

## Modelo de datos

Seis tablas MySQL:

| Tabla | Contenido |
|---|---|
| `studies` | Metadatos del estudio: identificador opaco (`CXR-{HEX}`), estado, tiempos, paciente referenciado por hash. |
| `detections` | Cajas predichas por el ensemble: `(x1, y1, x2, y2, score, label, model)`. |
| `images_original` | PNG original tras conversión desde DICOM/MHA. |
| `images_annotated` | PNG con overlay de bounding boxes para presentación. |
| `validations` | Validación radiológica: correcto / parcial / incorrecto, comentario, autor, timestamp. |
| `manual_annotations` | Cajas dibujadas por el radiólogo para corrección o aprendizaje futuro. |

## Flujo end-to-end

1. **Upload.** El radiólogo arrastra una radiografía al navegador. El frontend valida formato (PNG, JPG, DICOM, MHA) y tamaño (hasta 50 MB) y la sube vía HTTPS al backend.
2. **Persistencia.** El backend recibe la imagen, asigna un identificador opaco `CXR-{HEX12}`, persiste la imagen original en `images_original`, crea un registro en `studies` con estado `queued` y emite un mensaje a la cola `cxr.inference`. La transacción se hace en este orden para que, si la publicación AMQP fallase, el estudio quede recuperable.
3. **Inferencia.** El worker consume el mensaje, decodifica la imagen, valida sus dimensiones (rechaza < 10×10 px o > 16 384×16 384 px), ejecuta Faster R-CNN y YOLOv8 en paralelo sobre la misma GPU, aplica Weighted Box Fusion (IoU = 0.20, pesos proporcionales al NODE21 score) y publica el resultado en `cxr.results`.
4. **Render.** El backend recibe el resultado, persiste cajas en `detections` y la imagen anotada en `images_annotated`, actualiza el estado a `completed` y el frontend recibe la respuesta vía polling o WebSocket.

## Funcionalidades clave

- **Visor interactivo con zoom real.** Las cajas SVG se renderizan en una capa superpuesta a la imagen, manteniendo la relación de aspecto correcta y permitiendo zoom hasta 1:1 a pixel real.
- **Threshold slider dinámico.** El radiólogo desliza un umbral entre 0.005 y 1.0 y las detecciones aparecen o desaparecen en tiempo real, sin volver a invocar el modelo.
- **Módulo de validación radiológica.** Para cada estudio, marca cada detección como correcta, parcial o incorrecta, añade un comentario y se persiste con autor y timestamp.
- **Dibujo manual.** El radiólogo dibuja cajas adicionales si el modelo se ha perdido alguna lesión. Estas anotaciones manuales se exportan separadamente para reentrenamiento dirigido.
- **Historial paginado con búsqueda.** Listado de estudios con filtros por fecha, estado y validación.
- **Export CSV / JSON.** Exportación masiva en formatos abiertos para reentrenamiento o análisis externo.

## Latencias

| Etapa | Tiempo medio (GB10) |
|---|---|
| Upload + persistencia backend | 200 ms |
| Encolado RabbitMQ | 5 ms |
| Inferencia Faster R-CNN | 55 ms |
| Inferencia YOLOv8s | 25 ms |
| Weighted Box Fusion | 1 ms |
| Render + transferencia respuesta | 100–500 ms |
| **Total end-to-end** | **2–3 s** |

## Código fuente

El código completo del prototipo está incluido en este mismo repositorio, además de los repositorios espejo desplegables que se mantienen sincronizados:

| Componente | Ubicación en este repo | Espejo desplegable |
|---|---|---|
| Backend FastAPI | [`backend/`](../backend/) | <https://github.com/tcontesti/cxr-detection> |
| Frontend Vue 3 | [`frontend/`](../frontend/) | <https://github.com/tcontesti/cxr-frontend> |
| Worker GPU | [`spark/worker/`](../spark/worker/) | servidor hospital (no público) |
| Scripts entrenamiento | [`spark/scripts/`](../spark/scripts/) | servidor hospital |
| Pipeline reutilizable | [`spark/pipeline/`](../spark/pipeline/) | servidor hospital |

## Para desplegarlo localmente

Backend (Docker Compose):

```bash
cd backend
cp .env.example .env        # ajustar credenciales (placeholders CHANGE_ME_*)
docker compose up -d --build
# Verificar: curl http://localhost:9020/api/health
```

Frontend (Vite dev server):

```bash
cd frontend
npm install
npm run dev                 # abre http://localhost:5175
```

El worker corre sobre la Spark; ver [[Hardware-Spark-GPU]] para la instalación del servicio systemd.

## Hoja de ruta de producción

Tareas pendientes para llegar a un despliegue clínico productivo, documentadas en [`docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md`](../docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md):

1. Autenticación OIDC con Keycloak.
2. HTTPS con certificado del hospital.
3. Integración con PACS vía C-STORE DICOM.
4. Auditoría ENS (logs encadenados SHA-256).
5. Rate limiting y resource limits en Docker.
6. TLS en RabbitMQ y Dead Letter Queue.
7. Versionado de modelo embebido en cada resultado.
