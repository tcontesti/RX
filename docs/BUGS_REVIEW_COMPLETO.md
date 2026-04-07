# REVISIÓN DE BUGS — Prototipo CXR Detection
# Fecha revisión inicial: 2026-04-04
# Revisión exhaustiva final: 2026-04-12
# Total issues encontrados: 113 (37 backend + 34 frontend + 42 worker)
# Total fixes aplicados: 23 (11 ronda 1 + 12 ronda 2)

---

## RONDA 1 — Fixes iniciales (2026-04-04)

### Backend
1. ✅ Pool MySQL: pool_pre_ping=True + pool_recycle=300 (conexiones stale)
2. ✅ DetectionOut: from_attributes=True (serialización Pydantic)
3. ✅ Consumer: pool_recycle + pool_pre_ping (crash tras idle)
4. ✅ Tabla cxr_original_images creada
5. ✅ main.py: graceful shutdown con lifespan + CORS restrictivo + logging
6. ✅ health.py: verifica MySQL + RabbitMQ (antes solo devolvía "ok" sin verificar)
7. ✅ Dockerfile: single worker async + usuario non-root

### Frontend
8. ✅ Búsqueda solo con Enter (no en cada letra)
9. ✅ SVG overlay en vez de imagen anotada fija (slider funcional)
10. ✅ Zoom real (100% = 1px imagen = 1px pantalla)
11. ✅ Colores vivos para bounding boxes
12. ✅ Historial con fecha/hora, buscar, eliminar, borrar todo
13. ✅ useUpload.js: onUnmounted limpia polling, revokeObjectURL, max poll errors, validación tamaño
14. ✅ api.js: handleResponse() centralizado con error handling

### Worker
15. ✅ Config RabbitMQ: host, user, password actualizados
16. ✅ Threshold bajado a 0.01 (frontend filtra)

---

## RONDA 2 — Revisión exhaustiva final (2026-04-12)

### Spark (worker + scripts + pipeline)

| # | Severidad | Archivo | Fix |
|---|-----------|---------|-----|
| 17 | **CRITICO** | inference_worker.py | Graceful shutdown: signal_handler llama channel.stop_consuming() (antes SIGTERM no detenía el worker) |
| 18 | **ALTO** | inference_worker.py | Backoff exponencial en reconexión RabbitMQ (10s→20s→40s... max 300s, reset tras conexión exitosa) |
| 19 | **ALTO** | config.yaml | Config RabbitMQ corregida: host=`$RABBITMQ_HOST`, user=cxr_user, password=`$RABBITMQ_PASS` |
| 20 | **ALTO** | frcnn_builder.py | freeze_layers arreglado: busca prefijos cortos (layer1) y completos (backbone.body.layer1) — antes nunca congelaba nada |
| 21 | **MEDIO** | inference_worker.py | Validación de imagen: min 10×10, max 16384×16384 — rechaza corruptas, 1px, o enormes |
| 22 | **MEDIO** | evaluate.py | save_visualisations: añade .png al nombre si falta (evita FileNotFoundError) |

### Windows (backend + frontend)

| # | Severidad | Archivo | Fix |
|---|-----------|---------|-----|
| 23 | **CRITICO** | upload.py | Commit BD ANTES de publicar a RabbitMQ. Si publish falla, estudio queda como "queued" (recuperable). Antes se perdía el estudio |
| 24 | **CRITICO** | result_consumer.py | Idempotencia: si estudio ya está en estado terminal (completed/error), ackea mensaje duplicado sin re-procesar. Previene detecciones duplicadas |
| 25 | **ALTO** | results.py | Content-Type correcto según image_format (png→image/png, jpg→image/jpeg, dcm→application/dicom) |
| 26 | **ALTO** | .env.example | Passwords reales reemplazados por placeholders CHANGE_ME_* |
| 27 | **MEDIO** | results.py | DELETE /all con bulk SQL DELETE en vez de cargar todos los estudios en memoria (escala a miles) |
| 28 | **BAJO** | i18n/index.js | Key viewer duplicada eliminada en locale ca |

### Verificaciones post-fix

| Verificación | Resultado |
|-------------|-----------|
| Spark: sintaxis todos los .py | ✅ OK |
| Spark: imports venv | ✅ OK |
| Spark: worker --test | ✅ OK (8 detecciones, 375ms, FRCNN 59ms + YOLO 166ms) |
| Spark: checkpoints accesibles | ✅ OK (FRCNN 165MB, YOLO 22MB, VinDr 166MB) |
| Spark: imágenes | ✅ 7,148 en data/png_images/ |
| Windows: docker compose up --build | ✅ OK, todos los containers arrancaron |
| Windows: /api/health | ✅ {"api":"ok","database":"ok","rabbitmq":"ok","status":"ok"} |
| Windows: npm run build | ✅ OK, compilado en 1.05s sin errores ni warnings |

---

## PENDIENTE — Solo infraestructura/despliegue (no funcional)

### Para producción hospital (fase posterior)

| # | Componente | Issue | Prioridad |
|---|-----------|-------|-----------|
| 1 | Infra | HTTPS con certificado hospital | Alta |
| 2 | Infra | Auth Keycloak OIDC | Alta |
| 3 | Infra | Rate limiting en nginx | Media |
| 4 | Infra | Resource limits en Docker (CPU/RAM) | Media |
| 5 | Infra | TLS para RabbitMQ | Media |
| 6 | Worker | torch.load weights_only=False (inseguro) | Media |
| 7 | Worker | Dead Letter Queue | Media |
| 8 | Worker | Versionado de modelo en resultados | Baja |
| 9 | Backend | Request logging/audit trail | Media |
| 10 | Backend | Timestamps con timezone | Baja |
| 11 | Backend | Request ID / tracing | Baja |
| 12 | Frontend | Error boundaries Vue | Baja |
| 13 | Frontend | CSP headers | Baja |
| 14 | Frontend | Accesibilidad (aria-labels) | Baja |

### Para ENS/HIPAA (certificación)
- Auditoría ENS con SHA-256 chained logs
- Cifrado de datos sensibles (AES-256-GCM)
- PIN lock por inactividad
- Integración PACS (DICOM C-STORE)
- Certificación CE marking
