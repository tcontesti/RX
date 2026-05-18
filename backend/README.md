# backend/

Backend FastAPI containerizado del prototipo CXR. Asíncrono, con MySQL + RabbitMQ + Nginx empaquetados en Docker Compose. Es la copia documental del repositorio espejo desplegable.

- **Espejo desplegable:** <https://github.com/tcontesti/cxr-detection>

## Estructura

```
backend/
├── docker-compose.yml         orquestación de los 5 servicios
├── .env.example               variables de entorno (rellenar antes de arrancar)
├── mysql/
│   ├── init/01_create_db.sql  esquema inicial (6 tablas)
│   └── my.cnf                 configuración del servidor MySQL
├── nginx/nginx.conf           proxy y reglas CORS
├── rabbitmq/rabbitmq.conf     configuración del broker AMQP
├── services/cxr-svc/
│   ├── Dockerfile             imagen del backend
│   ├── requirements.txt       dependencias Python
│   └── app/
│       ├── main.py            entrypoint FastAPI
│       ├── config.py          variables de entorno tipadas
│       ├── database.py        SQLAlchemy async engine
│       ├── routers/           endpoints REST (health, upload, results, history, validation)
│       ├── models/            modelos ORM (6 tablas)
│       ├── schemas/           DTOs Pydantic
│       └── consumers/         consumer de cxr.results
└── shared/messaging.py        helpers AMQP (publish/consume)
```

## Desplegar en local

Requisitos: Docker Desktop o Docker Engine ≥ 24, con Docker Compose v2.

```bash
cp .env.example .env
# Editar .env con credenciales reales (CHANGE_ME_*)
docker compose up -d --build
```

Verificación:

```bash
curl http://localhost:9020/api/health
# {"status":"ok","mysql":"ok","rabbitmq":"ok"}
```

## Endpoints

Documentados automáticamente en Swagger: <http://localhost:9020/docs>.

| Router | Endpoint principal |
|---|---|
| `routers/health.py` | `GET /api/health` |
| `routers/upload.py` | `POST /api/upload` (subida CXR + encolado AMQP) |
| `routers/results.py` | `GET /api/study/{study_id}` (resultado con cajas) |
| `routers/history.py` | `GET /api/history` (paginado, búsqueda, filtros) |
| `routers/validation.py` | `POST /api/validation` (correcto / parcial / incorrecto) |

## Puertos

| Servicio | Puerto host | Notas |
|---|---|---|
| Backend FastAPI | 9020 | Swagger en `/docs` |
| Nginx | 8090 | Proxy reverso opcional |
| MySQL | (interno) | Sólo accesible desde la red Docker |
| RabbitMQ | 5672 | AMQP — usado por el worker GPU |
| RabbitMQ UI | 15674 | Panel administración |

## Seguridad

- `docker-compose.yml` aplica `read_only: true`, `no-new-privileges` y `tmpfs` en los servicios de aplicación.
- La red `backend_net` es `internal: true` (sin salida directa a internet).
- Las credenciales se inyectan únicamente vía `.env` (gitignored). El fichero `.env.example` contiene placeholders.

Pendientes documentados en [`../docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md`](../docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md): autenticación OIDC, TLS, rate limiting y Dead Letter Queue.

## Licencia

CC BY-NC 4.0 — ver [`../LICENSE`](../LICENSE).
