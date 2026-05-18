# frontend/

Frontend Vue 3 + Tailwind del prototipo CXR. Trilingüe (ca/es/en), dark mode, sin estado global pesado (Composition API + composables). Es la copia documental del repositorio espejo desplegable.

- **Espejo desplegable:** <https://github.com/tcontesti/cxr-frontend>

## Estructura

```
frontend/
├── index.html                 entrypoint Vite
├── package.json               dependencias (sin lock para evitar drift en el repo)
├── vite.config.js             dev server + proxy a /api
├── public/                    assets estáticos servidos tal cual
│   ├── favicon.svg
│   └── logo-husl.png
└── src/
    ├── App.vue                shell de la aplicación
    ├── main.js                bootstrap Vue + router + i18n
    ├── style.css              Tailwind base + utilities
    ├── router/index.js        rutas: AnalyzeView, HistoryView, ValidationStatsView
    ├── i18n/index.js          catálogo ca / es / en
    ├── lib/api.js             cliente REST (fetch hacia /api/cxr)
    ├── components/
    │   ├── CxrViewer.vue      visor de radiografía con zoom + overlay SVG
    │   ├── DropZone.vue       drag & drop PNG / DICOM / MHA
    │   ├── ThresholdSlider.vue umbral dinámico de detecciones
    │   ├── DetectionList.vue  lista de cajas con score
    │   ├── ValidationPanel.vue módulo de validación radiológica
    │   ├── StatsBar.vue       contadores agregados
    │   ├── StudyStatus.vue    badge de estado del estudio
    │   └── ConfirmModal.vue   diálogo de confirmación reutilizable
    ├── composables/
    │   ├── useUpload.js       lógica de subida + polling de resultados
    │   ├── useDetections.js   filtrado por threshold en cliente
    │   ├── useTheme.js        toggle dark mode persistido
    │   └── useConfirm.js      promesa-confirmación con modal
    └── views/
        ├── AnalyzeView.vue    flujo de análisis (upload + visor)
        ├── HistoryView.vue    historial paginado con filtros
        └── ValidationStatsView.vue  estadísticas de validación radiológica
```

## Arrancar en local

Requisitos: Node.js 20+ y npm.

```bash
npm install
npm run dev
```

Abre <http://localhost:5175>. El dev server proxy-pasa cualquier `/api/*` al backend en `http://localhost:9020` (configurado en `vite.config.js`). Asegúrate de tener el backend corriendo (ver [`../backend/README.md`](../backend/README.md)).

## Build de producción

```bash
npm run build
```

Genera `dist/` (gitignored) con HTML + JS + CSS estáticos. Se sirve detrás de un Nginx en producción.

## Stack

| Componente | Versión |
|---|---|
| Vue | 3.5 |
| Vite | 7 |
| Tailwind CSS | 4.2 |
| vue-router | 4 |
| vue-i18n | 9 |

## Idiomas

Catálogo completo en `src/i18n/index.js`: catalán (predeterminado en Mallorca), castellano y inglés. Toggle desde la barra superior, persistente vía `localStorage`.

## Comunicación con el backend

Toda la API se concentra en `src/lib/api.js`. Cliente fetch puro, sin axios. Endpoints relativos (`/api/cxr/*`) — no se hardcodean URLs absolutas, de modo que el mismo bundle funciona en dev (proxy Vite) y en producción (Nginx).

## Licencia

CC BY-NC 4.0 — ver [`../LICENSE`](../LICENSE).
