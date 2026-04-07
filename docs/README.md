# docs/

Documentación técnica del proyecto y landing page del repositorio.

## Landing page

- `index.html` — página de presentación que se sirve por GitHub Pages en <https://tcontesti.github.io/RX/>. Sin dependencias locales: usa Tailwind por CDN y Google Fonts.
- `assets/` — logos del Hospital Universitari Son Llàtzer y de la Universitat de les Illes Balears utilizados en la landing.
- `.nojekyll` — marca para que GitHub Pages no procese el directorio como un sitio Jekyll.

## Documentación técnica

| Archivo | Contenido |
|---|---|
| `DOCUMENTACION_DEFINITIVA.md` | Documento maestro consolidado. Punto de entrada recomendado. |
| `DOCUMENTACION_PROYECTO.md` | Roadmap de las cinco fases, resultados experimentales detallados y bitácora del proyecto. |
| `DOCUMENTACION_TECNICA.md` | Arquitectura del prototipo, esquema de la base de datos, formato de mensajes RabbitMQ y endpoints REST. |
| `PROPUESTA_ARQUITECTURA_HOSPITAL.md` | Diseño de la arquitectura para despliegue hospitalario (autenticación Keycloak, integración PACS, ENS). |
| `BEHRENDT_BUGS_ANALYSIS.md` | Auditoría del código de referencia (Behrendt et al., NODE21 Challenge winner). 26 bugs catalogados con su corrección. |
| `BUGS_REVIEW_COMPLETO.md` | 113 issues encontrados en el prototipo y trazabilidad de los 23 corregidos. |
| `WEIGHTS_AND_CHECKPOINTS.md` | Inventario de pesos entrenados con instrucciones de carga. |
| `INDICE.md` | Índice maestro de la documentación. |
| `README_ACADEMICO.md` | Documento académico paralelo para jornadas IA. |
| `Resumen_Ejecutivo_CXR.md` / `.tex` | Resumen ejecutivo (versión Markdown + fuente LaTeX). |

## Orden de lectura sugerido

1. **Lector clínico:** `README_ACADEMICO.md` → `Resumen_Ejecutivo_CXR.md` → `PROPUESTA_ARQUITECTURA_HOSPITAL.md`.
2. **Lector ingeniero:** `DOCUMENTACION_DEFINITIVA.md` → `DOCUMENTACION_TECNICA.md` → `WEIGHTS_AND_CHECKPOINTS.md`.
3. **Lector que audita:** `BEHRENDT_BUGS_ANALYSIS.md` → `BUGS_REVIEW_COMPLETO.md`.

## Regenerar la landing page

La landing es HTML estático. Para editarla basta con modificar `index.html` y empujar a la rama principal: GitHub Pages reconstruye automáticamente. Para activar GitHub Pages la primera vez:

```bash
gh repo edit --enable-pages --pages-branch master --pages-path /docs
```

Para verificar que está activa:

```bash
gh api repos/tcontesti/RX/pages
```
