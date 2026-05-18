# Índice del proyecto — CXR Nodule Detection

Última actualización: 2026-05-18

---

## Documentación técnica

| # | Documento | Descripción |
|---|---|---|
| 1 | [DOCUMENTACION_DEFINITIVA.md](DOCUMENTACION_DEFINITIVA.md) | **Documento maestro** — resumen ejecutivo, resultados, arquitectura, modelos, backend, frontend, worker, validación, despliegue, roadmap |
| 2 | [DOCUMENTACION_PROYECTO.md](DOCUMENTACION_PROYECTO.md) | Roadmap detallado con estado de cada fase y tarea |
| 3 | [DOCUMENTACION_TECNICA.md](DOCUMENTACION_TECNICA.md) | Arquitectura técnica, API (13 endpoints), modelo de datos (6 tablas), Docker, apéndices |
| 4 | [WEIGHTS_AND_CHECKPOINTS.md](WEIGHTS_AND_CHECKPOINTS.md) | Inventario de pesos y checkpoints con instrucciones de carga |
| 5 | [BEHRENDT_BUGS_ANALYSIS.md](BEHRENDT_BUGS_ANALYSIS.md) | Análisis de 26 bugs en el código del ganador NODE21 |
| 6 | [BUGS_REVIEW_COMPLETO.md](BUGS_REVIEW_COMPLETO.md) | 113 issues del prototipo identificados, 23 corregidos |
| 7 | [PROPUESTA_ARQUITECTURA_HOSPITAL.md](PROPUESTA_ARQUITECTURA_HOSPITAL.md) | Diseño de arquitectura del prototipo hospitalario |
| 8 | [README_ACADEMICO.md](README_ACADEMICO.md) | Versión académica para jornadas IA |
| 9 | [Resumen_Ejecutivo_CXR.md](Resumen_Ejecutivo_CXR.md) | Resumen ejecutivo (Markdown) |

## Documento académico

| # | Fichero | Descripción |
|---|---|---|
| 10 | [../memoria/Memoria.pdf](../memoria/Memoria.pdf) | Memoria académica compilada |
| 11 | [../memoria/source/Memoria_v2_IEEE.tex](../memoria/source/Memoria_v2_IEEE.tex) | Fuente LaTeX (formato IEEE) |
| 12 | [../memoria/source/Paper_definitivo.tex](../memoria/source/Paper_definitivo.tex) | Fuente del paper extendido |
| 13 | [../memoria/source/secciones_nuevas.tex](../memoria/source/secciones_nuevas.tex) | Secciones añadidas en la extensión (YOLO26, WBF ensemble, data leakage, prototipo hospital) |
| 14 | [../memoria/source/biblio_rectifier.bib](../memoria/source/biblio_rectifier.bib) | Bibliografía BibTeX |

## Código fuente

| # | Proyecto | Ubicación | Descripción |
|---|---|---|---|
| 15 | Backend FastAPI | <https://github.com/tcontesti/cxr-detection> | FastAPI + MySQL + RabbitMQ + Nginx |
| 16 | Frontend Vue | <https://github.com/tcontesti/cxr-frontend> | Vue 3 + Tailwind + i18n + dark mode |
| 17 | Worker GPU | [../spark/worker/](../spark/worker/) | Servicio de inferencia (systemd, RabbitMQ) |
| 18 | Scripts training | [../spark/scripts/](../spark/scripts/) | Entrenamiento, evaluación, ensemble WBF |
| 19 | Pipeline ML | [../spark/pipeline/](../spark/pipeline/) | Pipeline reutilizable para nuevos datasets |

## Proyecto original UIB

| # | Directorio | Descripción |
|---|---|---|
| 20 | [../original-project/](../original-project/) | Memoria v1, notebooks, posters y CSVs de evaluación del trabajo base (Marc Link Cladera + Antonio Contestí Coll, supervisor: Dr. Miquel Miró Nicolau) |
| 21 | [../original-project/PRACTICA_LINK_CONTESTI/](../original-project/PRACTICA_LINK_CONTESTI/) | Entrega oficial de la práctica |
| 22 | [../original-project/poster/](../original-project/poster/) | Posters de defensa |

## Reportes y figuras

| # | Directorio | Descripción |
|---|---|---|
| 23 | [../reports/](../reports/) | CSVs de evaluación y curvas FROC del ensemble |
