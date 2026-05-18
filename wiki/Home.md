# CXR Nodule Detection — Wiki

Material asociado al proyecto **"Detección automática de nódulos pulmonares en radiografías de tórax mediante deep learning"**, originado como trabajo final de la asignatura de Aprendizaje Automático en la Escola Politècnica Superior de la Universitat de les Illes Balears (curso 2025–2026) y extendido como prototipo asistencial en colaboración con el Hospital Universitari Son Llàtzer.

- **Autores:** Marc Link Cladera y Antonio Contestí Coll — `antonio.contesti1@estudiant.uib.cat`
- **Tutor:** Dr. Miquel Miró Nicolau — `miquel.miro@uib.es`
- **Colaboración clínica:** Dra. Ana Estremera Bauzá — jefa del Servei de Radiologia, Hospital Universitari Son Llàtzer
- **Referencia internacional:** Behrendt et al. — Hamburg University of Technology, ganadores del NODE21 Challenge

## Resumen ejecutivo

El trabajo aborda una pregunta concreta: qué ocurre cuando alguien externo al laboratorio de origen intenta reproducir, auditar y desplegar la implementación ganadora del NODE21 Challenge (Behrendt et al., 2023). La auditoría se realiza sobre la única implementación pública del estado del arte, con la dirección académica del Dr. Miquel Miró Nicolau (UIB) y la validación clínica de la Dra. Ana Estremera Bauzá (Hospital Universitari Son Llàtzer).

La primera contribución es metodológica: una **auditoría sistemática** del código publicado por Behrendt et al. que cataloga 26 bugs (ocho críticos), entre los que destacan el cálculo erróneo de la métrica oficial NODE21, la mezcla entre datos de entrenamiento y test al construir las augmentaciones offline, la ausencia de seeding determinista y la pérdida silenciosa de cajas anotadas tras el preprocesado. La conclusión, documentada con trazabilidad por línea, es que la implementación de referencia es inutilizable y sólo se rescatan ~20 líneas (SWA, WeightedSampler, gradient clipping).

La segunda contribución cuantifica un problema previamente no documentado: **el data leakage introducido por las augmentaciones offline mal agrupadas inflaba el score NODE21 en +6.7 puntos absolutos** (de 0.9695 a 0.9025 tras aplicar un split limpio con `StratifiedGroupKFold`). Es decir, la cifra publicada por la implementación de referencia mezcla en validación copias aumentadas de imágenes que están en entrenamiento. El protocolo limpio se documenta como hallazgo replicable.

La tercera contribución es el **ensemble Weighted Box Fusion** sobre dos detectores complementarios: Faster R-CNN con backbone ResNet-50+FPN preentrenado sobre VinDr-CXR, y YOLOv8s a resolución 1024×1024. El ensemble obtiene NODE21 = 0.9391, CM = 94.47 %, AUROC = 0.9683 y Sens@0.25 FP = 87.4 % en 81 ms por imagen, superando al ganador oficial NODE21 (CM = 83.90 %, 21 modelos en cascada) por +10.6 puntos absolutos en la métrica oficial. La pipeline reutilizable se documenta con la trazabilidad necesaria para reproducir cada cifra desde checkpoint.

La cuarta contribución es el **prototipo asistencial CXR Detection**: una aplicación web operativa con frontend Vue 3 trilingüe, backend FastAPI sobre Docker Compose, cola RabbitMQ y worker de inferencia en GPU NVIDIA Grace Hopper GB10. Incluye visor interactivo con zoom real, threshold slider dinámico, módulo de validación radiológica (correcto / parcial / incorrecto), corrección manual de bounding boxes y exportación CSV/JSON para reentrenamiento. Tiempos extremo a extremo de 2–3 segundos por radiografía. La arquitectura está documentada con vistas a integración futura con PACS y autenticación OIDC.

## Navegación del wiki

### Documento académico

| Página | Contenido |
|---|---|
| [[Memoria-academica]] | Estructura del PDF capítulo a capítulo, perfiles de lectura recomendados |
| [[Licencia-y-citacion]] | CC BY-NC 4.0, BibTeX, licencias de terceros |

### Prototipo asistencial

| Página | Contenido |
|---|---|
| [[Prototipo-CXR-Detection]] | Arquitectura Vue 3 + FastAPI + RabbitMQ + worker GPU, latencias, módulo de validación |

### Trabajo experimental

| Página | Contenido |
|---|---|
| [[Experimentos]] | Mapa de experimentos: FRCNN-VinDr, YOLOv8s, YOLO26s, ensemble WBF, ablation data leakage |
| [[Modelos]] | Fichas de los modelos: arquitecturas, pretraining, hiperparámetros y métricas |
| [[Dataset-NODE21]] | Composición, splits, protocolo anti-data-leakage y datasets complementarios |

### Infraestructura y reproducibilidad

| Página | Contenido |
|---|---|
| [[Hardware-Spark-GPU]] | NVIDIA Grace Hopper GB10, aarch64, CUDA 13, comandos básicos |
| [[Reproducibilidad]] | Cuatro escenarios prácticos: compilar paper, entrenar, evaluar, ejecutar prototipo |

### Marco

| Página | Contenido |
|---|---|
| [[Colaboraciones]] | UIB, IB-Salut Son Llàtzer, referencia internacional (TU Hamburg) |
