# Detección automática de nódulos pulmonares en radiografías de tórax mediante deep learning: del modelo al prototipo hospitalario

**Premios Salut Innova UIB-HLL Son Espases · 4ª edición (2026)**
**Categoría:** Inteligencia artificial y datos masivos (big data)

**Proponentes:**
- Antonio Contesti Coll (estudiante/graduado, EPS-UIB)
- Marc Link Cladera (proyecto base UIB, coautor)
- Dr. Miquel Miró Nicolau (supervisor académico UIB)
- Dra. Ana Estremera Rodrigo (jefa del Servicio de Radiología, Hospital Universitari Son Llàtzer, IB-Salut)

**Origen:** trabajo final de la asignatura de Aprendizaje Automático EPS-UIB (curso 2024–2025, Grado en Ingeniería Informática) — desarrollo en curso como prototipo hospitalario.

---

## 1. Justificación clínica

El **cáncer de pulmón es la primera causa de mortalidad por cáncer a nivel mundial**, con una supervivencia a 5 años inferior al 20% cuando se diagnostica en estadios avanzados, frente al 60% cuando se detecta en estadios tempranos. Los nódulos pulmonares son los indicadores precoces más relevantes, y la **radiografía de tórax (CXR) es el estudio de imagen más solicitado** en atención primaria y urgencias.

Sin embargo, **la detección de nódulos en CXR es notoriamente difícil**: estudios clínicos reportan tasas de falsos negativos del 20-50% por parte de radiólogos. Los nódulos pueden ser pequeños (<10 mm), estar superpuestos con costillas y clavículas, o camuflados en el parénquima pulmonar. En el Hospital Universitari Son Llàtzer, la presión asistencial sobre el servicio de radiología hace especialmente valioso disponer de una **segunda lectura algorítmica** que asista al radiólogo en el cribado inicial.

Este proyecto aborda dicho escenario con un sistema de inteligencia artificial que **localiza nódulos en menos de 3 segundos**, presentándolos con cajas delimitadoras superpuestas sobre la imagen, y permite al radiólogo validar, corregir o anotar manualmente los resultados.

## 2. Innovación propuesta

El sistema integra dos detectores de objetos de arquitecturas fundamentalmente distintas — **Faster R-CNN** (two-stage, preentrenado en el dominio radiológico VinDr-CXR) y **YOLOv8** (single-stage, anchor-free) — combinados mediante **Weighted Box Fusion (WBF)**, técnica que fusiona las predicciones de múltiples modelos calculando la media ponderada de coordenadas en lugar de eliminar boxes redundantes (NMS). Esta combinación produce localizaciones más precisas que cualquier modelo individual.

El prototipo se ha implementado como un **sistema web hospitalario end-to-end**: frontend Vue 3 trilingüe (catalán, castellano, inglés) con dark mode, backend FastAPI asíncrono containerizado con Docker, MySQL para persistencia, RabbitMQ como cola de mensajes asíncrona, y un worker de inferencia que ejecuta los modelos en una GPU NVIDIA GB10 (Grace Hopper, arquitectura aarch64). El sistema incluye un **módulo completo de validación radiológica**: el radiólogo puede marcar resultados como correctos, parciales o incorrectos, dibujar bounding boxes manuales sobre nódulos no detectados, marcar falsos positivos, y añadir anotaciones textuales. Todo el material validado se exporta como dataset etiquetado en CSV/JSON para el reentrenamiento continuo del modelo.

## 3. Apoyo al estado del arte

El proyecto toma como referencia el **trabajo ganador del NODE21 Challenge** (Behrendt et al., 2023, *Nature Scientific Reports*), competición internacional de referencia en detección de nódulos en CXR, y avanza más allá con **cinco contribuciones metodológicas propias**:

(i) **Auditoría sistemática del código del estado del arte**: el análisis exhaustivo del repositorio de Behrendt reveló 26 bugs, ocho de ellos críticos (incompatibilidades con PyTorch 2.x, Lightning 2.x y Pandas 2.x), que hacen el código inejecutable en versiones actuales. Solo se pudieron rescatar ~20 líneas útiles. Este hallazgo cuestiona la reproducibilidad de los resultados originales.

(ii) **Cuantificación de data leakage**: se descubrió que el split de entrenamiento/validación del proyecto base no agrupaba las imágenes augmentadas por imagen origen, produciendo una contaminación que inflaba el score NODE21 en **+6.7 puntos**. La corrección con `StratifiedGroupKFold` agrupando por imagen base produce el primer score honesto del pipeline.

(iii) **Importancia del score threshold**: la modificación del threshold de detección por defecto (0.5 → 0.005) mejoró el NODE21 score de 0.4546 a 0.8544 (+40 puntos), evidenciando un hallazgo metodológico de aplicabilidad general en detección de objetos con métrica FROC.

(iv) **Ensemble eficiente**: con solo **dos modelos** se supera el rendimiento del ensemble de 21 modelos del estado del arte (5 arquitecturas × 5 folds + WBF), reduciendo la latencia de inferencia de ~1.2s a 81ms.

(v) **Prototipo hospitalario funcional con módulo de validación**, que cierra el ciclo de mejora continua del modelo permitiendo el reentrenamiento con datos validados por radiólogos en activo.

## 4. Resultados clave

| Métrica | Valor | Comparación |
|---|---|---|
| **NODE21 Score** (ensemble WBF) | **0.9391** | Behrendt: ~0.7524 |
| **Competition Metric (CM)** | **94.47%** | Behrendt: 83.90% (+10 pp) |
| **AUROC a nivel imagen** | **96.83%** | Behrendt: 86.79% |
| **Sensibilidad @ 0.25 FP/img** | **87.4%** | mejor individual: 82.1% (+5.3 pp) |
| **Latencia de inferencia** | **81 ms/imagen** | Behrendt: ~1.2 s |
| **Latencia end-to-end (web)** | **2–3 s** | desde upload a resultado en pantalla |
| **Score honesto (sin leakage)** | **0.9025 (FRCNN), 0.9103 (YOLOv8)** | con leakage inflaba a 0.9695 |

Los scores corresponden al fold 0 de la partición `StratifiedGroupKFold` sobre el dataset NODE21 (4.882 radiografías; 1.134 con nódulos, 3.748 negativas). Test sets directamente no comparables con Behrendt (su test set es privado), pero la diferencia de +10 pp en Competition Metric sugiere un sistema competitivo con un orden de magnitud menos de complejidad.

## 5. Impacto económico y social

**Para el paciente:** reducción del riesgo de falsos negativos en CXR (donde la tasa actual es del 20-50%), aceleración del triaje en sospecha de cáncer de pulmón, y propuesta de DermApIxel como **segunda lectura algorítmica** complementaria al criterio del radiólogo.

**Para el sistema sanitario:** la asistencia automática al diagnóstico en CXR puede aliviar la presión sobre el Servicio de Radiología del Hospital Universitari Son Llàtzer, en línea con la transformación digital del IB-Salut. La latencia inferior a 3 segundos lo hace compatible con el flujo de trabajo radiológico estándar.

**Para la comunidad de investigación:** la publicación abierta del protocolo de auditoría de data leakage, del análisis crítico del código del estado del arte (26 bugs documentados), del prototipo hospitalario completo y del módulo de validación-anotación cubre huecos identificados en la literatura.

**Para la sociedad:** todo el código y la documentación se publican en GitHub bajo licencia abierta, facilitando la reproducción, la auditoría científica y la adopción educativa.

## 6. Cooperación UIB · IB-Salut

El proyecto nace de la articulación entre tres roles complementarios:

- **UIB (Escola Politècnica Superior)** — El **Dr. Miquel Miró Nicolau** aporta la dirección académica original del trabajo y el marco metodológico. **Marc Link Cladera** y **Antonio Contesti Coll** son coautores del proyecto base UIB (modelos de clasificación y detección inicial). Antonio Contesti lidera el desarrollo actual del prototipo hospitalario completo (backend, frontend, worker, módulo de validación) y la auditoría crítica del estado del arte.

- **IB-Salut, Hospital Universitari Son Llàtzer** — La **Dra. Ana Estremera Rodrigo**, jefa del Servicio de Radiología, aporta el conocimiento clínico, la validación del flujo de trabajo radiológico, los requisitos funcionales del prototipo (interfaz, casos de uso, integración con PACS) y la perspectiva del impacto real en consulta. La colaboración incluye el codiseño del módulo de validación y anotación.

## 7. Madurez del proyecto

El sistema es un **prototipo totalmente funcional** que puede demostrarse en vivo durante el Innovation Day. Se ha sometido a dos rondas de revisión exhaustiva de código (113 issues identificados, 23 bugs críticos corregidos), tiene reconexión automática a RabbitMQ con backoff exponencial, gestión de memoria GPU tras cada inferencia, rotación de logs, graceful shutdown, validación de tamaño/formato de archivo, y un health check real que verifica API + MySQL + RabbitMQ.

**Material disponible para evaluación:**

| Recurso | Estado |
|---|---|
| Memoria académica (UIB) | Disponible |
| Backend (FastAPI + Docker) | https://github.com/tcontesti/cxr-detection |
| Frontend (Vue 3) | https://github.com/tcontesti/cxr-frontend |
| Worker GPU + scripts entrenamiento | Disponible (servidor Spark del hospital) |
| Documentación técnica completa | INDICE.md + 6 documentos técnicos |
| Demostración en vivo | Prototipo operativo en LAN hospital |

**Trabajo futuro inmediato:** extensión multi-patología (tuberculosis con TBX11K, 22 patologías con VinDr-CXR), integración PACS por DICOM C-STORE, autenticación Keycloak OIDC, certificado HTTPS hospitalario, y publicación de un paper comparativo en *Nature Scientific Reports* o *Medical Image Analysis*.
