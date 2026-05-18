# Memoria académica

[← Home](Home.md)

La memoria académica está disponible en dos formatos en el repositorio:

- [`memoria/Memoria.pdf`](../memoria/Memoria.pdf) — PDF compilado, listo para lectura.
- [`memoria/source/`](../memoria/source/) — fuente LaTeX completa, lista para recompilar.

## Estructura del documento

El documento sigue el formato IEEE de doble columna. La estructura cubre los capítulos clásicos de un trabajo de detección por aprendizaje profundo en imagen médica.

| Capítulo | Contenido |
|---|---|
| 1. Introducción | Motivación clínica del cribado de nódulos pulmonares por radiografía de tórax. Justificación del enfoque detección + clasificación (no segmentación) en un contexto de cribado primario. |
| 2. Estado del arte | Revisión del NODE21 Challenge, de la implementación ganadora (Behrendt et al., 2023) y de los detectores aplicables a CXR. Datasets de referencia: NODE21, VinDr-CXR, ChestX-ray14, JSRT, PadChest, Open-I. |
| 3. Datos | Composición del dataset NODE21 (4 882 radiografías de cuatro fuentes), preprocesado MHA → PNG, splits StratifiedGroupKFold y protocolo anti-data-leakage. |
| 4. Metodología | Arquitectura Faster R-CNN ResNet-50 + FPN, arquitectura YOLOv8s, preentrenamiento sobre VinDr-CXR, ensemble Weighted Box Fusion, métricas (NODE21 CM, FROC, AUROC, Sens@0.25 FP). |
| 5. Auditoría del código de referencia | 26 bugs catalogados en la implementación de Behrendt et al., con trazabilidad por línea y propuesta de corrección. Cuantificación del impacto del data leakage en la métrica reportada. |
| 6. Resultados experimentales | Comparativa cuantitativa de Faster R-CNN VinDr, YOLOv8s, YOLO26s y ensemble WBF. Curvas FROC, matrices de confusión, ablation sobre el score threshold. |
| 7. Prototipo asistencial | Diseño Vue 3 + FastAPI + RabbitMQ + worker GPU, esquema de la base de datos MySQL (6 tablas), módulo de validación radiológica y propuesta de integración con PACS. |
| 8. Discusión | Por qué dos modelos vencen a veintiuno, importancia del preentrenamiento sobre VinDr-CXR, papel del score threshold y limitaciones del estado del arte. |
| 9. Conclusiones y trabajo futuro | Roadmap a cinco fases: nódulos (completado), tuberculosis (TBX11K), 22 patologías (VinDr-CXR multilabel), prototipo asistencial (completado), publicación. |
| Apéndices | Configuración del worker, variables de entorno, formato de mensajes RabbitMQ, inventario de checkpoints. |

## Perfiles de lectura recomendados

- **Lector clínico (radiólogo, jefe de servicio):** capítulos 1, 7 y 8. La sección 7 describe la herramienta tal y como la usará el radiólogo y la 8 discute las decisiones clínicas (cribado vs. diagnóstico, score threshold, integración con flujo de trabajo).
- **Lector ingeniero (interesado en reproducir):** capítulos 3, 4, 6 y los apéndices. La sección 3 documenta el preprocesado completo y la 6 detalla los hiperparámetros de cada experimento.
- **Lector que audita reproducibilidad:** capítulos 2, 5 y 8. La sección 5 es la auditoría sistemática de la implementación de referencia y la 8 enmarca por qué los resultados publicados deben tomarse con cautela.
- **Lector con prisa:** la sección de resumen ejecutivo (al inicio) y el capítulo 9 (conclusiones) cubren las cifras y el roadmap en pocas páginas.

## Recompilar el documento

Las instrucciones detalladas están en [`memoria/README.md`](../memoria/README.md). En resumen, con TeX Live o MiKTeX instalado:

```bash
cd memoria/source
latexmk -pdf Memoria_v2_IEEE.tex
cp Memoria_v2_IEEE.pdf ../Memoria.pdf
```

## Documentos complementarios

| Documento | Ubicación | Para qué leerlo |
|---|---|---|
| Documentación maestra | [`docs/DOCUMENTACION_DEFINITIVA.md`](../docs/DOCUMENTACION_DEFINITIVA.md) | Visión consolidada del proyecto completo |
| Auditoría del estado del arte | [`docs/BEHRENDT_BUGS_ANALYSIS.md`](../docs/BEHRENDT_BUGS_ANALYSIS.md) | 26 bugs en la implementación de Behrendt et al. |
| Revisión del prototipo | [`docs/BUGS_REVIEW_COMPLETO.md`](../docs/BUGS_REVIEW_COMPLETO.md) | 113 issues encontrados, 23 corregidos |
| Propuesta para hospital | [`docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md`](../docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md) | Diseño para despliegue clínico productivo |
| Resumen ejecutivo | [`docs/Resumen_Ejecutivo_CXR.md`](../docs/Resumen_Ejecutivo_CXR.md) | Versión corta para comités y stakeholders |
