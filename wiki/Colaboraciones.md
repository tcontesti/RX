# Colaboraciones

[← Home](Home.md)

El proyecto se desarrolla principalmente desde la UIB (dirección académica + implementación), con Antonio Contestí Coll articulando además el puente con el IB-Salut a través de su rol como Técnico de Gestión de Sistemas y Tecnologías de las Telecomunicaciones del Hospital Universitario Son Llàtzer. La referencia internacional auditada es TU Hamburg. La validación clínica prospectiva con un Servicio asistencial concreto del HUSL queda como fase posterior.

## i · Dirección académica — UIB

**Universitat de les Illes Balears**, Escola Politècnica Superior.

| Persona | Rol |
|---|---|
| **Dr. Miquel Miró Nicolau** | Tutor del trabajo final de la asignatura de Aprendizaje Automático. Profesor de la EPS. Define los objetivos académicos, supervisa la metodología y aprueba la memoria. |
| **Marc Link Cladera** | Estudiante de Ingeniería Informática (UIB · EPS), defensa de TFG prevista para junio de 2026. Coautor del trabajo original; responsable de la implementación inicial (clasificación binaria con EfficientNet, RetinaNet y baseline Faster R-CNN sobre NODE21). |
| **Antonio Contestí Coll** | Estudiante de Ingeniería Informática (UIB · EPS), defensa de TFG prevista para junio de 2026; Técnico de Gestión de Sistemas y Tecnologías de las Telecomunicaciones, Hospital Universitario Son Llàtzer (IB-Salut). Coautor del trabajo original; responsable de la extensión actual (auditoría sistemática del estado del arte, detector YOLO, ensemble WBF, prototipo asistencial). Mantiene el repositorio y desarrolla la integración hospitalaria. |

**Contacto académico:** `miquel.miro@uib.es`.

**Origen:** el trabajo nace como trabajo final de la asignatura de Aprendizaje Automático de Marc Link Cladera y Antonio Contestí Coll en la Escola Politècnica Superior de la UIB. La versión original (clasificación binaria de nódulos con EfficientNet-B0 multicanal) está completa y conservada en [`original-project/`](../original-project/). La extensión actual (detección + clasificación con ensemble + auditoría del estado del arte + prototipo web) se realiza tras la defensa, manteniendo la misma dirección académica.

## ii · Despliegue hospitalario previsto — IB-Salut

El sistema está diseñado para integrarse en un **Servicio de Radiología hospitalario del IB-Salut**, manteniendo el flujo de trabajo asistencial estándar (radiólogo, residente, telerradiología) y con preparación para integración PACS por DICOM C-STORE y autenticación corporativa OIDC (Keycloak).

**Trabajo previsto en esta fase:**

- Validación del prototipo sobre casuística clínica anonimizada del centro receptor.
- Definición operativa de los flujos de uso reales (informado masivo, urgencia, segunda lectura) con el equipo asistencial del Servicio que finalmente aloje el piloto.
- Cumplimiento de los requisitos no funcionales del despliegue (ENS, OIDC, auditoría con SHA-256 chained logs).
- Aportación del marco regulatorio aplicable (RGPD, Ley 41/2002 de autonomía del paciente, marcado CE como software de diagnóstico).

**Estado actual:** la colaboración formal con un Servicio asistencial concreto está pendiente de formalización y supeditada a la aprobación del protocolo por el Comité de Ética asistencial correspondiente. El diseño completo del despliegue hospitalario se documenta en [`docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md`](../docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md).

## iii · Referencia internacional — TU Hamburg

**Hamburg University of Technology · Institute of Medical Technology and Intelligent Systems (mTec).**

| Trabajo | Aporte al proyecto |
|---|---|
| **Behrendt et al. (2023).** *A systematic approach to deep learning-based nodule detection in chest radiographs.* *Scientific Reports*, 13, 10120. <https://doi.org/10.1038/s41598-023-37270-2> | Implementación ganadora del NODE21 Challenge (CM = 83.90 %). Es la única implementación pública del estado del arte para detección de nódulos en CXR, y el punto de partida obligado de la auditoría. |

**Naturaleza de la colaboración:** no hay colaboración directa con el grupo de Hamburg. Su código publicado es **objeto de auditoría** en este trabajo. La auditoría está disponible en [`docs/BEHRENDT_BUGS_ANALYSIS.md`](../docs/BEHRENDT_BUGS_ANALYSIS.md) y documenta 26 bugs (ocho críticos) en su implementación, incluyendo la cuantificación del data leakage de +6.7 puntos NODE21.

La memoria académica cita y referencia el trabajo de Behrendt et al. como baseline obligado y como motivación para el protocolo limpio adoptado en este proyecto.

## iv · Colaboraciones futuras

Líneas de continuación tras el cierre de la primera versión del proyecto:

| Línea | Estado | Notas |
|---|---|---|
| **Fase 2 — Tuberculosis sobre TBX11K** | Pendiente | 11 200 radiografías con anotación de TB activa, latente y cicatricial. Ampliación natural del prototipo asistencial. |
| **Fase 3 — Multipatología sobre VinDr-CXR 22 clases** | En curso | Dataset descargándose vía PhysioNet. Permitirá clasificación multietiqueta y detección conjunta. |
| **Fase 5 — Publicación** | 35 % | Paper comparativo orientado a *Scientific Reports* (Nature). Incluye la auditoría sistemática, la cuantificación del data leakage y el ensemble. |
| **Reproducibility group en imaging médico** | Potencial | Conversaciones preliminares con grupos especializados en reproducibilidad metodológica (TU Eindhoven, Imperial College). El protocolo anti-leakage exportable es de interés para esos grupos. |

## v · Reconocimientos

- Equipo de servidores de la UIB / IB-Salut por el acceso a la NVIDIA DGX Spark.
- Organizadores del NODE21 Challenge (Radboud University Medical Center) por la curación y publicación del dataset.
- Equipos de PhysioNet, VinDr, Open-I y NIH por mantener accesibles los datasets de imagen torácica.
- Comunidad de torchvision y Ultralytics por las implementaciones de Faster R-CNN y YOLO usadas.
