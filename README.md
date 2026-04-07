# rx-cxr-nodule-detection

Material asociado al proyecto **"Detección automática de nódulos pulmonares en radiografías de tórax mediante deep learning"**, originado como Trabajo de Fin de Grado de Marc Link Cladera y Antonio Contestí Coll en la Escola Politècnica Superior de la Universitat de les Illes Balears, bajo la dirección del Dr. Miquel Miró Nicolau, y extendido como prototipo asistencial en colaboración con la Dra. Ana Estremera Bauzá (jefa del Servei de Radiologia, Hospital Universitari Son Llàtzer).

El proyecto aborda la detección automática de nódulos pulmonares sobre radiografías de tórax combinando dos detectores complementarios (Faster R-CNN con preentrenamiento sobre VinDr-CXR y YOLOv8s) mediante un ensemble Weighted Box Fusion. La pipeline se acompaña de una auditoría sistemática del código de referencia del estado del arte (Behrendt et al., ganador del NODE21 Challenge), una cuantificación del data leakage introducido por augmentaciones offline mal agrupadas y un prototipo web operativo con módulo de validación radiológica.

## Estructura del repositorio

```
rx-cxr-nodule-detection/
├── tfg/                 documento académico (LaTeX + PDF)
│   ├── MemoriaTFG.pdf
│   ├── source/          fuente LaTeX completa
│   └── README.md        instrucciones de compilación
├── docs/                documentación técnica y landing page
│   ├── index.html       página de presentación (GitHub Pages)
│   ├── assets/
│   └── *.md             documentación arquitectura, bugs, propuesta hospital
├── spark/               código que corre sobre la GPU
│   ├── worker/          servicio de inferencia (RabbitMQ + GPU)
│   ├── scripts/         entrenamiento, evaluación y preprocesado
│   ├── pipeline/        pipeline reutilizable (data, models, training)
│   └── README.md        cómo reproducir los experimentos
├── original-project/    proyecto base UIB (Marc Link + Antonio Contestí)
│   ├── Memoria_v1.tex   memoria del TFG original
│   ├── PRACTICA_LINK_CONTESTI/   notebooks y CSVs de evaluación
│   ├── poster/          posters de defensa
│   └── images/          figuras de la memoria v1
├── reports/             CSVs y figuras de evaluación
└── wiki/                wiki del proyecto (10 páginas)
```

Los servicios web del prototipo (backend FastAPI y frontend Vue 3) se alojan en repositorios independientes:

- Backend: <https://github.com/tcontesti/cxr-detection>
- Frontend: <https://github.com/tcontesti/cxr-frontend>

## Cómo navegar este material

**Si quieres leer la memoria:** abre `tfg/MemoriaTFG.pdf`.

**Si quieres compilar el documento desde fuente:** sigue `tfg/README.md`. Requiere una distribución LaTeX (TeX Live recomendado) y el archivo `biblio_rectifier.bib`.

**Si quieres entender el prototipo asistencial:** consulta `docs/PROPUESTA_ARQUITECTURA_HOSPITAL.md` y `docs/DOCUMENTACION_TECNICA.md`. La arquitectura general (frontend Vue → backend FastAPI → RabbitMQ → worker GPU) está descrita allí; los servicios desplegables están en los repositorios `cxr-detection` y `cxr-frontend`.

**Si quieres reproducir un experimento concreto:** consulta `spark/README.md` y la página *Experimentos* del wiki. Cada experimento documenta su script, su checkpoint y la métrica esperada.

**Si quieres revisar la auditoría del código de referencia:** abre `docs/BEHRENDT_BUGS_ANALYSIS.md` (26 bugs catalogados en la implementación de Behrendt et al.) y `docs/BUGS_REVIEW_COMPLETO.md` (113 issues encontrados en el prototipo y trazabilidad de los 23 corregidos).

## Reproducibilidad

Los experimentos se ejecutaron sobre una NVIDIA DGX Spark con chip Grace Hopper GB10, arquitectura aarch64 y CUDA 13.0. La memoria documenta en el capítulo de metodología las dependencias y los protocolos completos.

Los datasets utilizados (NODE21, VinDr-CXR, ChestX-ray14, JSRT, PadChest, Open-I) están disponibles a través de sus repositorios oficiales según sus respectivas licencias. Este trabajo no redistribuye material de dichos datasets. Los pesos preentrenados de los modelos base (Faster R-CNN con backbone ResNet-50 desde torchvision, YOLOv8s desde Ultralytics) están disponibles a través de sus librerías originales.

## Colaboraciones

Este proyecto se desarrolla en colaboración entre la Universitat de les Illes Balears (Dr. Miquel Miró Nicolau, Escola Politècnica Superior) y el Servei de Radiologia del Hospital Universitari Son Llàtzer (Dra. Ana Estremera Bauzá), con el objetivo de validar un asistente de cribado sobre casuística clínica real. El trabajo extiende y audita la implementación de referencia publicada por Behrendt et al. (Hamburg University of Technology), ganadora del NODE21 Challenge, cuyos resultados se documentan en *Scientific Reports* (Nature, 2023).

## Licencia

Documento académico y código liberados bajo Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Consultar `LICENSE` para detalles. El material de terceros (modelos base, datasets, implementaciones de referencia) conserva su licencia original.

## Cita

Si este trabajo te resulta útil en tu investigación, por favor cítalo como:

```bibtex
@misc{link_contesti_2026_cxr,
  author       = {Link Cladera, Marc and Contest\'i Coll, Antonio},
  title        = {Detecci\'on autom\'atica de n\'odulos pulmonares en
                  radiograf\'ias de t\'orax mediante deep learning},
  year         = {2026},
  howpublished = {Bachelor's Thesis, Escola Polit\`ecnica Superior,
                  Universitat de les Illes Balears},
  note         = {Supervisor: Dr. Miquel Mir\'o Nicolau. Clinical
                  collaboration: Dra. Ana Estremera Bauz\'a (Hospital
                  Universitari Son Ll\`atzer)}
}
```

## Contacto

- **Antonio Contestí Coll** — `antonio.contesti1@estudiant.uib.cat`
- **Tutor:** Dr. Miquel Miró Nicolau — `miquel.miro@uib.es`
- **Colaboración clínica:** Dra. Ana Estremera Bauzá — Servei de Radiologia, Hospital Universitari Son Llàtzer
