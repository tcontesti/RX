# Licencia y citación

[← Home](Home.md)

## Licencia

El material original de este repositorio está liberado bajo **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. El texto completo de la licencia está en [`LICENSE`](../LICENSE) en la raíz del repositorio.

En resumen, la licencia permite:

- **Compartir** el material (copiar y redistribuir en cualquier medio o formato).
- **Adaptar** el material (remezclar, transformar y construir a partir de él).

Bajo las condiciones siguientes:

- **Atribución** — Debes dar crédito apropiado, proporcionar un enlace a la licencia e indicar si has hecho cambios. Puedes hacerlo de cualquier forma razonable, pero no de manera que sugiera que el autor te respalda a ti o a tu uso.
- **No comercial** — No puedes usar el material para fines comerciales.

### Alcance de la licencia

| Componente | Licencia | Notas |
|---|---|---|
| Memoria académica (`memoria/`) | CC BY-NC 4.0 | LaTeX source + PDF compilado |
| Worker GPU, scripts y pipeline (`spark/`) | CC BY-NC 4.0 | Sólo uso académico y no comercial |
| Documentación (`docs/`) | CC BY-NC 4.0 | Documentos técnicos y landing page |
| Reportes de evaluación (`reports/`, `memoria/source/*.png`) | CC BY-NC 4.0 | Cifras y figuras propias |
| Proyecto base UIB (`original-project/`) | CC BY-NC 4.0 | Por Marc Link Cladera y Antonio Contestí Coll, supervisor Dr. Miquel Miró Nicolau |

## Material de terceros

El material referenciado y evaluado por este trabajo conserva su licencia original. **Este repositorio no redistribuye** ninguno de los siguientes:

### Datasets

| Dataset | Licencia | Fuente |
|---|---|---|
| NODE21 | CC BY-NC-ND 4.0 | <https://node21.grand-challenge.org/> |
| VinDr-CXR | Restricciones de PhysioNet | <https://physionet.org/content/vindr-cxr/> |
| ChestX-ray14 (NIH) | Open access | <https://nihcc.app.box.com/v/ChestXray-NIHCC> |
| JSRT | Acceso bajo solicitud | <http://db.jsrt.or.jp/eng.php> |
| PadChest | CC BY-SA 4.0 | <https://bimcv.cipf.es/bimcv-projects/padchest/> |
| Open-I (NLM) | Open access | <https://openi.nlm.nih.gov/> |
| TBX11K (futuro) | CC BY-NC 4.0 | <https://mmcheng.net/tb/> |

### Implementaciones de referencia

| Trabajo | Licencia | Uso en este proyecto |
|---|---|---|
| Behrendt et al., *Scientific Reports* (2023) | Repositorio público con licencia propia | Auditado en `docs/BEHRENDT_BUGS_ANALYSIS.md`. No redistribuido. |

### Pesos preentrenados

| Modelo | Licencia | Fuente |
|---|---|---|
| Faster R-CNN ResNet-50+FPN (COCO) | BSD-3-Clause | `torchvision.models.detection` |
| YOLOv8s, YOLO26s | AGPL-3.0 | `ultralytics` |

## Cómo citar este trabajo

### BibTeX

```bibtex
@misc{link_contesti_2026_cxr,
  author       = {Link Cladera, Marc and Contest\'i Coll, Antonio},
  title        = {Detecci\'on autom\'atica de n\'odulos pulmonares en
                  radiograf\'ias de t\'orax mediante deep learning},
  year         = {2026},
  howpublished = {Final project for the course of Machine Learning, Escola Polit\`ecnica Superior,
                  Universitat de les Illes Balears},
  note         = {Supervisor: Dr. Miquel Mir\'o Nicolau. Clinical
                  collaboration: Dra. Ana Estremera Rodrigo\'a (Hospital
                  Universitari Son Ll\`atzer)}
}
```

### Texto plano

> Link Cladera, M., & Contestí Coll, A. (2026). *Detección automática de nódulos pulmonares en radiografías de tórax mediante deep learning*. trabajo final de la asignatura de Aprendizaje Automático, Escola Politècnica Superior, Universitat de les Illes Balears, Palma de Mallorca. Supervisor: Dr. Miquel Miró Nicolau. Colaboración clínica: Dra. Ana Estremera Rodrigo (Hospital Universitari Son Llàtzer).

## Citas obligatorias derivadas

Si reproduces los resultados experimentales o usas los modelos derivados, cita además:

- **NODE21 Challenge** — para el dataset y la métrica oficial:
  > Sogancioglu, E., van Ginneken, B., Behrendt, F., Bengs, M., Schlaefer, A., et al. (2024). Nodule detection and generation on chest X-rays: NODE21 Challenge. *IEEE Transactions on Medical Imaging*. <https://doi.org/10.1109/TMI.2024.3382042>

- **Behrendt et al.** — para la implementación de referencia auditada:
  > Behrendt, F., Bengs, M., Bhattacharya, D., Krüger, J., Opfer, R., & Schlaefer, A. (2023). A systematic approach to deep learning-based nodule detection in chest radiographs. *Scientific Reports*, 13, 10120. <https://doi.org/10.1038/s41598-023-37270-2>

- **VinDr-CXR** — si usas el preentrenamiento sobre VinDr:
  > Nguyen, H. Q., Lam, K., Le, L. T., et al. (2022). VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. *Scientific Data*, 9, 429.

- **Weighted Box Fusion** — para el método de ensemble:
  > Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117.

Ver [[Memoria-academica]] para el resto de referencias bibliográficas.
