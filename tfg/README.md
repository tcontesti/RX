# tfg/

Documento académico del proyecto. Dos formatos:

- `MemoriaTFG.pdf` — PDF compilado, listo para lectura.
- `source/` — fuente LaTeX completa, lista para recompilar.

## Estructura de la fuente

```
tfg/source/
├── Memoria_v2_IEEE.tex     documento principal (formato IEEE)
├── Paper_definitivo.tex    versión paper (extendida)
├── secciones_nuevas.tex    secciones añadidas en la extensión hospitalaria
├── biblio_rectifier.bib    bibliografía BibTeX
├── images/                 figuras integradas en el documento
└── *.png                   FROC curves, matrices de confusión y curvas de entrenamiento
```

## Compilación

Requisitos: una distribución LaTeX moderna con `pdflatex`, `biber` (o `bibtex`) y los paquetes habituales de IEEEtran (`IEEEtran.cls`, `amsmath`, `graphicx`, `hyperref`, `xcolor`, `listings`).

Distribuciones recomendadas:

- **Windows:** [MiKTeX](https://miktex.org/) (instala los paquetes que falten bajo demanda).
- **Linux/macOS:** [TeX Live](https://www.tug.org/texlive/) completo, o el subconjunto `texlive-full` de la distribución.

Para compilar el documento principal:

```bash
cd tfg/source
pdflatex Memoria_v2_IEEE.tex
biber    Memoria_v2_IEEE          # o: bibtex Memoria_v2_IEEE
pdflatex Memoria_v2_IEEE.tex
pdflatex Memoria_v2_IEEE.tex
```

Con `latexmk` (mucho más cómodo):

```bash
cd tfg/source
latexmk -pdf Memoria_v2_IEEE.tex
```

Para regenerar el PDF que se muestra en el repositorio:

```bash
cp Memoria_v2_IEEE.pdf ../MemoriaTFG.pdf
```

## Citar este documento

```bibtex
@misc{link_contesti_2026_cxr,
  author       = {Link Cladera, Marc and Contest\'i Coll, Antonio},
  title        = {Detecci\'on autom\'atica de n\'odulos pulmonares en
                  radiograf\'ias de t\'orax mediante deep learning},
  year         = {2026},
  howpublished = {Bachelor's Thesis, Escola Polit\`ecnica Superior,
                  Universitat de les Illes Balears},
  note         = {Supervisor: Dr. Miquel Mir\'o Nicolau}
}
```

## Licencia

El documento académico está liberado bajo Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Consultar `../LICENSE` en la raíz del repositorio.
