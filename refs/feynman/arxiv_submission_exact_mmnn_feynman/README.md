# arXiv source bundle

Compile with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` contains the bibliography directly. All seven external PDF figures
are in `figures/`; the MMNN/Feynman schematic is native TikZ. No shell escape,
external data, or generated auxiliary file is required.
