# arXiv submission bundle

Compile from this directory with:

```sh
pdflatex main.tex
pdflatex main.tex
```

The bundle is self-contained: `main.tex`, `neurips_2026.sty`, and every referenced PDF figure are included. Replace the `Anonymous Authors` line before public submission.

The reproducible power-law experiment is maintained outside this minimal bundle at `experiments/feynman/run_powerlaw_early_stopping.py`, with tests in `tests/test_powerlaw_weingarten.py`.
