NeurIPS 2026 supplemental source bundle
=======================================

This anonymized bundle contains the LaTeX source, appendix, checklist, generated figures, and Python scripts needed to reproduce the paper PDF and the spectral-bias figures.

Contents
--------

- `neurips/neurips_2026.tex`: main paper source.
- `neurips/neurips_appendix_after_checklist.tex`: supplementary appendix included after the checklist.
- `neurips/checklist.tex`: completed NeurIPS checklist.
- `neurips/*.sty`: local style dependencies used by the paper.
- `lr_spectral_workshop/figures/*.png`: generated figures used by the paper.
- `lr_spectral_workshop/figdata/`: deterministic generated arrays and intermediate figure data.
- `lr_spectral_workshop/generate_experiments.py`: figure-generation script.
- `lr_spectral_workshop/scripts/generate_experiments.py`: compact script copy used for reproduction.

Compile the paper
-----------------

From the bundle root:

```bash
cd neurips
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

Regenerate figures
------------------

From the bundle root:

```bash
cd lr_spectral_workshop
python generate_experiments.py
```

The experiments are synthetic deterministic Fourier/kernel-limit and CPWL diagnostics. They do not require external datasets or pretrained models.
