# ICML Paper: Hierarchical Feature Construction in Low-Rank Neural Networks

## Structure

- `main.tex` - Main paper file
- `sections/` - Individual sections
  - `introduction.tex` - Introduction and motivation
  - `related.tex` - Related work
  - `setup.tex` - Architecture and setup
  - `theory.tex` - Theoretical results on hierarchical construction
  - `experiments.tex` - Experimental evidence
  - `instability.tex` - Instability as reinforcement mechanism
  - `discussion.tex` - Discussion and implications
  - `conclusion.tex` - Conclusion
- `appendix/` - Appendices
  - `theory.tex` - Additional theoretical results
  - `experiments.tex` - Additional experimental details
  - `proofs.tex` - Full proofs
- `references.bib` - Bibliography

## Main Claims

1. **Hierarchical Construction**: Low-rank networks ($r \asymp d^\beta$ for $\beta \in (0,1)$) construct features hierarchically through sequential frequency band activation.

2. **High-Rank Failure**: Networks with rank too high ($r \asymp d$ or full-rank) fail to develop hierarchical structure and learn flat representations.

3. **Instability-Reinforcement**: Instability near sharp minima acts as a reinforcement mechanism that drives hierarchical construction.

## Key Results from meanfield_lowrank.tex

The paper incorporates results on:
- Mean-field dynamics in extensive-rank regime
- Sequential frequency band activation
- Instability = reinforcement mechanism
- Stepwise loss curves corresponding to hierarchical construction

## To Do

- [ ] Add actual experimental results and figures
- [ ] Complete theoretical proofs in appendix
- [ ] Add more references
- [ ] Create figures for stepwise loss, frequency spectrum, Hessian spikes
- [ ] Fill in quantitative results from experiments
- [ ] Add code repository link

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Note: Requires `icml2026.sty` style file (download from ICML website).
