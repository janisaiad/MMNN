# Paper Compilation Guide

## Available Versions

### 1. **templateArxiv.tex** - arXiv Preprint Version
- **Style**: PRIMEarxiv (arXiv-compatible format)
- **Pages**: 19 pages
- **Output**: `templateArxiv.pdf`
- **Features**: 
  - Table of contents
  - Full detailed appendices
  - Extended format suitable for preprints

### 2. **templateNeurIPS.tex** - NeurIPS Conference Version  
- **Style**: neurips_2023 (NeurIPS 2023/2024 format)
- **Pages**: 19 pages (may need to trim for 9-page main + unlimited appendix)
- **Output**: `templateNeurIPS.pdf`
- **Features**:
  - Two-column format
  - Appendix after bibliography
  - Standard conference formatting

## Compilation Instructions

### For arXiv version:
```bash
pdflatex templateArxiv.tex
bibtex templateArxiv
pdflatex templateArxiv.tex
pdflatex templateArxiv.tex
```

### For NeurIPS version:
```bash
pdflatex templateNeurIPS.tex
bibtex templateNeurIPS
pdflatex templateNeurIPS.tex
pdflatex templateNeurIPS.tex
```

## File Structure

All content is modular via `\input` commands:

```
templateArxiv.tex / templateNeurIPS.tex  ← Main file (skeleton only)
├── introduction.tex     ← Introduction + Related Work
├── definition.tex       ← Model Definition + NTK Recursion
├── rkhs.tex            ← RKHS Theory
├── spectra.tex         ← Spectral Theory
├── conclusion.tex      ← Conclusion
├── appendix.tex        ← All proofs and technical details
└── references.bib      ← Bibliography database
```

## NeurIPS Submission Guidelines

### Page Limits (NeurIPS 2024)
- **Main paper**: 9 pages (including figures, tables, but excluding references)
- **References**: Unlimited
- **Appendix**: Unlimited (after references)
- **Total with appendix**: No hard limit

### Preparing for Submission

#### Option 1: Use as-is (19 pages may exceed main limit)
Current structure puts everything in main body. You may need to:
1. Move some content from main sections to appendix
2. Condense introduction/related work
3. Remove or shrink figures

#### Option 2: Camera-ready mode
Change line 4 in `templateNeurIPS.tex`:
```latex
\usepackage[final]{neurips_2023}  % instead of [preprint]
```

#### Option 3: Anonymous submission mode
```latex
\usepackage{neurips_2023}  % no options = review mode
```

### Recommended Adjustments for 9-page Limit

If your main content exceeds 9 pages:

1. **Move to appendix**:
   - Detailed proofs (keep theorem statements)
   - Extended related work
   - Additional experimental details
   - Fisher-Kibble technical details

2. **Condense**:
   - Introduction (focus on contributions)
   - Related work (cite more, explain less)
   - Proof sketches instead of full proofs

3. **Structure for NeurIPS**:
   ```
   Main (≤9 pages):
   - Abstract
   - Introduction (1.5 pages)
   - Model & NTK Framework (2 pages)
   - RKHS Theory (1.5 pages)  
   - Spectral Theory (1.5 pages)
   - Conclusion (0.5 pages)
   - References (unlimited)
   
   Appendix (unlimited):
   - All proofs
   - Extended related work
   - Additional experiments
   - Technical lemmas
   ```

## Quick Commands

### Check page count (main only, excluding references):
```bash
pdfinfo templateNeurIPS.pdf | grep Pages
```

### Recompile both versions:
```bash
make all  # if Makefile exists
# OR
bash compile_all.sh
```

### Clean auxiliary files:
```bash
rm -f *.aux *.log *.out *.bbl *.blg *.toc
```

## Notes

- Both versions use the same content files (`introduction.tex`, `definition.tex`, etc.)
- Only the main template file differs (arXiv vs NeurIPS styling)
- Bibliography uses `plain` style for NeurIPS, `unsrt` for arXiv
- All citations and cross-references are preserved in both versions

