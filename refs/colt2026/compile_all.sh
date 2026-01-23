#!/bin/bash
# Compile both arXiv and NeurIPS versions of the paper

set -e  # we exit on any error

echo "=================================================="
echo "Compiling arXiv version..."
echo "=================================================="
pdflatex -interaction=nonstopmode TOCOMPILE_templateArxiv.tex > /dev/null
bibtex TOCOMPILE_templateArxiv
pdflatex -interaction=nonstopmode TOCOMPILE_templateArxiv.tex > /dev/null
pdflatex -interaction=nonstopmode TOCOMPILE_templateArxiv.tex > /dev/null
echo "✓ TOCOMPILE_templateArxiv.pdf compiled successfully"
pdfinfo TOCOMPILE_templateArxiv.pdf | grep Pages

echo ""
echo "=================================================="
echo "Compiling NeurIPS version..."
echo "=================================================="
pdflatex -interaction=nonstopmode TOCOMPILE_templateNeurIPS.tex > /dev/null
bibtex TOCOMPILE_templateNeurIPS
pdflatex -interaction=nonstopmode TOCOMPILE_templateNeurIPS.tex > /dev/null
pdflatex -interaction=nonstopmode TOCOMPILE_templateNeurIPS.tex > /dev/null
echo "✓ TOCOMPILE_templateNeurIPS.pdf compiled successfully"
pdfinfo TOCOMPILE_templateNeurIPS.pdf | grep Pages

echo ""
echo "=================================================="
echo "Both versions compiled successfully!"
echo "=================================================="
echo "arXiv version:  TOCOMPILE_templateArxiv.pdf"
echo "NeurIPS version: TOCOMPILE_templateNeurIPS.pdf"

