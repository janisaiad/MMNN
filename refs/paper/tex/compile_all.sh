#!/bin/bash
# Compile both arXiv and NeurIPS versions of the paper

set -e  # we exit on any error

echo "=================================================="
echo "Compiling arXiv version..."
echo "=================================================="
pdflatex -interaction=nonstopmode templateArxiv.tex > /dev/null
bibtex templateArxiv
pdflatex -interaction=nonstopmode templateArxiv.tex > /dev/null
pdflatex -interaction=nonstopmode templateArxiv.tex > /dev/null
echo "✓ templateArxiv.pdf compiled successfully"
pdfinfo templateArxiv.pdf | grep Pages

echo ""
echo "=================================================="
echo "Compiling NeurIPS version..."
echo "=================================================="
pdflatex -interaction=nonstopmode templateNeurIPS.tex > /dev/null
bibtex templateNeurIPS
pdflatex -interaction=nonstopmode templateNeurIPS.tex > /dev/null
pdflatex -interaction=nonstopmode templateNeurIPS.tex > /dev/null
echo "✓ templateNeurIPS.pdf compiled successfully"
pdfinfo templateNeurIPS.pdf | grep Pages

echo ""
echo "=================================================="
echo "Both versions compiled successfully!"
echo "=================================================="
echo "arXiv version:  templateArxiv.pdf"
echo "NeurIPS version: templateNeurIPS.pdf"

