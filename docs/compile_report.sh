#!/bin/bash
# Recompile the CS229 final project report PDF from LaTeX source.
# Usage: ./compile_report.sh

set -e

NAME="CS_229_Final_project_Report_JagatBrahma"

pdflatex -interaction=nonstopmode "$NAME.tex"
bibtex "$NAME"
pdflatex -interaction=nonstopmode "$NAME.tex"
pdflatex -interaction=nonstopmode "$NAME.tex"

# Clean up auxiliary files
rm -f "$NAME.aux" "$NAME.log" "$NAME.out" "$NAME.bbl" "$NAME.blg"

echo "Done: $NAME.pdf"
