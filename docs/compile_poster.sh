#!/usr/bin/env bash
# Compile poster.tex to poster.pdf
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error poster.tex 2>&1 | tail -3
rm -f poster.aux poster.log poster.out poster.toc
echo "Done: poster.pdf"
