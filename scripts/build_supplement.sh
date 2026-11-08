#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../supplement"
if command -v bibtex >/dev/null 2>&1; then BIB=bibtex; else BIB=bibtex8; fi
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
"$BIB" supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
