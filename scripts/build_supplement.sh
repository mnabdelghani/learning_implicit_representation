#!/usr/bin/env bash
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: $0 [supplement|technical_archive]" >&2
  exit 2
fi
document=${1-supplement}
case "$document" in
  supplement|technical_archive) ;;
  *)
    echo "Usage: $0 [supplement|technical_archive]" >&2
    exit 2
    ;;
esac
cd "$(dirname "$0")/../supplement"
pdflatex -interaction=nonstopmode -halt-on-error "$document.tex"
if grep -Fq '\bibdata' "$document.aux"; then
  if command -v bibtex >/dev/null 2>&1; then BIB=bibtex; else BIB=bibtex8; fi
  "$BIB" "$document"
fi
pdflatex -interaction=nonstopmode -halt-on-error "$document.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$document.tex"
