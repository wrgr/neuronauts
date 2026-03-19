#!/usr/bin/env bash
# Render docs/whitepaper.md to a pretty PDF for review.
# Requires: pandoc, pdflatex (MacTeX or BasicTeX)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INPUT="docs/whitepaper.md"
OUTPUT="docs/whitepaper.pdf"

echo "Rendering $INPUT -> $OUTPUT"
pandoc "$INPUT" \
  -o "$OUTPUT" \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -V block-headings \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V toccolor=gray \
  --toc \
  --number-sections \
  -V papersize=letter \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=article

echo "Done. Output: $OUTPUT"
