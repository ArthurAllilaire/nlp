#!/usr/bin/env bash
# build_report.sh — convert REPORT.md to REPORT.pdf
# Run from the project root: bash build_report.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Dependency check ──────────────────────────────────────────────────────────
if ! command -v pandoc &>/dev/null; then
    echo "ERROR: pandoc not found. Install with:"
    echo "  sudo apt install pandoc"
    exit 1
fi

if ! command -v lualatex &>/dev/null; then
    echo "ERROR: lualatex not found. Install with:"
    echo "  sudo apt install texlive-luatex texlive-latex-extra texlive-fonts-recommended"
    exit 1
fi

echo "Building REPORT.pdf ..."

pandoc REPORT.md \
    --output REPORT.pdf \
    --pdf-engine=pdflatex \
    --highlight-style=tango \
    --toc \
    --toc-depth=2 \
    --resource-path=. \
    2>&1

echo ""
echo "Done -> REPORT.pdf"
