#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v python >/dev/null 2>&1; then
  python make_figures.py
fi

xelatex -interaction=nonstopmode -halt-on-error FBG_FiLM_Model_Evolution_Report_CN.tex
xelatex -interaction=nonstopmode -halt-on-error FBG_FiLM_Model_Evolution_Report_CN.tex
echo "Built: FBG_FiLM_Model_Evolution_Report_CN.pdf"
