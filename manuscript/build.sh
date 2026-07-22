#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -synctex=1 main.tex
