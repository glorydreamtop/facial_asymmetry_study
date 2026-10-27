#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" 01_prepare.py
"$PYTHON" 02_axis_models.py
"$PYTHON" 03_global_fa.py
"$PYTHON" 04_tables.py
"$PYTHON" 05_docx.py
