from __future__ import annotations

import sys
from pathlib import Path


def setup_project_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    pipeline_root = Path(__file__).resolve().parents[1]
    for path in (repo_root, pipeline_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root
