"""Bridge to the original tested implementation in ``python/ds_rangenet_v3.py``.

The canonical model code predates this ``src/`` package and is kept intact so
that nothing about the paper's tested architecture changes. This helper puts the
repository ``python/`` directory on ``sys.path`` and imports the module, so the
new package can re-export it without duplicating a single line of model code.
"""
from __future__ import annotations

import importlib
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

ds_rangenet_v3 = importlib.import_module("ds_rangenet_v3")
