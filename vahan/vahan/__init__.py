"""
ALL-MAXED INIT 🚀
==================
Unifies imports for VAHAN Analytics package across local, GitHub, and Streamlit Cloud.

Handles nested structure:
Vahan-Analysis/
└── vahan/
    ├── streamlit_app.py
    └── vahan/
        ├── api.py
        ├── parsing.py
        ├── charts.py
        ├── metrics.py
"""

import os
import sys
import importlib
import logging

# ------------------------------
# 🧭 PATH RESOLUTION FIX
# ------------------------------
_pkg_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_pkg_dir, ".."))

# Ensure both parent and package paths are on sys.path
for p in {_pkg_dir, _parent_dir}:
    if p not in sys.path:
        sys.path.insert(0, p)

# ------------------------------
# 🧱 SAFE IMPORT HELPERS
# ------------------------------
def _try_import(name):
    """Try to import a module; return None on failure."""
    try:
        return importlib.import_module(f"vahan.{name}")
    except Exception as e:
        logging.warning(f"[vahan.init] ⚠️ Failed to import {name}: {e}")
        return None

# ------------------------------
# 📦 IMPORT CORE MODULES
# ------------------------------
api = _try_import("api")
charts = _try_import("charts")
metrics = _try_import("metrics")
parsing = _try_import("parsing")

# ------------------------------
# 🔗 RE-EXPORT MAIN FUNCTIONS
# ------------------------------
build_params = getattr(api, "build_params", None)
get_json = getattr(api, "get_json", None)

# ------------------------------
# 🧩 PACKAGE EXPORTS
# ------------------------------
__all__ = [
    "api",
    "charts",
    "metrics",
    "parsing",
    "build_params",
    "get_json",
]

# ------------------------------
# 🪵 LOG INITIALIZATION
# ------------------------------
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logging.info("✅ VAHAN package initialized (ALL-MAXED MODE)")
