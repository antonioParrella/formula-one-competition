"""Path bootstrap.

Puts the parent repo's ``src/`` directory (Scorer, DRIVER_MAP, OpenF1
helpers) and this project root on ``sys.path`` so modules here can do
``from scorer import Scorer`` without packaging either tree.

Import this before any ``src/`` import:

    import _paths  # noqa: F401
    from scorer import Scorer
"""

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent      # f1-tipping/
REPO_ROOT = PKG_ROOT.parent                     # formula-one-competition/
SRC_DIR = REPO_ROOT / "src"

for _p in (str(PKG_ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows consoles default to cp1252, which can't print the table rules /
# flags used in reports. Reconfigure rather than strip the output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
