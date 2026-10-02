"""
SURVILLENCE TRAFFIC — Backwards-Compatible Backend Launcher
============================================================
The project now has one authoritative backend: the root app.py.
This wrapper keeps older workflows that launch backend/server.py working
without accidentally starting the obsolete duplicate server.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("SURVILLENCE TRAFFIC — Unified backend compatibility launcher")
    print("Authoritative entrypoint: app.py")
    print(f"Opening: http://localhost:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
