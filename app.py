"""Vercel ASGI entrypoint; the same API also runs with Uvicorn locally."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from courtiq.api import app  # noqa: E402,F401
