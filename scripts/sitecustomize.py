"""Runtime compatibility patches for RecClaw scripts.

Older RecBole LightGCN builds normalized adjacency matrices with
``dok_matrix.update(dict_of_coordinates)``. Newer SciPy versions intentionally
disable direct DOK update, which crashes before LightGCN training starts. This
small startup patch restores the coordinate-dict update behavior without
editing RecBole core source.
"""

from __future__ import annotations

import os

if os.environ.get("RECCLAW_ENABLE_SCIPY_DOK_PATCH") != "1":
    dok_matrix = None  # type: ignore[assignment]
else:
    try:
        from scipy.sparse import dok_matrix
    except Exception:  # pragma: no cover - SciPy may be absent in lightweight tests.
        dok_matrix = None  # type: ignore[assignment]


def _recclaw_dok_update(self, values) -> None:  # type: ignore[no-untyped-def]
    if hasattr(values, "items"):
        iterator = values.items()
    else:
        iterator = values
    for key, value in iterator:
        self[key] = value


if dok_matrix is not None:
    try:
        dok_matrix.update = _recclaw_dok_update  # type: ignore[method-assign]
    except Exception:
        pass
