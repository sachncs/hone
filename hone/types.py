"""Shared type aliases used across the hone package."""

from __future__ import annotations

JsonScalar = str | int | float | bool | None
JsonObject = dict[str, object]
