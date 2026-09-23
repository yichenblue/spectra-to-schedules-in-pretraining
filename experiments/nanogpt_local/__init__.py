"""Local nanoGPT backbone and two-stage pilot utilities.

The model symbols are loaded lazily so the pure-NumPy experiment analyses can
be imported in lightweight environments that intentionally do not install
PyTorch.
"""

from __future__ import annotations

from typing import Any


__all__ = ["GPT", "GPTConfig", "load_checkpoint"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .model import GPT, GPTConfig, load_checkpoint

        return {"GPT": GPT, "GPTConfig": GPTConfig, "load_checkpoint": load_checkpoint}[
            name
        ]
    raise AttributeError(name)
