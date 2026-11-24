"""The MDP of a task: reward, termination and observation terms, and tools to choose them.

The registry (``register_term``, ``list_terms``, ``get_term``) and the built-in terms need only numpy, so configs can be
validated without an engine. The tools that look at a live environment (``catalog``, ``ObsSelector``, ``preview``) need
torch and are imported on first use. The config classes (``MdpConfig``, ``RewardTerm``, ``TerminationTerm``, ``reward``,
``termination``) are the ones of ``robotenv.config``, re-exported here. Guide: ``docs/configs/mdp.md``.
"""

from __future__ import annotations

from typing import Any

from .registry import KINDS, REQUIRED, Info, Param, Term, get_term, list_terms, register_term, unregister_term

__all__ = [
    "KINDS",
    "REQUIRED",
    "Info",
    "MdpConfig",
    "ObsSelector",
    "Param",
    "RewardTerm",
    "Term",
    "TerminationTerm",
    "catalog",
    "get_term",
    "list_terms",
    "preview",
    "register_term",
    "reward",
    "termination",
    "unregister_term",
]

_LAZY = {
    "MdpConfig": "robotenv.config.mdp",
    "RewardTerm": "robotenv.config.mdp",
    "TerminationTerm": "robotenv.config.mdp",
    "reward": "robotenv.config.mdp",
    "termination": "robotenv.config.mdp",
    "catalog": "robotenv.mdp.tools",
    "ObsSelector": "robotenv.mdp.tools",
    "preview": "robotenv.mdp.tools",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'robotenv.mdp' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
