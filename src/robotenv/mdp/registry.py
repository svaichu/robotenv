"""The registry of MDP terms: named reward, termination and observation functions. numpy only.

A *term* is a function the environment evaluates every step on an :class:`~robotenv.mdp.context.MdpContext`, a view
of the robot state, the task state and the action. Configs (``config.mdp``) refer to terms by name and carry their
parameters as data, so a reward, a termination rule or the set of observations is part of the ``env_hash``. The
functions themselves live here, next to their metadata (parameters and defaults, shape, availability), so a config can
be validated without torch or an engine: the term modules never import one at module level.

Three kinds of term:

``reward``
    ``fn(ctx, **params) -> (N,) or (N, 1)``. The config weights it; the sum is the step's reward.
``termination``
    ``fn(ctx, **params) -> bool (N,) or (N, 1)``; the config says what a true value means (success, failure or time out).
``observation``
    ``fn(ctx) -> (N, *shape)``; the name is the key in the observation TensorDict (``"proprio/qpos"`` is
    ``td["proprio", "qpos"]``). No parameters: the config only chooses which keys exist. ``shape`` and ``dtype`` are
    declared, so the specs are known before anything runs.

Providers register their own terms with :func:`register_term`; the names of a provider's terms are conventionally
prefixed with its namespace so they never collide with the built-in ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..errors import ConfigError

__all__ = [
    "KINDS",
    "REQUIRED",
    "Info",
    "Param",
    "Term",
    "get_term",
    "info_from_robot",
    "list_terms",
    "register_term",
    "unregister_term",
]

KINDS = ("reward", "termination", "observation")
_SCALARS = (bool, int, float, str)


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()


@dataclass(frozen=True)
class Info:
    """What a term's availability and shape may depend on: the robot, as numbers. Built from a config
    (:func:`info_from_robot`) or from the compiled robot by the engine side."""

    n: int
    action_dim: int
    has_gripper: bool
    has_ee: bool


def info_from_robot(robot: Any) -> Info:
    """The :class:`Info` of a robot config (``ParametricArm`` or ``ExternalRobot``)."""
    gripper = getattr(robot, "gripper", None) is not None
    ee = getattr(robot, "end_effector", True) is not None
    n = int(robot.n)
    return Info(n=n, action_dim=n + (1 if gripper else 0), has_gripper=gripper, has_ee=ee)


@dataclass(frozen=True)
class Param:
    """One parameter of a reward or termination term: a scalar (bool, int, float or str) with a default or ``REQUIRED``.

    ``obs_key`` marks a string that names an observation key (``"proprio/ee_pos"``): it is checked against the
    observation terms, and against the task's own keys (``task/...``) when the environment is built.
    """

    default: Any = REQUIRED
    obs_key: bool = False
    doc: str = ""
    of: type | None = None  # the type of a required parameter (a default gives its own)

    def kind(self) -> type:
        if self.obs_key:
            return str
        if self.default is REQUIRED:
            return self.of or float
        return type(self.default)


@dataclass(frozen=True)
class Term:
    kind: str
    name: str
    fn: Callable[..., Any]
    params: Mapping[str, Param] = field(default_factory=dict)
    doc: str = ""
    # observation terms
    shape: Callable[[Info], tuple[int, ...]] | None = None
    dtype: str = "float32"
    # availability: ``None`` when the term can run on the robot, otherwise the reason it cannot
    available: Callable[[Info], str | None] | None = None
    # the term needs a task that defines its own reward / success (``task_reward``, ``task_success``)
    needs_task: bool = False

    def why_unavailable(self, info: Info) -> str | None:
        return None if self.available is None else self.available(info)

    def resolve_params(self, given: Mapping[str, Any], path: str = "") -> dict[str, Any]:
        """``given`` checked against the declared parameters, completed with their defaults; the result is what is
        stored and hashed. Unknown names and missing required values raise ``ConfigError``."""
        out: dict[str, Any] = {}
        unknown = sorted(set(given) - set(self.params))
        if unknown:
            known = ", ".join(self.params) or "none"
            raise ConfigError(path, f"{self.kind} term {self.name!r} has no parameter {unknown[0]!r}; its parameters: {known}")
        for key, p in self.params.items():
            where = f"{path}.{key}" if path else key
            if key not in given:
                if p.default is REQUIRED:
                    raise ConfigError(where, f"{self.kind} term {self.name!r} requires this parameter")
                out[key] = p.default
                continue
            value = given[key]
            want = p.kind()
            if want is float:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ConfigError(where, f"expected a number, got {type(value).__name__}")
                value = float(value)
                if value != value or value in (float("inf"), float("-inf")):
                    raise ConfigError(where, "must be finite")
            elif want is bool:
                if not isinstance(value, bool):
                    raise ConfigError(where, f"expected a bool, got {type(value).__name__}")
            elif want is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ConfigError(where, f"expected an integer, got {type(value).__name__}")
            elif want is str:
                if not isinstance(value, str) or not value:
                    raise ConfigError(where, "must be a non-empty string")
            out[key] = value
        return out


_REGISTRY: dict[tuple[str, str], Term] = {}
_STATE = {"builtins": False}


def _load_builtins() -> None:
    if not _STATE["builtins"]:
        _STATE["builtins"] = True
        from . import terms  # noqa: F401  (registers the built-in terms)


def register_term(
    kind: str,
    name: str,
    *,
    params: Mapping[str, Param] | None = None,
    shape: Callable[[Info], tuple[int, ...]] | None = None,
    dtype: str = "float32",
    available: Callable[[Info], str | None] | None = None,
    needs_task: bool = False,
    replace: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register ``fn`` as the term ``name`` of ``kind``. The docstring is the description shown by
    ``robotenv.mdp.catalog``. Registering a taken name is an error unless ``replace`` is true."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not isinstance(name, str) or not name:
        raise ValueError("a term needs a name")
    if kind == "observation":
        if shape is None:
            raise ValueError(f"observation term {name!r} must declare its shape")
        if params:
            raise ValueError(f"observation term {name!r} takes no parameters: the config only selects the key")
    for pname, p in (params or {}).items():
        if p.default is not REQUIRED and not isinstance(p.default, _SCALARS):
            raise ValueError(f"parameter {pname!r} of {name!r}: the default must be a bool, int, float or str")
        if p.of is not None and p.of not in _SCALARS:
            raise ValueError(f"parameter {pname!r} of {name!r}: 'of' must be bool, int, float or str")

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _load_builtins()  # so a name that a built-in term already has is caught (re-entrant: the flag is set first)
        key = (kind, name)
        if key in _REGISTRY and not replace and _REGISTRY[key].fn is not fn:
            raise ValueError(f"{kind} term {name!r} is already registered; pass replace=True to override it")
        doc = " ".join((fn.__doc__ or "").split())
        _REGISTRY[key] = Term(
            kind=kind, name=name, fn=fn, params=dict(params or {}), doc=doc, shape=shape, dtype=dtype,
            available=available, needs_task=needs_task,
        )
        return fn

    return wrap


def unregister_term(kind: str, name: str) -> None:
    _REGISTRY.pop((kind, name), None)


def list_terms(kind: str) -> tuple[str, ...]:
    """Names of the registered terms of ``kind``, sorted."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    _load_builtins()
    return tuple(sorted(n for k, n in _REGISTRY if k == kind))


def get_term(kind: str, name: str, path: str = "") -> Term:
    """The term ``name`` of ``kind``; ``ConfigError`` (at ``path``) listing the known names when there is none."""
    _load_builtins()
    term = _REGISTRY.get((kind, name))
    if term is None:
        raise ConfigError(path, f"unknown {kind} term {name!r}; registered: {list(list_terms(kind))}")
    return term
