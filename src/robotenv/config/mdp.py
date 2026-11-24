"""The MDP of a task as data: which rewards, terminations and observations the environment computes. numpy only.

``TaskConfig.mdp`` is ``None`` (the task provider's own reward, success rule and observations) or an :class:`MdpConfig`.
Each of its three parts is also ``None`` to keep that part as the provider has it, so a config can change the reward and
nothing else. Terms are named functions of ``robotenv.mdp`` (``robotenv.mdp.catalog`` lists them) with their parameters
as data; their parameters are completed with the registered defaults when the term is built, so the stored (and hashed)
term never depends on a default that changes later.

* ``rewards``: a tuple of :class:`RewardTerm`; the step reward is the sum of ``weight * term``. ``()`` is a reward of 0.
* ``terminations``: a tuple of :class:`TerminationTerm`. Each says what a true value means: ``"success"`` (the task is
  solved: the ``success`` flag, and the episode ends), ``"failure"`` (the episode ends: ``terminated``) or ``"timeout"``
  (the episode is cut: ``truncated``). ``max_episode_steps`` always truncates as well.
* ``observations``: the keys of the observation TensorDict, ``("proprio/qpos", "task/target_pos", ...)``. Nothing else
  about observations is configured: scaling, stacking and noise are TorchRL transforms of the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .._canon import as_bool, as_float, check_keys, within
from ..errors import ConfigError
from ..mdp.registry import Info, get_term, list_terms

__all__ = [
    "MdpConfig",
    "RewardTerm",
    "TerminationTerm",
    "TERMINATION_KINDS",
    "check_observation_key",
    "reward",
    "termination",
]

TERMINATION_KINDS = ("success", "failure", "timeout")
TASK_PREFIX = "task/"


def _pairs(params: Any, path: str) -> tuple[tuple[str, Any], ...]:
    if isinstance(params, Mapping):
        items = list(params.items())
    else:
        try:
            items = [tuple(p) for p in params]
            if any(len(p) != 2 for p in items):
                raise TypeError
        except TypeError:
            raise ConfigError(path, "expected a mapping of parameter name to value") from None
    for k, _ in items:
        if not isinstance(k, str) or not k:
            raise ConfigError(path, f"parameter names must be strings, got {k!r}")
    return tuple(sorted((k, v) for k, v in items))


def _term_params(kind: str, fn: Any, params: Any, path: str) -> tuple[str, tuple[tuple[str, Any], ...]]:
    if not isinstance(fn, str) or not fn:
        raise ConfigError(f"{path}.fn" if path else "fn", "must be the name of a registered term")
    term = get_term(kind, fn, f"{path}.fn" if path else "fn")
    resolved = term.resolve_params(dict(_pairs(params, f"{path}.params" if path else "params")), f"{path}.params" if path else "params")
    for name, spec in term.params.items():
        if spec.obs_key:
            check_observation_key(resolved[name], f"{path}.params.{name}" if path else f"params.{name}")
    return fn, tuple(sorted(resolved.items()))


def check_observation_key(key: Any, path: str) -> None:
    """``key`` must be a registered observation key or a task's (``task/...``, checked when the environment is built)."""
    if not isinstance(key, str) or not key:
        raise ConfigError(path, "an observation key must be a non-empty string")
    if key.startswith(TASK_PREFIX):
        if len(key) == len(TASK_PREFIX):
            raise ConfigError(path, "'task/' needs the name of one of the task's observations: 'task/<name>'")
        return
    if key not in list_terms("observation"):
        raise ConfigError(
            path,
            f"unknown observation key {key!r}; built-in keys: {list(list_terms('observation'))}; "
            "keys of the task are written 'task/<name>' (the task lists them when the environment is built)",
        )


@dataclass(frozen=True, slots=True)
class RewardTerm:
    """``weight * fn(ctx, **params)``, summed with the other terms. ``params`` is a mapping; it is stored as sorted
    ``(name, value)`` pairs, completed with the term's defaults. Negative weights make penalties."""

    fn: str
    weight: float
    params: Any

    def __post_init__(self):
        fn, params = _term_params("reward", self.fn, self.params, "")
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "weight", as_float(self.weight, "weight"))
        object.__setattr__(self, "params", params)

    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def to_dict(self) -> dict[str, Any]:
        return {"fn": self.fn, "weight": self.weight, "params": self.params_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RewardTerm":
        check_keys(d, {"fn", "weight", "params"}, "")
        return cls(fn=d["fn"], weight=d["weight"], params=d["params"])


@dataclass(frozen=True, slots=True)
class TerminationTerm:
    """A condition ``fn(ctx, **params)`` that is true or false per env.

    ``kind``: ``"success"`` sets the ``success`` flag (the environment's ``success`` is the OR of its success terms, or the
    task's own test when there are none); ``"failure"`` ends the episode as ``terminated``; ``"timeout"`` ends it as
    ``truncated``. A success term also ends the episode (``terminated``) when ``terminate`` is true; with ``terminate``
    false it only sets the flag. ``terminate`` must be true for the other kinds.
    """

    fn: str
    params: Any
    kind: str
    terminate: bool

    def __post_init__(self):
        fn, params = _term_params("termination", self.fn, self.params, "")
        if self.kind not in TERMINATION_KINDS:
            raise ConfigError("kind", f"must be one of {list(TERMINATION_KINDS)}, got {self.kind!r}")
        as_bool(self.terminate, "terminate")
        if not self.terminate and self.kind != "success":
            raise ConfigError("terminate", f"must be true for kind {self.kind!r}: only a success term can just set the flag")
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "params", params)

    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def to_dict(self) -> dict[str, Any]:
        return {"fn": self.fn, "params": self.params_dict(), "kind": self.kind, "terminate": self.terminate}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TerminationTerm":
        check_keys(d, {"fn", "params", "kind", "terminate"}, "")
        return cls(fn=d["fn"], params=d["params"], kind=d["kind"], terminate=d["terminate"])


def reward(fn: str, weight: float = 1.0, **params: Any) -> RewardTerm:
    """``RewardTerm(fn, weight, params)`` with keyword parameters: ``reward("distance", -1.0, a="proprio/ee_pos", b="task/target_pos")``."""
    return RewardTerm(fn=fn, weight=weight, params=params)


def termination(fn: str, kind: str = "failure", *, terminate: bool = True, **params: Any) -> TerminationTerm:
    """``TerminationTerm(fn, params, kind, terminate)`` with keyword parameters: ``termination("abs_above", key="proprio/qvel", limit=50.0)``."""
    return TerminationTerm(fn=fn, params=params, kind=kind, terminate=terminate)


def _tuple_of(cls: type, value: Any, path: str) -> tuple | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise ConfigError(path, f"expected a sequence of {cls.__name__} or None, got {type(value).__name__}")
    items = tuple(value)
    for i, item in enumerate(items):
        if not isinstance(item, cls):
            raise ConfigError(f"{path}[{i}]", f"expected a {cls.__name__}, got {type(item).__name__}")
    return items


@dataclass(frozen=True, slots=True)
class MdpConfig:
    """What the environment computes besides the physics. A part that is ``None`` stays as the task provider has it."""

    rewards: Any
    terminations: Any
    observations: Any

    def __post_init__(self):
        object.__setattr__(self, "rewards", _tuple_of(RewardTerm, self.rewards, "rewards"))
        object.__setattr__(self, "terminations", _tuple_of(TerminationTerm, self.terminations, "terminations"))
        obs = self.observations
        if obs is not None:
            if isinstance(obs, (str, bytes)) or not hasattr(obs, "__iter__"):
                raise ConfigError("observations", f"expected a sequence of keys or None, got {type(obs).__name__}")
            keys = tuple(obs)
            for i, k in enumerate(keys):
                check_observation_key(k, f"observations[{i}]")
            obs = tuple(sorted(set(keys)))  # the order of a selection is not information: the TensorDict is keyed
        object.__setattr__(self, "observations", obs)

    @property
    def is_default(self) -> bool:
        """True when nothing is configured: the provider's own reward, termination and observations."""
        return self.rewards is None and self.terminations is None and self.observations is None

    def check_robot(self, robot: Any, path: str = "mdp") -> None:
        """The part of the validation that needs the robot: observation keys the robot cannot provide."""
        from ..mdp.registry import info_from_robot

        info = info_from_robot(robot)
        keys: list[tuple[str, str]] = [(f"{path}.observations[{i}]", k) for i, k in enumerate(self.observations or ())]
        for attr, kind in (("rewards", "reward"), ("terminations", "termination")):
            for i, t in enumerate(getattr(self, attr) or ()):
                spec = get_term(kind, t.fn)
                keys += [(f"{path}.{attr}[{i}].params.{n}", dict(t.params)[n]) for n, p in spec.params.items() if p.obs_key]
        for where, key in keys:
            reason = _unavailable(key, info)
            if reason:
                raise ConfigError(where, f"observation key {key!r} is not available: {reason}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rewards": None if self.rewards is None else [t.to_dict() for t in self.rewards],
            "terminations": None if self.terminations is None else [t.to_dict() for t in self.terminations],
            "observations": None if self.observations is None else list(self.observations),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MdpConfig":
        check_keys(d, set(), "", optional={"rewards", "terminations", "observations"})

        def terms(name: str, term_cls: type):
            raw = d.get(name)
            if raw is None:
                return None
            if isinstance(raw, (str, bytes, Mapping)) or not hasattr(raw, "__iter__"):
                raise ConfigError(name, f"expected a list of mappings, got {type(raw).__name__}")
            return tuple(within(f"{name}[{i}]", term_cls.from_dict, r) for i, r in enumerate(raw))

        return cls(
            rewards=terms("rewards", RewardTerm),
            terminations=terms("terminations", TerminationTerm),
            observations=d.get("observations"),
        )

    def _short(self) -> str:
        """One line, for the rows of ``TaskConfig`` and ``BaseEnvConfig``."""
        parts = []
        if self.rewards is not None:
            parts.append("rewards=[" + ", ".join(f"{t.weight:+g}*{t.fn}" for t in self.rewards) + "]")
        if self.terminations is not None:
            parts.append("terminations=[" + ", ".join(f"{t.kind}:{t.fn}" for t in self.terminations) + "]")
        if self.observations is not None:
            parts.append("observations=[" + ", ".join(self.observations) + "]")
        return "MdpConfig(" + " ".join(parts) + ")"

    _oneline = _short

    def __repr__(self) -> str:
        def fmt_term(t) -> str:
            args = ", ".join(f"{k}={v!r}" for k, v in t.params)
            return f"{t.fn}({args})"

        rows = ["MdpConfig"]
        if self.rewards is None:
            rows.append("  rewards       the task's own")
        else:
            rows.append("  rewards")
            rows += [f"    {t.weight:+g} * {fmt_term(t)}" for t in self.rewards] or ["    (none: reward 0)"]
        if self.terminations is None:
            rows.append("  terminations  the task's own")
        else:
            rows.append("  terminations")
            rows += [f"    {t.kind}{'' if t.terminate else ' (flag only)'}: {fmt_term(t)}" for t in self.terminations] or [
                "    (none)"
            ]
        rows.append("  observations  " + ("the task's own" if self.observations is None else ", ".join(self.observations) or "(none)"))
        return "\n".join(rows)


def _unavailable(key: str, info: Info) -> str | None:
    if key.startswith(TASK_PREFIX):
        return None
    return get_term("observation", key).why_unavailable(info)
