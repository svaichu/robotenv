"""Choosing an MDP interactively: ``catalog`` (what exists for this robot and task), ``ObsSelector`` (pick observation
keys and see the result) and ``preview`` (run the environment and look at what comes out). Needs torch for ``preview`` and
for the task's keys; the rest reads the registry. Intended for a notebook or a REPL; every object prints as a table.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .._pretty import indent, pre_html, table
from ..config.mdp import TASK_PREFIX, MdpConfig
from ..errors import ConfigError
from .registry import KINDS, REQUIRED, Info, get_term, info_from_robot, list_terms

__all__ = ["Catalog", "CatalogRow", "ObsSelector", "Preview", "catalog", "preview"]


def _resolved(cfg: Any) -> Any:
    """An ``EnvConfig`` from an ``EnvConfig`` or a ``BaseEnvConfig``."""
    return cfg.resolve() if hasattr(cfg, "resolve") and not hasattr(cfg, "hash_dict") else cfg


@dataclass(frozen=True)
class CatalogRow:
    kind: str
    name: str
    shape: tuple[int, ...] | None  # observation keys: the shape of one env's value
    dtype: str | None
    params: str  # reward and termination terms: ``name=default`` pairs, ``name`` when required
    doc: str
    available: bool
    reason: str | None  # why it is not available


class Catalog(list):
    """The rows of :func:`catalog`; ``rows`` is the list itself. Prints as a table."""

    def __init__(self, rows: Iterable[CatalogRow], kind: str, notes: Iterable[str] = ()):
        super().__init__(rows)
        self.kind = kind
        self.notes = tuple(notes)

    def names(self, available_only: bool = True) -> list[str]:
        return [r.name for r in self if r.available or not available_only]

    def __repr__(self) -> str:
        if self.kind == "observation":
            rows = [(r.name, "x".join(map(str, r.shape)) if r.shape is not None else "?", r.dtype, r.doc if r.available else f"[n/a: {r.reason}] {r.doc}") for r in self]
            lines = table(["key", "shape", "dtype", "description"], rows, prefix="")
        else:
            rows = [(r.name, r.params, r.doc if r.available else f"[n/a: {r.reason}] {r.doc}") for r in self]
            lines = table([self.kind, "parameters", "description"], rows, prefix="")
        return "\n".join([*lines, *(f"note: {n}" for n in self.notes)])

    def _repr_html_(self) -> str:
        return pre_html(repr(self))


def _params_text(term: Any) -> str:
    return ", ".join(n if p.default is REQUIRED else f"{n}={p.default!r}" for n, p in term.params.items())


def _task_rows(cfg: Any) -> tuple[list[CatalogRow], list[str]]:
    """The task's own observation keys. Builds the task's engine side, which may need the provider's library."""
    from ..config.providers import get_task_provider

    try:
        task = get_task_provider(cfg.task.params.provider).build(cfg, cfg.sim.simulator)
        spec = task.observation_spec(1)
    except Exception as e:  # a provider whose library is missing, or a scene that cannot load
        return [], [f"the keys of task {cfg.task.id!r} are not listed: {type(e).__name__}: {e}"]
    rows = []
    for key in spec.keys():
        leaf = spec[key]
        dtype = str(leaf.dtype).replace("torch.", "")
        rows.append(CatalogRow("observation", TASK_PREFIX + key, tuple(leaf.shape[1:]), dtype, "", "an observation of the task", True, None))
    return rows, []


def catalog(cfg: Any, kind: str = "observation", *, task_keys: bool = True) -> Catalog:
    """What an MDP of ``cfg`` can use.

    ``kind`` is ``"observation"``, ``"reward"`` or ``"termination"``. For observations, shapes are resolved against the
    robot of ``cfg`` (``proprio/qpos`` is ``n`` long for an ``n``-joint arm), keys the robot cannot provide are marked
    unavailable, and the task's own keys (``task/...``) are listed when the task can be built (``task_keys=False`` skips
    that). Rewards and terminations list their parameters; their ``obs_key`` parameters take any observation key.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    cfg = _resolved(cfg)
    info = info_from_robot(cfg.robot)
    rows: list[CatalogRow] = []
    for name in list_terms(kind):
        term = get_term(kind, name)
        reason = term.why_unavailable(info)
        shape = None if term.shape is None else tuple(term.shape(info))
        rows.append(CatalogRow(kind, name, shape, term.dtype if kind == "observation" else None, _params_text(term), term.doc, reason is None, reason))
    notes: list[str] = []
    if kind == "observation" and task_keys:
        extra, notes = _task_rows(cfg)
        rows += extra
    return Catalog(rows, kind, notes)


def _flat_size(shape: tuple[int, ...]) -> int:
    return int(math.prod(shape)) if shape else 1


class ObsSelector:
    """Pick the observation keys of an environment, and see what you picked.

    Immutable: every call returns a new selector. ``ObsSelector(cfg)`` starts from the keys ``cfg`` already selects, or from
    nothing when it takes the task's own. ::

        sel = ObsSelector(cfg).select("proprio/qpos", "proprio/ee_pos", "task/target_pos")
        print(sel)                 # the keys, their shapes and the total size
        cfg2 = sel.apply()         # an EnvConfig with task.mdp.observations set (rewards and terminations kept)
    """

    __slots__ = ("_cfg", "_catalog", "_keys")

    def __init__(self, cfg: Any, keys: Iterable[str] | None = None, *, _catalog: Catalog | None = None):
        self._cfg = _resolved(cfg)
        self._catalog = _catalog if _catalog is not None else catalog(self._cfg)
        if keys is None:
            mdp = self._cfg.task.mdp
            keys = () if mdp is None or mdp.observations is None else mdp.observations
        self._keys = tuple(sorted(set(keys)))

    def _derive(self, keys: Iterable[str]) -> "ObsSelector":
        return ObsSelector(self._cfg, keys, _catalog=self._catalog)

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def _row(self, key: str) -> CatalogRow:
        for r in self._catalog:
            if r.name == key:
                return r
        known = self._catalog.names(available_only=False)
        hint = f" ({self._catalog.notes[0]})" if self._catalog.notes and key.startswith(TASK_PREFIX) else ""
        raise ConfigError("observations", f"unknown observation key {key!r}; known: {known}{hint}")

    def select(self, *keys: str) -> "ObsSelector":
        """Add keys. Raises ``ConfigError`` for an unknown key or one the robot cannot provide."""
        for key in keys:
            row = self._row(key)
            if not row.available:
                raise ConfigError("observations", f"observation key {key!r} is not available: {row.reason}")
        return self._derive((*self._keys, *keys))

    def deselect(self, *keys: str) -> "ObsSelector":
        """Remove keys (keys that were not selected are ignored)."""
        return self._derive(k for k in self._keys if k not in keys)

    def select_group(self, prefix: str) -> "ObsSelector":
        """Add every available key under ``prefix`` (``"proprio"``, ``"task"``)."""
        pre = prefix.rstrip("/") + "/"
        found = [r.name for r in self._catalog if r.name.startswith(pre) and r.available]
        if not found:
            raise ConfigError("observations", f"no available keys under {prefix!r}")
        return self._derive((*self._keys, *found))

    def clear(self) -> "ObsSelector":
        return self._derive(())

    def mdp(self) -> MdpConfig:
        """The ``MdpConfig`` of the current config with ``observations`` replaced by the selection."""
        mdp = self._cfg.task.mdp
        if mdp is None:
            return MdpConfig(None, None, self._keys)
        return replace(mdp, observations=self._keys)

    def apply(self) -> Any:
        """The ``EnvConfig`` with the selection written into ``task.mdp.observations``."""
        task = replace(self._cfg.task, mdp=self.mdp())
        return type(self._cfg)(robot=self._cfg.robot, control=self._cfg.control, task=task, sim=self._cfg.sim)

    @property
    def size(self) -> int:
        """Total number of numbers per env across the selected keys (booleans and counters included)."""
        return sum(_flat_size(self._row(k).shape or ()) for k in self._keys)

    def __repr__(self) -> str:
        if not self._keys:
            return "ObsSelector: nothing selected (the environment would observe only its flags)"
        rows = [
            (k, "x".join(map(str, self._row(k).shape or ())) or "scalar", self._row(k).dtype) for k in self._keys
        ]
        return "\n".join([f"ObsSelector: {len(self._keys)} keys, {self.size} numbers per env", *table(["key", "shape", "dtype"], rows)])

    def _repr_html_(self) -> str:
        return pre_html(repr(self))

    def widget(self):
        """A checklist of the available keys for a notebook (needs ``ipywidgets``). Ticking a box edits the selection; the
        widget's ``.selector`` holds the current :class:`ObsSelector` and its output shows the shapes and total size."""
        try:
            import ipywidgets as w
        except ImportError as e:
            raise ImportError("the key checklist needs ipywidgets: pip install ipywidgets") from e
        from IPython.display import display  # ipywidgets depends on IPython

        out = w.Output()
        box = w.VBox()
        state = {"sel": self}

        def show() -> None:
            out.clear_output()
            with out:
                print(state["sel"])

        def on_change(change) -> None:
            key, on = change["owner"].description, change["new"]
            state["sel"] = state["sel"].select(key) if on else state["sel"].deselect(key)
            box.selector = state["sel"]
            show()

        boxes = []
        for r in self._catalog:
            if not r.available:
                continue
            shape = "x".join(map(str, r.shape or ())) or "scalar"
            cb = w.Checkbox(value=r.name in self._keys, description=r.name, indent=False, tooltip=f"{shape}: {r.doc}")
            cb.observe(on_change, names="value")
            boxes.append(cb)
        box.children = (*boxes, out)
        box.selector = self
        show()
        display(box)
        return box


@dataclass(frozen=True)
class PreviewRow:
    key: str
    shape: tuple[int, ...]
    dtype: str
    low: float | None
    high: float | None
    mean: float | None


class Preview(list):
    """The rows of :func:`preview`: what the environment's TensorDict holds after a few steps. Prints as a table."""

    def __init__(self, rows: Iterable[PreviewRow], env_hash: str, steps: int, reward_terms: dict[str, float]):
        super().__init__(rows)
        self.env_hash, self.steps, self.reward_terms = env_hash, steps, reward_terms

    def __repr__(self) -> str:
        def num(x):
            return "" if x is None else f"{x:.4g}"

        rows = [(r.key, "x".join(map(str, r.shape)) or "scalar", r.dtype, num(r.low), num(r.high), num(r.mean)) for r in self]
        lines = [f"Preview  env_hash={self.env_hash}  after {self.steps} steps of random actions", *table(["key", "shape", "dtype", "min", "max", "mean"], rows)]
        if self.reward_terms:
            lines += ["reward terms (mean over envs, last step)", *indent(" ".join(f"{k}={v:.4g}" for k, v in self.reward_terms.items())).split("\n")]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return pre_html(repr(self))


def preview(cfg: Any, steps: int = 5, *, num_envs: int | None = None) -> Preview:
    """Build the environment of ``cfg``, reset it, take ``steps`` random actions and report every entry of the TensorDict
    (shape, dtype, range), the reward, and the mean of each reward term. ``num_envs`` overrides ``cfg.sim.num_envs`` for
    the preview. Catches a bad scale or a term that is constant before a training run."""
    import torch

    from ..envs import make_env

    cfg = _resolved(cfg)
    if num_envs is not None:
        cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, num_envs=num_envs))
    env = make_env(cfg)
    try:
        td = env.reset()
        for _ in range(steps):
            td["action"] = env.action_spec.rand()
            td = env.step(td)["next"]
        rows: list[PreviewRow] = []

        def walk(node: Any, prefix: tuple[str, ...]) -> None:
            for key in node.keys():
                value = node[key]
                if hasattr(value, "keys"):
                    walk(value, (*prefix, key))
                    continue
                name = "/".join((*prefix, key))
                t = value.detach().cpu()
                shape = tuple(t.shape[1:])
                if t.dtype == torch.bool:
                    f = t.float()
                    rows.append(PreviewRow(name, shape, "bool", float(f.min()), float(f.max()), float(f.mean())))
                else:
                    f = t.float()
                    rows.append(PreviewRow(name, shape, str(t.dtype).replace("torch.", ""), float(f.min()), float(f.max()), float(f.mean())))

        walk(td, ())
        rows.sort(key=lambda r: r.key)
        terms = {k: float(v.float().mean()) for k, v in getattr(env, "reward_terms", {}).items()}
        return Preview(rows, env.env_hash, steps, terms)
    finally:
        env.close()
