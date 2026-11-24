"""The engine side of ``TaskConfig.mdp``: resolves the configured terms against a live task and robot and evaluates them.

Both simulators' environments own one :class:`MdpManager` and ask it for the observation specs and values, the reward
and the termination flags, so a term runs unchanged on the CPU and the GPU simulator. With ``mdp = None`` the manager
reproduces what the environments did before the MDP was configurable (the provider's reward and success, every
observation), so an environment that does not use it behaves exactly as it did. Needs torch and torchrl.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded

from ..config.mdp import TASK_PREFIX, MdpConfig
from ..errors import ConfigError
from .context import MdpContext, RobotState
from .registry import Info, Term, get_term
from .terms import DEFAULT_EPISODE, DEFAULT_PROPRIO

__all__ = ["MdpManager"]

_DTYPES = {"float32": torch.float32, "int64": torch.int64, "bool": torch.bool}


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    """``{"proprio/qpos": x, "success": y}`` -> ``{"proprio": {"qpos": x}, "success": y}``."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        *groups, leaf = key.split("/")
        for g in groups:
            node = node.setdefault(g, {})
        node[leaf] = value
    return out


class MdpManager:
    """``cfg`` is the resolved ``EnvConfig``; ``sim`` the bound simulation (``tasks.base.SimView``) the environment runs;
    ``action_dim`` and ``has_gripper`` describe the action space and the observations the *simulator* provides (the GPU
    simulator has no gripper, so ``proprio/gripper`` is unavailable there even on a robot that has one)."""

    def __init__(self, cfg: Any, task: Any, sim: Any, *, action_dim: int, has_gripper: bool):
        self.cfg = cfg
        self.task = task
        self.sim = sim
        self.device = sim.device
        self.num_envs = sim.num_envs
        self.max_episode_steps = cfg.task.max_episode_steps
        iface = sim.iface
        self.info = Info(n=iface.n, action_dim=action_dim, has_gripper=has_gripper, has_ee=iface.ee_site_id is not None)
        self._has_gripper = has_gripper
        mdp: MdpConfig | None = cfg.task.mdp
        self.mdp = mdp
        self._terminate_on_success = bool(getattr(cfg.task.params, "terminate_on_success", False))

        self._task_spec = task.observation_spec(self.num_envs)
        self._task_keys = tuple(self._task_spec.keys())
        explicit = mdp is not None and mdp.observations is not None
        self._legacy_layout = not explicit
        self.keys: tuple[str, ...] = self._selected_keys(mdp)
        problems = self._check(mdp)
        if problems:
            lines = "\n".join(f"  - {p}" for p in problems)
            raise ConfigError("task.mdp", f"the mdp does not fit this task and robot ({len(problems)} problem"
                              f"{'s' if len(problems) != 1 else ''}):\n{lines}")
        self._rewards = None if mdp is None or mdp.rewards is None else [(t, get_term("reward", t.fn)) for t in mdp.rewards]
        self._terms = (
            None if mdp is None or mdp.terminations is None else [(t, get_term("termination", t.fn)) for t in mdp.terminations]
        )
        self._success_ids = [] if self._terms is None else [i for i, (t, _) in enumerate(self._terms) if t.kind == "success"]
        self.reward_term_names: tuple[str, ...] = self._unique_names([t.fn for t, _ in self._rewards or []])
        self.last_reward_terms: dict[str, torch.Tensor] = {}

    # --- resolving the configuration -------------------------------------------------------

    def _selected_keys(self, mdp: MdpConfig | None) -> tuple[str, ...]:
        if mdp is not None and mdp.observations is not None:
            return mdp.observations
        keys = [TASK_PREFIX + k for k in self._task_keys]
        keys += list(DEFAULT_PROPRIO) + (["proprio/gripper"] if self._has_gripper else []) + list(DEFAULT_EPISODE)
        return tuple(keys)

    def _problem_with_key(self, key: str) -> str | None:
        if key.startswith(TASK_PREFIX):
            if key[len(TASK_PREFIX):] not in self._task_keys:
                return f"the task {self.cfg.task.id!r} observes {[TASK_PREFIX + k for k in self._task_keys]}, not {key!r}"
            return None
        reason = get_term("observation", key).why_unavailable(self.info)
        return None if reason is None else f"{key!r} is not available here: {reason}"

    def _check(self, mdp: MdpConfig | None) -> list[str]:
        if mdp is None:
            return []
        problems: list[str] = []
        for i, key in enumerate(mdp.observations or ()):
            msg = self._problem_with_key(key)
            if msg:
                problems.append(f"observations[{i}]: {msg}")
        for attr, kind in (("rewards", "reward"), ("terminations", "termination")):
            for i, t in enumerate(getattr(mdp, attr) or ()):
                spec = get_term(kind, t.fn)
                for name, p in spec.params.items():
                    if p.obs_key:
                        msg = self._problem_with_key(dict(t.params)[name])
                        if msg:
                            problems.append(f"{attr}[{i}].params.{name}: {msg}")
        return problems

    @staticmethod
    def _unique_names(names: list[str]) -> tuple[str, ...]:
        seen: dict[str, int] = {}
        out = []
        for n in names:
            seen[n] = seen.get(n, 0) + 1
            out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
        return tuple(out)

    # --- specs -----------------------------------------------------------------------------

    def observation_spec(self) -> Composite:
        N, dev = self.num_envs, self.device
        flat: dict[str, Any] = {}
        for key in self.keys:
            if key.startswith(TASK_PREFIX):
                flat[key] = self._task_spec[key[len(TASK_PREFIX):]].to(dev)
                continue
            term: Term = get_term("observation", key)
            shape, dtype = (N, *term.shape(self.info)), _DTYPES[term.dtype]
            flat[key] = (
                Binary(shape=shape, dtype=dtype, device=dev) if dtype is torch.bool else Unbounded(shape=shape, dtype=dtype, device=dev)
            )

        def build(node: dict[str, Any]) -> Composite:
            return Composite(
                **{k: build(v) if isinstance(v, dict) else v for k, v in node.items()}, shape=(N,), device=dev
            )

        nested = _nest(flat)
        if self._legacy_layout:
            nested = {"task": nested.get("task", {}), **{k: v for k, v in nested.items() if k != "task"}}
        return build(nested)

    # --- per step --------------------------------------------------------------------------

    def context(
        self,
        task_obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        prev_action: torch.Tensor,
        episode_step: torch.Tensor,
    ) -> MdpContext:
        return MdpContext(
            sim=self.sim,
            task=self.task,
            task_obs=task_obs,
            robot=RobotState(self.sim, self._has_gripper),
            action=action,
            prev_action=prev_action,
            episode_step=episode_step,
            max_episode_steps=self.max_episode_steps,
            observation=self._observation,
            success=self._success,
        )

    def _observation(self, key: str, ctx: MdpContext) -> torch.Tensor:
        term = get_term("observation", key)
        reason = term.why_unavailable(self.info)
        if reason is not None:
            raise ConfigError(key, f"observation key {key!r} is not available here: {reason}")
        return term.fn(ctx)

    def observations(self, ctx: MdpContext) -> TensorDict:
        """The selected observation keys as a nested TensorDict (``"proprio/qpos"`` is ``td["proprio", "qpos"]``)."""
        flat = {key: ctx.get(key) for key in self.keys}
        flat = {k: (v.clone() if k == "episode_step" else v) for k, v in flat.items()}
        nested = _nest(flat)
        if self._legacy_layout:
            nested = {"task": nested.get("task", {}), **{k: v for k, v in nested.items() if k != "task"}}

        def to_td(node: dict[str, Any]) -> TensorDict:
            return TensorDict(
                {k: to_td(v) if isinstance(v, dict) else v for k, v in node.items()},
                batch_size=torch.Size([self.num_envs]),
                device=self.device,
            )

        return to_td(nested)

    # --- terms -----------------------------------------------------------------------------

    def _value(self, ctx: MdpContext, index: int) -> torch.Tensor:
        """The value of termination term ``index``, ``(N, 1)`` bool, computed once per context."""
        cache = ctx._cache
        value = cache.get(("termination", index))
        if value is None:
            term, spec = self._terms[index]
            value = spec.fn(ctx, **dict(term.params)).reshape(self.num_envs, 1).to(torch.bool)
            cache[("termination", index)] = value
        return value

    def _success(self, ctx: MdpContext) -> torch.Tensor:
        if not self._success_ids:
            return ctx.task_success
        out = self._value(ctx, self._success_ids[0])
        for i in self._success_ids[1:]:
            out = out | self._value(ctx, i)
        return out

    def reward(self, ctx: MdpContext) -> torch.Tensor:
        """``(N, 1)`` float32: the sum of the weighted reward terms, or the task's own reward when none are configured."""
        if self._rewards is None:
            return ctx.task_reward
        total = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.last_reward_terms = {}
        for name, (term, spec) in zip(self.reward_term_names, self._rewards):
            value = spec.fn(ctx, **dict(term.params)).reshape(self.num_envs, 1).to(torch.float32) * term.weight
            self.last_reward_terms[name] = value
            total = total + value
        return total

    def done(self, ctx: MdpContext) -> tuple[torch.Tensor, torch.Tensor]:
        """``(terminated, truncated)``, each ``(N, 1)`` bool."""
        truncated = ctx.episode_step >= self.max_episode_steps
        if self._terms is None:
            terminated = ctx.success.clone() if self._terminate_on_success else torch.zeros_like(truncated)
            return terminated, truncated
        terminated = torch.zeros_like(truncated)
        for i, (term, _) in enumerate(self._terms):
            value = self._value(ctx, i)
            if term.kind == "timeout":
                truncated = truncated | value
            elif term.terminate:
                terminated = terminated | value
        return terminated, truncated
