"""The built-in terms: observation keys, rewards and terminations. Importing this module needs no torch or engine.

The functions use tensor methods (or import torch inside the body), so the registry can be read by the numpy-only
config layer. Docstrings are the one-line descriptions ``robotenv.mdp.catalog`` shows.

Conventions: a reward term returns the *raw* quantity (a distance, a squared norm) and the config gives its sign and
weight, so ``-1.0 * distance`` is a penalty; a termination term returns a bool.
"""

from __future__ import annotations

from .registry import REQUIRED, Info, Param, register_term


def _need(attr: str, what: str):
    return lambda info: None if getattr(info, attr) else f"the robot has no {what}"


# ================================================================================================
# observation keys. ``proprio/*``: the robot; ``success``, ``episode_step``, ``time_remaining``: the episode.
# The keys of the task (``task/...``) belong to its provider.
# ================================================================================================


@register_term("observation", "proprio/qpos", shape=lambda i: (i.n,))
def _qpos(ctx):
    """Joint positions (rad, or m for a slide joint)."""
    return ctx.robot.qpos


@register_term("observation", "proprio/qvel", shape=lambda i: (i.n,))
def _qvel(ctx):
    """Joint velocities."""
    return ctx.robot.qvel


@register_term("observation", "proprio/effort", shape=lambda i: (i.n,))
def _effort(ctx):
    """Actuator forces / torques of the driven joints."""
    return ctx.robot.effort


@register_term("observation", "proprio/gripper", shape=lambda i: (1,), available=_need("has_gripper", "gripper"))
def _gripper(ctx):
    """Gap between the gripper fingers (m)."""
    return ctx.robot.gripper


@register_term("observation", "proprio/ee_pos", shape=lambda i: (3,), available=_need("has_ee", "end effector"))
def _ee_pos(ctx):
    """End effector position in the robot's base frame (m)."""
    return ctx.robot.ee_pos


@register_term("observation", "proprio/qpos_norm", shape=lambda i: (i.n,))
def _qpos_norm(ctx):
    """Joint positions scaled to [-1, 1] by the joint range (unlimited joints stay raw)."""
    import torch

    lo, hi = ctx.robot.joint_range[:, 0], ctx.robot.joint_range[:, 1]
    finite = torch.isfinite(lo) & torch.isfinite(hi)
    mid = torch.where(finite, (hi + lo) / 2.0, torch.zeros_like(lo))
    half = torch.where(finite, (hi - lo) / 2.0, torch.ones_like(lo)).clamp(min=1e-9)
    return (ctx.robot.qpos - mid) / half


@register_term("observation", "proprio/last_action", shape=lambda i: (i.action_dim,))
def _last_action(ctx):
    """The action applied in the last step, in [-1, 1] (zeros after a reset)."""
    return ctx.action


@register_term("observation", "success", shape=lambda i: (1,), dtype="bool")
def _success(ctx):
    """Whether the task is currently solved."""
    return ctx.success


@register_term("observation", "episode_step", shape=lambda i: (1,), dtype="int64")
def _episode_step(ctx):
    """Steps taken in the current episode."""
    return ctx.episode_step


@register_term("observation", "time_remaining", shape=lambda i: (1,))
def _time_remaining(ctx):
    """Fraction of the episode left, from 1 at the start to 0 at ``max_episode_steps``."""
    left = 1.0 - ctx.episode_step.to(ctx.robot.qpos.dtype) / float(ctx.max_episode_steps)
    return left.clamp(min=0.0)


# The set of keys of an environment whose config says nothing (``mdp.observations is None``): everything the
# environment reported before the MDP was configurable. ``proprio/gripper`` is added for a robot with a gripper.
DEFAULT_PROPRIO = ("proprio/qpos", "proprio/qvel", "proprio/effort")
DEFAULT_EPISODE = ("success", "episode_step")


# ================================================================================================
# rewards. Each returns (N,) or (N, 1); the config weights it.
# ================================================================================================

_KEY = Param(REQUIRED, obs_key=True, doc="an observation key")


@register_term("reward", "task_reward", needs_task=True)
def _task_reward(ctx):
    """The task provider's own reward (what the environment gives when no rewards are configured)."""
    return ctx.task_reward


@register_term("reward", "constant", params={"value": Param(1.0, doc="the value every step")})
def _constant(ctx, value):
    """A constant every step (a living bonus when positive)."""
    return ctx.column(value)


@register_term("reward", "success_bonus")
def _success_bonus(ctx):
    """1 on the steps where the task is solved, else 0."""
    return ctx.success.to(ctx.robot.qpos.dtype)


@register_term("reward", "distance", params={"a": Param(REQUIRED, obs_key=True), "b": Param(REQUIRED, obs_key=True)})
def _distance(ctx, a, b):
    """Euclidean distance between two observation keys, for example ``a="proprio/ee_pos"`` and ``b="task/target_pos"``."""
    return (ctx.get(a) - ctx.get(b)).norm(dim=-1, keepdim=True)


@register_term(
    "reward",
    "exp_distance",
    params={"a": Param(REQUIRED, obs_key=True), "b": Param(REQUIRED, obs_key=True), "scale": Param(0.1, doc="metres")},
)
def _exp_distance(ctx, a, b, scale):
    """``exp(-distance / scale)``: 1 on the goal, falling off over ``scale``. A bounded, positive shaping term."""
    return (-(ctx.get(a) - ctx.get(b)).norm(dim=-1, keepdim=True) / scale).exp()


@register_term("reward", "component", params={"key": _KEY, "index": Param(0, doc="which entry of the last axis")})
def _component(ctx, key, index):
    """One entry of an observation key, for example the height ``index=2`` of ``task/cube_pos``."""
    return ctx.get(key)[..., index : index + 1]


@register_term("reward", "sum_squares", params={"key": _KEY})
def _sum_squares(ctx, key):
    """Sum of squares of an observation key (a penalty on joint velocity: ``key="proprio/qvel"``)."""
    return ctx.get(key).pow(2).sum(dim=-1, keepdim=True)


@register_term("reward", "sum_abs", params={"key": _KEY})
def _sum_abs(ctx, key):
    """Sum of absolute values of an observation key (a penalty on effort: ``key="proprio/effort"``)."""
    return ctx.get(key).abs().sum(dim=-1, keepdim=True)


@register_term("reward", "action_sq")
def _action_sq(ctx):
    """Squared norm of the action: a penalty on large commands."""
    return ctx.action.pow(2).sum(dim=-1, keepdim=True)


@register_term("reward", "action_rate_sq")
def _action_rate_sq(ctx):
    """Squared change of the action since the previous step: a penalty on jerky commands."""
    return (ctx.action - ctx.prev_action).pow(2).sum(dim=-1, keepdim=True)


@register_term("reward", "joint_limit_proximity", params={"margin": Param(0.9, doc="fraction of the half range")})
def _joint_limit_proximity(ctx, margin):
    """How far joints are inside the outer ``1 - margin`` of their range, summed over joints (unlimited joints count 0)."""
    import torch

    lo, hi = ctx.robot.joint_range[:, 0], ctx.robot.joint_range[:, 1]
    finite = torch.isfinite(lo) & torch.isfinite(hi)
    normed = ctx.get("proprio/qpos_norm").abs()
    over = (normed - margin).clamp(min=0.0) * finite.to(normed.dtype)
    return over.sum(dim=-1, keepdim=True)


# ================================================================================================
# terminations. Each returns a bool (N,) or (N, 1); the config says what true means.
# ================================================================================================


@register_term("termination", "task_success", needs_task=True)
def _task_success(ctx):
    """The task provider's own success test."""
    return ctx.task_success


@register_term(
    "termination",
    "distance_below",
    params={"a": Param(REQUIRED, obs_key=True), "b": Param(REQUIRED, obs_key=True), "tolerance": Param(REQUIRED, of=float, doc="metres")},
)
def _distance_below(ctx, a, b, tolerance):
    """True where two observation keys are closer than ``tolerance`` (a goal reached)."""
    return (ctx.get(a) - ctx.get(b)).norm(dim=-1, keepdim=True) < tolerance


@register_term("termination", "abs_above", params={"key": _KEY, "limit": Param(REQUIRED, of=float)})
def _abs_above(ctx, key, limit):
    """True where any entry of an observation key exceeds ``limit`` in magnitude, or is not finite (a blown-up state)."""
    x = ctx.get(key)
    return ((x.abs() > limit) | ~x.isfinite()).any(dim=-1, keepdim=True)


@register_term("termination", "time_out", params={"steps": Param(REQUIRED, of=int)})
def _time_out(ctx, steps):
    """True once the episode has taken ``steps`` steps (a shorter limit than ``max_episode_steps``)."""
    return ctx.episode_step >= steps
