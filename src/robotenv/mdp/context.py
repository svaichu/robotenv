"""The view of the world a term sees: :class:`MdpContext`, with the robot state in :class:`RobotState`. Needs torch.

One context is made per environment step (and per reset); it caches what is read, so a quantity used by several terms
(the end effector position, say) is computed once. Every tensor lives on ``ctx.device`` with leading dimension
``ctx.num_envs``; ``float32`` unless said otherwise. Terms written against this class run unchanged on every simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from ..errors import ConfigError

if TYPE_CHECKING:
    from ..tasks.base import SimView, TaskImpl

__all__ = ["MdpContext", "RobotState"]


class RobotState:
    """The robot's own state, read from the simulation on first use.

    ``qpos``, ``qvel`` and ``effort`` are ``(N, n)`` in joint order (joint positions in rad or m, velocities, and the
    actuator forces); ``ee_pos`` is the end effector in the robot's base frame ``(N, 3)``; ``gripper`` is the gap between
    the fingers ``(N, 1)`` metres. ``joint_range`` is ``(n, 2)`` with ``-inf`` / ``+inf`` where a joint is unlimited.
    ``gripper`` and ``ee_pos`` raise ``ValueError`` on a robot that has none (the observation terms are unavailable
    there, so a validated config never reaches that).
    """

    def __init__(self, sim: "SimView", has_gripper: bool):
        self._sim = sim
        self._has_gripper = has_gripper
        self._cache: dict[str, torch.Tensor] = {}
        self.n: int = sim.iface.n
        self.device = sim.device
        self.joint_range = torch.as_tensor(sim.iface.joint_range, dtype=torch.float32, device=sim.device)

    def _read(self, name: str, read: Callable[[], Any]) -> torch.Tensor:
        value = self._cache.get(name)
        if value is None:
            value = self._cache[name] = torch.as_tensor(read()).to(device=self.device, dtype=torch.float32)
        return value

    @property
    def qpos(self) -> torch.Tensor:
        return self._read("qpos", lambda: self._sim.qpos)

    @property
    def qvel(self) -> torch.Tensor:
        return self._read("qvel", lambda: self._sim.qvel)

    @property
    def effort(self) -> torch.Tensor:
        return self._read("effort", lambda: self._sim.actuator_force)

    @property
    def gripper(self) -> torch.Tensor:
        if not self._has_gripper:
            raise ValueError("this robot has no gripper")
        return self._read("gripper", lambda: self._sim.gripper_opening)

    @property
    def ee_pos(self) -> torch.Tensor:
        site = self._sim.iface.ee_site_id
        if site is None:
            raise ValueError("this robot has no end effector")
        return self._read("ee_pos", lambda: self._sim.site_pos(site, "base"))


class MdpContext:
    """What a term is called with.

    ``robot``
        The :class:`RobotState`.
    ``get(key)``
        Any observation key, selected or not (``"proprio/ee_pos"``, ``"task/target_pos"``), cached per context. This is
        how a reward reads the goal or an object: ``ctx.get("task/target_pos")``.
    ``action`` / ``prev_action``
        The action applied in this step and the one before it, ``(N, action_dim)`` in ``[-1, 1]``; zeros at a reset.
    ``episode_step``
        ``(N, 1)`` int64, steps taken in the episode. ``max_episode_steps`` is the config's limit.
    ``success``
        Bool ``(N, 1)``: the success the environment reports (see ``TerminationTerm``). ``task_success`` and
        ``task_reward`` are the provider's own.
    ``sim`` and ``task``
        The simulation (``tasks.base.SimView``) and the task, for terms that need more than the above.
    """

    def __init__(
        self,
        *,
        sim: "SimView",
        task: "TaskImpl",
        task_obs: dict[str, torch.Tensor],
        robot: RobotState,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        episode_step: torch.Tensor,
        max_episode_steps: int,
        observation: Callable[[str, "MdpContext"], torch.Tensor],
        success: Callable[["MdpContext"], torch.Tensor],
    ):
        self.sim, self.task, self.task_obs, self.robot = sim, task, task_obs, robot
        self.action, self.prev_action = action, prev_action
        self.episode_step, self.max_episode_steps = episode_step, max_episode_steps
        self.num_envs: int = sim.num_envs
        self.device = sim.device
        self._observation, self._success = observation, success
        self._cache: dict[Any, torch.Tensor] = {}

    # --- observation keys ----------------------------------------------------------------

    def get(self, key: str) -> torch.Tensor:
        """The value of observation key ``key``, ``(N, ...)``."""
        value = self._cache.get(key)
        if value is None:
            if key.startswith("task/"):
                name = key[len("task/"):]
                if name not in self.task_obs:
                    raise ConfigError(key, f"the task observes {['task/' + k for k in self.task_obs]}, not {key!r}")
                value = self.task_obs[name]
            else:
                value = self._observation(key, self)
            self._cache[key] = value
        return value

    # --- success and the provider's own reward --------------------------------------------

    @property
    def success(self) -> torch.Tensor:
        value = self._cache.get("\0success")
        if value is None:
            value = self._cache["\0success"] = self._success(self)
        return value

    @property
    def task_success(self) -> torch.Tensor:
        value = self._cache.get("\0task_success")
        if value is None:
            value = self._cache["\0task_success"] = self.task.success(self.sim, self.task_obs)
        return value

    @property
    def task_reward(self) -> torch.Tensor:
        value = self._cache.get("\0task_reward")
        if value is None:
            value = self._cache["\0task_reward"] = self.task.reward(self.sim, self.task_obs)
        return value

    # --- small helpers for terms ----------------------------------------------------------

    def column(self, value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``(N, 1)`` filled with ``value``."""
        return torch.full((self.num_envs, 1), value, dtype=dtype, device=self.device)
