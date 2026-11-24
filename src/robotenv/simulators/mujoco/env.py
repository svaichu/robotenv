"""The CPU MuJoCo simulator's environment: any task of any provider, on any robot of the config.

Owns what is the same for every task: the compiled robot, the batched simulation, the control law, resets with
a mask, episode bookkeeping and rendering. The task (``tasks.base.TaskImpl``) adds its scene assets before the model
is compiled and supplies the ``task`` observation group, success and reward. Never autoresets; honours
``td["_reset"]``.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from ...config import EnvConfig
from ...tasks.base import mount_site_of as _mount_site
from ...envs.spaces import SpacesMixin
from ...mdp.manager import MdpManager
from ..base import task_seed
from .compile import compile_robot
from .sim import BatchedSim

class MujocoEnv(SpacesMixin, EnvBase):
    """Batched TorchRL environment: ``cfg.robot`` under ``cfg.control`` doing ``task`` in MuJoCo."""

    batch_locked = True

    def __init__(self, cfg: EnvConfig, task):
        cfg = cfg.resolve()
        self.cfg = cfg
        self.robot_hash = cfg.robot.config_hash
        self.env_hash = cfg.env_hash
        N = cfg.sim.num_envs
        super().__init__(device=torch.device("cpu"), batch_size=torch.Size([N]))

        self._task = task
        self._compiled = compile_robot(cfg, scene_hook=task.attach, mount_site=_mount_site(task))
        self._iface = self._compiled.interface
        self.world_mount = self._iface.mount  # the base in the world, read from the compiled model; recorded with env_hash
        self._sim = BatchedSim(self._compiled, N, cfg.sim.n_substeps)
        self._renderer = None  # created by the first render()
        n = self._iface.n
        self._n = n
        self._has_gripper = self._iface.has_gripper
        m = self._iface.action_dim  # n joint commands, plus the gripper opening when there is a gripper
        task.bind(self._sim)

        # Action [-1, 1] -> actuator control: ctrl = action * scale + offset.
        # Delta mode integrates the action into a target first (see _step); velocity mode scales it.
        self._mode = cfg.control.mode
        self._range = self._iface.joint_range
        self._q_target = np.zeros((N, n))
        if self._mode in ("joint_pos_delta", "joint_vel"):
            self._scale = np.full(n, cfg.control.action_scale)
            self._offset = np.zeros(n)
        elif self._mode == "joint_pos_abs":
            # [-1, 1] spans the actuator's ctrlrange (an external robot's own servo, which need not equal the joint range)
            rng = self._iface.joint_range if self._iface.ctrl_range is None else self._iface.ctrl_range
            lo, hi = rng[:, 0], rng[:, 1]
            finite = np.isfinite(lo) & np.isfinite(hi)
            lo, hi = np.where(finite, lo, 0.0), np.where(finite, hi, 0.0)  # no inf - inf below
            self._scale = np.where(finite, (hi - lo) / 2.0, np.pi)
            self._offset = np.where(finite, (hi + lo) / 2.0, 0.0)
        else:  # joint_torque
            self._scale = self._iface.effort_limit.copy()
            if self._iface.ctrl_range is not None:  # an external robot: the ends of the actuator's ctrlrange
                top = np.abs(self._iface.ctrl_range).max(axis=1)
                self._scale = np.where(np.isfinite(top), top, self._scale)
            self._offset = np.zeros(n)

        params = cfg.task.params
        self._max_episode_steps = cfg.task.max_episode_steps
        self._reset_noise = float(getattr(params, "reset_noise", 0.0))

        # Two seeded streams. ``_gen`` draws the reset noise of the robot (its shape depends on the number of joints).
        # ``_task_gen`` is for the task alone: what the task draws (an object placement, a goal) is then the same for
        # every embodiment that runs it with the same seed and ``num_envs``, whatever the robot.
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(cfg.sim.seed)
        self._task_gen = torch.Generator(device="cpu")
        self._task_gen.manual_seed(task_seed(cfg.sim.seed))
        self._episode_step = torch.zeros(N, 1, dtype=torch.int64)

        f32 = torch.float32
        # The MDP of the task (``cfg.task.mdp``): which observations exist, the reward and the termination rules.
        self._mdp = MdpManager(cfg, task, self._sim, action_dim=m, has_gripper=self._has_gripper)
        self._last_action = torch.zeros(N, m, dtype=f32)  # the action applied in the last step, zeros after a reset
        self.observation_spec = self._mdp.observation_spec()
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(N, m), dtype=f32)
        self.reward_spec = Unbounded(shape=(N, 1), dtype=f32)
        self.done_spec = Composite(
            done=Binary(shape=(N, 1), dtype=torch.bool),
            terminated=Binary(shape=(N, 1), dtype=torch.bool),
            truncated=Binary(shape=(N, 1), dtype=torch.bool),
            shape=(N,),
        )

    @property
    def q_target(self) -> np.ndarray:
        """Position targets, ``(N, n)``; meaningful in ``joint_pos_delta`` mode."""
        return self._q_target.copy()

    @property
    def interface(self):
        return self._iface

    @property
    def task(self):
        """The engine side of the task (``tasks.base.TaskImpl``)."""
        return self._task

    @property
    def mdp(self) -> MdpManager:
        """The resolved MDP (``cfg.task.mdp``): the observation keys, reward and termination terms."""
        return self._mdp

    @property
    def reward_terms(self) -> dict[str, torch.Tensor]:
        """The last step's weighted reward terms, ``{name: (N, 1)}``; empty when the reward is the task's own."""
        return dict(self._mdp.last_reward_terms)

    # --- sampling -------------------------------------------------------------------

    def _rand(self, *shape, dtype=torch.float64) -> torch.Tensor:
        return torch.rand(*shape, generator=self._gen, dtype=dtype)

    def _task_rand(self, *shape, dtype=torch.float64) -> torch.Tensor:
        return torch.rand(*shape, generator=self._task_gen, dtype=dtype)

    # --- observations ---------------------------------------------------------------

    def _observe(self, action: torch.Tensor | None = None, prev_action: torch.Tensor | None = None):
        """The observation TensorDict and the context the terms were evaluated on (it also serves reward and done)."""
        s = self._sim
        ctx = self._mdp.context(
            self._task.observe(s),
            self._last_action if action is None else action,
            self._last_action if prev_action is None else prev_action,
            self._episode_step,
        )
        return self._mdp.observations(ctx), ctx

    # --- EnvBase interface ----------------------------------------------------------

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        N, n = self.batch_size[0], self._n
        mask = torch.ones(N, dtype=torch.bool)
        if tensordict is not None and tensordict.get("_reset", None) is not None:
            mask = tensordict.get("_reset").reshape(N).to(torch.bool)

        def injected(key):
            if tensordict is not None and tensordict.get(key, None) is not None:
                return tensordict.get(key)
            return None

        # Random draws always cover the whole batch so that rng use does not depend on the mask.
        noise = (self._rand(N, n) * 2.0 - 1.0) * self._reset_noise
        qpos = torch.from_numpy(self._iface.home_qpos).expand(N, n) + noise
        qpos = np.clip(qpos.numpy(), self._iface.joint_range[:, 0], self._iface.joint_range[:, 1])
        qpos_in, qvel_in = injected("qpos"), injected("qvel")
        if qpos_in is not None:
            qpos = qpos_in.to(torch.float64).numpy()
        qvel = None if qvel_in is None else qvel_in.to(torch.float64).numpy()

        self._sim.reset(mask=mask.numpy(), qpos=qpos, qvel=qvel)
        if self._mode == "joint_pos_delta":
            self._q_target[mask.numpy()] = self._sim.qpos[mask.numpy()]
        self._task.reset(self._sim, mask, self._task_rand, injected)
        self._episode_step[mask] = 0
        self._last_action[mask] = 0.0

        out, _ = self._observe()
        for key in ("done", "terminated", "truncated"):
            prev = injected(key)
            flags = torch.zeros(N, 1, dtype=torch.bool)
            if prev is not None:
                flags = prev.to(torch.bool).clone()
                flags[mask] = False
            out.set(key, flags)
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action").to(torch.float64).clamp(-1.0, 1.0).numpy()
        n = self._n
        arm_action = action[:, :n]
        if self._mode == "joint_pos_delta":
            self._q_target = np.clip(
                self._q_target + arm_action * self._scale, self._range[:, 0], self._range[:, 1]
            )
            ctrl = self._q_target
        else:
            ctrl = arm_action * self._scale + self._offset
        if self._has_gripper:
            # The last action component is the opening, [-1, 1] -> [0, max_opening], in every control mode.
            opening = (action[:, n:] + 1.0) / 2.0 * self._iface.gripper_max_opening
            ctrl = np.concatenate([ctrl, opening], axis=1)
        self._sim.set_ctrl(ctrl)
        self._sim.step()
        self._episode_step += 1

        prev_action, self._last_action = self._last_action, torch.from_numpy(action).to(torch.float32)
        out, ctx = self._observe(self._last_action, prev_action)
        terminated, truncated = self._mdp.done(ctx)
        out.set("reward", self._mdp.reward(ctx))
        out.set("terminated", terminated)
        out.set("truncated", truncated)
        out.set("done", terminated | truncated)
        return out

    def _set_seed(self, seed: int | None):
        if seed is not None:
            self._gen.manual_seed(int(seed))
            self._task_gen.manual_seed(task_seed(int(seed)))
        return seed

    # --- rendering (needs an OpenGL context: MUJOCO_GL=egl on headless machines) ----------------------------

    def render(
        self,
        env_index: int = 0,
        *,
        camera=None,
        size: tuple[int, int] = (240, 320),
        overlays: bool = True,
    ) -> np.ndarray:
        """Render the live state of one env to ``uint8 (H, W, 3)`` (Gymnasium's ``rgb_array``).

        Shows the whole scene plus the overlays the task draws (``TaskImpl.overlays``). ``camera`` is ``None`` (the
        ``"iso"`` view framed on the workspace), a view name (``iso``, ``front``, ``side``, ``top``), a
        ``robotenv.viz.CameraSpec``, a ``mujoco.MjvCamera``, or the name / id of a camera in the model.
        Rendering never changes the simulation.
        """
        from .render import get_renderer

        return get_renderer(self).render(env_index, camera=camera, size=size, overlays=overlays)

    def close(self, *, raise_if_closed: bool = True):
        renderer, self._renderer = getattr(self, "_renderer", None), None
        if renderer is not None:
            renderer.close()
        super().close(raise_if_closed=raise_if_closed)
