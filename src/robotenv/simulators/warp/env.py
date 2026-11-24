"""The MuJoCo Warp simulator's environment: the GPU twin of ``MujocoEnv``.

Same specs, keys, control law, masked reset and seeding rules, on ``cfg.sim.device``, driven by ``WarpBatchedSim`` and
any task written against ``SimView``. Both random streams are CPU generators seeded exactly as in ``MujocoEnv``; the
draws are moved to the device, so the same seed and ``num_envs`` give the same draws as the CPU simulator.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from ...config import ConfigError, EnvConfig
from ...tasks.base import mount_site_of as _mount_site
from ...envs.spaces import SpacesMixin
from ...mdp.manager import MdpManager
from ..base import task_seed
from ..mujoco.compile import compile_robot
from .sim import WarpBatchedSim, check_model_supported


class WarpEnv(SpacesMixin, EnvBase):
    """Batched TorchRL environment: ``cfg.robot`` under ``cfg.control`` doing ``task`` in MuJoCo Warp."""

    batch_locked = True

    def __init__(self, cfg: EnvConfig, task):
        cfg = cfg.resolve()
        self.cfg = cfg
        self.robot_hash = cfg.robot.config_hash
        self.env_hash = cfg.env_hash
        N = cfg.sim.num_envs
        device = torch.device(cfg.sim.device)
        super().__init__(device=device, batch_size=torch.Size([N]))

        self._task = task
        self._compiled = compile_robot(cfg, scene_hook=task.attach, mount_site=_mount_site(task))
        self._iface = self._compiled.interface
        self.world_mount = self._iface.mount  # the base in the world, read from the compiled model; recorded with env_hash
        self._sim = WarpBatchedSim(self._compiled, N, cfg.sim.n_substeps, cfg.sim.device, cfg.sim.options)
        self._renderer = None  # created by the first render()
        n = self._iface.n
        self._n = n
        task.bind(self._sim)
        problems = check_model_supported(self._sim.model)
        if problems:
            lines = "\n".join(f"  - {p}" for p in problems)
            raise ConfigError("", f"simulator 'warp' cannot run this model ({len(problems)} problems):\n{lines}")
        self._sim.finalize()

        f32 = dict(dtype=torch.float32, device=device)
        self._mode = cfg.control.mode
        self._range = torch.as_tensor(self._iface.joint_range, **f32)
        self._q_target = torch.zeros(N, n, **f32)
        if self._mode in ("joint_pos_delta", "joint_vel"):
            self._scale = torch.full((n,), float(cfg.control.action_scale), **f32)
            self._offset = torch.zeros(n, **f32)
        elif self._mode == "joint_pos_abs":
            # [-1, 1] spans the actuator's ctrlrange for an external robot (its own servo), the joint range otherwise
            rng = self._iface.joint_range if self._iface.ctrl_range is None else self._iface.ctrl_range
            lo, hi = rng[:, 0], rng[:, 1]
            finite = np.isfinite(lo) & np.isfinite(hi)
            lo, hi = np.where(finite, lo, 0.0), np.where(finite, hi, 0.0)  # no inf - inf below
            self._scale = torch.as_tensor(np.where(finite, (hi - lo) / 2.0, np.pi), **f32)
            self._offset = torch.as_tensor(np.where(finite, (hi + lo) / 2.0, 0.0), **f32)
        else:  # joint_torque
            scale = self._iface.effort_limit.copy()
            if self._iface.ctrl_range is not None:  # an external robot: the ends of the actuator's ctrlrange
                top = np.abs(self._iface.ctrl_range).max(axis=1)
                scale = np.where(np.isfinite(top), top, scale)
            self._scale = torch.as_tensor(scale, **f32)
            self._offset = torch.zeros(n, **f32)

        params = cfg.task.params
        self._max_episode_steps = cfg.task.max_episode_steps
        self._reset_noise = float(getattr(params, "reset_noise", 0.0))

        # Two seeded CPU streams, as in MujocoEnv: the robot's reset noise, and the task's own draws.
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(cfg.sim.seed)
        self._task_gen = torch.Generator(device="cpu")
        self._task_gen.manual_seed(task_seed(cfg.sim.seed))
        self._episode_step = torch.zeros(N, 1, dtype=torch.int64, device=device)

        f32 = torch.float32
        # The MDP of the task (``cfg.task.mdp``), as in MujocoEnv. This simulator has no gripper: its action is the n joint
        # commands and ``proprio/gripper`` is not available.
        self._mdp = MdpManager(cfg, task, self._sim, action_dim=n, has_gripper=False)
        self._last_action = torch.zeros(N, n, dtype=f32, device=device)
        self.observation_spec = self._mdp.observation_spec()
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(N, n), dtype=f32, device=device)
        self.reward_spec = Unbounded(shape=(N, 1), dtype=f32, device=device)
        self.done_spec = Composite(
            done=Binary(shape=(N, 1), dtype=torch.bool, device=device),
            terminated=Binary(shape=(N, 1), dtype=torch.bool, device=device),
            truncated=Binary(shape=(N, 1), dtype=torch.bool, device=device),
            shape=(N,),
            device=device,
        )

    @property
    def q_target(self) -> np.ndarray:
        """Position targets, ``(N, n)`` float64 on the host; meaningful in ``joint_pos_delta`` mode."""
        return self._q_target.detach().cpu().numpy().astype(np.float64)

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
        ctx = self._mdp.context(
            self._task.observe(self._sim),
            self._last_action if action is None else action,
            self._last_action if prev_action is None else prev_action,
            self._episode_step,
        )
        return self._mdp.observations(ctx), ctx

    # --- EnvBase interface ----------------------------------------------------------

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        N, n = self.batch_size[0], self._n
        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        if tensordict is not None and tensordict.get("_reset", None) is not None:
            mask = tensordict.get("_reset").reshape(N).to(self.device, torch.bool)

        def injected(key):
            if tensordict is not None and tensordict.get(key, None) is not None:
                return tensordict.get(key)
            return None

        # Random draws always cover the whole batch so that rng use does not depend on the mask. The arithmetic is the
        # CPU env's (float64 on the host); only the result is stored as float32 on the device.
        noise = (self._rand(N, n) * 2.0 - 1.0) * self._reset_noise
        qpos = torch.from_numpy(self._iface.home_qpos).expand(N, n) + noise
        qpos = torch.from_numpy(np.clip(qpos.numpy(), self._iface.joint_range[:, 0], self._iface.joint_range[:, 1]))
        qpos_in, qvel_in = injected("qpos"), injected("qvel")
        if qpos_in is not None:
            qpos = qpos_in.to(torch.float64)

        self._sim.reset(mask=mask, qpos=qpos, qvel=qvel_in)
        if self._mode == "joint_pos_delta":
            self._q_target[mask] = self._sim.qpos[mask]
        self._task.reset(self._sim, mask, self._task_rand, injected)
        self._episode_step[mask] = 0
        self._last_action[mask] = 0.0

        out, _ = self._observe()
        for key in ("done", "terminated", "truncated"):
            prev = injected(key)
            flags = torch.zeros(N, 1, dtype=torch.bool, device=self.device)
            if prev is not None:
                flags = prev.to(self.device, torch.bool).clone()
                flags[mask] = False
            out.set(key, flags)
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action").to(self.device, torch.float32).clamp(-1.0, 1.0)
        if self._mode == "joint_pos_delta":
            lo, hi = self._range[:, 0], self._range[:, 1]
            self._q_target = torch.minimum(torch.maximum(self._q_target + action * self._scale, lo), hi)
            self._sim.set_ctrl(self._q_target)
        else:
            self._sim.set_ctrl(action * self._scale + self._offset)
        self._sim.step()
        self._sim.check_overflow()
        self._episode_step += 1

        prev_action, self._last_action = self._last_action, action.clone()
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

    def render(self, env_index: int = 0, *, camera=None, size: tuple[int, int] = (240, 320), overlays: bool = True):
        """Render one env to ``uint8 (H, W, 3)`` from a host copy of its state; see ``MujocoEnv.render``."""
        from ..mujoco.render import get_renderer

        return get_renderer(self).render(env_index, camera=camera, size=size, overlays=overlays)

    def close(self, *, raise_if_closed: bool = True):
        renderer, self._renderer = getattr(self, "_renderer", None), None
        if renderer is not None:
            renderer.close()
        self._sim.close()
        super().close(raise_if_closed=raise_if_closed)
