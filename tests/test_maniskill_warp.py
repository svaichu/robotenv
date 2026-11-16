"""ManiSkill robots on the Warp simulator against the CPU simulator (Warp's CPU device; CUDA where it exists, marker ``gpu``).

Measured on Warp's CPU device, 30 steps of small random actions from the keyframe, 4 envs, free space, ``implicitfast``
(largest absolute difference; the tolerances are those of ``test_warp_parity.py``: qpos 1e-4, qvel 1e-3, effort 1e-2):

| robot       | control                        | qpos   | qvel   | effort |
|-------------|--------------------------------|--------|--------|--------|
| panda_stick | native, joint_pos_delta        | 1e-6   | 1e-5   | 5e-4   |
| panda       | native, joint_pos_delta        | 1e-6   | 1e-5   | 4e-4   |
| both        | FixedGains(60, 6), abs         | 7e-7   | 6e-6   | 2e-5   |
| both        | InertiaScaledGains(20, 1), abs | 1e-6   | 1e-4   | 3e-4   |
| panda_stick | native, joint_pos_abs          | 5e-3   | 6e-2   | 1.4    |
| panda       | native, joint_pos_abs          | 6e-6   | 4e-4   | 2e-2   |

The last two rows are **not asserted**: an absolute target far from the keyframe saturates ManiSkill's servos at their
force limit (kp 1000, limit 100), and a saturated stiff servo amplifies float32 rounding. It is the class of mismatch
of the Menagerie Panda's native servos; the numbers are in docs/robots/maniskill.md. Nothing in ``simulators/warp``
was changed for these robots: gravity compensation, the finger equality and the contact excludes are taken as they are.
"""

import numpy as np
import pytest

pytest.importorskip("mani_skill")
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("torchrl")
pytest.importorskip("mujoco_warp")

from torchrl.envs.utils import check_env_specs

from robotenv.config import BaseEnvConfig, FixedGains, InertiaScaledGains, NativeGains
from robotenv.robot.providers.maniskill import PROVIDER
from robotenv.simulators.warp.sim import check_model_supported
from tests.parity import build, errors, on_warp, run, twin
from tests.test_warp_env import DEVICES
from tests.test_warp_parity import TOL

pytestmark = pytest.mark.maniskill

ARMS = ["panda", "panda_stick"]


def config(name, num_envs=4):
    robot = PROVIDER.describe(name)
    return robot, BaseEnvConfig().robot(robot).sim(num_envs=num_envs, scene="plain_floor", integrator="implicitfast")


def check(cfg, device, n, seed=0, scale=0.05, steps=30):
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, cfg.sim.num_envs, n)) * scale
    cpu, gpu = run(twin(cfg), actions, None), run(on_warp(cfg, device), actions, None)
    worst = errors(gpu, cpu)
    for key, err in worst.items():
        name = key.split("/")[-1]
        if name in TOL:
            assert err <= TOL[name], f"{key}: {err:.2e} > {TOL[name]:.0e}"
    return worst


@pytest.mark.parametrize("name", ARMS)
def test_the_provider_and_warp_accept_the_model(name):
    robot, base = config(name)
    assert PROVIDER.validate(robot, "warp") == []
    env = build(twin(base.resolve()))
    assert check_model_supported(env._compiled.model) == []  # the finger equality, gravcomp and meshes included
    env.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ARMS)
@pytest.mark.parametrize("gains", [NativeGains(), InertiaScaledGains(20.0, 1.0)], ids=lambda g: type(g).__name__)
def test_the_arms_reset_and_step_on_warp(device, name, gains):
    robot, base = config(name, num_envs=2)
    cfg = on_warp(base.control(gains=gains).resolve(), device)
    env = build(cfg)
    check_env_specs(env)
    env.close()
    env = build(cfg)
    td = env.reset()
    for i in range(20):
        td["action"] = torch.zeros(2, robot.n, device=env.device) if i < 10 else env.action_spec.rand()
        td = env.step(td)["next"]
        assert all(torch.isfinite(v).all() for v in td.values(True, True) if v.dtype.is_floating_point), f"step {i}"
    env.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ARMS)
def test_parity_with_maniskills_own_servos_in_delta_mode(device, name):
    robot, base = config(name)
    check(base.control(mode="joint_pos_delta", action_scale=0.05).resolve(), device, robot.n)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ARMS)
@pytest.mark.parametrize("gains", [FixedGains(60.0, 6.0), InertiaScaledGains(20.0, 1.0)], ids=lambda g: type(g).__name__)
def test_parity_with_our_gains(device, name, gains):
    robot, base = config(name)
    check(base.control(gains=gains, mode="joint_pos_abs").resolve(), device, robot.n)


def test_the_keyframe_and_the_held_fingers_start_the_same_in_both():
    _, base = config("panda", num_envs=3)
    cfg = base.resolve()
    cpu, warp = build(twin(cfg)), build(on_warp(cfg, "cpu"))
    cpu.reset(), warp.reset()
    assert np.allclose(warp._sim._d.qpos.numpy(), np.stack([d.qpos for d in cpu._sim.data]), atol=1e-6)
    assert np.allclose(warp._sim._d.qpos.numpy()[:, -2:], 0.04, atol=1e-6)  # the fingers, at the keyframe's value
    cpu.close(), warp.close()
