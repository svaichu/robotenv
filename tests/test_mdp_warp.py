"""One MDP, two simulators: the same rewards, terminations and observation keys on the CPU and the Warp simulator."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("torchrl")
pytest.importorskip("mujoco_warp")

from torchrl.envs.utils import check_env_specs  # noqa: E402

from robotenv.config import ConfigError, MdpConfig, reward, termination  # noqa: E402
from tests.parity import build, errors, on_warp, random_case, run, twin, with_sim  # noqa: E402
from tests.test_mdp import DIST, make_cfg  # noqa: E402
from tests.test_warp_env import DEVICES  # noqa: E402

TOL = {"qpos": 1e-4, "qpos_norm": 1e-4, "ee_pos": 1e-4, "reward": 1e-3, "last_action": 1e-6, "target_pos": 0.0}

MDP = MdpConfig(
    rewards=(reward("distance", -1.0, **DIST), reward("sum_squares", -0.01, key="proprio/qpos"),
             reward("action_rate_sq", -0.1), reward("joint_limit_proximity", -1.0, margin=0.2)),
    terminations=(termination("time_out", "timeout", steps=7),),
    observations=("proprio/qpos", "proprio/qpos_norm", "proprio/ee_pos", "proprio/last_action", "task/target_pos", "time_remaining"),
)


def cfg_with(mdp):
    cfg = with_sim(make_cfg(n=4, num_envs=5, max_episode_steps=100), scene="empty")
    return replace(cfg, task=replace(cfg.task, mdp=mdp))


@pytest.mark.parametrize("device", DEVICES)
def test_the_same_mdp_gives_the_same_numbers_on_both_simulators(device):
    cfg = cfg_with(MDP)
    actions, inject = random_case(4, 5, 12)
    cpu, gpu = run(twin(cfg), actions, inject), run(on_warp(cfg, device), actions, inject)
    assert {"proprio/qpos", "proprio/last_action", "reward", "time_remaining"} <= set(cpu) == set(gpu)
    assert "proprio/qvel" not in cpu  # only what was selected
    for key, err in errors(gpu, cpu).items():
        name = key.split("/")[-1]
        assert err <= TOL.get(name, 1e-6), f"{key}: {err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_specs_flags_and_terms_on_warp(device):
    env = build(on_warp(cfg_with(MDP), device))
    check_env_specs(env)
    td = env.reset()
    assert set(td["proprio"].keys()) == {"qpos", "qpos_norm", "ee_pos", "last_action"} and td["proprio", "last_action"].shape == (5, 4)
    for _ in range(7):
        td["action"] = env.action_spec.rand()
        td = env.step(td)["next"]
    assert td["truncated"].all() and not td["terminated"].any()  # the time_out term at 7 steps
    assert list(env.reward_terms) == ["distance", "sum_squares", "action_rate_sq", "joint_limit_proximity"]
    env.close()


@pytest.mark.parametrize("device", DEVICES)
def test_gripper_keys_are_not_available_on_this_simulator(device):
    # the robot has no gripper here, so the config check already says so; the simulator never sees the key
    with pytest.raises(ConfigError, match="no gripper"):
        cfg_with(MdpConfig(None, None, ("proprio/gripper",)))
