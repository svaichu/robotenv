"""``robotdesigner.eliminate_joints`` and the ``reduced`` robot provider, on the test-only robot provider (no download)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("torchrl")

from robotdesigner.reduce import eliminate_joints

from robotenv.config import (
    ConfigError,
    ExternalRobot,
    get_robot_provider,
    list_robot_providers,
    robot_from_dict,
)
from robotenv.envs import make_env
from tests.test_mujoco_external import reach_cfg
from tests.toy_robot_provider import GRIPPER4_ARM, PROVIDER

ORIGINAL = PROVIDER.describe("gripper4", driven=GRIPPER4_ARM)


def world_pos(spec, body, qpos):
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[: len(qpos)] = qpos
    mujoco.mj_kinematics(m, d)
    return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)].copy()


def test_the_provider_is_registered_by_the_entry_point():
    assert "reduced" in list_robot_providers()


def test_eliminating_one_joint_removes_the_hand_and_what_refers_to_it():
    r = eliminate_joints(ORIGINAL, 1)
    assert isinstance(r, ExternalRobot) and r.provider == "reduced" and r.name == "toyrobot/gripper4"
    assert [j.name for j in r.joints] == ["j1", "j2", "j3"] and r.home_qpos == ORIGINAL.home_qpos[:3]
    assert r.end_effector.body == "l3" and r.end_effector.offset.pos == (0.2, 0.0, 0.01)
    assert r.source == ORIGINAL.source + "#cut=hand" and r.label == "gripper4-1"
    m = get_robot_provider("reduced").spec(r).compile()
    assert (m.njnt, m.nu, m.ntendon, m.neq, m.nexclude, m.nsensor) == (3, 3, 0, 0, 0, 1)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand") == -1
    assert np.allclose(m.key_qpos[0], [0.1, -0.2, 0.3]) and np.allclose(m.key_ctrl[0], [0.1, -0.2, 0.3])


@pytest.mark.parametrize("k, removed", [(1, "hand"), (2, "l3")])
def test_the_new_end_effector_is_where_the_first_removed_joint_was(k, removed):
    r = eliminate_joints(ORIGINAL, k)
    q = np.random.default_rng(k).uniform(-1.0, 1.0, size=(5, r.n))
    ee = get_robot_provider("reduced").kinematics(r).forward_ee_batch(q)
    spec = PROVIDER.spec(ORIGINAL)
    for qi, e in zip(q, ee):
        assert np.allclose(e, world_pos(spec, removed, qi), atol=1e-9)


def test_reducing_twice_nests_and_matches_reducing_at_once():
    twice = eliminate_joints(eliminate_joints(ORIGINAL, 1), 1)
    once = eliminate_joints(ORIGINAL, 2)
    assert twice.name == "reduced/toyrobot/gripper4" and twice.n == once.n == 2
    assert twice.end_effector == once.end_effector
    q = np.random.default_rng(0).uniform(-1.0, 1.0, size=(4, 2))
    a = get_robot_provider("reduced").kinematics(twice).forward_ee_batch(q)
    b = get_robot_provider("reduced").kinematics(once).forward_ee_batch(q)
    assert np.allclose(a, b)


def test_a_robot_by_name_is_described_first():
    assert eliminate_joints("toyrobot/gripper4", 1, driven=GRIPPER4_ARM) == eliminate_joints(ORIGINAL, 1)
    assert get_robot_provider("reduced").describe("toyrobot/gripper4", eliminate=2, driven=GRIPPER4_ARM).n == 2


def test_round_trip_and_hash():
    r = eliminate_joints(ORIGINAL, 1)
    assert robot_from_dict(r.to_dict()) == r and r.config_hash != ORIGINAL.config_hash
    assert r.config_hash != eliminate_joints(ORIGINAL, 2).config_hash


@pytest.mark.parametrize("k", [0, 4, True, 1.0])
def test_k_must_leave_a_joint(k):
    with pytest.raises(ConfigError) as e:
        eliminate_joints(ORIGINAL, k)
    assert e.value.path == "k"


def test_branches_are_not_a_chain():
    with pytest.raises(ConfigError, match="serial chain"):
        eliminate_joints(PROVIDER.describe("floater"), 1)


def test_a_changed_original_is_reported_as_stale():
    r = eliminate_joints(ORIGINAL, 1)
    stale = ExternalRobot.from_dict({**r.to_dict(), "source": "0000#cut=hand"})
    with pytest.raises(ConfigError, match="changed"):
        get_robot_provider("reduced").spec(stale)
    assert [p.path for p in get_robot_provider("reduced").validate(stale, "warp")] == ["sim.simulator"]


def test_make_env_builds_and_steps_the_reduced_robot():
    r = eliminate_joints(ORIGINAL, 2)
    env = make_env(reach_cfg(r, num_envs=2))
    td = env.rollout(5)
    assert td["action"].shape == (2, 5, 2)
    assert all(np.isfinite(v.numpy()).all() for v in td["next"].values(True, True) if v.dtype.is_floating_point)
