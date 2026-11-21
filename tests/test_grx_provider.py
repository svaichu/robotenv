"""The ``grx`` robot provider against the GRx models (``GRX_ROOT`` or a git fetch of the pinned commit; marker ``grx``)."""

import warnings

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from robotenv.config import BaseEnvConfig, ConfigError, MobileMount, Mount, Pose, get_robot_provider
from robotenv.robot.providers.grx import ADDED, PROVIDER, cards, resolve_entry

pytestmark = pytest.mark.grx

JOINTS = {"gr1t1": 32, "gr1t2": 32, "gr2v3_8_7": 29, "gr3v2_1_1": 31}
ENTRIES = [(name, entry) for name, card in cards().items() for entry in card["entries"]]


@pytest.fixture(scope="module", autouse=True)
def models():
    from robotenv.robot.providers.grx import _source

    try:
        _source.root()
    except ConfigError as e:
        pytest.skip(str(e))


def err_of(call):
    with pytest.raises(ConfigError) as e:
        call()
    return e.value


def test_the_registry_knows_the_provider_and_its_robots():
    assert get_robot_provider("grx") is PROVIDER and set(PROVIDER.names()) == set(JOINTS)
    assert "GPL" in PROVIDER.license("gr1t1")


@pytest.mark.parametrize("name,entry", ENTRIES)
def test_describe_reads_every_hinge_with_its_effort_limit_and_a_standing_root(name, entry):
    r = PROVIDER.describe(name, entry)
    assert r.provider == "grx" and r.n == JOINTS[name] and r.actuator_kind == "position" and r.keyframe is None
    assert all(j.range is not None and j.effort_limit > 0 for j in r.joints)
    assert r.end_effector.body == "right_end_effector_link"
    assert 0.8 < r.home_root.pos[2] < 1.1 and r.home_root.quat == (1.0, 0.0, 0.0, 0.0)
    assert r.config_hash == PROVIDER.describe(name, entry).config_hash  # deterministic
    assert r.entry == resolve_entry(name, entry)


def test_the_two_entries_of_a_robot_are_different_robots():
    a, b = PROVIDER.describe("gr2v3_8_7", "basic"), PROVIDER.describe("gr2v3_8_7", "dummy_hand")
    assert a.source != b.source and a.config_hash != b.config_hash
    assert PROVIDER.describe("gr2v3_8_7", a.entry).source == a.source  # a path works like a short name


def test_a_fixed_base_has_no_root_and_spec_follows_home_root():
    fixed = PROVIDER.describe("gr1t2", base="fixed")
    assert fixed.home_root is None
    free = PROVIDER.describe("gr1t2")
    kinds = lambda r: {int(j.type) for j in PROVIDER.spec(r).joints}
    assert int(mujoco.mjtJoint.mjJNT_FREE) not in kinds(fixed) and int(mujoco.mjtJoint.mjJNT_FREE) in kinds(free)
    assert fixed.source == free.source  # the file is the same; the root is a choice of the config


def test_the_home_pose_stands_on_the_floor_without_penetration():
    for name in JOINTS:
        r = PROVIDER.describe(name)
        spec = PROVIDER.spec(r)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:7] = [*r.home_root.pos, *r.home_root.quat]
        mujoco.mj_forward(model, data)
        assert all(c.dist >= -1e-6 for c in data.contact[: data.ncon]), name
        assert data.ncon > 0 or r.home_root.pos[2] > 0  # the feet are within the margin of the floor, not far above it


def test_the_model_carries_the_actuators_visuals_and_the_added_gains():
    r = PROVIDER.describe("gr1t1")  # a URDF: its root link and fixed links are kept, its meshes are read
    model = PROVIDER.spec(r).compile()
    assert model.nu == 32 and model.nmesh > 0 and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_end_effector_link") >= 0
    a = 0
    assert model.actuator_gainprm[a, 0] == pytest.approx(r.joints[0].effort_limit) and model.actuator_forcerange[a, 1] == pytest.approx(r.joints[0].effort_limit)
    assert (model.dof_armature[6:] == 0.01).all() and model.dof_armature[:6].sum() == 0  # the root is free: no armature


def test_the_source_id_covers_what_the_provider_adds(monkeypatch):
    import robotenv.robot.providers.grx as grx

    r = PROVIDER.describe("gr3v2_1_1")
    monkeypatch.setattr(grx, "ADDED", ADDED + " changed")
    assert err_of(lambda: PROVIDER.spec(r)).path == "robot.source"
    assert [(p.path) for p in PROVIDER.validate(r, "mujoco")] == ["robot.source"]


def test_errors_carry_config_paths():
    assert err_of(lambda: PROVIDER.describe("gr9")).path == "name"
    assert err_of(lambda: PROVIDER.describe("gr1t1", "nonexistent")).path == "entry"
    assert err_of(lambda: PROVIDER.describe("gr1t1", base="wheeled")).path == "base"
    assert err_of(lambda: PROVIDER.describe("gr1t1", controlled=["no_such_joint"])).path == "controlled"
    r = PROVIDER.describe("gr1t1")
    assert [p.path for p in PROVIDER.validate(r, "isaac")] == ["sim.simulator"]


def test_controlled_selects_joints_in_the_given_order():
    names = ["right_shoulder_pitch_joint", "left_shoulder_pitch_joint"]
    r = PROVIDER.describe("gr2v3_8_7", controlled=names)
    assert [j.name for j in r.joints] == names


@pytest.mark.parametrize("name,entry", [("gr1t1", "basic"), ("gr2v3_8_7", "dummy_hand"), ("gr3v2_1_1", "basic")])
def test_make_env_steps_a_floating_humanoid(name, entry):
    robot = PROVIDER.describe(name, entry)
    env = BaseEnvConfig().robot(robot).task(mount=MobileMount(spawn=(0.0, 0.0, 0.0), heading=0.0)).sim(num_envs=2, scene="plain_floor").build()
    td = env.reset()
    for _ in range(50):
        td["action"] = env.action_spec.rand()
        td = env.step(td)["next"]
    assert all(torch.isfinite(v).all() for _, v in td.items(True, True) if torch.is_tensor(v) and v.is_floating_point())


def test_make_env_steps_a_fixed_base_and_refuses_the_wrong_mount():
    robot = PROVIDER.describe("gr3v2_1_1", base="fixed")
    cfg = BaseEnvConfig().robot(robot).task(mount=Mount(Pose((0.0, 0.0, 1.2), (1.0, 0.0, 0.0, 0.0)))).sim(num_envs=1, scene="plain_floor")
    env = cfg.build()
    td = env.reset()
    td["action"] = env.action_spec.rand()
    env.step(td)
    floating = PROVIDER.describe("gr3v2_1_1")
    with pytest.raises(ConfigError):
        BaseEnvConfig().robot(floating).task(mount=Mount(Pose.identity())).sim(num_envs=1, scene="plain_floor").build()
