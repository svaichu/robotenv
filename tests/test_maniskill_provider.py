"""The ``maniskill`` robot provider against the real library (marker ``maniskill``; the robots tested ship in the wheel).

ManiSkill is a robot provider only: these tests run its robots in tasks of other providers, on our simulators.
"""

import importlib.util
import warnings
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mani_skill")
mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from torchrl.envs.utils import check_env_specs

from robotenv.config import (
    BaseEnvConfig,
    ConfigError,
    ControlConfig,
    EnvConfig,
    ExternalRobot,
    FixedBox,
    FixedGains,
    InertiaScaledGains,
    Mount,
    MujocoOptions,
    NativeGains,
    Pose,
    SimConfig,
    TaskConfig,
    get_robot_provider,
)
from robotenv.config import kinematics as kin
from robotenv.envs import make_env
from robotenv.robot.providers.maniskill import PROVIDER, _model, cards
from tests.toy_provider import ReachParams

pytestmark = pytest.mark.maniskill

ARM = tuple(f"panda_joint{i}" for i in range(1, 8))
REST = (0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4)
ARMS = ["panda", "panda_stick"]


def err_of(build):
    with pytest.raises(ConfigError) as e:
        build()
    return e.value


def idle(robot, **sim):
    """``empty/idle`` on the floor, with the integrator the model asks for."""
    return BaseEnvConfig().robot(robot).sim(**{"num_envs": 2, "scene": "plain_floor", "integrator": "implicitfast", **sim})


def rollout(env, steps=20):
    td = env.reset()
    finite = lambda t: all(torch.isfinite(v).all() for v in t.values(True, True) if v.dtype.is_floating_point)
    assert finite(td)
    for i in range(steps):
        td["action"] = torch.zeros(env.batch_size[0], env.action_spec.shape[-1]) if i < steps // 2 else env.action_spec.rand()
        td = env.step(td)["next"]
        assert finite(td), f"step {i}"
    return td


# --- describe -------------------------------------------------------------------------------------------------


def test_the_provider_is_registered_and_lists_maniskills_agents():
    assert get_robot_provider("maniskill") is PROVIDER
    assert {"panda", "panda_stick", "so100", "fetch"} <= set(PROVIDER.names()) and len(PROVIDER.names()) >= 40


def test_panda_has_seven_driven_joints_maniskills_keyframe_and_its_tcp_link():
    r = PROVIDER.describe("panda")
    assert isinstance(r, ExternalRobot) and r.provider == "maniskill" and r.name == "panda" and r.entry == "panda_v2.urdf"
    assert r.n == 7 and tuple(j.name for j in r.joints) == ARM and r.actuator_kind == "position"
    assert r.joints[0].range == pytest.approx((-2.8973, 2.8973)) and {j.effort_limit for j in r.joints} == {100.0}
    assert r.keyframe == "rest" and r.home_qpos == pytest.approx(REST) and r.home_root is None
    assert r.end_effector.body == "panda_hand_tcp" and r.end_effector.site is None and r.end_effector.offset == Pose.identity()
    stick = PROVIDER.describe("panda_stick")
    assert stick.n == 7 and stick.entry == "panda_stick.urdf" and stick.config_hash != r.config_hash


def test_a_finger_is_driven_only_when_it_is_asked_for():
    r = PROVIDER.describe("panda", controlled=[*ARM, "panda_finger_joint1"])
    assert r.n == 8 and r.joints[-1].name == "panda_finger_joint1" and r.joints[-1].range == pytest.approx((0.0, 0.04))
    assert r.home_qpos[-1] == 0.04
    env = idle(r).build()
    td = rollout(env)
    assert td["proprio", "qpos"].shape[-1] == 8
    env.close()


def test_describe_refuses_unknown_names_and_bad_arguments():
    e = err_of(lambda: PROVIDER.describe("pandaa"))
    assert e.path == "name" and "'panda'" in e.reason
    assert err_of(lambda: PROVIDER.describe("panda", "other.urdf")).path == "entry"
    assert PROVIDER.describe("panda", "panda_v2.urdf").n == 7
    assert err_of(lambda: PROVIDER.describe("panda", controlled=["nope"])).path == "controlled"
    e = err_of(lambda: PROVIDER.describe("panda", keyframe="away"))
    assert e.path == "keyframe" and "['rest']" in e.reason
    with pytest.warns(UserWarning, match="panda_joint4.* clipped"):  # without a keyframe home is zero, outside that joint's range
        bare = PROVIDER.describe("panda", keyframe=None)
    assert bare.keyframe is None and bare.home_qpos[0] == 0.0 and bare.home_qpos[3] == pytest.approx(-0.0698)


def test_a_home_value_outside_its_range_is_clipped_and_said(monkeypatch):
    facts = _model.facts_of("panda")
    key = facts.keyframes[0]
    wide = replace(facts, keyframes=(replace(key, qpos=(5.0, *key.qpos[1:])),))  # panda_joint1 beyond its 2.8973
    monkeypatch.setattr(_model, "facts_of", lambda name: wide)
    with pytest.warns(UserWarning, match="outside its range .* clipped"):
        r = PROVIDER.describe("panda")
    assert r.home_qpos[0] == pytest.approx(2.8973)


def test_a_controller_without_a_force_limit_needs_effort_limit(monkeypatch):
    facts = _model.facts_of("panda")
    free = replace(facts, servos=tuple(replace(s, force_limit=None) for s in facts.servos))
    monkeypatch.setattr(_model, "facts_of", lambda name: free)
    e = err_of(lambda: PROVIDER.describe("panda"))
    assert e.path == "effort_limit" and "'panda_joint1'" in e.reason and "effort_limit=" in e.reason
    assert PROVIDER.describe("panda", effort_limit=87.0).joints[0].effort_limit == 87.0
    limits = {n: 10.0 * (i + 1) for i, n in enumerate(ARM)}
    assert PROVIDER.describe("panda", effort_limit=limits).joints[2].effort_limit == 30.0


def test_the_config_round_trips_and_the_hash_follows_the_source():
    r = PROVIDER.describe("panda")
    again = ExternalRobot.from_dict(r.to_dict())
    assert again == r and again.config_hash == r.config_hash
    assert replace(r, source="0" * 64).config_hash != r.config_hash


def test_every_card_matches_its_model():
    for name, card in cards().items():
        assert name in PROVIDER.names(), name
        try:
            model = _model.open_spec(_model.facts_of(name)).compile()
        except ConfigError:
            continue  # a robot whose files are not on disk: the sweep records it
        for n in card.get("controlled", []):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0, f"{name}: {n}"
        ee = card.get("end_effector")
        if ee is not None:
            assert (ee.get("site") is None) != (ee.get("body") is None)
            kind, ref = (mujoco.mjtObj.mjOBJ_SITE, ee["site"]) if ee.get("site") else (mujoco.mjtObj.mjOBJ_BODY, ee["body"])
            assert mujoco.mj_name2id(model, kind, ref) >= 0, f"{name}: {ref}"
        if card.get("keyframe") is not None:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, card["keyframe"]) >= 0


# --- validate, licence ------------------------------------------------------------------------------------------


def test_validate_problems_have_paths():
    r = PROVIDER.describe("panda")
    assert PROVIDER.validate(r, "mujoco") == [] and PROVIDER.validate(r, "warp") == []
    assert [p.path for p in PROVIDER.validate(r, "other")] == ["sim.simulator"]
    assert [p.path for p in PROVIDER.validate(replace(r, source="stale"), "mujoco")] == ["robot.source"]
    unknown = PROVIDER.validate(replace(r, name="pandaa"), "mujoco")
    assert [p.path for p in unknown] == ["robot.name"] and "'panda'" in unknown[0].message
    assert [p.path for p in PROVIDER.validate(replace(r, entry="other.urdf"), "mujoco")] == ["robot.entry"]


def test_a_robot_whose_files_are_not_on_disk_is_refused_with_the_download_command(monkeypatch, tmp_path):
    from mani_skill.agents.registration import REGISTERED_AGENTS

    cls = REGISTERED_AGENTS["panda"].agent_cls
    _model.facts_of.cache_clear()
    monkeypatch.setattr(cls, "urdf_path", str(tmp_path / "robots" / "nowhere" / "panda.urdf"))
    try:
        e = err_of(lambda: PROVIDER.describe("panda"))
        assert e.path == "name" and "not on disk" in e.reason and "mani_skill.utils.download_asset" in e.reason
        problems = PROVIDER.validate(replace(PANDA, source="x"), "mujoco")
        assert [p.path for p in problems] == ["robot.name"] and "download_asset" in problems[0].message
    finally:
        monkeypatch.undo()
        _model.facts_of.cache_clear()


def test_the_download_hint_names_maniskills_asset_id():
    from mani_skill import ASSET_DIR

    hint = _model._download_hint(str(ASSET_DIR / "robots" / "ur10e" / "ur10e.xml"))
    assert hint == "python -m mani_skill.utils.download_asset ur10e"


def test_licence_and_explain():
    assert "LICENSE-3RD-PARTY" in PROVIDER.license("panda")
    assert err_of(lambda: PROVIDER.license("nope")).path == "name"
    lines = BaseEnvConfig().robot(PROVIDER.describe("panda")).explain().splitlines()
    assert any("robot.license" in line and "LICENSE-3RD-PARTY" in line for line in lines)


def test_make_env_reports_a_stale_source_and_an_unknown_robot():
    for bad, where in ((replace(PANDA, source="stale"), "robot.source"), (replace(PANDA, name="nope_robot"), "robot.name")):
        e = err_of(lambda bad=bad: idle(bad).build())
        assert where in str(e)


# --- make_env on the CPU simulator ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARMS)
def test_make_env_holds_the_keyframe_with_maniskills_own_servos(name):
    env = idle(PROVIDER.describe(name)).control(mode="joint_pos_delta", action_scale=0.05).build()
    td = env.reset()
    assert td["proprio", "qpos"][0].numpy() == pytest.approx(REST, abs=1e-6)
    for _ in range(20):
        td["action"] = torch.zeros(2, 7)  # a zero delta: the target stays the keyframe
        td = env.step(td)["next"]
    assert td["proprio", "qpos"][0].numpy() == pytest.approx(REST, abs=1e-4)
    env.close()


@pytest.mark.parametrize("gains", [NativeGains(), FixedGains(200.0, 20.0), InertiaScaledGains(20.0, 1.0)], ids=lambda g: type(g).__name__)
def test_make_env_runs_under_native_and_our_gains(gains):
    env = idle(PANDA).control(gains=gains).build()
    check_env_specs(env)
    env.close()
    env = idle(PANDA).control(gains=gains).build()
    td = rollout(env)
    assert td["proprio", "qpos"].shape == (2, 7)
    model = env._compiled.model
    servo = env.interface.actuator_ids
    if isinstance(gains, NativeGains):
        assert (model.actuator_gainprm[servo, 0] == 1000.0).all()
    else:
        assert model.nu == 9 and (model.actuator_gainprm[servo, 0] != 1000.0).all()  # ours on the arm, ManiSkill's on the fingers
    env.close()


def test_native_gains_under_an_explicit_integrator_are_reported():
    with pytest.warns(UserWarning, match="asks for the integrator 'implicitfast'"):
        idle(PANDA, integrator="euler").build().close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idle(PANDA).build().close()


def test_the_oracle_matches_the_simulator():
    mount = Pose((0.3, -0.2, 0.1), (float(np.cos(0.3)), 0.0, 0.0, float(np.sin(0.3))))
    env = idle(PANDA, num_envs=1).task(mount=Mount(mount)).build()
    env.reset()
    sim_ee = env._sim.data[0].site_xpos[env.interface.ee_site_id]
    assert np.allclose(kin.forward_ee_batch(PANDA, np.array([PANDA.home_qpos]), mount)[0], sim_ee, atol=1e-6)
    q = kin.sample_joint_positions(PANDA, 4, np.random.default_rng(0))
    assert q.shape == (4, 7) and (q >= [j.range[0] for j in PANDA.joints]).all()
    env.close()


# --- tasks of other providers on a ManiSkill robot --------------------------------------------------------------


def test_a_goal_task_of_another_provider_runs_on_a_maniskill_robot():
    cfg = EnvConfig(
        robot=PANDA,
        control=ControlConfig("joint_pos_abs", NativeGains(), None),
        task=TaskConfig(ReachParams("reach", 0.0, False), 20, 0.02, FixedBox((0.3, -0.2, 0.2), (0.6, 0.2, 0.6)), mount=Mount()),
        sim=SimConfig("mujoco", 3, "cpu", 0.002, "implicitfast", 50, 0, "plain_floor", options=MujocoOptions(False)),
    )
    env = make_env(cfg)
    td = rollout(env)
    assert td["task", "ee_pos"].shape == (3, 3) and torch.isfinite(td["reward"]).all()
    env.close()


@pytest.mark.robosuite
@pytest.mark.skipif(importlib.util.find_spec("robosuite") is None, reason="needs the robosuite extra")
def test_robosuite_lift_runs_on_a_maniskill_robot_mounted_on_the_scenes_frame():
    from robotenv.tasks.robosuite import LiftObject, RobosuiteParams

    params = RobosuiteParams("Lift", {}, LiftObject("cube", 0.84), 0.0, False)
    env = BaseEnvConfig().robot(PANDA).task(params, mount=Mount()).sim(num_envs=2, scene="empty", integrator="implicitfast").build()
    td = rollout(env)
    assert env.world_mount.pos == pytest.approx((-0.56, 0.0, 0.912))
    assert torch.isfinite(td["task", "cube_pos"]).all() and td["task", "ee_pos"].shape == (2, 3)
    env.close()


PANDA = PROVIDER.describe("panda")
