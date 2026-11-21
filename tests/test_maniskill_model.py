"""``maniskill``: a ManiSkill agent's model as the ``MjSpec`` the compile of an external robot expects (marker ``maniskill``)."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mani_skill")
mujoco = pytest.importorskip("mujoco")

from robotenv.config import ConfigError
from robotenv.robot.providers.maniskill import PROVIDER, _model
from robotenv.simulators.mujoco.external import actuator_kind

pytestmark = pytest.mark.maniskill

ARM = tuple(f"panda_joint{i}" for i in range(1, 8))
REST = (0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4)
ARMS = ["panda", "panda_stick"]  # the arms whose URDF compiles as shipped
O = mujoco.mjtObj


def model_of(name):
    return _model.open_spec(_model.facts_of(name)).compile()


def test_the_agent_facts_are_read_without_building_an_environment():
    f = _model.facts_of("panda")
    assert f.kind == "urdf" and f.entry == "panda_v2.urdf" and f.fix_root_link and f.gravity_compensated
    assert f.ee_link == "panda_hand_tcp" and f.arm_joints == ARM and [k.name for k in f.keyframes] == ["rest"]
    assert [s.joint for s in f.servos] == [*ARM, "panda_finger_joint1", "panda_finger_joint2"]
    assert {(s.kp, s.kd, s.force_limit) for s in f.servos} == {(1000.0, 100.0, 100.0)}
    assert dict(f.friction) == {"panda_leftfinger": 2.0, "panda_rightfinger": 2.0} and f.srdf.endswith("panda_v2.srdf")
    assert f.joint_order == (*ARM, "panda_finger_joint1", "panda_finger_joint2")


@pytest.mark.parametrize("name", ARMS)
def test_every_controlled_joint_has_one_position_actuator_with_maniskills_gains(name):
    m = model_of(name)
    servos = _model.facts_of(name).servos
    assert m.nu == len(servos) >= 7
    for a, s in enumerate(servos):
        assert int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        assert m.joint(int(m.actuator_trnid[a, 0])).name == s.joint and actuator_kind(m, a) == "position"
        assert m.actuator_gainprm[a, 0] == s.kp and tuple(m.actuator_biasprm[a, :3]) == (0.0, -s.kp, -s.kd)
        j = int(m.actuator_trnid[a, 0])  # the force limit is the joint's actuator-force range, not the actuator's own
        assert not m.actuator_forcelimited[a] and m.jnt_actfrclimited[j] and tuple(m.jnt_actfrcrange[j]) == (-s.force_limit, s.force_limit)
        assert m.actuator_ctrllimited[a] and tuple(m.actuator_ctrlrange[a]) == tuple(m.jnt_range[m.actuator_trnid[a, 0]])


def test_fixed_links_are_kept_so_the_end_effector_link_is_a_body():
    m = model_of("panda")
    assert mujoco.mj_name2id(m, O.mjOBJ_BODY, "panda_hand_tcp") >= 0 and mujoco.mj_name2id(m, O.mjOBJ_BODY, "panda_link0") >= 0
    assert m.nq == 9 and m.neq == 1  # the finger mimic stays an equality constraint
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # what ManiSkill's damping needs
    assert m.ngeom > 0 and (m.geom_contype > 0).all()  # collision geometry only: visuals are discarded


def test_the_keyframe_is_mapped_by_joint_name_and_holds_the_fingers():
    m = model_of("panda")
    key = mujoco.mj_name2id(m, O.mjOBJ_KEY, "rest")
    assert key >= 0
    q = {m.joint(j).name: m.key_qpos[key][m.jnt_qposadr[j]] for j in range(m.njnt)}
    assert [q[n] for n in ARM] == pytest.approx(REST) and q["panda_finger_joint1"] == q["panda_finger_joint2"] == 0.04
    assert tuple(m.key_ctrl[key]) == pytest.approx((*REST, 0.04, 0.04))  # the ctrl of every servo is its joint's value


def test_the_keyframe_does_not_depend_on_the_order_the_joints_are_listed_in():
    facts = _model.facts_of("panda")
    order = facts.joint_order[::-1]
    key = facts.keyframes[0]
    permuted = replace(facts, joint_order=order, keyframes=(replace(key, qpos=key.qpos[::-1]),))
    assert np.array_equal(_model.open_spec(permuted).compile().key_qpos, model_of("panda").key_qpos)


@pytest.mark.parametrize("name", ARMS)
def test_the_model_holds_its_keyframe_and_does_not_touch_itself(name):
    m = model_of(name)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0 and m.nexclude > 0  # the SRDF's disabled pairs are contact excludes
    q0 = d.qpos.copy()
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.abs(d.qpos - q0).max() < 1e-6 and d.ncon == 0  # gravity is compensated, as in ManiSkill


def test_gravity_compensation_and_link_friction_follow_the_agent():
    m = model_of("panda")
    assert (m.body_gravcomp[1:] == 1.0).all() and m.body_gravcomp[0] == 0.0
    finger = mujoco.mj_name2id(m, O.mjOBJ_BODY, "panda_leftfinger")
    link = mujoco.mj_name2id(m, O.mjOBJ_BODY, "panda_link3")
    assert (m.geom_friction[m.geom_bodyid == finger, 0] == 2.0).all() and (m.geom_friction[m.geom_bodyid == link, 0] == 1.0).all()
    heavy = replace(_model.facts_of("panda"), gravity_compensated=False)
    assert (_model.open_spec(heavy).compile().body_gravcomp == 0.0).all()


def test_spec_returns_independent_specs():
    robot = PROVIDER.describe("panda")
    a, b = PROVIDER.spec(robot), PROVIDER.spec(robot)
    for key in list(a.keys):
        a.delete(key)
    a.delete(a.actuators[0])
    assert a.compile().nu == 8 and b.compile().nu == 9 and b.compile().nkey == 1


def test_the_source_is_a_content_id_and_a_changed_one_is_refused():
    robot = PROVIDER.describe("panda")
    assert robot.source == _model.source_id(_model.facts_of("panda")) and len(robot.source) == 64
    assert robot.source != PROVIDER.describe("panda_stick").source
    # the agent facts are part of it: other gains are another robot
    facts = _model.facts_of("panda")
    softer = replace(facts, servos=tuple(replace(s, kp=500.0) for s in facts.servos))
    assert _model.source_id(softer) != robot.source
    with pytest.raises(ConfigError) as e:
        PROVIDER.spec(replace(robot, source="0" * 64))
    assert e.value.path == "robot.source" and e.value.reason == "the model changed since this config was written"


@pytest.mark.parametrize("name", ["so100", "koch-v1.1", "allegro_hand_right", "panda"])
def test_a_saturated_servo_on_a_light_link_stays_stable_and_reaches_its_target(name):
    m = model_of(name)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    target = m.actuator_ctrlrange.mean(axis=1)  # far from the keyframe: kp times the error exceeds the force limit
    d.ctrl[:] = target
    for _ in range(1500):
        mujoco.mj_step(m, d)
    assert d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    reached = np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)])
    assert np.abs(reached - target).max() < 1e-2


# --- repairs: mesh paths, mesh formats, links without mass, floating bases ---------------------------------------


def test_a_package_path_is_looked_up_from_the_urdfs_directory(tmp_path):
    (tmp_path / "robot" / "meshes").mkdir(parents=True)
    (tmp_path / "robot" / "meshes" / "a.stl").write_bytes(b"x")
    urdf_dir = str(tmp_path / "robot")
    here = str(tmp_path / "robot" / "meshes" / "a.stl")
    assert _model._resolve_mesh("meshes/a.stl", urdf_dir) == here
    assert _model._resolve_mesh("package://robot/meshes/a.stl", urdf_dir) == here  # the package is the directory itself
    assert _model._resolve_mesh("package:/robot/meshes/a.stl", urdf_dir) == here
    assert _model._resolve_mesh("package://robot/meshes/a.stl", str(tmp_path)) == here  # or a directory below
    assert model_of("koch-v1.1").nq == 6  # a shipped URDF with such paths


def test_a_mesh_mujoco_cannot_decode_is_converted_once_into_the_cache(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("ROBOTENV_CACHE_DIR", str(tmp_path))
    assert _model.cache_dir() == str(tmp_path / "maniskill" / "meshes")
    _model._urdf_document.cache_clear()
    before = _model.CONVERSIONS[0]
    try:
        assert model_of("so100").nq == 6  # its collision meshes include .ply files
        converted = _model.CONVERSIONS[0] - before
        cached = sorted(p.name for p in (tmp_path / "maniskill" / "meshes").iterdir())
        assert converted == len(cached) > 0 and all(n.endswith(".stl") for n in cached)
        _model._urdf_document.cache_clear()
        assert model_of("so100").nq == 6 and _model.CONVERSIONS[0] - before == converted  # nothing converted twice
        source = _model.source_id(_model.facts_of("so100"))
    finally:
        _model._urdf_document.cache_clear()
    monkeypatch.delenv("ROBOTENV_CACHE_DIR")
    _model.source_id.cache_clear()
    assert _model.source_id(_model.facts_of("so100")) == source  # the id hashes the original files, not the cache
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.stl", "*.ply"], cwd=root, capture_output=True, text=True, check=True).stdout
    assert dirty == "", "a mesh was written into the repository tree"


def test_a_moving_link_without_mass_gets_a_minimal_one():
    # the floating gripper is moved by six joints whose intermediate links are empty; MuJoCo refuses those as they are
    m = model_of("floating_panda_gripper")
    assert m.nq == 8
    light = [m.body(b).name for b in range(1, m.nbody) if m.body_mass[b] == _model.EMPTY_LINK_MASS]
    assert len(light) >= 5 and m.body_mass.max() > 1000 * _model.EMPTY_LINK_MASS


def test_a_repeated_inertial_element_counts_once():
    assert model_of("trifingerpro").nq == 9


def test_a_floating_base_agent_gets_a_free_root_with_the_keyframes_pose():
    f = _model.facts_of("unitree_g1")
    assert not f.fix_root_link and not f.gravity_compensated
    m = model_of("unitree_g1")
    root = mujoco.mj_name2id(m, O.mjOBJ_JOINT, _model.ROOT_JOINT)
    assert root >= 0 and m.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE and m.nq == len(f.joint_order) + 7
    key = mujoco.mj_name2id(m, O.mjOBJ_KEY, "standing")
    a = m.jnt_qposadr[root]
    standing = next(k for k in f.keyframes if k.name == "standing")
    assert tuple(m.key_qpos[key][a : a + 7]) == pytest.approx((*standing.pos, *standing.quat)) and standing.pos[2] > 0.5
    assert (m.body_gravcomp == 0.0).all()  # a floating robot carries its weight, in ManiSkill too
    robot = PROVIDER.describe("unitree_g1")
    assert robot.home_root.pos == pytest.approx(standing.pos) and robot.n == 37 and robot.end_effector is None


def test_jointed_links_never_collide_even_where_the_parent_is_welded_to_the_world():
    m = model_of("allegro_hand_right")  # no SRDF: every exclude is a pair of jointed links
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert m.nexclude == m.nbody - 2 and d.ncon == 0
