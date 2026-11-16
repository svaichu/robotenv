"""Printing configs (issue 61): ``repr`` is a readable table, empty things are left out, and it never raises."""

from robotenv._pretty import fmt, kv, table
from robotenv.config import (
    BaseEnvConfig, EndEffectorRef, EnvConfig, ExternalJoint, ExternalRobot, Pose, ToolSpec,
)


def arm_env(n: int, **kw) -> EnvConfig:
    kw.setdefault("axes", "z" * n)
    return BaseEnvConfig().robot(n=n, **kw).resolve()


def test_env_repr_starts_with_the_hash_and_lists_each_section():
    env = arm_env(3)
    lines = repr(env).splitlines()
    assert lines[0] == f"EnvConfig  env_hash={env.env_hash}"
    for header in ("ParametricArm  n=3", "ControlConfig", "TaskConfig  empty/idle", "SimConfig  mujoco"):
        assert any(line.startswith("  " + header) for line in lines), header


def test_the_arm_prints_one_row_per_joint():
    for n in (1, 3, 7):
        arm = arm_env(n).robot
        # a header, a column line, then one row per joint
        assert len(repr(arm).splitlines()) == 2 + n


def test_empty_columns_and_unset_parts_are_left_out():
    text = repr(arm_env(2).robot)
    for word in ("inertia", "joint_frame", "tool", "gripper", "end_effector", "None"):
        assert word not in text
    tooled = arm_env(2, tool=ToolSpec("g", (0, 0, 0), (1, 0, 0, 0))).robot
    assert "tool  ToolSpec(" in repr(tooled)
    posed = arm_env(2, end_effector=Pose((0, 0, 0.1), (1, 0, 0, 0))).robot
    assert "end_effector  pos=(0, 0, 0.1)" in repr(posed)


def test_an_external_robot_prints_its_joints_and_facts():
    robot = ExternalRobot(
        provider="toyrobot", name="arm3", entry="arm3.xml", source="s",
        joints=(ExternalJoint("j1", (-2.0, 2.0), 40.0), ExternalJoint("j2", None, 10.0)),
        home_qpos=(0.0, 0.5), home_root=None, keyframe=None,
        end_effector=EndEffectorRef("ee", None, Pose.identity()), actuator_kind="position", label="mine",
    )
    lines = repr(robot).splitlines()
    assert lines[0] == f"ExternalRobot  toyrobot/arm3  n=2  hash={robot.config_hash}  label=mine"
    assert lines[2].split() == ["0", "j1", "(-2,", "2)", "40", "0"]
    assert lines[3].split() == ["1", "j2", "10", "0.5"]  # no range: the cell is blank
    assert "  end_effector   site ee" in lines and "  keyframe" not in repr(robot)


def test_a_section_prints_itself_and_leaves_out_what_is_none():
    env = arm_env(1)
    assert repr(env.control).splitlines()[0] == "ControlConfig" and "action_scale" not in repr(env.control)
    assert "n_substeps" in repr(env.sim) and repr(env.task).startswith("TaskConfig  empty/idle")
    assert "params" not in repr(env.task)  # EmptyParams has only a name, which the header shows


def test_leaves_keep_their_one_line_repr():
    assert repr(Pose.identity()) == "Pose(pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))"
    assert "\n" not in repr(arm_env(1).control.gains)


def test_base_env_repr_is_grouped_and_tags_each_value():
    cfg = BaseEnvConfig().robot(n=3, axes="zzz").sim(num_envs=4)
    lines = repr(cfg).splitlines()
    assert lines[0].startswith("BaseEnvConfig (defaults version")
    assert [line for line in lines if not line.startswith("    ")][1:] == ["  robot", "  control", "  task", "  sim"]
    assert any(line.split() == ["n", "3", "[set]"] for line in lines)
    assert any(line.split() == ["num_envs", "4", "[set]"] for line in lines)
    assert any(line.split()[-1] == "[default]" and line.split()[0] == "dt" for line in lines)
    assert "None" not in repr(cfg)


def test_base_env_repr_shows_a_ready_robot_as_its_table():
    robot = ExternalRobot(
        provider="toyrobot", name="arm3", entry="arm3.xml", source="s",
        joints=(ExternalJoint("j1", (-2.0, 2.0), 40.0),), home_qpos=(0.0,), home_root=None, keyframe=None,
        end_effector=EndEffectorRef("ee", None, Pose.identity()), actuator_kind="position",
    )
    text = repr(BaseEnvConfig().robot(robot))
    assert "  robot  [set]\n    ExternalRobot  toyrobot/arm3  n=1" in text
    assert any(line.split()[:2] == ["0", "j1"] for line in text.splitlines())


def test_explain_rows_stay_on_one_line_for_a_ready_robot():
    arm = arm_env(3).robot
    cfg = BaseEnvConfig().robot(arm)
    rows = [line for line in cfg.explain().splitlines() if line.startswith("  robot")]
    assert len(rows) == 1 and rows[0].endswith(f"ParametricArm(n=3, hash={arm.config_hash})  [set]")


def test_repr_html_wraps_the_text_in_pre():
    env = arm_env(1)
    html = env._repr_html_()
    assert html.startswith("<pre>") and html.endswith("</pre>") and "EnvConfig" in html
    assert BaseEnvConfig()._repr_html_().startswith("<pre>")


def test_printing_changes_neither_equality_nor_the_hash():
    a, b = arm_env(2), arm_env(2)
    repr(a)
    assert a == b and a.env_hash == b.env_hash


def test_the_formatter():
    assert fmt(None) == "" and fmt(0.5) == "0.5" and fmt(-0.0) == "0" and fmt((1.0, 2)) == "(1, 2)" and fmt(True) == "True"
    assert kv([("a", 1), ("bbb", None), ("cc", "x")]) == ["  a   1", "  cc  x"]
    assert table(["i", "x", "y"], [[0, None, 1], [1, None, 22]]) == ["  i  y", "  0  1", "  1  22"]
