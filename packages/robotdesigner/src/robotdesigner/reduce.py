"""Eliminate joints of an external robot from the end-effector end: the arm is cut short, its last links removed.

``eliminate_joints(robot, k)`` takes an ``ExternalRobot`` of any robot provider (or ``"provider/name"``, which is
described first) and returns the ``ExternalRobot`` of the same model without its last ``k`` driven joints. Everything
from the body that carries the first removed joint onwards is removed (links, hand, fingers and what refers to them:
actuators, tendons, equalities, sensors). The new end effector is the frame of that removed body, on its parent, so it
sits where the removed joint was.

The result names the provider ``reduced``, which this package registers with robotenv (entry point
``robotenv.robot.providers``), so ``make_env`` builds it like any other robot. It refers to the original model, not a
copy: ``name`` is ``"<provider>/<name>"`` of the original, ``entry`` the original's and ``source`` the original's
followed by ``#cut=<body>``; reducing a reduced robot again nests the same way. A changed original model is reported
as for the original.

Needs robotenv and MuJoCo; both are imported when called, so ``import robotdesigner`` stays numpy only.
"""

from __future__ import annotations

from typing import Any

from .errors import ConfigError

NAMESPACE = "reduced"
CUT = "#cut="


def _robotenv():
    try:
        import mujoco  # noqa: F401

        from robotenv.config import (
            EndEffectorRef,
            ExternalRobot,
            Pose,
            get_robot_provider,
        )
    except ImportError as e:
        raise ConfigError("", f"eliminating joints needs robotenv with MuJoCo (robotenv[env]): {e}") from e
    return EndEffectorRef, ExternalRobot, Pose, get_robot_provider


def split(robot: Any) -> tuple[Any, str]:
    """The original robot a ``reduced`` robot refers to, and the body cut away. Config side apart from robotenv."""
    _, ExternalRobot, _, _ = _robotenv()
    provider, sep, name = robot.name.partition("/")
    source, cut, body = robot.source.rpartition(CUT)
    if not sep or not provider or not name:
        raise ConfigError("robot.name", f"expected '<provider>/<name>' of the original robot, got {robot.name!r}")
    if not cut or not body:
        raise ConfigError("robot.source", f"expected '<source of the original>{CUT}<body>', got {robot.source!r}")
    original = ExternalRobot(
        provider=provider, name=name, entry=robot.entry, source=source, joints=robot.joints,
        home_qpos=robot.home_qpos, home_root=robot.home_root, keyframe=robot.keyframe, end_effector=None,
        actuator_kind=robot.actuator_kind,
    )
    return original, body


def _subtree(model: Any, body: int) -> set[int]:
    inside = {body}
    for b in range(body + 1, model.nbody):  # bodies are numbered depth first: a subtree is contiguous
        if int(model.body_parentid[b]) in inside:
            inside.add(b)
    return inside


def _ancestors(model: Any, body: int) -> set[int]:
    out = set()
    while body > 0:
        body = int(model.body_parentid[body])
        out.add(body)
    return out


def eliminate_joints(robot: Any, k: int, *, label: str | None = None, **describe_options: Any) -> Any:
    """The ``ExternalRobot`` of ``robot`` without its last ``k`` driven joints and the links they move.

    ``robot`` is an ``ExternalRobot`` or ``"provider/name"`` (described with ``describe_options``). ``1 <= k < n``.
    The driven joints must form a serial chain toward the end effector: the kept ones on the path from the root to
    the cut, the removed ones below it. ``label`` defaults to the original's label (if any) and ``-k``.
    """
    import mujoco

    EndEffectorRef, ExternalRobot, Pose, get_robot_provider = _robotenv()
    from ._cut import cut_spec

    if isinstance(robot, str):
        provider, sep, name = robot.partition("/")
        if not sep:
            raise ConfigError("robot", f"expected 'provider/name' or an ExternalRobot, got {robot!r}")
        robot = get_robot_provider(provider).describe(name, **describe_options)
    elif describe_options:
        raise ConfigError("robot", "describe options are for a robot given as 'provider/name'")
    if not isinstance(robot, ExternalRobot):
        raise ConfigError("robot", f"expected an ExternalRobot, got {type(robot).__name__}")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k < robot.n:
        raise ConfigError("k", f"must be an integer with 1 <= k < n = {robot.n}, got {k!r}")

    spec = get_robot_provider(robot.provider).spec(robot)
    model = spec.compile()

    def joint_id(name: str) -> int:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ConfigError("robot.joints", f"the model has no joint {name!r}")
        return j

    keep, drop = robot.joints[: robot.n - k], robot.joints[robot.n - k :]
    cut = int(model.jnt_bodyid[joint_id(drop[0].name)])
    below, above = _subtree(model, cut), _ancestors(model, cut)
    for j in drop:
        if int(model.jnt_bodyid[joint_id(j.name)]) not in below:
            raise ConfigError(
                "k", f"joint {j.name!r} is not below joint {drop[0].name!r}: the driven joints are not a serial chain"
            )
    for j in keep:
        if int(model.jnt_bodyid[joint_id(j.name)]) not in above:
            raise ConfigError(
                "k", f"joint {j.name!r} is not above joint {drop[0].name!r}: the driven joints are not a serial chain"
            )
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cut)
    parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[cut]))
    if not body or not parent:
        raise ConfigError("robot", f"the body of joint {drop[0].name!r} and its parent need names in the model")

    reduced = cut_spec(spec, body).compile()  # the cut must compile, and the keyframe must survive it
    if robot.keyframe is not None and mujoco.mj_name2id(reduced, mujoco.mjtObj.mjOBJ_KEY, robot.keyframe) < 0:
        raise ConfigError("robot.keyframe", f"the cut model has no keyframe {robot.keyframe!r}")

    kind = robot.actuator_kind
    if kind == "mixed":
        from robotenv.simulators.mujoco.external import actuator_kind

        trn = {int(reduced.actuator_trnid[a, 0]): a for a in range(reduced.nu)
               if reduced.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        kinds = {actuator_kind(reduced, trn[j]) for j in
                 (mujoco.mj_name2id(reduced, mujoco.mjtObj.mjOBJ_JOINT, x.name) for x in keep) if j in trn}
        kind = kinds.pop() if len(kinds) == 1 else "mixed"

    offset = Pose(tuple(float(x) for x in model.body_pos[cut]), tuple(float(x) for x in model.body_quat[cut]))
    return ExternalRobot(
        provider=NAMESPACE,
        name=f"{robot.provider}/{robot.name}",
        entry=robot.entry,
        source=f"{robot.source}{CUT}{body}",
        joints=keep,
        home_qpos=robot.home_qpos[: robot.n - k],
        home_root=robot.home_root,
        keyframe=robot.keyframe,
        end_effector=EndEffectorRef(site=None, body=parent, offset=offset),
        actuator_kind=kind,
        label=(f"{robot.label or robot.name}-{k}" if label is None else label),
    )
