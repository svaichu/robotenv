"""Engine side of ``maniskill.describe``: the numbers an ``ExternalRobot`` needs, from the agent class and its model."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np

from ....config import EndEffectorRef, ExternalJoint, ExternalRobot, Pose
from ....errors import ConfigError
from . import _model

_HINGE, _SLIDE, _FREE = _model._HINGE, _model._SLIDE, _model._FREE
O = mujoco.mjtObj


def _end_effector_from_card(card: dict) -> EndEffectorRef:
    offset = card.get("offset")
    return EndEffectorRef(
        site=card.get("site"),
        body=card.get("body"),
        offset=Pose.identity() if offset is None else Pose(tuple(offset), (1.0, 0.0, 0.0, 0.0)),
    )


def describe(name, entry, controlled, end_effector, keyframe, effort_limit) -> ExternalRobot:
    from . import DEFAULT, cards

    facts = _model.facts_of(name)
    card = cards().get(name, {})
    if entry is not None and entry != facts.entry:
        raise ConfigError("entry", f"maniskill/{name} has one model, {facts.entry!r}; got {entry!r}")
    model = _model.open_spec(facts).compile()
    servo = {s.joint: s for s in facts.servos}
    actuated = {
        int(model.actuator_trnid[a, 0]) for a in range(model.nu) if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
    }

    # --- the driven joints ---
    if controlled is None:
        controlled = card.get("controlled") or facts.arm_joints or tuple(s.joint for s in facts.servos)
    driven = []
    for n in controlled:
        j = mujoco.mj_name2id(model, O.mjOBJ_JOINT, n)
        if j < 0:
            raise ConfigError("controlled", f"maniskill/{name} has no joint {n!r}")
        if int(model.jnt_type[j]) not in (_HINGE, _SLIDE):
            raise ConfigError("controlled", f"{n!r} is not a hinge or a slide: only those are driven")
        if j not in actuated:
            raise ConfigError("controlled", f"joint {n!r} of maniskill/{name} is not driven by its joint-position controller")
        driven.append(j)
    if not driven:
        raise ConfigError("name", f"maniskill/{name} has no joint that its joint-position controller drives")

    # --- ranges, effort limits ---
    joints = []
    for j in driven:
        n = model.joint(j).name
        rng = tuple(float(x) for x in model.jnt_range[j]) if model.jnt_limited[j] else None
        limit = servo[n].force_limit if n in servo else None
        if limit is None:
            a = next(a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == j)
            if model.actuator_forcelimited[a]:  # an MJCF's own actuator
                limit = float(np.abs(model.actuator_forcerange[a]).max())
        if limit is None and facts.kind == "mjcf" and model.jnt_actfrclimited[j]:  # from the motor the spec replaced
            limit = float(np.abs(model.jnt_actfrcrange[j]).max())
        if limit is None and isinstance(effort_limit, dict) and n in effort_limit:
            limit = float(effort_limit[n])
        elif limit is None and effort_limit is not None and not isinstance(effort_limit, dict):
            limit = float(effort_limit)
        if limit is None:
            raise ConfigError(
                "effort_limit",
                f"joint {n!r} of maniskill/{name} has no force limit in its controller: pass "
                f"effort_limit={{{n!r}: <N·m or N>}} or one number for all joints",
            )
        joints.append(ExternalJoint(name=n, range=rng, effort_limit=limit))

    # --- keyframe, home posture, root ---
    names = [k.name for k in facts.keyframes]
    if keyframe is DEFAULT:
        keyframe = card["keyframe"] if "keyframe" in card else ("rest" if "rest" in names else (names[0] if names else None))
    key_id = None
    if keyframe is not None:
        key_id = mujoco.mj_name2id(model, O.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ConfigError("keyframe", f"maniskill/{name} has no keyframe {keyframe!r}; it has {names}")
    state = model.qpos0 if key_id is None else model.key_qpos[key_id]
    home = []
    for j, joint in zip(driven, joints):
        q = float(state[model.jnt_qposadr[j]])
        if joint.range is not None and not joint.range[0] <= q <= joint.range[1]:
            clipped = min(max(q, joint.range[0]), joint.range[1])
            warnings.warn(
                f"maniskill/{name}: home value {q:.4f} of {joint.name!r} is outside its range {list(joint.range)}; clipped to {clipped:.4f}",
                stacklevel=3,
            )
            q = clipped
        home.append(q)
    free = [j for j in range(model.njnt) if int(model.jnt_type[j]) == _FREE]
    home_root = None
    if free:
        a = model.jnt_qposadr[free[0]]
        home_root = Pose(tuple(float(x) for x in state[a : a + 3]), tuple(float(x) for x in state[a + 3 : a + 7]))

    # --- end effector ---
    if end_effector is None and "end_effector" in card:
        end_effector = _end_effector_from_card(card["end_effector"])
    if end_effector is None and facts.ee_link is not None and mujoco.mj_name2id(model, O.mjOBJ_BODY, facts.ee_link) >= 0:
        end_effector = EndEffectorRef(site=None, body=facts.ee_link, offset=Pose.identity())

    from ....simulators.mujoco.external import actuator_kind

    kinds = {actuator_kind(model, a) for a in range(model.nu) if int(model.actuator_trnid[a, 0]) in driven}
    return ExternalRobot(
        provider="maniskill", name=name, entry=facts.entry, source=_model.source_id(facts), joints=tuple(joints),
        home_qpos=tuple(home), home_root=home_root, keyframe=keyframe, end_effector=end_effector,
        actuator_kind=next(iter(kinds)) if len(kinds) == 1 else "mixed",
    )
