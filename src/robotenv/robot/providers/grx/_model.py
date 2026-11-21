"""Engine side of the ``grx`` provider: open a model as a ``mujoco.MjSpec`` with the additions, and read its numbers."""

from __future__ import annotations

import os
import re
import warnings

import mujoco
import numpy as np

from ....config import EndEffectorRef, ExternalJoint, ExternalRobot, Pose
from ....errors import ConfigError
from ....simulators.mujoco.external import actuator_kind
from . import ARMATURE, DEFAULT_END_EFFECTOR, KD_PER_EFFORT, KP_PER_EFFORT, model_path, resolve_entry, source_id

O = mujoco.mjtObj
_HINGE, _SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
_ON = mujoco.mjtLimited.mjLIMITED_TRUE
ROOT_JOINT = "root"
CLEARANCE = 0.005  # m between the lowest collision geometry and the floor at the home pose
# MuJoCo's URDF import fuses fixed links (the root link into the world, the end-effector links away) and drops visual
# geometry by default; the GRx models need neither.
_COMPILER = '<mujoco><compiler fusestatic="false" discardvisual="false" strippath="false" balanceinertia="true"/></mujoco>'


def open_spec(name: str, entry: str, floating: bool) -> mujoco.MjSpec:
    """The model of ``name`` with the additions of the provider (see the package docstring)."""
    path = model_path(name, entry)
    if path.suffix == ".urdf":
        text = path.read_text()
        if "<mujoco>" not in text:
            text = re.sub(r"(<robot\b[^>]*>)", lambda m: m.group(1) + _COMPILER, text, count=1)
        spec = mujoco.MjSpec.from_string(text)
        spec.modelfiledir = os.fspath(path.parent)  # mesh paths are relative to the file
    else:
        spec = mujoco.MjSpec.from_file(os.fspath(path))
    spec.compiler.degree = False
    root = spec.worldbody.first_body()
    if floating:
        root.add_freejoint(name=ROOT_JOINT)
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint.armature = ARMATURE
        effort = float(np.abs(joint.actfrcrange).max())
        if effort <= 0.0:
            continue  # no effort limit: the joint stays passive
        act = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        kp, kd = KP_PER_EFFORT * effort, KD_PER_EFFORT * effort
        act.gainprm[0] = kp
        act.biasprm[:3] = [0.0, -kp, -kd]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.forcelimited = _ON
        act.forcerange = [-effort, effort]
        act.ctrllimited = _ON
        act.ctrlrange = list(joint.range)
    return spec


def _lowest_collision_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    low = np.inf
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        low = min(low, center[2] - float(np.abs(rot[2]) @ model.geom_aabb[g, 3:]))
    return float(low)


def describe(name, entry, base, controlled, end_effector) -> ExternalRobot:
    if base not in ("floating", "fixed"):
        raise ConfigError("base", f"expected 'floating' or 'fixed', got {base!r}")
    entry = resolve_entry(name, entry)
    floating = base == "floating"
    spec = open_spec(name, entry, floating=False)  # a standalone, fixed compile: the posture and the heights
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    actuated = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    if controlled is None:
        driven = list(actuated)
    else:
        driven = []
        for n in controlled:
            j = mujoco.mj_name2id(model, O.mjOBJ_JOINT, n)
            if j < 0:
                raise ConfigError("controlled", f"grx/{name} has no joint {n!r}")
            if j not in actuated:
                raise ConfigError("controlled", f"joint {n!r} of grx/{name} has no effort limit, so no actuator")
            driven.append(j)
    if not driven:
        raise ConfigError("name", f"grx/{name} has no joint that could be driven")
    joints, home = [], []
    for j in driven:
        n = model.joint(j).name
        rng = tuple(float(x) for x in model.jnt_range[j]) if model.jnt_limited[j] else None
        a = actuated[j]
        joints.append(ExternalJoint(name=n, range=rng, effort_limit=float(np.abs(model.actuator_forcerange[a]).max())))
        q = float(model.qpos0[model.jnt_qposadr[j]])
        if rng is not None and not rng[0] <= q <= rng[1]:
            clipped = min(max(q, rng[0]), rng[1])
            warnings.warn(f"grx/{name}: home value {q:.4f} of {n!r} is outside its range {list(rng)}; clipped to {clipped:.4f}", stacklevel=3)
            q = clipped
        home.append(q)

    home_root = None
    if floating:
        base_pos = data.xpos[1]  # the root body of the fixed compile, at the origin of the model's frame
        home_root = Pose((float(base_pos[0]), float(base_pos[1]), round(float(base_pos[2] - _lowest_collision_z(model, data) + CLEARANCE), 4)), (1.0, 0.0, 0.0, 0.0))

    if end_effector is None:
        if mujoco.mj_name2id(model, O.mjOBJ_BODY, DEFAULT_END_EFFECTOR) >= 0:
            end_effector = EndEffectorRef(site=None, body=DEFAULT_END_EFFECTOR, offset=Pose.identity())
    kinds = {actuator_kind(model, actuated[j]) for j in driven}
    return ExternalRobot(
        provider="grx", name=name, entry=entry, source=source_id(name, entry), joints=tuple(joints), home_qpos=tuple(home),
        home_root=home_root, keyframe=None, end_effector=end_effector, actuator_kind=next(iter(kinds)) if len(kinds) == 1 else "mixed",
    )
