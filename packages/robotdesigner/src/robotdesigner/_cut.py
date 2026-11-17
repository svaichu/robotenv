"""Cut a MuJoCo model at a body: the body and its subtree are removed. Engine side (MuJoCo), no robotenv.

``spec.delete`` removes the subtree and every element that refers into it (actuators, tendons, equalities, contact
excludes and pairs, sensors). Keyframes are not remapped by MuJoCo, so they are rebuilt here, joint by joint and actuator by
actuator, from the compile of the uncut model: a keyframe of the cut model holds the same state for what is left.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ConfigError

_FREE_QPOS = {0: 7, 1: 4}  # mjJNT_FREE, mjJNT_BALL: qpos sizes; hinge and slide have 1
_FREE_DOF = {0: 6, 1: 3}


def _key_state(model: Any, spec: Any) -> list[dict[str, Any]]:
    """Every keyframe of ``model`` (the compile of ``spec``), as values per spec element (joint, actuator)."""
    joints, actuators = list(spec.joints), list(spec.actuators)
    out = []
    for k in range(model.nkey):
        qpos, qvel, ctrl, act = [], [], [], []
        for j in joints:
            t = int(model.jnt_type[j.id])
            qa, da = int(model.jnt_qposadr[j.id]), int(model.jnt_dofadr[j.id])
            qpos.append((j, model.key_qpos[k, qa : qa + _FREE_QPOS.get(t, 1)].copy()))
            qvel.append((j, model.key_qvel[k, da : da + _FREE_DOF.get(t, 1)].copy()))
        for a in actuators:
            ctrl.append((a, float(model.key_ctrl[k, a.id])))
            adr, num = int(model.actuator_actadr[a.id]), int(model.actuator_actnum[a.id])
            if adr >= 0:
                act.append((a, model.key_act[k, adr : adr + num].copy()))
        out.append({
            "qpos": qpos, "qvel": qvel, "ctrl": ctrl, "act": act, "time": float(model.key_time[k]),
            "mpos": model.key_mpos[k].copy(), "mquat": model.key_mquat[k].copy(),
        })
    return out


def cut_spec(spec: Any, body: str) -> Any:
    """Remove body ``body`` and its subtree from ``spec`` (edited in place and returned), keeping the keyframes."""
    target = spec.body(body)
    if target is None:
        raise ConfigError("body", f"the model has no body {body!r}")
    if target == spec.worldbody:
        raise ConfigError("body", "cannot remove the world body")
    model = spec.compile()
    state = _key_state(model, spec)
    nmocap = model.nmocap
    spec.delete(target)
    for key in spec.keys:  # cleared, then refilled once the addresses of the cut model are known
        key.qpos, key.qvel, key.ctrl, key.act, key.mpos, key.mquat = [], [], [], [], [], []
    cut = spec.compile()
    if cut.nmocap != nmocap:
        raise ConfigError("body", f"the subtree of {body!r} holds a mocap body; cutting it is not supported")
    alive_joints, alive_actuators = list(spec.joints), list(spec.actuators)
    for key, s in zip(spec.keys, state):
        qpos, qvel = cut.qpos0.copy(), np.zeros(cut.nv)
        for j, v in s["qpos"]:
            if j in alive_joints:
                qa = int(cut.jnt_qposadr[j.id])
                qpos[qa : qa + len(v)] = v
        for j, v in s["qvel"]:
            if j in alive_joints:
                da = int(cut.jnt_dofadr[j.id])
                qvel[da : da + len(v)] = v
        ctrl, act = np.zeros(cut.nu), np.zeros(cut.na)
        for a, v in s["ctrl"]:
            if a in alive_actuators:
                ctrl[a.id] = v
        for a, v in s["act"]:
            if a in alive_actuators:
                adr = int(cut.actuator_actadr[a.id])
                act[adr : adr + len(v)] = v
        key.qpos, key.qvel, key.ctrl, key.act = qpos.tolist(), qvel.tolist(), ctrl.tolist(), act.tolist()
        key.mpos, key.mquat, key.time = s["mpos"].ravel().tolist(), s["mquat"].ravel().tolist(), s["time"]
    return spec
