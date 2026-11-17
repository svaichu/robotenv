"""A robot provider for the tests only: namespace ``toyrobot``, two small models written inline (no download).

``arm3`` is a 3-hinge arm with position actuators and a ``home`` keyframe; ``floater`` has a free root, two hinge legs with
torque actuators and a keyframe. ``tests/conftest.py`` registers the provider lazily; it shows what a robot provider looks like.
The engine side (``describe``, ``spec``, ``kinematics``) imports MuJoCo inside the methods, so importing this module needs
no engine.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from robotenv.errors import ConfigError, Problem
from robotenv.robot import EndEffectorRef, ExternalJoint, ExternalRobot, Pose

ARM3 = """
<mujoco model="arm3">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="l1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2" damping="0.5" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
        <body name="l2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
          <body name="l3" pos="0.3 0 0">
            <joint name="j3" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
            <site name="ee" pos="0.3 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100" forcerange="-40 40"/>
    <position name="a2" joint="j2" kp="100" forcerange="-30 30"/>
    <position name="a3" joint="j3" kp="100" forcerange="-20 20"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.1 -0.2 0.3"/>
  </keyframe>
</mujoco>
"""

FLOATER = """
<mujoco model="floater">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.05 0.03" mass="2"/>
      <body name="leg1" pos="0.1 0.05 0">
        <joint name="hip1" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.2"/>
      </body>
      <body name="leg2" pos="0.1 -0.05 0">
        <joint name="hip2" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="hip1" ctrlrange="-1 1" gear="10"/>
    <motor name="m2" joint="hip2" ctrlrange="-1 1" gear="10"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.5 1 0 0 0 0.1 0.2"/>
  </keyframe>
</mujoco>
"""

ARMFINGER = """
<mujoco model="armfinger">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="l1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
        <body name="l2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
          <body name="hand" pos="0.3 0 0">
            <joint name="finger" type="slide" axis="0 1 0" range="0 0.04"/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100" ctrlrange="-3 3" forcerange="-40 40"/>
    <position name="a2" joint="j2" kp="100" ctrlrange="-1.5 1.5" forcerange="-30 30"/>
    <position name="af" joint="finger" kp="50" ctrlrange="0 0.04"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.1 -0.2 0.02" ctrl="0.1 -0.2 0.03"/>
  </keyframe>
</mujoco>
"""

PASSIVE = """
<mujoco model="passive">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="l1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
        <site name="ee" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A 4-joint arm with a two-finger hand: a tendon, an equality, a contact exclude and sensors that refer into the hand,
# for cutting the arm short (``robotdesigner.eliminate_joints``). Describe it with ``driven=GRIPPER4_ARM``.
GRIPPER4 = """
<mujoco model="gripper4">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="l1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
        <body name="l2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
          <body name="l3" pos="0.3 0 0" euler="0 0 0.3">
            <joint name="j3" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
            <body name="hand" pos="0.2 0 0.01">
              <joint name="j4" type="hinge" axis="1 0 0" range="-1 1"/>
              <geom type="box" size="0.02 0.04 0.02" mass="0.3"/>
              <site name="ee" pos="0.05 0 0"/>
              <body name="f1" pos="0.03 0.02 0">
                <joint name="fj1" type="slide" axis="0 1 0" range="0 0.04"/>
                <geom type="box" size="0.01 0.005 0.01" mass="0.05"/>
              </body>
              <body name="f2" pos="0.03 -0.02 0">
                <joint name="fj2" type="slide" axis="0 -1 0" range="0 0.04"/>
                <geom type="box" size="0.01 0.005 0.01" mass="0.05"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="l3" body2="hand"/></contact>
  <tendon><fixed name="split"><joint joint="fj1" coef="0.5"/><joint joint="fj2" coef="0.5"/></fixed></tendon>
  <equality><joint joint1="fj1" joint2="fj2"/></equality>
  <actuator>
    <position name="a1" joint="j1" kp="100" forcerange="-40 40"/>
    <position name="a2" joint="j2" kp="100" forcerange="-30 30"/>
    <position name="a3" joint="j3" kp="100" forcerange="-20 20"/>
    <position name="a4" joint="j4" kp="50" forcerange="-10 10"/>
    <position name="grip" joint="fj1" kp="20" ctrlrange="0 0.04"/>
  </actuator>
  <sensor><jointpos joint="j2"/><jointpos joint="fj1"/><framepos objtype="site" objname="ee"/></sensor>
  <keyframe>
    <key name="home" qpos="0.1 -0.2 0.3 0.4 0.01 0.01" ctrl="0.1 -0.2 0.3 0.4 0.02"/>
  </keyframe>
</mujoco>
"""
GRIPPER4_ARM = ("j1", "j2", "j3", "j4")

# The same arm with damped servos (kv): the dynamics are not chaotic, which the CPU-versus-Warp parity tests need.
ARM3D = (
    ARM3.replace('model="arm3"', 'model="arm3d"')
    .replace(' damping="0.5" armature="0.01"', "")
    .replace('type="hinge"', 'type="hinge" damping="0.5" armature="0.01"')
    .replace('kp="100"', 'kp="100" kv="10"')
)

# The damped arm with an implicit integrator asked for in its own option, as a model with stiff servos does.
ARM3I = ARM3D.replace('model="arm3d"', 'model="arm3i"').replace("<worldbody>", '<option integrator="implicitfast"/>\n  <worldbody>', 1)

MODELS = {
    "arm3": ("arm3.xml", ARM3),
    "arm3d": ("arm3d.xml", ARM3D),
    "arm3i": ("arm3i.xml", ARM3I),
    "floater": ("floater.xml", FLOATER),
    "armfinger": ("armfinger.xml", ARMFINGER),
    "passive": ("passive.xml", PASSIVE),
    "gripper4": ("gripper4.xml", GRIPPER4),
}


def source_of(xml: str) -> str:
    return hashlib.sha256(xml.encode()).hexdigest()[:16]


class ToyRobots:
    """The robot provider: ``describe`` reads the model, ``spec`` opens it, ``kinematics`` is the MuJoCo oracle."""

    def names(self) -> tuple[str, ...]:
        return tuple(MODELS)

    def describe(self, name: str, driven: tuple[str, ...] | None = None, **options: Any) -> ExternalRobot:
        """``driven`` names the joints a controller drives (default: every joint with an actuator)."""
        import mujoco

        if name not in MODELS:
            raise ConfigError("name", f"unknown toy robot {name!r}; known: {list(MODELS)}")
        entry, xml = MODELS[name]
        model = mujoco.MjModel.from_xml_string(xml)
        joints, kinds = [], set()
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if driven is not None and jname not in driven:
                continue
            lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (None, None)
            effort = float(np.abs(model.actuator_forcerange[a]).max()) if model.actuator_forcelimited[a] else float(
                np.abs(model.actuator_ctrlrange[a]).max() * model.actuator_gear[a, 0]
            )
            joints.append(ExternalJoint(
                name=jname, range=None if lo is None else (float(lo), float(hi)), effort_limit=effort,
            ))
            bias = model.actuator_biasprm[a]
            kinds.add("position" if bias[1] < 0 else "velocity" if bias[2] < 0 else "torque")
        key = model.key_qpos[0]
        qpos = [float(key[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)]]) for j in joints]
        free = any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(model.njnt))
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if site >= 0:
            ee = EndEffectorRef(site="ee", body=None, offset=Pose.identity())
        elif body >= 0:
            ee = EndEffectorRef(site=None, body="hand", offset=Pose((0.05, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
        else:
            ee = None
        return ExternalRobot(
            provider="toyrobot", name=name, entry=entry, source=source_of(xml), joints=tuple(joints),
            home_qpos=tuple(qpos),
            home_root=Pose(tuple(float(x) for x in key[:3]), tuple(float(x) for x in key[3:7])) if free else None,
            keyframe="home", end_effector=ee,
            actuator_kind=kinds.pop() if len(kinds) == 1 else "mixed",
        )

    def spec(self, robot: ExternalRobot) -> Any:
        import mujoco

        if robot.name not in MODELS:
            raise ConfigError("robot.name", f"unknown toy robot {robot.name!r}; known: {list(MODELS)}")
        entry, xml = MODELS[robot.name]
        if robot.entry != entry:
            raise ConfigError("robot.entry", f"unknown entry {robot.entry!r}; the model has {entry!r}")
        if source_of(xml) != robot.source:
            raise ConfigError("robot.source", "the model changed since this config was written")
        return mujoco.MjSpec.from_string(xml)

    def validate(self, robot: ExternalRobot, simulator: str) -> list[Problem]:
        if simulator != "mujoco":
            return [Problem("sim.simulator", f"toyrobot robots run on 'mujoco', not on {simulator!r}")]
        return []

    def kinematics(self, robot: ExternalRobot) -> Any:
        from robotenv.simulators.mujoco.external import MujocoKinematics

        return MujocoKinematics(robot, self.spec(robot))


PROVIDER = ToyRobots()
