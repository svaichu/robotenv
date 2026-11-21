"""The robot provider ``grx``: the GRx humanoids of Fourier, from the repository ``FFTAI/Wiki-GRx-Models``.

Independent of every other provider. Importing this module imports no engine; each method imports what it needs when
it is called. No model file is copied into this repository: ``_source`` finds a checkout (``GRX_ROOT``) or fetches the
pinned commit into a cache directory with ``git``. The models are GPL-3.0 (:meth:`GrxRobots.license`).

The models are URDF (GR1) and MJCF (GR2, GR3) files without actuators, without a free root and without a keyframe;
:func:`robotenv.robot.providers.grx._model.open_spec` adds them, and what it adds is :data:`ADDED`, part of the
robot's ``source``, so changing a gain changes the robot's hash.

- one **position actuator per hinge**, ``kp = KP_PER_EFFORT * effort`` and ``kd = KD_PER_EFFORT * effort``, force
  limited to the joint's ``actuatorfrcrange`` (the effort limit of the model); ``NativeGains`` therefore drives every
  joint with these gains;
- ``ARMATURE`` on every hinge, so that explicit integration of the light wrist and head joints is stable;
- a free joint on the root body when the robot is described with ``base="floating"`` (the default), at the pose
  ``describe`` computes: standing at the zero posture with the lowest collision geometry just above the floor.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from ....errors import ConfigError, Problem

__all__ = ["PROVIDER", "GrxRobots", "cards"]

KP_PER_EFFORT = 1.0  # N·m/rad of stiffness per N·m of effort limit
KD_PER_EFFORT = 0.05  # N·m·s/rad of damping per N·m of effort limit
ARMATURE = 0.01  # kg·m² added to every hinge
ADDED = f"grx-1 kp={KP_PER_EFFORT} kd={KD_PER_EFFORT} armature={ARMATURE}"
STALE = "the model changed since this config was written"
SIMULATORS = frozenset({"mujoco", "warp"})
DEFAULT_END_EFFECTOR = "right_end_effector_link"


def cards() -> dict[str, dict[str, Any]]:
    """The cards of ``cards.json`` by robot name."""
    data = json.loads(resources.files(__package__).joinpath("cards.json").read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _card(name: str) -> dict[str, Any]:
    table = cards()
    if name not in table:
        raise ConfigError("name", f"the grx provider has no robot {name!r}; it has {sorted(table)}")
    return table[name]


def resolve_entry(name: str, entry: str | None) -> str:
    """The model file (relative to the robot's directory) for ``entry``: a short name of the card, a path of the card, or
    the default. ``ConfigError`` at ``entry`` otherwise."""
    card = _card(name)
    entries = card["entries"]
    if entry is None:
        return next(iter(entries.values()))
    if entry in entries:
        return entries[entry]
    if entry in entries.values():
        return entry
    raise ConfigError("entry", f"grx/{name} has no model {entry!r}; it has {list(entries)}")


def model_path(name: str, entry: str):
    from . import _source

    return _source.root() / "GRX" / _card(name)["dir"] / resolve_entry(name, entry)


def source_id(name: str, entry: str) -> str:
    from . import _source

    path = model_path(name, entry)
    if not path.is_file():
        raise ConfigError("entry", f"the model file of grx/{name} is missing: {path}")
    return _source.content_id(path, ADDED)


class GrxRobots:
    """``RobotProvider`` of the GRx humanoids."""

    def names(self) -> tuple[str, ...]:
        return tuple(cards())

    def license(self, name: str) -> str:
        _card(name)
        from . import _source

        return _source.license_text()

    def describe(
        self,
        name: str,
        entry: str | None = None,
        *,
        base: str = "floating",
        controlled: tuple[str, ...] | list[str] | None = None,
        end_effector: Any = None,
    ):
        """Open the model of robot ``name`` and return its :class:`ExternalRobot`.

        ``entry`` is a short name of the card (``basic``, ``dummy_hand``) or a model file; the default is the first.
        ``base`` is ``"floating"`` (a free root, started standing; the task takes a ``MobileMount``) or ``"fixed"``
        (the root body is welded at the mount). ``controlled`` names the driven joints (default: every hinge).
        ``end_effector`` is an ``EndEffectorRef`` (default: the body ``right_end_effector_link``).
        """
        from ._model import describe

        return describe(name, entry, base, controlled, end_effector)

    def spec(self, robot):
        """A fresh, editable ``mujoco.MjSpec`` of ``robot``, after checking ``robot.source`` against the files."""
        from ._model import open_spec

        try:
            entry = resolve_entry(robot.name, robot.entry)
            if entry != robot.entry:
                raise ConfigError("entry", f"{robot.entry!r} is not a model file of grx/{robot.name}")
            if source_id(robot.name, entry) != robot.source:
                raise ConfigError("source", STALE)
        except ConfigError as e:
            raise e.prefixed("robot") from None
        return open_spec(robot.name, entry, floating=robot.home_root is not None)

    def validate(self, robot, simulator: str) -> list[Problem]:
        problems: list[Problem] = []
        if simulator not in SIMULATORS:
            problems.append(Problem("sim.simulator", f"grx robots run on {sorted(SIMULATORS)}, not on {simulator!r}"))
        try:
            entry = resolve_entry(robot.name, robot.entry)
            if entry != robot.entry:
                problems.append(Problem("robot.entry", f"{robot.entry!r} is not a model file of grx/{robot.name}"))
            elif source_id(robot.name, entry) != robot.source:
                problems.append(Problem("robot.source", STALE))
        except ConfigError as e:
            e = e.prefixed("robot")
            problems.append(Problem(e.path, e.reason))
        return problems

    def kinematics(self, robot):
        from ....simulators.mujoco.external import MujocoKinematics

        return MujocoKinematics(robot, self.spec(robot))


PROVIDER = GrxRobots()
