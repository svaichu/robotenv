"""The robot provider ``maniskill``: robots of ManiSkill, through the ``mani_skill`` library.

**ManiSkill is a robot provider only for now.** No ManiSkill task is provided and SAPIEN, ManiSkill's simulator,
simulates nothing here: the simulators stay MuJoCo and MuJoCo Warp. A ManiSkill robot becomes an
:class:`~robotenv.robot.ExternalRobot` like any other, so it runs in the tasks of every task provider.

Importing this module imports no engine, not ``mani_skill`` and not ``sapien``; each method imports what it needs
when it is called. No model file is copied into this repository: the models are those of the installed library (and of
its asset directory for the robots it downloads), and meshes MuJoCo cannot read are converted into a cache directory
outside the repository tree.

``describe`` reads ManiSkill's agent class (model file, joints, keyframe, end-effector link, the gains of its
joint-position controller) and returns an ``ExternalRobot``; ``spec`` opens the model as a ``mujoco.MjSpec`` with one
position actuator per controlled joint, carrying those gains, so ``NativeGains`` means ManiSkill's own PD controller.
``cards.json`` holds, per robot, what cannot be derived.
"""

from __future__ import annotations

import difflib
import json
from importlib import resources
from typing import Any

from ....errors import ConfigError, Problem

__all__ = ["PROVIDER", "ManiSkillRobots", "cards"]

DEFAULT = object()  # "use the card's keyframe, else the agent's own"
EXTRA = "pip install 'robotenv[maniskill]' (Python 3.10 to 3.12)"
STALE = "the model changed since this config was written"
SIMULATORS = frozenset({"mujoco", "warp"})


def cards() -> dict[str, dict[str, Any]]:
    """The cards of ``cards.json`` by robot name."""
    data = json.loads(resources.files(__package__).joinpath("cards.json").read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _agents() -> dict[str, Any]:
    """ManiSkill's agent registry (uid -> spec); ``ConfigError`` at ``provider`` when the library is missing."""
    try:
        import mani_skill.agents.robots  # noqa: F401  (registers the agents)
        from mani_skill.agents.registration import REGISTERED_AGENTS
    except ImportError as e:
        raise ConfigError("provider", f"the maniskill provider needs the mani_skill library: {EXTRA}") from e
    return REGISTERED_AGENTS


def _agent_class(name: str):
    """The agent class of robot ``name``; ``ConfigError`` at ``name`` with the near matches."""
    registry = _agents()
    if name not in registry:
        near = difflib.get_close_matches(str(name), sorted(registry), n=3, cutoff=0.5)
        hint = f"; did you mean {near}?" if near else f"; known: {sorted(registry)}"
        raise ConfigError("name", f"ManiSkill has no robot {name!r}{hint}")
    return registry[name].agent_cls


class ManiSkillRobots:
    """``RobotProvider`` of ManiSkill's robots (its *agents*). A robot's name is the agent's uid."""

    def names(self) -> tuple[str, ...]:
        try:
            return tuple(sorted(_agents()))
        except ConfigError:
            return ()

    def license(self, name: str) -> str:
        """Where the licence of robot ``name`` is stated. ManiSkill's code is Apache-2.0; the robot models come from
        their vendors under their own terms, which the library lists in one file."""
        _agent_class(name)
        return "see LICENSE-3RD-PARTY of the mani_skill distribution (ManiSkill itself: Apache-2.0)"

    def describe(
        self,
        name: str,
        entry: str | None = None,
        *,
        controlled: tuple[str, ...] | list[str] | None = None,
        end_effector: Any = None,
        keyframe: Any = DEFAULT,
        effort_limit: float | dict[str, float] | None = None,
    ):
        """Read ManiSkill's agent ``name`` and return its :class:`ExternalRobot`.

        ``entry`` is the agent's model file; an agent has one, so the argument is only checked. ``controlled`` names
        the driven joints (default: the card's, else the agent's arm joints, else every joint of its joint-position
        controller). ``end_effector`` is an ``EndEffectorRef`` (default: the card's, else the agent's end-effector
        link, else none). ``keyframe`` defaults to the card's, else ``rest`` if the agent has it, else the agent's
        first keyframe, else none. ``effort_limit`` supplies the limit of joints whose controller has none: a number,
        or ``{joint name: limit}``.
        """
        _agents()  # the library first: its absence is a ConfigError that names the extra
        from ._describe import describe

        return describe(name, entry, controlled, end_effector, keyframe, effort_limit)

    def spec(self, robot):
        """A fresh, editable ``mujoco.MjSpec`` of ``robot``, after checking ``robot.source`` against the model."""
        _agents()
        from . import _model

        try:
            facts = _model.facts_of(robot.name)
        except ConfigError as e:
            raise e.prefixed("robot") from None
        if _model.source_id(facts) != robot.source:
            raise ConfigError("robot.source", STALE)
        return _model.open_spec(facts)

    def validate(self, robot, simulator: str) -> list[Problem]:
        problems: list[Problem] = []
        if simulator not in SIMULATORS:
            problems.append(
                Problem("sim.simulator", f"maniskill robots run on {sorted(SIMULATORS)}, not on {simulator!r}")
            )
        try:
            _agents()
            from . import _model

            facts = _model.facts_of(robot.name)
            if facts.entry != robot.entry:
                problems.append(
                    Problem("robot.entry", f"the model of maniskill/{robot.name} is {facts.entry!r}, not {robot.entry!r}")
                )
            elif _model.source_id(facts) != robot.source:
                problems.append(Problem("robot.source", STALE))
        except ConfigError as e:
            e = e.prefixed("robot")
            problems.append(Problem(e.path, e.reason))
        return problems

    def kinematics(self, robot):
        from ....simulators.mujoco.external import MujocoKinematics

        return MujocoKinematics(robot, self.spec(robot))


PROVIDER = ManiSkillRobots()
