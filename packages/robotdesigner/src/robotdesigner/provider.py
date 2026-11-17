"""The robot provider ``reduced``: robots of other providers with joints eliminated (see ``robotdesigner.reduce``).

Registered with robotenv through the entry point ``robotenv.robot.providers``. Importing this module imports neither
robotenv nor an engine; every method does, when called.
"""

from __future__ import annotations

from typing import Any


class ReducedRobots:
    """Robots of other providers cut short at the end-effector end. The model is the original's, cut when opened."""

    def names(self) -> tuple[str, ...]:
        return ("<provider>/<name>",)

    def describe(self, name: str, eliminate: int = 1, **options: Any) -> Any:
        """``name`` is ``"<provider>/<name>"`` of the original, described with ``options``; ``eliminate`` joints are
        removed from the end-effector end."""
        from .reduce import eliminate_joints

        return eliminate_joints(name, eliminate, **options)

    def spec(self, robot: Any) -> Any:
        """The original's ``MjSpec`` (its provider checks its source), with the cut body removed."""
        from robotenv.config import get_robot_provider

        from ._cut import cut_spec
        from .reduce import split

        original, body = split(robot)
        try:
            spec = get_robot_provider(original.provider).spec(original)
            return cut_spec(spec, body)
        except Exception as e:
            from .errors import ConfigError

            if isinstance(e, ConfigError):
                raise
            raise ConfigError("robot", f"the model of {robot.name} could not be cut: {type(e).__name__}: {e}") from e

    def validate(self, robot: Any, simulator: str) -> list:
        from robotenv.config import get_robot_provider
        from robotenv.errors import ConfigError, Problem

        from .reduce import split

        try:
            original, _ = split(robot)
            return get_robot_provider(original.provider).validate(original, simulator)
        except ConfigError as e:
            return [Problem(e.path, e.reason)]

    def kinematics(self, robot: Any) -> Any:
        from robotenv.simulators.mujoco.external import MujocoKinematics

        return MujocoKinematics(robot, self.spec(robot))


PROVIDER = ReducedRobots()
