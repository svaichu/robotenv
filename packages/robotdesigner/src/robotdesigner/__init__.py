"""Parametric robot arms as frozen, validated, hashable data (numpy only; no simulator is imported)."""

from __future__ import annotations

from .arm import Capsule, JointSpec, LinkSpec, ParametricArm, Segment, ToolSpec
from .errors import ConfigError
from .generators import chain
from .gripper import TwoFingerGripper
from .pose import Pose

__version__ = "0.1.0"

__all__ = [
    "Capsule", "ConfigError", "JointSpec", "LinkSpec", "ParametricArm", "Pose", "Segment", "ToolSpec",
    "TwoFingerGripper", "chain", "eliminate_joints",
]


def __getattr__(name):
    # Lazy: the tool needs robotenv and MuJoCo, which ``import robotdesigner`` does not import.
    if name == "eliminate_joints":
        from .reduce import eliminate_joints

        return eliminate_joints
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
