"""A camera mounted on a robot. numpy only; the MuJoCo simulators add it to the model (``add_cameras``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .._canon import as_float, as_str, check_keys, within
from ..errors import ConfigError
from .pose import Pose

# Names render() treats as framed views of the whole scene (``robotenv.viz.VIEWS``); a robot camera must not shadow one.
RESERVED_NAMES = ("iso", "front", "side", "top")


@dataclass(frozen=True, slots=True)
class CameraMount:
    """A camera fixed to a frame of the robot.

    ``name`` is the camera's name in the model and what ``env.render(camera=name)`` takes. ``parent`` is the frame it is
    fixed to: ``"base"`` (the mount frame, for either kind of robot); for a ``ParametricArm`` also ``"ee"`` (the end
    effector) or ``"link_{i}"`` (the frame of link ``i``); for an ``ExternalRobot`` the name of a body of the model.
    ``pose`` is the camera frame in the ``parent`` frame, with MuJoCo's convention: the camera looks along its ``-z``
    axis, with ``+y`` up in the image. ``fovy`` is the vertical field of view in degrees; the image size is the one
    given to ``render``.
    """

    name: str
    parent: str
    pose: Pose
    fovy: float = 45.0

    def __post_init__(self):
        as_str(self.name, "name")
        if self.name in RESERVED_NAMES:
            raise ConfigError("name", f"{self.name!r} is the name of a scene view; choose another")
        as_str(self.parent, "parent")
        if not isinstance(self.pose, Pose):
            raise ConfigError("pose", "expected a Pose")
        fovy = as_float(self.fovy, "fovy")
        if not 0.0 < fovy < 180.0:
            raise ConfigError("fovy", "must be in (0, 180) degrees")
        object.__setattr__(self, "fovy", fovy)

    def _short(self) -> str:
        return f"{self.name} on {self.parent} {self.pose._short()} fovy={self.fovy:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parent": self.parent, "pose": self.pose.to_dict(), "fovy": self.fovy}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CameraMount":
        check_keys(d, {"name", "parent", "pose"}, "", optional={"fovy"})
        return cls(
            name=d["name"], parent=d["parent"], pose=within("pose", Pose.from_dict, d["pose"]), fovy=d.get("fovy", 45.0)
        )


def check_cameras(cameras: Any) -> tuple[CameraMount, ...]:
    """``cameras`` as a tuple of ``CameraMount`` with distinct names; ``ConfigError`` otherwise."""
    if isinstance(cameras, (str, bytes)) or not hasattr(cameras, "__iter__"):
        raise ConfigError("cameras", "expected a sequence of CameraMount")
    cameras = tuple(cameras)
    names: list[str] = []
    for i, c in enumerate(cameras):
        if not isinstance(c, CameraMount):
            raise ConfigError(f"cameras[{i}]", "expected a CameraMount")
        if c.name in names:
            raise ConfigError(f"cameras[{i}].name", f"duplicate camera name {c.name!r}")
        names.append(c.name)
    return cameras


def cameras_from_dict(items: Any) -> tuple[CameraMount, ...]:
    if isinstance(items, (str, bytes)) or not hasattr(items, "__iter__"):
        raise ConfigError("cameras", "expected a sequence")
    return check_cameras(tuple(within(f"cameras[{i}]", CameraMount.from_dict, c) for i, c in enumerate(items)))
