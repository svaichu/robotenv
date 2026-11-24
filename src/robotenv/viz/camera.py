"""Camera framing from configs alone: numpy only, no engine import until ``CameraSpec.to_mjv``."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import FixedBox, world_mount
from ..config import kinematics as kin

# (azimuth, elevation) in degrees, world frame. Azimuth 0 looks along +x; the camera sits opposite to its view
# direction, so "front" (180) looks at the robot from +x towards -x.
VIEWS: dict[str, tuple[float, float]] = {
    "iso": (135.0, -25.0),
    "front": (180.0, -15.0),
    "side": (90.0, -10.0),
    "top": (90.0, -89.0),
}


@dataclass(frozen=True)
class CameraSpec:
    """A free camera: look-at point, distance, azimuth and elevation (degrees), as in MuJoCo."""

    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float

    def to_mjv(self):
        import mujoco

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.distance = self.distance
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        return cam


def _parts(cfg):
    """``cfg`` is an ``EnvConfig`` (robot, mount and target region) or a bare ``ParametricArm`` (at the origin)."""
    robot = getattr(cfg, "robot", cfg)
    task = getattr(cfg, "task", None)
    return robot, getattr(task, "target_region", None), None if task is None else world_mount(cfg)


def workspace_points(cfg, n_samples: int = 512, seed: int = 0) -> np.ndarray:
    """World-frame points the camera should cover: base, floor under the base, end-effector samples over the
    joint ranges, the home pose, and the corners of a ``FixedBox`` target region. Uses its own random
    generator, so it never touches an environment's random stream."""
    robot, region, mount = _parts(cfg)
    rng = np.random.default_rng(seed)
    T = np.eye(4) if mount is None else kin.pose_to_matrix(mount)
    base = T[:3, 3]
    pts = [base[None], np.array([[base[0], base[1], 0.0]])]
    if getattr(robot, "end_effector", True) is None:
        # A robot without an end effector (a quadruped) has no reach to sample: frame a metre around where it starts.
        root = getattr(robot, "home_root", None)
        centre = base if root is None else T[:3, :3] @ np.asarray(root.pos) + base
        pts.append(centre + 0.6 * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float))
    else:
        pts.append(kin.forward_ee_batch(robot, kin.sample_joint_positions(robot, n_samples, rng), mount))
        pts.append(kin.forward_ee_batch(robot, np.asarray(robot.home_qpos, dtype=float)[None], mount))
    if isinstance(region, FixedBox):
        lo, hi = np.asarray(region.low, dtype=float), np.asarray(region.high, dtype=float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pts.append(corners @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts, axis=0)


def auto_camera(
    *cfgs,
    view: str = "iso",
    fovy: float = 45.0,
    aspect: float = 4.0 / 3.0,
    margin: float = 1.15,
    n_samples: int = 512,
) -> CameraSpec:
    """A camera that sees the whole workspace of every config in ``cfgs``.

    Giving several configs returns ONE camera that covers all of them, so clips of different embodiments are
    drawn at the same scale. ``fovy`` (degrees) and ``aspect`` (width / height) are those of the render.
    """
    if not cfgs:
        raise ValueError("auto_camera needs at least one config")
    pts = np.concatenate([workspace_points(c, n_samples) for c in cfgs], axis=0)
    return fit_camera(pts, view=view, fovy=fovy, aspect=aspect, margin=margin)


def grid_offsets(n: int, spacing: float, ncols: int | None = None) -> np.ndarray:
    """``(n, 3)`` world offsets that lay ``n`` copies of a world out on a grid in the xy plane, ``spacing`` apart.
    Env 0 stays at the origin; rows grow along +y and columns along +x."""
    if n < 1:
        raise ValueError("nothing to lay out")
    cols = min(int(ncols) if ncols else math.ceil(math.sqrt(n)), n)
    if cols < 1:
        raise ValueError(f"ncols must be >= 1, got {ncols}")
    k = np.arange(n)
    return np.stack([(k % cols) * spacing, (k // cols) * spacing, np.zeros(n)], axis=1).astype(float)


def default_spacing(cfg, n_samples: int = 512) -> float:
    """Distance between neighbouring envs of a grid: the xy extent of the workspace plus 40 percent, at least 0.5 m."""
    pts = workspace_points(cfg, n_samples)
    return max(1.4 * float(np.ptp(pts[:, :2], axis=0).max()), 0.5)


def grid_camera(
    cfg,
    offsets: np.ndarray,
    *,
    view: str = "iso",
    fovy: float = 45.0,
    aspect: float = 4.0 / 3.0,
    margin: float = 1.1,
    n_samples: int = 512,
) -> CameraSpec:
    """One camera that sees the workspace of ``cfg`` copied to every offset of a grid (see ``grid_offsets``)."""
    pts = workspace_points(cfg, n_samples)
    offsets = np.asarray(offsets, dtype=float).reshape(-1, 3)
    allpts = (pts[None] + offsets[:, None]).reshape(-1, 3)
    return fit_camera(allpts, view=view, fovy=fovy, aspect=aspect, margin=margin)


def fit_camera(pts: np.ndarray, *, view: str, fovy: float, aspect: float, margin: float) -> CameraSpec:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; available: {sorted(VIEWS)}")
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    radius = max(float(np.linalg.norm(pts - center, axis=1).max()), 0.1)
    # The tighter of the vertical and horizontal half angles decides how far back the camera must sit.
    half = math.atan(math.tan(math.radians(fovy) / 2.0) * min(1.0, aspect))
    az, el = VIEWS[view]
    return CameraSpec(tuple(float(x) for x in center), margin * radius / math.sin(half), az, el)
