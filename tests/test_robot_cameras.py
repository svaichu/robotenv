"""Cameras mounted on a robot: validation, serialization, hashing; the MuJoCo model carries them at the right place."""

import numpy as np
import pytest

from robotenv.config import BaseEnvConfig, Chain, ConfigError, EnvConfig
from robotenv.robot import CameraMount, ParametricArm, Pose, chain

KW = dict(link_length=0.3, mass=1.0, radius=0.02, damping=0.1, armature=0.01, frictionloss=0.0, effort_limit=50.0)
WRIST = CameraMount("wrist", "ee", Pose((0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)), fovy=60.0)


def test_no_cameras_keeps_the_hash_and_the_dict():
    arm = chain(2, "zy", **KW)
    assert "cameras" not in arm.to_dict() and arm.cameras == ()


def test_cameras_change_the_hash_and_round_trip():
    arm, cam = chain(2, "zy", **KW), chain(2, "zy", cameras=[WRIST], **KW)
    assert cam.config_hash != arm.config_hash
    assert ParametricArm.from_dict(cam.to_dict()) == cam


@pytest.mark.parametrize(
    "camera, path",
    [
        (CameraMount("x", "link_2", Pose.identity()), "cameras[0].parent"),
        (CameraMount("x", "wrist_link", Pose.identity()), "cameras[0].parent"),
    ],
)
def test_parent_must_be_a_frame_of_the_arm(camera, path):
    with pytest.raises(ConfigError) as e:
        chain(2, "zy", cameras=[camera], **KW)
    assert e.value.path == path


def test_names_are_distinct_and_not_scene_views():
    with pytest.raises(ConfigError) as e:
        chain(2, "zy", cameras=[WRIST, WRIST], **KW)
    assert e.value.path == "cameras[1].name"
    with pytest.raises(ConfigError):
        CameraMount("iso", "base", Pose.identity())
    with pytest.raises(ConfigError):
        CameraMount("c", "base", Pose.identity(), fovy=180.0)


def test_mujoco_model_places_the_cameras():
    mujoco = pytest.importorskip("mujoco")
    from robotenv.simulators.mujoco.compile import compile_robot

    cams = [WRIST, CameraMount("fixed", "base", Pose((0.5, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))),
            CameraMount("l1", "link_1", Pose.identity())]
    cfg = BaseEnvConfig().robot(Chain(n=3, axes="zyy", total_length=0.9, total_mass=3.0, radius=0.02, damping=0.1,
                                      armature=0.01, frictionloss=0.0, effort_limit=50.0, cameras=cams)).resolve()
    assert EnvConfig.from_dict(cfg.to_dict()) == cfg
    model = compile_robot(cfg).model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pos = {n: data.cam_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)] for n in ("wrist", "fixed", "l1")}
    np.testing.assert_allclose(pos["wrist"], [0.9, 0.0, 0.05], atol=1e-9)  # the end effector at q = 0, plus the offset
    np.testing.assert_allclose(pos["fixed"], [0.5, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(pos["l1"], [0.3, 0.0, 0.0], atol=1e-9)
    assert model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")] == 60.0
