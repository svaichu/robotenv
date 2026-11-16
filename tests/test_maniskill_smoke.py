"""The ``mani_skill`` library as the ``maniskill`` robot provider needs it (marker ``maniskill``; no download, no GPU)."""

import os

import pytest

pytest.importorskip("mani_skill")
mujoco = pytest.importorskip("mujoco")

import mani_skill.agents.robots  # noqa: F401  (registers the agents)
from mani_skill.agents.registration import REGISTERED_AGENTS

pytestmark = pytest.mark.maniskill


def test_the_panda_agent_is_registered_and_its_model_ships_in_the_wheel():
    cls = REGISTERED_AGENTS["panda"].agent_cls
    assert cls.urdf_path.endswith("panda_v2.urdf") and os.path.exists(cls.urdf_path)
    assert cls.ee_link_name == "panda_hand_tcp" and list(cls.keyframes) == ["rest"]


def test_the_panda_urdf_compiles_in_mujoco_without_actuators():
    model = mujoco.MjModel.from_xml_path(REGISTERED_AGENTS["panda"].agent_cls.urdf_path)
    assert model.nq == 9 and model.nu == 0


def test_sapien_gives_its_joint_order_without_a_renderer():
    from robotenv.robot.providers.maniskill._model import sapien_joint_order

    panda = REGISTERED_AGENTS["panda"].agent_cls
    order = sapien_joint_order(panda.urdf_path, True)
    assert order == (*(f"panda_joint{i}" for i in range(1, 8)), "panda_finger_joint1", "panda_finger_joint2")
    # A branching robot: SAPIEN's order is breadth first, MuJoCo's depth first, so a keyframe must be mapped by name.
    hand = REGISTERED_AGENTS["allegro_hand_right"].agent_cls
    order = sapien_joint_order(hand.urdf_path, True)
    assert order[:5] == ("joint_0.0", "joint_4.0", "joint_8.0", "joint_12.0", "joint_1.0") and len(order) == 16
