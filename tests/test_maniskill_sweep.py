"""The status of every ManiSkill agent as a robot here (marker ``maniskill``).

``tests/maniskill_expected.json`` records, per agent uid, ``status`` (``ok``, or ``refused: <reason>``) and, for the
``ok`` ones, ``holds``: whether the driven joints stay within ``HOLD_TOLERANCE`` of the keyframe over ``HOLD_STEPS``
steps with a zero delta action (``null`` when the robot has no keyframe). ``ok`` means the robot builds and steps with
finite numbers; it does not mean that it stands or that it moves as in ManiSkill.

A fixed-base robot is swept in the ``empty`` scene and a floating one on ``plain_floor``: several agents are hands
and grippers whose model, mounted at the origin, reaches below z = 0, and a floor there would be a collision the robot
does not have in ManiSkill. A run in which MuJoCo reports an unstable simulation (a bad ``qacc``, after which it resets
the state) is ``refused: unstable`` even though the numbers stay finite.

``python -m tests.test_maniskill_sweep`` rewrites the file from the sweep. A robot recorded as ``refused: not
downloaded`` is not compared when its files are on disk: the file was written without ManiSkill's downloaded assets.
"""

import json
import warnings
from pathlib import Path

import pytest

pytest.importorskip("mani_skill")
mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from robotenv.config import BaseEnvConfig, ConfigError, snapped
from robotenv.robot.providers.maniskill import PROVIDER

pytestmark = pytest.mark.maniskill

EXPECTED = Path(__file__).with_name("maniskill_expected.json")
NOT_DOWNLOADED = "refused: not downloaded"
HOLD_STEPS, HOLD_TOLERANCE = 200, 0.05  # control steps; rad or m


def _unstable(env) -> bool:
    return any(int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0 for d in env._sim.data)


def _finite(td) -> bool:
    return all(torch.isfinite(v).all() for v in td.values(True, True) if v.dtype.is_floating_point)


def status_of(name: str) -> dict:
    """``{"status": ..., "holds": ...}`` of agent ``name``, with the reasons the expected file uses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            robot = PROVIDER.describe(name)
        except ConfigError as e:
            if "not on disk" in e.reason:
                return {"status": NOT_DOWNLOADED, "holds": None}
            if e.path == "effort_limit":
                return {"status": "refused: needs effort_limit", "holds": None}
            reason = "the model does not compile" if "does not compile" in e.reason else "the agent cannot be read"
            return {"status": f"refused: {reason}", "holds": None}
        scene = "empty" if robot.home_root is None else "plain_floor"
        base = BaseEnvConfig().robot(robot).task(mount=snapped(robot)).sim(num_envs=2, scene=scene, integrator="implicitfast")
        try:
            env = base.build()
        except ConfigError:
            return {"status": "refused: make_env", "holds": None}
        td = env.reset()
        assert _finite(td), f"{name}: reset"
        for i in range(20):
            td["action"] = torch.zeros(2, env.action_spec.shape[-1]) if i < 10 else env.action_spec.rand()
            td = env.step(td)["next"]
            if not _finite(td):
                env.close()
                return {"status": "refused: not finite", "holds": None}
        unstable = _unstable(env)
        env.close()
        if unstable:
            return {"status": "refused: unstable", "holds": None}
        if robot.keyframe is None:
            return {"status": "ok", "holds": None}
        env = base.control(mode="joint_pos_delta", action_scale=0.05).build()
        td = env.reset()
        home = td["proprio", "qpos"].clone()
        for _ in range(HOLD_STEPS):
            td["action"] = torch.zeros(2, robot.n)
            td = env.step(td)["next"]
        drift = (td["proprio", "qpos"] - home).abs().max().item()
        env.close()
        return {"status": "ok", "holds": bool(drift < HOLD_TOLERANCE)}


def test_the_expected_file_lists_exactly_maniskills_agents():
    expected = json.loads(EXPECTED.read_text())
    assert sorted(expected) == sorted(PROVIDER.names()), "the library has agents the expected file does not list, or the reverse"
    assert sum(v["status"] == "ok" for v in expected.values()) >= 15  # the file is not empty of robots that run


@pytest.mark.parametrize("name", sorted(json.loads(EXPECTED.read_text())) if EXPECTED.exists() else [])
def test_an_agent_has_its_expected_status(name):
    expected = json.loads(EXPECTED.read_text())[name]
    got = status_of(name)
    if expected["status"] == NOT_DOWNLOADED and got["status"] != NOT_DOWNLOADED:
        pytest.skip(f"{name} is on disk here; the expected file was written without it (it is {got})")
    assert got == expected, "status changed; if intended, python -m tests.test_maniskill_sweep"


if __name__ == "__main__":
    out = {n: status_of(n) for n in PROVIDER.names()}
    EXPECTED.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    count = lambda pred: sum(pred(v) for v in out.values())
    print(
        f"wrote {EXPECTED}: {count(lambda v: v['status'] == 'ok')} ok ({count(lambda v: v['holds'] is True)} hold their keyframe), "
        f"{count(lambda v: v['status'] == NOT_DOWNLOADED)} not downloaded, "
        f"{count(lambda v: v['status'] not in ('ok', NOT_DOWNLOADED))} refused otherwise"
    )
