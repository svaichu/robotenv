# Robot provider: reduced (fewer joints, from the end-effector end)

`robotdesigner.eliminate_joints(robot, k)` takes an `ExternalRobot` of any robot provider and returns the same robot
without its last `k` driven joints and the links they move. It answers questions like "how does a Panda with four
joints compare with the full one?" on the real model, not a parametric copy of it. The result is an `ExternalRobot` of
the robot provider `reduced`, which the `robotdesigner` package registers with robotenv (entry point
`robotenv.robot.providers`), so `make_env` builds it like any other robot.

```python no-run
from robotdesigner import eliminate_joints
from robotenv.config import get_robot_provider

panda = get_robot_provider("menagerie").describe("franka_emika_panda")   # 7 driven joints
arm4 = eliminate_joints(panda, 3)                                         # joints 1 to 4; link5 onwards is gone
arm4 = eliminate_joints("menagerie/franka_emika_panda", 3)                # the same: the robot is described first
print(arm4)                                                               # provider reduced, n=4, end effector on link4
```

## What is removed and where the end effector goes

The body that carries the first removed joint (joint `n - k + 1`) is removed with its whole subtree: the links, the hand
and its fingers. MuJoCo then drops everything that refers into that subtree: actuators (also those of the fingers,
which were held, not driven), tendons, equality constraints, contact excludes and pairs, sensors. The keyframes are kept,
remapped joint by joint and actuator by actuator, so `keyframe` still sets the state of whatever is left.

The new end effector is the frame of the removed body, given on its parent (`EndEffectorRef(body=<parent>,
offset=<the removed body's pose in the parent>)`): it sits where the first removed joint was. With the home posture
cut to the kept joints, the kept part of the arm is placed exactly as in the original.

The driven joints must form a serial chain toward the end effector: the kept ones on the path from the root to the cut,
the removed ones below it. A robot whose driven joints branch (legs of a floating base) is refused with `ConfigError`
at `k`, and so is `k` outside `1 <= k < n`.

## How the reduced robot refers to its model

No model file is written. The reduced robot refers to the original and is cut when opened:

| field | value |
|---|---|
| `provider` | `reduced` |
| `name` | `<provider>/<name>` of the original, as `menagerie/franka_emika_panda` |
| `entry` | the original's |
| `source` | the original's, then `#cut=<body>` (the removed body): hashed, so `k` is part of the robot hash |
| `joints`, `home_qpos` | the original's first `n - k` |
| `home_root`, `keyframe` | the original's |
| `actuator_kind` | the original's, recomputed over the kept joints when it was `mixed` |
| `label` | the original's label (or name), then `-k`, unless given |

A reduced robot is plain config data like any `ExternalRobot`: it can be written, read back and hashed without an
engine, and the cut is part of its identity.

```python
from robotenv.config import EndEffectorRef, ExternalJoint, ExternalRobot, Pose, list_robot_providers, robot_from_dict

assert "reduced" in list_robot_providers()           # registered by robotdesigner's entry point, nothing loaded
arm = ExternalRobot(
    provider="reduced", name="toyrobot/gripper4", entry="gripper4.xml", source="3a9c#cut=hand",
    joints=(ExternalJoint("j1", (-2.0, 2.0), 40.0), ExternalJoint("j2", (-1.5, 1.5), 30.0), ExternalJoint("j3", (-1.5, 1.5), 20.0)),
    home_qpos=(0.1, -0.2, 0.3), home_root=None, keyframe="home",
    end_effector=EndEffectorRef(site=None, body="l3", offset=Pose((0.2, 0.0, 0.01), (1.0, 0.0, 0.0, 0.0))),
    actuator_kind="position",
)
assert robot_from_dict(arm.to_dict()) == arm
shorter = ExternalRobot.from_dict({**arm.to_dict(), "source": "3a9c#cut=l3"})
assert shorter.config_hash != arm.config_hash       # a different cut is a different robot
```

`spec(robot)` asks the original's provider for its model, so a changed original is reported as stale exactly as for
the original. `validate` is the original provider's. A reduced robot can be reduced again; the names and sources nest
(`reduced/menagerie/franka_emika_panda`, `<source>#cut=link7#cut=link6`) and the result is the same robot as reducing
at once, up to those strings.

The provider's `describe("menagerie/franka_emika_panda", eliminate=3, **options)` does the same with the original's
`describe` options.

## Requirements

The tool and the provider need robotenv with MuJoCo (`robotenv[env]`) and the provider of the original robot; they are
imported when called, so `import robotdesigner` stays numpy only. Tests: `tests/test_reduce.py`, on the test-only robot
provider (no download).
