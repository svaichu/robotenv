# Robot provider: `grx`

The robot provider `grx` turns the humanoids of Fourier's [Wiki-GRx-Models](https://github.com/FFTAI/Wiki-GRx-Models) (GR1, GR2, GR3) into [`ExternalRobot`](../configs/robot.md#externalrobot-a-model-that-lives-elsewhere)s.
It is independent of the other providers: it reads the model files itself and needs no library, only `git` and the network once (the `grx` extra is `robotenv[env]`).

**No model is copied into this repository.** `GRX_ROOT` names a checkout of the repository (it must contain `GRX/`); without it, the first use fetches the pinned commit with `git` into a cache directory
(`GRX_CACHE_DIR`, default `~/.cache/robotenv/grx`; about 250 MB with its history). The models are **GPL-3.0**: `PROVIDER.license(name)` says so and `BaseEnvConfig.explain()` shows it as `robot.license`.

The config side needs neither git nor an engine, so a config naming this provider can be written, hashed and stored anywhere:

```python
from robotenv.config import EndEffectorRef, ExternalJoint, ExternalRobot, Pose, list_robot_providers
from robotenv.robot.providers.grx import cards

assert "grx" in list_robot_providers()
gr3_like = ExternalRobot(
    provider="grx", name="gr3v2_1_1", entry="mjcf/gr3v2_1_1.xml", source="b9b185cef59fbe96",
    joints=(ExternalJoint("waist_yaw_joint", (-2.618, 2.618), 140.4),),
    home_qpos=(0.0,), home_root=Pose((0.0, 0.0, 0.932), (1.0, 0.0, 0.0, 0.0)), keyframe=None,
    end_effector=EndEffectorRef(site=None, body="right_end_effector_link", offset=Pose.identity()), actuator_kind="position",
)
assert gr3_like.n == 1 and set(cards()) == {"gr1t1", "gr1t2", "gr2v3_8_7", "gr3v2_1_1"}
```

With the models available, `describe` opens one and fills the numbers in:

```python no-run
from robotenv.config import BaseEnvConfig, MobileMount
from robotenv.robot.providers.grx import PROVIDER

gr1 = PROVIDER.describe("gr1t2")                          # 32 hinges, a free root, standing; entry "basic"
gr3 = PROVIDER.describe("gr3v2_1_1", "dummy_hand")        # the model with the hand shapes
env = BaseEnvConfig().robot(gr1).task(mount=MobileMount(spawn=(0.0, 0.0, 0.0), heading=0.0)).sim(num_envs=4, scene="plain_floor").build()
torso = PROVIDER.describe("gr2v3_8_7", base="fixed")      # the root welded at the mount: an upper body on a stand
```

## Robots and entries

| robot | model files | hinges | format |
|---|---|---|---|
| `gr1t1`, `gr1t2` | `basic_urdf/<name>.urdf`, `basic_urdf/<name>_dummy_hand.urdf` | 32 | URDF |
| `gr2v3_8_7` | `mjcf/gr2v3_8_7.xml`, `mjcf/gr2v3_8_7_dummy_hand.xml` | 29 | MJCF |
| `gr3v2_1_1` | `mjcf/gr3v2_1_1.xml`, `mjcf/gr3v2_1_1_dummy_hand.xml` | 31 | MJCF |

`entry` is `basic` (the default), `dummy_hand`, or the file path; `ExternalRobot.entry` holds the path. The models with the Fourier six-degree-of-freedom hand and the legacy GR1 and `GR2v3.0.1` directories are not
listed: the finger joints of the hand models have no effort limits and are coupled by mimic joints that MuJoCo does not read (`cards.json`, the data table, is where a model would be added).

## What `describe` and `spec` do

The files have no actuators, no free root and no keyframe, so the provider adds them when it opens a model (`spec`), and what it adds is part of `source`:

- **actuators**: one position actuator per hinge, `kp = 1.0 x effort` and `kd = 0.05 x effort` (N·m/rad), forced limited to the joint's `actuatorfrcrange` (the model's own effort limit; the range comes from the
  joint). The models state no gains, so these are ours, chosen so that a fixed-base robot holds its home posture within 0.07 rad and steps stably under the default integrator. `NativeGains` drives with them;
  `FixedGains` and `InertiaScaledGains` replace them. Every hinge also gets an `armature` of 0.01 kg·m².
- **root**: `base="floating"` (default) adds a free joint on the root body and `home_root` is the pose that stands the robot at the zero posture with its lowest collision geometry 5 mm above the floor (0.93 to
  1.01 m). A floating robot takes a `MobileMount`. `base="fixed"` has no free joint and `home_root` is `None`; the root body is welded at the mount (a `Mount`), so place the mount at the height you want.
- **URDF** (GR1): the root link and fixed links stay as bodies (the end-effector link is a fixed link), visual meshes are kept, and meshes are read from the checkout.
- **driven joints**: every hinge, in the model's order; `controlled=[...]` selects some, in the order given. Joints not driven keep their actuator and are left out of the action: their actuators hold control zero.
- **end effector**: the body `right_end_effector_link` (every model has it and `left_end_effector_link`); pass `end_effector=EndEffectorRef(...)` for another.
- **keyframe**: none; the home posture is the model's zero posture, `home_qpos`.

`spec(robot)` fails with `robot.source: the model changed since this config was written` when the files (the model, the meshes it names) or the provider's additions differ from those the config was made against.
`validate` reports an unsupported simulator (`sim.simulator`), an entry that is not a model of the robot (`robot.entry`) and a stale source (`robot.source`); a missing checkout and a failed fetch are a `ConfigError`
at `provider` that says to set `GRX_ROOT`.

## What to expect

The zero posture is not a balanced stance: a floating robot driven with its home posture settles, then topples. The provider gives the embodiment, not a controller. Running on the CPU simulator
(`mujoco`) is tested; the models go to the Warp simulator through the same `MjSpec`, which is **not** tested in this repository's CI.
