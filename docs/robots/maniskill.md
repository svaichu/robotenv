# Robot provider: `maniskill`

The robot provider `maniskill` turns a robot of [ManiSkill](https://github.com/haosulab/ManiSkill) (an *agent*, named by its uid) into an
[`ExternalRobot`](../configs/robot.md#externalrobot-a-model-that-lives-elsewhere), through the `mani_skill` library. Install it with `pip install 'robotenv[maniskill]'`
(**Python 3.10 to 3.12**: ManiSkill pins a dependency, `mplib`, that has no wheel for Python 3.13, so on 3.13 the extra installs nothing and the provider reports the library as missing).

## ManiSkill is a robot provider only, for now

ManiSkill brings **robots only**. It is not a task provider and not a simulator:

- **No ManiSkill task is provided.** A ManiSkill scene is not a model file: it is Python code that calls the builders of SAPIEN, ManiSkill's simulator, and a task's success and reward read
  SAPIEN's state. Building such a scene also needs a Vulkan driver, even with rendering switched off. Bringing the tasks is a possible later step, not part of this provider.
- **SAPIEN simulates nothing here.** The simulators stay MuJoCo and MuJoCo Warp. `sapien` is imported, because ManiSkill's agent classes import it, and it is asked once per robot for its joint
  order, in a bare physics scene without a renderer.

A ManiSkill robot is a plain `ExternalRobot`, so it runs in the tasks of every task provider (`empty/idle`, [robosuite](../tasks/robosuite.md), your own) on both simulators.

**No model is copied into this repository.** The models are those of the installed library (and of ManiSkill's asset directory for the robots it downloads); meshes MuJoCo cannot read are
converted into a cache directory outside the repository (below). ManiSkill itself is Apache-2.0; the robot models come from their vendors under their own terms, which the library lists in its
`LICENSE-3RD-PARTY`. `BaseEnvConfig.explain()` shows that pointer as `robot.license`.

The config side needs neither the library nor an engine, so a config naming this provider can be written, hashed and stored anywhere:

```python
from robotenv.config import EndEffectorRef, ExternalJoint, ExternalRobot, Pose, list_robot_providers
from robotenv.robot.providers.maniskill import cards

assert "maniskill" in list_robot_providers()
panda_like = ExternalRobot(
    provider="maniskill", name="panda", entry="panda_v2.urdf", source="9aa42519",
    joints=(ExternalJoint("panda_joint1", (-2.8973, 2.8973), 100.0),),
    home_qpos=(0.0,), home_root=None, keyframe="rest",
    end_effector=EndEffectorRef(site=None, body="panda_hand_tcp", offset=Pose.identity()), actuator_kind="position",
)
assert panda_like.n == 1 and panda_like.provider == "maniskill"
assert cards()["so100"]["end_effector"]["body"] == "Fixed_Jaw_tip"   # data, not code
```

With the library installed, `describe` reads the agent and fills these numbers in:

```python no-run
from robotenv.config import BaseEnvConfig
from robotenv.robot.providers.maniskill import PROVIDER

panda = PROVIDER.describe("panda")                      # ManiSkill's uid: seven arm joints, the fingers held, the TCP link as end effector
env = BaseEnvConfig().robot(panda).sim(num_envs=4, scene="plain_floor", integrator="implicitfast").build()
g1 = PROVIDER.describe("unitree_g1")                    # a floating base: a free root with the pose of the keyframe
PROVIDER.describe("panda", controlled=[*(j.name for j in panda.joints), "panda_finger_joint1"])   # drive a finger too
```

**Set `integrator="implicitfast"`** (or `"implicit"`) with ManiSkill's own servos. PhysX integrates a joint drive implicitly and ManiSkill's damping is tuned for that; under `euler`, the
default, or `rk4` the servos chatter (measured on the Panda at `dt = 0.002`: 0.24 rad of drift in one second at rest, none under `implicitfast`). The model states the integrator it needs and
`make_env` warns when `sim.integrator` is an explicit one; it does not change it. With `FixedGains` or `InertiaScaledGains` any integrator does.

## What `describe` does

`describe(name, entry=None, *, controlled=None, end_effector=None, keyframe=..., effort_limit=None)` reads ManiSkill's agent class, without building a ManiSkill environment:

- **name**: the agent's uid (`PROVIDER.names()` lists them). **entry**: the agent's model file; an agent has one, so the argument is only checked.
- **controlled**: the card's, else the agent's arm joints (`arm_joint_names`), else every joint of its joint-position controller. A finger or gripper joint is driven only when it is listed.
- **range / effort_limit**: the joint range of the model; the force limit of the agent's joint-position controller. Where the controller has none, `describe` raises a `ConfigError` at
  `effort_limit` that names the joint and the argument to pass.
- **keyframe / home**: the card's keyframe, else the agent's `rest`, else its first keyframe, else none. `home_qpos` is that keyframe at the driven joints; for a floating base `home_root` is the
  keyframe's root pose. A home value outside its range is clipped, with a warning.
- **end_effector**: the card's, else the body of the agent's end-effector link (`ee_link_name`), else none.
- **actuator_kind**: `position`. **source**: a content id over the model file, the collision meshes and SRDF it uses, and the agent facts read (joints, gains, keyframes). A ManiSkill
  upgrade that changes any of them gives a different robot hash, and `spec` refuses the old config with `robot.source: the model changed since this config was written`.

A **card** (`robot/providers/maniskill/cards.json`) holds what cannot be derived: an end effector where the agent names none, the driven joints where the gripper is one of the controller's
joints. Most robots need none. More than one end effector is not supported, so a hand has none.

`validate` reports the library missing (naming the extra), an unknown robot (with near matches), files that are not on disk (with ManiSkill's download command), a stale source and a simulator
other than `mujoco` and `warp`, each with its config path.

## What `spec` builds, and how it differs from the robot in ManiSkill

A ManiSkill model is a URDF (for a few agents an MJCF) written for SAPIEN and PhysX. `spec(robot)` opens it as a `mujoco.MjSpec` and adds what the compile of an external robot expects:

| in ManiSkill | here |
|---|---|
| a PD joint drive per controlled joint (stiffness, damping, force limit) | a position actuator with `kp` = stiffness and `kd` = damping; the force limit is the joint's actuator-force range. With `NativeGains` these are the servos; `FixedGains` and `InertiaScaledGains` replace them on the driven joints |
| "balance passive force": gravity switched off on the links of a fixed robot | MuJoCo's gravity compensation (`gravcomp = 1`) on every body of such a robot. A floating robot carries its weight in both |
| keyframes, in SAPIEN's joint order (breadth first) | the same keyframes, mapped onto MuJoCo's joints by name |
| jointed links never collide; an SRDF disables more pairs | a contact exclude for every pair of jointed links and for every pair of the SRDF |
| a friction coefficient on single links (the Panda's fingers) | the same coefficient on the geoms of those links; SAPIEN's contact patch radius has no counterpart and is dropped |
| a root link that is not fixed | a free joint on the root body |
| visual meshes and collision meshes | **collision geometry only**: a render shows the collision meshes |
| PhysX contact and solver settings | MuJoCo's, and the scene's options win ([simulators](../simulators.md#external-robots-on-the-cpu-simulator)) |

So the robot has ManiSkill's kinematics, masses, gains and limits, and MuJoCo's contact dynamics: a policy trained in ManiSkill is not expected to transfer unchanged.

Repairs, each a rule applied to every model that needs it: `package://` mesh paths are resolved against the directory tree of the URDF; a collision mesh in a format MuJoCo does not decode
(`.ply`, `.glb`) is converted once to STL; inertia is balanced; a moving link without mass (the intermediate links of a floating gripper) gets a minimal one; fixed links are kept as bodies
(MuJoCo's URDF import would fuse them, and the end-effector link is one).

**Cache.** Converted meshes go to `$ROBOTENV_CACHE_DIR/maniskill/meshes`, else `~/.cache/robotenv/maniskill/meshes` (`XDG_CACHE_HOME` is honoured). A file is named after the content of the
original mesh, so a changed mesh is converted again, and the robot's `source` hashes the originals, not the cache.

**Robots that need a download.** About half of ManiSkill's agents use models that are not in the wheel. The provider downloads nothing: such a robot is refused with the command that fetches
it, for example `python -m mani_skill.utils.download_asset ur10e`.

## What is driven, what is held

As for every external robot ([menagerie](menagerie.md#what-is-driven-what-is-held)): `ExternalRobot.joints` are driven, one action component each; the other servos (the fingers) are **held** at
the keyframe's value on every step, and there is no gripper action. A fixed-base robot takes a `Mount`, a floating one a `MobileMount`.

## Warp

The Panda and the Panda with a stick run on the Warp simulator with nothing changed in it: gravity compensation, the finger equality, the contact excludes and the joint force ranges are taken
as they are, with `WarpOptions(njmax=128, nconmax=32)`. Against the CPU simulator, over 30 steps of small random actions from the keyframe (4 envs, free space, `implicitfast`), the largest
differences are 1e-6 rad in `qpos`, 8e-5 rad/s in `qvel` and 5e-4 in `effort`, with ManiSkill's own servos in delta and absolute mode and with our gains: inside the tolerances of the parity
tests. This was measured on Warp's CPU device; **CUDA is not verified**. The other robots of the table below were not run on Warp.

## Which robots run

`tests/maniskill_expected.json` records the status of every agent when described with no arguments and run with the `empty/idle` task (2 envs, 10 zero steps and 10 random steps,
`implicitfast`). A fixed-base robot is run in the `empty` scene and a floating one on `plain_floor`: a hand mounted at the origin reaches below z = 0. The suite fails when a robot flips, and the
table below is checked against that file. Today 19 run,
19 are not downloaded and 5 are refused for another reason.

- **ok** means the robot builds and steps with finite numbers and a stable simulation. It does not mean that it stands or that it moves as in ManiSkill.
- **holds keyframe**: with a zero delta action for 200 control steps the driven joints stay within 0.05 rad (or m) of the keyframe; `-` where there is no keyframe. 16 do. `unitree_g1` does
  not: it is a free-standing humanoid without a balance controller.
- **refused: not downloaded**: the model is not in the wheel. These 19 robots are **untested**: the file was written where ManiSkill's download is not reachable.
- **refused: unstable**: MuJoCo reports a diverging simulation. The Inspire hands couple finger joints through mimic constraints on links with an inertia of 2e-9 kg m².
- **refused: needs effort_limit**: the model (the MJCF `humanoid`) has no force limit; `describe(..., effort_limit=...)` supplies one.

| robot | status | holds keyframe |
|---|---|---|
| `allegro_hand_left` | ok | yes |
| `allegro_hand_right` | ok | yes |
| `allegro_hand_right_touch` | ok | yes |
| `anymal_c` | refused: not downloaded | - |
| `dclaw` | ok | yes |
| `fetch` | ok | yes |
| `fixed_inspire_hand_left` | refused: unstable | - |
| `fixed_inspire_hand_right` | refused: unstable | - |
| `floating_ability_hand_right` | ok | yes |
| `floating_inspire_hand_left` | refused: unstable | - |
| `floating_inspire_hand_right` | refused: unstable | - |
| `floating_panda_gripper` | ok | yes |
| `floating_robotiq_2f_85_gripper` | refused: not downloaded | - |
| `googlerobot` | refused: not downloaded | - |
| `humanoid` | refused: needs effort_limit | - |
| `koch-v1.1` | ok | yes |
| `panda` | ok | yes |
| `panda_stick` | ok | yes |
| `panda_wristcam` | ok | yes |
| `so100` | ok | yes |
| `stompy` | refused: not downloaded | - |
| `trifingerpro` | ok | - |
| `unitree_g1` | ok | no |
| `unitree_g1_simplified_legs` | ok | yes |
| `unitree_g1_simplified_upper_body` | ok | yes |
| `unitree_g1_simplified_upper_body_right_arm` | ok | - |
| `unitree_g1_simplified_upper_body_with_head_camera` | ok | yes |
| `unitree_go2` | refused: not downloaded | - |
| `unitree_go2_simplified_locomotion` | refused: not downloaded | - |
| `unitree_h1` | refused: not downloaded | - |
| `unitree_h1_simplified` | refused: not downloaded | - |
| `ur_10e` | refused: not downloaded | - |
| `widowx250s` | refused: not downloaded | - |
| `widowx250s_bridgedataset_flat_table` | refused: not downloaded | - |
| `widowx250s_bridgedataset_sink` | refused: not downloaded | - |
| `widowxai` | refused: not downloaded | - |
| `widowxai_wristcam` | refused: not downloaded | - |
| `xarm6_nogripper` | refused: not downloaded | - |
| `xarm6_nogripper_wristcam` | refused: not downloaded | - |
| `xarm6_robotiq` | refused: not downloaded | - |
| `xarm6_robotiq_wristcam` | refused: not downloaded | - |
| `xarm7_ability` | ok | yes |
| `xlerobot` | refused: not downloaded | - |

## Limits

- **Robot provider only**: no ManiSkill task, scene or object (above).
- **No gripper action**, one end effector at most, no end-effector control modes; ManiSkill's other controllers (end-effector pose, velocity) are not used, only the gains of its joint-position controller.
- **Collision geometry only** in simulation and in renders.
- **Python 3.10 to 3.12**, and no automatic download.
- Everything in [the limits of external robots](menagerie.md#limits) applies.

How a provider like this one is written: [writing a robot provider](writing-a-robot-provider.md).
