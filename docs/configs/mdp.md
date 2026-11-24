# MDP: rewards, terminations and observations

`TaskConfig.mdp` says what the environment computes on top of the physics: the **reward**, the **termination** rules and the **observation keys**. Each is a list of named *terms*
with parameters as data, so the whole MDP is part of the `env_hash`. Terms read the robot state (joint positions, velocities, efforts, the end effector, the gripper), the task's
own observations and the action, through one context that is the same on the CPU and the GPU simulator.

```python
from robotenv.config import (
    BaseEnvConfig, ConfigError, MdpConfig, RewardTerm, TerminationTerm, reward, termination,
)
from robotenv.mdp import get_term, list_terms
```

`mdp = None` (the default) is the task provider's own reward, success rule and observations: nothing changes for a config that does not use it, and its hash is the one it always had.

## `MdpConfig`

`MdpConfig(rewards, terminations, observations)`; each part is **`None`** (the provider's own) or a tuple.

| part | value | meaning |
|---|---|---|
| `rewards` | tuple of `RewardTerm(fn, weight, params)`, or `None` | the step reward is the sum of `weight * term`; `()` is a reward of 0 |
| `terminations` | tuple of `TerminationTerm(fn, params, kind, terminate)`, or `None` | when an episode ends and what that means (below) |
| `observations` | tuple of keys, or `None` | which entries the observation TensorDict has; nothing else about observations is configured |

Terms are made with the helpers `reward(fn, weight, **params)` and `termination(fn, kind, **params)`. Parameters are checked against the term's declaration, completed with
its defaults and stored, so a hash never depends on a default that changes later.

```python
mdp = MdpConfig(
    rewards=(
        reward("distance", -1.0, a="proprio/ee_pos", b="task/target_pos"),   # -1 * ||ee - target||
        reward("sum_squares", -0.01, key="proprio/qvel"),                   # a penalty on joint speed
        reward("action_rate_sq", -0.1),                                     # a penalty on jerky commands
    ),
    terminations=(
        termination("distance_below", "success", a="proprio/ee_pos", b="task/target_pos", tolerance=0.02),
        termination("abs_above", "failure", key="proprio/qvel", limit=100.0),   # the state blew up
        termination("time_out", "timeout", steps=150),
    ),
    observations=("proprio/qpos", "proprio/qvel", "proprio/ee_pos", "task/target_pos"),
)
assert mdp.rewards[0] == RewardTerm("distance", -1.0, {"a": "proprio/ee_pos", "b": "task/target_pos"})
assert reward("exp_distance", 1.0, a="proprio/ee_pos", b="task/target_pos").params_dict()["scale"] == 0.1   # the default is stored
assert MdpConfig.from_dict(mdp.to_dict()) == mdp                                   # a plain dict, so JSON and YAML files work
assert MdpConfig(None, None, ("proprio/qvel", "proprio/qpos")).observations == ("proprio/qpos", "proprio/qvel")   # a set: order is not information
```

Put it in a task with `mdp=` (it is the last field of `TaskConfig`, and the only one that may be left out), or through the builder:

```python
cfg = BaseEnvConfig().robot(n=3, axes="zyy", total_length=0.9, total_mass=3.0)
with_mdp = cfg.task(mdp=MdpConfig(None, None, ("proprio/qpos", "proprio/ee_pos")))
assert with_mdp.resolve().env_hash != cfg.resolve().env_hash
assert cfg.task(mdp=MdpConfig(None, None, None)).resolve().env_hash == cfg.resolve().env_hash    # nothing configured is no mdp
assert with_mdp.with_overrides({"task.mdp.observations": ["proprio/qvel"]}).resolve().task.mdp.observations == ("proprio/qvel",)
# on the command line: ["task.mdp={\"rewards\": []}"]  (a JSON literal for the whole part; null removes the mdp)
```

## Observation keys

The observation is a nested TensorDict: `"proprio/qpos"` is `td["proprio", "qpos"]`. `MdpConfig.observations` lists the keys that exist; the environment computes only those.
With `observations=None` the environment has everything it always had (the task's keys, `proprio/qpos`, `qvel`, `effort`, `gripper` for a robot with one, `success`, `episode_step`).

| key | shape | content |
|---|---|---|
| `proprio/qpos`, `proprio/qvel`, `proprio/effort` | `[n]` | joint positions, velocities, actuator forces |
| `proprio/qpos_norm` | `[n]` | positions scaled to [-1, 1] by the joint range (unlimited joints stay raw) |
| `proprio/ee_pos` | `[3]` | end effector in the robot's base frame (a robot with an end effector) |
| `proprio/gripper` | `[1]` | gap between the fingers in m (a robot with a gripper) |
| `proprio/last_action` | `[action_dim]` | the action applied in the last step, zeros after a reset |
| `success`, `episode_step`, `time_remaining` | `[1]` | success flag (bool), steps taken (int64), fraction of the episode left |
| `task/<name>` | the task's | the task's own observations: the keys the provider lists (`task/ee_pos`, `task/<object>_pos` for robosuite) |

A key the robot cannot provide is an error with the reason; an unknown key lists the valid ones:

```python
try:
    cfg.task(mdp=MdpConfig(None, None, ("proprio/gripper",))).resolve()
except ConfigError as e:
    assert e.path == "task.mdp.observations[0]" and "no gripper" in str(e)
try:
    MdpConfig(None, None, ("proprio/joint_angles",))
except ConfigError as e:
    assert e.path == "observations[0]" and "proprio/qpos" in str(e)
```

`task/<name>` keys are checked when the environment is built, because only the task knows what it observes; the error lists the task's keys. Scaling, normalisation,
frame stacking, noise and flattening are not configured here: they are TorchRL transforms of the environment (`CatTensors`, `ObservationNorm`, `FrameStack`, ...).
A key can also be read by a reward or a termination without being observed: `reward("distance", ..., a="proprio/ee_pos", ...)` works whatever `observations` says.

## Rewards

`fn(ctx, **params)` returns one value per env; the config weights it. Terms return the *raw* quantity and the weight gives the sign: `-1.0 * distance` is a penalty.

| term | parameters | value |
|---|---|---|
| `task_reward` | | the task provider's own reward |
| `distance` | `a`, `b` | `‖a − b‖` of two observation keys |
| `exp_distance` | `a`, `b`, `scale=0.1` | `exp(−‖a − b‖ / scale)`: 1 on the goal, bounded |
| `component` | `key`, `index=0` | one entry of an observation key, for example a height |
| `sum_squares`, `sum_abs` | `key` | a squared / absolute norm of an observation key |
| `action_sq`, `action_rate_sq` | | `‖a‖²`, `‖aₜ − aₜ₋₁‖²` |
| `joint_limit_proximity` | `margin=0.9` | how far joints are in the outer `1 − margin` of their range |
| `success_bonus` | | 1 where the task is solved |
| `constant` | `value=1.0` | the same every step |

`rewards=None` keeps the provider's reward; to add a penalty to it, list `reward("task_reward", 1.0)` among the terms. The weighted value of each term of the last step is
`env.reward_terms` (`{name: [N, 1]}`; a repeated term is `distance#2`).

## Terminations and success

A `TerminationTerm` is a condition per env; its `kind` says what true means:

| kind | effect |
|---|---|
| `"success"` | the env's `success` flag is the OR of its success terms (the task's own test when there are none), and the episode ends as `terminated`; `terminate=False` sets the flag only |
| `"failure"` | the episode ends as `terminated` |
| `"timeout"` | the episode ends as `truncated` |

`max_episode_steps` always truncates, whatever the terms say. Terms: `task_success` (the provider's test), `distance_below(a, b, tolerance)`, `abs_above(key, limit)` (also true for a
non-finite value), `time_out(steps)`. `terminations=None` is the old behaviour (`terminated` follows `success` when the task's params say `terminate_on_success`); with terminations
configured, say it with a success term and leave `terminate_on_success` false: setting both is an error.

```python
try:
    termination("time_out", "failure", terminate=False, steps=10)
except ConfigError as e:
    assert e.path == "terminate"           # only a success term can leave the episode running
```

## Choosing terms interactively

```python no-run
from robotenv.mdp import ObsSelector, catalog, preview

catalog(cfg)                          # the keys that exist for this robot and task: shape, dtype, why one is unavailable
catalog(cfg, "reward")                # the reward terms with their parameters; also "termination"

sel = ObsSelector(cfg).select_group("proprio").select("task/target_pos").deselect("proprio/effort")
print(sel)                            # the keys, their shapes and the total number of values per env
cfg2 = sel.apply()                    # an EnvConfig with task.mdp.observations set (rewards and terminations kept)
sel.widget()                          # in a notebook: a checklist (needs ipywidgets); .selector is the current selection

preview(cfg2, steps=5)                # builds the env, takes random actions and prints each entry's shape, dtype and range
```

`ObsSelector` is immutable (every call returns a new one) and raises a `ConfigError` for an unknown or unavailable key. `catalog` resolves shapes against the robot
(`proprio/qpos` has `n` entries for an `n`-joint arm) and lists the task's keys when the task can be built. `preview` is the quick check that a term is not constant or off by a factor.

## Writing a term

Terms are registered by name, so a provider (or a project) adds its own. A term is a plain function of the context; the decorator declares its parameters.

```python
from robotenv.mdp import Param, register_term, unregister_term

@register_term("reward", "mine/near_home", params={"radius": Param(0.1, doc="metres")})
def near_home(ctx, radius):
    """1 where every joint is within ``radius`` of zero."""
    return (ctx.robot.qpos.abs() < radius).all(dim=-1, keepdim=True).float()

assert "mine/near_home" in list_terms("reward") and reward("mine/near_home").params_dict() == {"radius": 0.1}
unregister_term("reward", "mine/near_home")
```

The function receives an `MdpContext` (`robotenv.mdp.context`): `ctx.robot` (`qpos`, `qvel`, `effort`, `ee_pos`, `gripper`, `joint_range`), `ctx.get(key)` (any observation key,
`"task/..."` included), `ctx.action` and `ctx.prev_action` in [-1, 1], `ctx.episode_step`, `ctx.success`, `ctx.task_success`, `ctx.task_reward`, and `ctx.sim` / `ctx.task` when more
is needed. Tensors are float32 on `ctx.device` with leading dimension `ctx.num_envs`. Observation terms return `[N, *shape]` and declare `shape` (a function of the robot) and
`dtype`; they take no parameters. Terms may use torch inside the function body, so the config layer can still be imported without it.

Registered terms must be registered in every process that builds a config that names them, like task providers (see [cluster](../cluster.md)).

## Errors

```python
def error_of(build):
    try:
        build()
    except ConfigError as e:
        return e
    raise AssertionError("no error")

assert error_of(lambda: reward("nope")).path == "fn"
assert error_of(lambda: reward("distance", -1.0, a="proprio/ee_pos")).path == "params.b"                 # a required parameter
assert error_of(lambda: reward("distance", -1.0, c=1, a="proprio/ee_pos", b="proprio/qpos")).path == "params"
assert error_of(lambda: termination("time_out", "timeout", steps=2.5)).path == "params.steps"
assert error_of(lambda: MdpConfig(("distance",), None, None)).path == "rewards[0]"
assert error_of(lambda: MdpConfig.from_dict({"rewards": [{"fn": "x", "weight": 1.0, "params": {}}]})).path == "rewards[0].fn"
```

Everything about the *engine* (does the task observe this key, does the simulator offer it) is reported by `make_env` with the path (`task.mdp.rewards[0].params.b`).
