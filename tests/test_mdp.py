"""The MDP of a task as data (``TaskConfig.mdp``): config rules, hashing, and the CPU environment that evaluates it.

The config part needs only numpy. The environment part runs ``toy/reach`` (``tests/toy_provider``) on the CPU simulator.
"""

import json
from dataclasses import replace

import pytest

import tests.conftest  # noqa: F401  (registers the toy task provider)
from robotenv.config import (
    ConfigError,
    ControlConfig,
    EnvConfig,
    FixedBox,
    FixedGains,
    MdpConfig,
    Mount,
    MujocoOptions,
    RewardTerm,
    SimConfig,
    TaskConfig,
    TerminationTerm,
    TwoFingerGripper,
    chain,
    reward,
    termination,
)
from robotenv.mdp import Param, get_term, list_terms, register_term, unregister_term
from tests.toy_provider import ReachParams

COMMON = dict(radius=0.03, damping=0.5, armature=0.01, frictionloss=0.0, effort_limit=50.0)
DIST = dict(a="proprio/ee_pos", b="task/target_pos")


def make_cfg(mdp=None, n=2, num_envs=3, max_episode_steps=20, terminate=False, gripper=None, mode="joint_pos_abs"):
    arm = chain(n, "zy" * (n // 2), total_length=0.9, total_mass=2.0, range=(-2.0, 2.0), gripper=gripper, **COMMON)
    return EnvConfig(
        robot=arm,
        control=ControlConfig(mode, FixedGains(60.0, 6.0), None),
        task=TaskConfig(ReachParams("reach", 0.0, terminate), max_episode_steps, 0.02,
                        FixedBox((0.2, -0.3, 0.0), (0.6, 0.3, 0.4)), Mount(), mdp),
        sim=SimConfig("mujoco", num_envs, "cpu", 0.002, "euler", 50, 0, "plain_floor", options=MujocoOptions(False)),
    )


# --- the config side (numpy only) -------------------------------------------------------------------------------


def test_nothing_configured_is_the_same_config_and_hash_as_no_mdp():
    plain = make_cfg()
    assert plain.task.mdp is None and "mdp" not in plain.task.to_dict()
    empty = make_cfg(MdpConfig(None, None, None))
    assert empty.task.mdp is None and empty.env_hash == plain.env_hash


def test_a_configured_mdp_changes_the_hash_and_the_parts_are_independent():
    base = make_cfg().env_hash
    rewards = make_cfg(MdpConfig((reward("distance", -1.0, **DIST),), None, None))
    obs = make_cfg(MdpConfig(None, None, ("proprio/qpos",)))
    assert len({base, rewards.env_hash, obs.env_hash}) == 3
    assert rewards.task.mdp.terminations is None and rewards.task.mdp.observations is None


def test_terms_store_their_defaults_so_the_hash_does_not_depend_on_a_later_default():
    t = reward("joint_limit_proximity", 2.0)
    assert t.params_dict() == {"margin": 0.9}
    assert t == RewardTerm("joint_limit_proximity", 2.0, {"margin": 0.9}) == RewardTerm("joint_limit_proximity", 2, ())


def test_observation_selection_is_a_set_the_order_is_not_information():
    a = MdpConfig(None, None, ("proprio/qvel", "proprio/qpos", "proprio/qvel"))
    b = MdpConfig(None, None, ["proprio/qpos", "proprio/qvel"])
    assert a == b and a.observations == ("proprio/qpos", "proprio/qvel")


def test_dict_and_json_round_trip():
    mdp = MdpConfig(
        rewards=(reward("distance", -1.0, **DIST), reward("sum_squares", -0.01, key="proprio/qvel"), reward("action_rate_sq", -0.1)),
        terminations=(termination("distance_below", "success", tolerance=0.05, **DIST),
                      termination("abs_above", "failure", key="proprio/qvel", limit=50.0),
                      termination("time_out", "timeout", steps=10)),
        observations=("proprio/qpos", "task/target_pos"),
    )
    cfg = make_cfg(mdp)
    again = EnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.env_hash == cfg.env_hash
    assert MdpConfig.from_dict(mdp.to_dict()) == mdp


def test_terms_validate_name_parameters_and_types():
    with pytest.raises(ConfigError, match="unknown reward term 'nope'"):
        reward("nope")
    with pytest.raises(ConfigError, match="requires this parameter"):
        reward("distance", -1.0, a="proprio/ee_pos")
    with pytest.raises(ConfigError, match="no parameter 'c'"):
        reward("distance", -1.0, c=1, **DIST)
    with pytest.raises(ConfigError, match="unknown observation key 'proprio/zzz'"):
        reward("sum_squares", -1.0, key="proprio/zzz")
    with pytest.raises(ConfigError, match="expected an integer"):
        termination("time_out", "timeout", steps=2.5)
    with pytest.raises(ConfigError, match="expected a number"):
        reward("exp_distance", 1.0, scale="far", **DIST)
    with pytest.raises(ConfigError, match="expected a number"):
        RewardTerm("constant", True, {})
    with pytest.raises(ConfigError, match="task/"):
        MdpConfig(None, None, ("task/",))
    with pytest.raises(ConfigError, match="rewards\\[0\\]"):
        MdpConfig(("distance",), None, None)
    with pytest.raises(ConfigError, match="expected a sequence of keys"):
        MdpConfig(None, None, "proprio/qpos")


def test_only_a_success_term_may_leave_the_episode_running():
    assert termination("task_success", "success", terminate=False).terminate is False
    with pytest.raises(ConfigError, match="only a success term"):
        termination("task_success", "failure", terminate=False)
    with pytest.raises(ConfigError, match="kind"):
        termination("task_success", "win")


def test_from_dict_rejects_unknown_keys_and_names_the_term_that_is_wrong():
    with pytest.raises(ConfigError, match="unknown keys: speed"):
        MdpConfig.from_dict({"speed": 1})
    with pytest.raises(ConfigError, match=r"rewards\[1\]"):
        MdpConfig.from_dict({"rewards": [{"fn": "constant", "weight": 1.0, "params": {}}, {"fn": "x", "weight": 1.0, "params": {}}]})


def test_the_robot_decides_which_keys_exist():
    with pytest.raises(ConfigError, match="task.mdp.observations\\[0\\].*no gripper"):
        make_cfg(MdpConfig(None, None, ("proprio/gripper",)))
    make_cfg(MdpConfig(None, None, ("proprio/gripper",)), gripper=TwoFingerGripper.default())
    with pytest.raises(ConfigError, match="rewards\\[0\\].params.key.*no gripper"):
        make_cfg(MdpConfig((reward("sum_squares", 1.0, key="proprio/gripper"),), None, None))


def test_explicit_terminations_and_terminate_on_success_must_not_both_say_it():
    mdp = MdpConfig(None, (termination("task_success", "success"),), None)
    with pytest.raises(ConfigError, match="terminate_on_success"):
        make_cfg(mdp, terminate=True)
    make_cfg(mdp, terminate=False)
    make_cfg(MdpConfig((reward("constant", 1.0),), None, None), terminate=True)  # rewards do not conflict


def test_registry_lists_and_rejects_duplicates():
    assert {"proprio/qpos", "proprio/ee_pos", "success"} <= set(list_terms("observation"))
    assert {"distance", "task_reward", "action_rate_sq"} <= set(list_terms("reward"))
    assert {"task_success", "time_out"} <= set(list_terms("termination"))
    with pytest.raises(ValueError, match="already registered"):
        register_term("reward", "distance")(lambda ctx: 0)
    with pytest.raises(ValueError, match="declare its shape"):
        register_term("observation", "mine/x")(lambda ctx: 0)
    with pytest.raises(ValueError, match="no parameters"):
        register_term("observation", "mine/x", shape=lambda i: (1,), params={"k": Param(1.0)})(lambda ctx: 0)


def test_a_registered_term_is_usable_in_a_config_and_unknown_after_unregistering():
    @register_term("reward", "test_pkg/twice", params={"k": Param(2.0)})
    def twice(ctx, k):
        """twice"""
        return ctx.column(k)

    try:
        assert get_term("reward", "test_pkg/twice").doc == "twice"
        assert reward("test_pkg/twice").params_dict() == {"k": 2.0}
    finally:
        unregister_term("reward", "test_pkg/twice")
    with pytest.raises(ConfigError, match="unknown reward term"):
        reward("test_pkg/twice")


# --- the environment (CPU) ---------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("torchrl")

from torchrl.envs.utils import check_env_specs  # noqa: E402

from robotenv.envs import make_env  # noqa: E402


def step(env, td, action):
    td["action"] = action
    return env.step(td)["next"]


def test_no_mdp_keeps_every_observation_of_before():
    td = make_env(make_cfg()).reset()
    assert set(td.keys()) >= {"task", "proprio", "success", "episode_step"}
    assert set(td["task"].keys()) == {"ee_pos", "target_pos"} and set(td["proprio"].keys()) == {"qpos", "qvel", "effort"}


def test_observations_selects_exactly_those_keys_and_the_specs_agree():
    cfg = make_cfg(MdpConfig(None, None, ("proprio/qpos", "proprio/ee_pos", "task/target_pos", "time_remaining")))
    env = make_env(cfg)
    check_env_specs(env)
    assert set(env.observation_space) == {"task", "proprio", "time_remaining"}
    assert set(env.observation_space["proprio"]) == {"qpos", "ee_pos"} and set(env.observation_space["task"]) == {"target_pos"}
    assert env.observation_space["proprio"]["ee_pos"] == torch.Size([3])
    td = env.reset()
    assert td["proprio", "qpos"].shape == (3, 2) and td["time_remaining"].tolist() == [[1.0]] * 3
    assert "episode_step" not in td.keys() and "qvel" not in td["proprio"].keys()


def test_a_selection_without_task_keys_has_no_task_group():
    env = make_env(make_cfg(MdpConfig(None, None, ("proprio/qpos",))))
    assert "task" not in env.reset().keys()


def test_an_empty_selection_leaves_only_the_flags():
    env = make_env(make_cfg(MdpConfig(None, None, ())))
    check_env_specs(env)
    td = env.reset()
    assert set(td.keys()) == {"done", "terminated", "truncated"}


def test_a_task_key_the_task_does_not_observe_is_an_error_at_build_time_that_lists_the_keys():
    with pytest.raises(ConfigError, match=r"observations\[0\].*task/target_pos"):
        make_env(make_cfg(MdpConfig(None, None, ("task/nonsense",))))
    with pytest.raises(ConfigError, match=r"rewards\[0\].params.b"):
        make_env(make_cfg(MdpConfig((reward("distance", -1.0, a="proprio/ee_pos", b="task/nope"),), None, None)))


def test_observed_values_are_the_robot_state():
    env = make_env(make_cfg(MdpConfig(None, None, ("proprio/qpos", "proprio/qpos_norm", "proprio/last_action", "proprio/ee_pos", "task/ee_pos"))))
    td = env.reset()
    assert torch.equal(td["proprio", "last_action"], torch.zeros(3, 2))
    nxt = step(env, td, torch.full((3, 2), 0.5))
    assert torch.equal(nxt["proprio", "last_action"], torch.full((3, 2), 0.5))
    assert torch.allclose(nxt["proprio", "ee_pos"], nxt["task", "ee_pos"], atol=1e-6)  # the task's key is the same site
    assert torch.allclose(nxt["proprio", "qpos_norm"], nxt["proprio", "qpos"] / 2.0, atol=1e-6)  # range (-2, 2), mid 0


def test_rewards_none_is_the_tasks_own_and_task_reward_term_is_the_same_number():
    sel = ("proprio/ee_pos", "task/target_pos")
    own = make_env(make_cfg(MdpConfig(None, None, sel)))
    via = make_env(make_cfg(MdpConfig((reward("task_reward", 1.0),), None, sel)))
    a, b = own.reset(), via.reset()
    act = torch.full((3, 2), 0.3)
    ra, rb = step(own, a, act)["reward"], step(via, b, act.clone())["reward"]
    assert torch.equal(ra, rb) and (ra < 0).all()


def test_weighted_terms_sum_and_each_is_available_by_name():
    mdp = MdpConfig(
        rewards=(reward("distance", -2.0, **DIST), reward("sum_squares", -0.5, key="proprio/qvel"),
                 reward("constant", 0.25), reward("distance", -1.0, **DIST)),
        terminations=None,
        observations=("proprio/ee_pos", "proprio/qvel", "task/target_pos"),
    )
    env = make_env(make_cfg(mdp))
    td = env.reset()
    nxt = step(env, td, torch.full((3, 2), 0.7))
    dist = (nxt["proprio", "ee_pos"] - nxt["task", "target_pos"]).norm(dim=-1, keepdim=True)
    expected = -2.0 * dist - 0.5 * nxt["proprio", "qvel"].pow(2).sum(-1, keepdim=True) + 0.25 - dist
    assert torch.allclose(nxt["reward"], expected, atol=1e-5)
    terms = env.reward_terms
    assert list(terms) == ["distance", "sum_squares", "constant", "distance#2"]
    assert torch.allclose(sum(terms.values()), nxt["reward"])
    assert nxt["reward"].dtype == torch.float32 and nxt["reward"].shape == (3, 1)


def test_an_empty_reward_tuple_is_zero_reward():
    env = make_env(make_cfg(MdpConfig((), None, None)))
    assert torch.equal(step(env, env.reset(), torch.ones(3, 2))["reward"], torch.zeros(3, 1))


def test_action_terms_use_the_action_and_the_one_before():
    mdp = MdpConfig((reward("action_rate_sq", 1.0), reward("action_sq", 10.0)), None, ("proprio/qpos",))
    env = make_env(make_cfg(mdp))
    td = env.reset()
    a1, a2 = torch.full((3, 2), 0.5), torch.full((3, 2), -0.25)
    r1 = step(env, td, a1)["reward"]
    assert torch.allclose(r1, torch.full((3, 1), 2 * 0.25 + 10 * 2 * 0.25))  # (0.5 - 0)^2 * 2 + 10 * 0.5^2 * 2
    td2 = env.reset()
    step(env, td2, a1)
    r2 = step(env, td2.clone(), a2)["reward"]
    assert torch.allclose(r2, torch.full((3, 1), 2 * 0.75**2 + 10 * 2 * 0.0625))


def test_reset_clears_the_last_action_of_the_reset_envs_only():
    env = make_env(make_cfg(MdpConfig(None, None, ("proprio/last_action",))))
    td = env.reset()
    nxt = step(env, td, torch.ones(3, 2))
    nxt["_reset"] = torch.tensor([[True], [False], [False]])
    out = env.reset(nxt)
    assert out["proprio", "last_action"][0].abs().sum() == 0 and (out["proprio", "last_action"][1:] == 1).all()


def test_joint_limit_proximity_is_zero_inside_and_grows_outside_the_margin():
    from tensordict import TensorDict

    env = make_env(make_cfg(MdpConfig((reward("joint_limit_proximity", 1.0, margin=0.5),), None, ("proprio/qpos_norm",)), n=2))
    inside = step(env, env.reset(), torch.zeros(3, 2))
    assert torch.equal(inside["reward"], torch.zeros(3, 1))
    near_limit = TensorDict({"qpos": torch.tensor([[1.9, -1.9]] * 3)}, batch_size=[3])  # range (-2, 2): |q_norm| = 0.95
    out = step(env, env.reset(near_limit), torch.zeros(3, 2))
    assert (out["reward"] > 0.5).all() and (out["reward"] < 0.95).all()  # about 2 * (0.95 - 0.5), a little less after a step


def test_component_and_exp_distance_and_success_bonus():
    mdp = MdpConfig(
        (reward("component", 1.0, key="task/target_pos", index=2), reward("exp_distance", 1.0, scale=0.5, **DIST),
         reward("success_bonus", 100.0)), None, ("task/target_pos", "proprio/ee_pos", "success"))
    env = make_env(make_cfg(mdp))
    nxt = step(env, env.reset(), torch.zeros(3, 2))
    d = (nxt["proprio", "ee_pos"] - nxt["task", "target_pos"]).norm(dim=-1, keepdim=True)
    expected = nxt["task", "target_pos"][:, 2:3] + (-d / 0.5).exp() + 100.0 * nxt["success"].float()
    assert torch.allclose(nxt["reward"], expected, atol=1e-5)


def test_timeout_term_truncates_and_max_episode_steps_still_does():
    mdp = MdpConfig(None, (termination("time_out", "timeout", steps=3),), ("episode_step",))
    env = make_env(make_cfg(mdp, max_episode_steps=20))
    td, flags = env.reset(), []
    for _ in range(4):
        td = step(env, td, torch.zeros(3, 2))
        flags.append((bool(td["truncated"][0]), bool(td["terminated"][0])))
    assert flags == [(False, False), (False, False), (True, False), (True, False)]
    short = make_env(make_cfg(MdpConfig(None, (), None), max_episode_steps=2))
    td = short.reset()
    td = step(short, td, torch.zeros(3, 2))
    assert not td["truncated"].any()
    assert step(short, td, torch.zeros(3, 2))["truncated"].all()


def test_failure_term_terminates():
    mdp = MdpConfig(None, (termination("abs_above", "failure", key="proprio/qvel", limit=0.01),), ("proprio/qvel",))
    env = make_env(make_cfg(mdp))
    nxt = step(env, env.reset(), torch.ones(3, 2))
    assert nxt["terminated"].all() and nxt["done"].all() and not nxt["truncated"].any()
    calm = make_env(make_cfg(MdpConfig(None, (termination("abs_above", "failure", key="proprio/qvel", limit=1e6),), None)))
    assert not step(calm, calm.reset(), torch.ones(3, 2))["terminated"].any()


def test_success_terms_define_the_flag_and_end_the_episode_unless_told_not_to():
    near = termination("distance_below", "success", tolerance=100.0, **DIST)  # always true
    ends = make_env(make_cfg(MdpConfig(None, (near,), ("success",))))
    nxt = step(ends, ends.reset(), torch.zeros(3, 2))
    assert nxt["success"].all() and nxt["terminated"].all()
    flag_only = make_env(make_cfg(MdpConfig(None, (termination("distance_below", "success", terminate=False, tolerance=100.0, **DIST),), ("success",))))
    nxt = step(flag_only, flag_only.reset(), torch.zeros(3, 2))
    assert nxt["success"].all() and not nxt["terminated"].any()
    far = make_env(make_cfg(MdpConfig(None, (termination("distance_below", "success", tolerance=1e-6, **DIST),), ("success",))))
    assert not step(far, far.reset(), torch.zeros(3, 2))["success"].any()


def test_without_success_terms_the_flag_is_the_tasks_own_and_terminate_on_success_still_works():
    cfg = make_cfg(MdpConfig((reward("constant", 1.0),), None, ("success",)), terminate=True)
    env = make_env(cfg)
    td = env.reset()
    nxt = step(env, td, torch.zeros(3, 2))
    assert nxt["terminated"].tolist() == nxt["success"].tolist()


def test_task_success_term_is_the_providers_rule():
    env = make_env(make_cfg(MdpConfig(None, (termination("task_success", "success"),), ("success",))))
    plain = make_env(make_cfg(MdpConfig(None, None, ("success",))))
    act = torch.zeros(3, 2)
    assert torch.equal(step(env, env.reset(), act)["success"], step(plain, plain.reset(), act)["success"])


def test_the_selection_does_not_change_the_physics_or_the_task_stream():
    full = make_env(make_cfg())
    some = make_env(make_cfg(MdpConfig(None, None, ("task/target_pos", "proprio/qpos"))))
    a, b = full.reset(), some.reset()
    assert torch.equal(a["task", "target_pos"], b["task", "target_pos"])
    for _ in range(3):
        a, b = step(full, a, torch.full((3, 2), 0.4)), step(some, b, torch.full((3, 2), 0.4))
    assert torch.equal(a["proprio", "qpos"], b["proprio", "qpos"])


def test_a_gripper_robot_offers_the_gripper_key_and_a_default_selection_still_contains_it():
    gripper = TwoFingerGripper.default()
    cfg = make_cfg(None, n=2, gripper=gripper)
    assert "gripper" in make_env(cfg).reset()["proprio"].keys()
    env = make_env(make_cfg(MdpConfig(None, None, ("proprio/gripper", "proprio/last_action")), gripper=gripper))
    td = env.reset()
    assert td["proprio", "gripper"].shape == (3, 1) and td["proprio", "last_action"].shape == (3, 3)  # n joints + the opening
    check_env_specs(env)


def test_replace_swaps_only_the_mdp():
    cfg = make_cfg()
    other = replace(cfg, task=replace(cfg.task, mdp=MdpConfig((), None, None)))
    assert other.env_hash != cfg.env_hash and other.robot == cfg.robot


# --- the builder layer and the interactive tools --------------------------------------------------------------------

from robotenv.config import BaseEnvConfig  # noqa: E402
from robotenv.mdp import ObsSelector, catalog, preview  # noqa: E402


def base():
    return BaseEnvConfig().task(ReachParams("reach", 0.0, False), success_tolerance=0.02,
                                target_region=FixedBox((0.2, -0.3, 0.0), (0.6, 0.3, 0.4))).robot(n=3, axes="zyy").sim(num_envs=2)


def test_builder_writes_the_mdp_and_round_trips_it_through_dict_overrides_and_validation():
    mdp = MdpConfig((reward("distance", -1.0, **DIST),), None, ("proprio/qpos", "task/target_pos"))
    b = base().task(mdp=mdp)
    assert b.resolve().task.mdp == mdp and BaseEnvConfig.from_dict(b.to_dict()) == b
    assert base().to_dict()["task"]["mdp"] is None and base().resolve().task.mdp is None
    assert b.with_overrides({"task.mdp.observations": ["proprio/qvel"]}).resolve().task.mdp.observations == ("proprio/qvel",)
    assert b.with_overrides(["task.mdp=null"]).resolve().env_hash == base().resolve().env_hash
    only_obs = base().with_overrides(['task.mdp={"observations": ["proprio/qpos"]}']).resolve().task.mdp
    assert only_obs.observations == ("proprio/qpos",) and only_obs.rewards is None
    assert "task.mdp" in b.diff()
    with pytest.raises(ConfigError, match="task.mdp.observations\\[0\\]"):
        b.with_overrides({"task.mdp.observations": ["nope"]}).resolve()


def test_catalog_resolves_shapes_against_the_robot_and_lists_the_tasks_keys():
    c = catalog(base())
    shapes = {r.name: r.shape for r in c}
    assert shapes["proprio/qpos"] == (3,) and shapes["proprio/ee_pos"] == (3,) and shapes["task/target_pos"] == (3,)
    assert not {r.name: r for r in c}["proprio/gripper"].available
    assert "proprio/gripper" not in c.names() and "proprio/gripper" in c.names(available_only=False)
    assert "proprio/qpos" in repr(c) and "n/a" in repr(c)
    assert "distance" in catalog(base(), "reward").names() and "time_out" in catalog(base(), "termination").names()
    assert not any(r.name.startswith("task/") for r in catalog(base(), task_keys=False))
    with pytest.raises(ValueError):
        catalog(base(), "action")


def test_catalog_declared_shapes_are_the_shapes_the_environment_produces():
    cfg = base().resolve()
    keys = tuple(n for n in catalog(cfg).names() if n != "proprio/gripper")
    env = make_env(replace(cfg, task=replace(cfg.task, mdp=MdpConfig(None, None, keys))))
    td = env.reset()
    for row in catalog(cfg):
        if row.name in keys:
            value = td[tuple(row.name.split("/"))]
            assert tuple(value.shape[1:]) == row.shape, row.name
            assert str(value.dtype).replace("torch.", "") == row.dtype, row.name


def test_selector_is_immutable_validates_and_applies():
    cfg = base().resolve()
    sel = ObsSelector(cfg)
    assert sel.keys == () and "nothing selected" in repr(sel)
    more = sel.select("proprio/qpos", "task/target_pos").select("proprio/qpos")
    assert sel.keys == () and more.keys == ("proprio/qpos", "task/target_pos") and more.size == 6
    assert more.deselect("proprio/qpos", "not selected").keys == ("task/target_pos",)
    assert set(sel.select_group("proprio").keys) == {n for n in catalog(cfg).names() if n.startswith("proprio/")}
    assert more.clear().keys == ()
    with pytest.raises(ConfigError, match="not available"):
        sel.select("proprio/gripper")
    with pytest.raises(ConfigError, match="unknown observation key 'task/zz'"):
        sel.select("task/zz")
    with pytest.raises(ConfigError, match="no available keys under 'nope'"):
        sel.select_group("nope")
    applied = more.apply()
    assert applied.task.mdp.observations == more.keys and applied.env_hash != cfg.env_hash
    # rewards and terminations the config already has are kept
    with_rewards = replace(cfg, task=replace(cfg.task, mdp=MdpConfig((reward("constant", 1.0),), None, None)))
    kept = ObsSelector(with_rewards).select("proprio/qvel").apply().task.mdp
    assert kept.rewards == with_rewards.task.mdp.rewards and kept.observations == ("proprio/qvel",)
    assert ObsSelector(applied).keys == more.keys  # starts from what the config selects
    assert ObsSelector(base()).keys == ()  # a BaseEnvConfig is accepted


def test_selector_widget_needs_ipywidgets_or_works():
    sel = ObsSelector(base().resolve()).select("proprio/qpos")
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="ipywidgets"):
            sel.widget()
        return
    box = sel.widget()
    boxes = [c for c in box.children if hasattr(c, "description")]
    assert next(c for c in boxes if c.description == "proprio/qpos").value is True
    next(c for c in boxes if c.description == "task/target_pos").value = True
    assert box.selector.keys == ("proprio/qpos", "task/target_pos")
    next(c for c in boxes if c.description == "proprio/qpos").value = False
    assert box.selector.keys == ("task/target_pos",)


def test_preview_reports_every_entry_and_the_reward_terms():
    mdp = MdpConfig((reward("distance", -1.0, **DIST), reward("constant", 0.5)), None, ("proprio/qpos", "proprio/ee_pos", "task/target_pos"))
    p = preview(base().task(mdp=mdp), steps=3, num_envs=4)
    keys = {r.key for r in p}
    assert {"proprio/qpos", "proprio/ee_pos", "task/target_pos", "reward", "done"} <= keys and "proprio/qvel" not in keys
    qpos = next(r for r in p if r.key == "proprio/qpos")
    assert qpos.shape == (3,) and qpos.low <= qpos.mean <= qpos.high
    assert set(p.reward_terms) == {"distance", "constant"} and p.reward_terms["constant"] == 0.5
    assert "after 3 steps" in repr(p) and "proprio/qpos" in repr(p)
