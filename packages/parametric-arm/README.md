# parametric-arm

Parametric robot arms as frozen, validated, hashable data. Numpy only: no simulator is imported.

```python
from parametric_arm import chain

arm = chain(3, "zyy", total_length=0.9, total_mass=3.0,
            radius=0.02, damping=0.1, armature=0.01, frictionloss=0.0, effort_limit=50.0)
print(arm.n, arm.config_hash)
```

Contents: `ParametricArm` (with `JointSpec`, `LinkSpec`, `Segment`, `Capsule`, `ToolSpec`), `TwoFingerGripper`, `Pose`, the `chain()` generator and `ConfigError`.

This package is the lowest layer of the `robotenv` repository: `robotenv` depends on it and re-exports these names from `robotenv.robot` and `robotenv.config`.
