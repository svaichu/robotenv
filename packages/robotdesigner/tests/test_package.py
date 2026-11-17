import subprocess
import sys

import robotdesigner as pa

COMMON = dict(radius=0.02, damping=0.1, armature=0.01, frictionloss=0.0, effort_limit=50.0)


def test_chain_roundtrip_and_hash():
    arm = pa.chain(3, "zyy", total_length=0.9, total_mass=3.0, **COMMON)
    again = pa.ParametricArm.from_dict(arm.to_dict())
    assert again == arm and again.config_hash == arm.config_hash


def test_errors_carry_the_path():
    try:
        pa.chain(0, "z", link_length=0.5, mass=1.0, **COMMON)
    except pa.ConfigError as e:
        assert e.path
    else:
        raise AssertionError("n=0 must be rejected")


def test_imports_without_robotenv_or_an_engine():
    code = (
        "import sys\n"
        "for m in ['robotenv', 'torch', 'mujoco', 'warp']: sys.modules[m] = None\n"
        "import robotdesigner, robotdesigner.reduce, robotdesigner.provider\n"
        "assert 'eliminate_joints' in robotdesigner.__all__\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
