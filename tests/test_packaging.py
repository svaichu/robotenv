import subprocess
import sys

BLOCKED = ["torch", "torchrl", "tensordict", "mujoco", "mujoco_warp", "warp", "robosuite"]


def test_config_import_has_no_engine():
    code = (
        "import sys\n"
        f"for m in {BLOCKED!r}: sys.modules[m] = None\n"
        "import robotenv.config\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_viz_import_has_no_engine():
    code = (
        "import sys\n"
        f"for m in {BLOCKED!r}: sys.modules[m] = None\n"
        "import robotenv.viz\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_import_is_lazy_and_exposes_make_env():
    code = (
        "import sys\n"
        f"for m in {BLOCKED!r}: sys.modules[m] = None\n"
        "import robotenv\n"
        "assert 'make_env' in dir(robotenv) and 'make_env' in robotenv.__all__\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    code = "import robotenv, robotenv.envs as e; assert robotenv.make_env is e.make_env"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_provider_package_imports_without_an_engine():
    code = (
        "import sys, pkgutil\n"
        f"for m in {BLOCKED!r}: sys.modules[m] = None\n"
        "import importlib, robotenv.tasks as t\n"
        "names = [m.name for m in pkgutil.iter_modules(t.__path__) if m.ispkg]\n"
        "assert names\n"
        "for n in names: importlib.import_module('robotenv.tasks.' + n)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_menagerie_provider_imports_without_an_engine_or_the_library():
    code = (
        "import sys\n"
        f"for m in {BLOCKED + ['mujoco_menagerie']!r}: sys.modules[m] = None\n"
        "import robotenv.config as c\n"
        "from robotenv.robot.providers.menagerie import PROVIDER, cards\n"
        "assert 'menagerie' in c.list_robot_providers() and PROVIDER.names() == () and len(cards()) > 20\n"
        "try:\n"
        "    PROVIDER.describe('franka_emika_panda')\n"
        "except c.ConfigError as e:\n"
        "    assert e.path == 'provider' and 'robotenv[menagerie]' in e.reason, str(e)\n"
        "else:\n"
        "    raise SystemExit('no error')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
