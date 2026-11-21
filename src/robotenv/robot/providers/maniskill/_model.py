"""Engine side of the ``maniskill`` provider: read an agent class, open its model as a ``mujoco.MjSpec``.

A ManiSkill agent is a Python class: a model file (a URDF, for a few an MJCF), keyframes, and controller configs for
SAPIEN. :func:`facts_of` reads what is needed from the class without building a ManiSkill environment, and
:func:`open_spec` turns the model into what the compile of an external robot expects (``simulators/mujoco/external.py``):

- fixed links are kept as bodies (MuJoCo's URDF import would fuse them, and an agent's end-effector link is one);
- one position actuator per joint of the agent's joint-position controller, with ManiSkill's stiffness as ``kp``, its
  damping as ``kd`` and its force limit as the joint's actuator-force range (a URDF has no actuators);
- the agent's keyframes, mapped onto MuJoCo's joints **by name**: a ManiSkill keyframe is in SAPIEN's joint order
  (breadth first), MuJoCo's is depth first;
- contact excludes for every pair of jointed links (SAPIEN never collides those; MuJoCo does where the parent is
  welded to the world) and for the pairs the model's SRDF disables;
- gravity compensation on every body where ManiSkill "balances the passive force" (its default; for PhysX that means
  gravity is switched off on the robot's links), as MuJoCo's ``gravcomp``: the stiffness and force limit of the
  controller are tuned for a robot that does not carry its own weight;
- a free joint on the root body of an agent whose root link is not fixed, with the root pose of each keyframe;
- the friction ManiSkill gives single links. SAPIEN's contact patch radius has no MuJoCo counterpart and is dropped.

Repairs, applied to every model that needs them: ``package://`` mesh paths are resolved, collision meshes in a
format MuJoCo does not decode are converted once into a cache directory outside the repository (:func:`cache_dir`),
inertia is balanced, and a moving link without mass gets a minimal one.

Visual geometry is discarded: the model that is simulated, and rendered, is the collision geometry. The contact and
solver options of the model are not carried over (the scene's win, as for every external robot).
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np

from ...._canon import canonical_json
from ....errors import ConfigError
from . import _agent_class

_HINGE, _SLIDE, _FREE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_FREE))
_JOINT = mujoco.mjtObj.mjOBJ_JOINT
_ON = mujoco.mjtLimited.mjLIMITED_TRUE
NO_LIMIT = 1e9  # ManiSkill's "no force limit" is 1e10; anything this large is not a limit
POSITION_MODES = ("pd_joint_pos",)  # the controller whose gains become the native actuators
ROOT_JOINT = "floating_base"  # the free joint added to the root body of a floating-base agent
MUJOCO_MESHES = (".stl", ".obj", ".msh")  # the mesh formats MuJoCo decodes
EMPTY_LINK_MASS, EMPTY_LINK_INERTIA = 1e-4, 1e-7  # kg, kg m^2: see _give_mass_to_empty_moving_links


@dataclass(frozen=True)
class Servo:
    """One joint of the agent's joint-position controller: ``kp`` = stiffness, ``kd`` = damping, ``force_limit`` or ``None``."""

    joint: str
    kp: float
    kd: float
    force_limit: float | None


@dataclass(frozen=True)
class Keyframe:
    name: str
    qpos: tuple[float, ...]  # in ``AgentFacts.joint_order``
    pos: tuple[float, float, float]  # the root pose of the keyframe, in the model's world frame
    quat: tuple[float, float, float, float]


@dataclass(frozen=True)
class AgentFacts:
    """What the provider reads from a ManiSkill agent class. Hashable; everything but ``path`` and ``srdf`` (absolute
    paths) enters the robot's ``source``."""

    uid: str
    path: str  # the model file, absolute
    entry: str  # its file name
    kind: str  # "urdf" or "mjcf"
    fix_root_link: bool
    disable_self_collisions: bool
    gravity_compensated: bool  # ManiSkill's ``balance_passive_force`` of the joint-position controller
    keyframes: tuple[Keyframe, ...]
    ee_link: str | None
    arm_joints: tuple[str, ...]
    servos: tuple[Servo, ...]
    joint_order: tuple[str, ...]  # SAPIEN's active joints, in its order: the order of a keyframe's qpos
    friction: tuple[tuple[str, float], ...]  # (link, friction coefficient)
    srdf: str | None  # the SRDF beside the model, absolute, or None

    def hashed(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("path", "srdf")}
        d["keyframes"] = [[k.name, k.qpos, k.pos, k.quat] for k in self.keyframes]
        d["servos"] = [[s.joint, s.kp, s.kd, s.force_limit] for s in self.servos]
        return d


# --- reading the agent class ----------------------------------------------------------------------------------------


def _download_hint(path: str) -> str:
    """The ManiSkill command that fetches the files of ``path``, as far as its asset table tells."""
    generic = "python -m mani_skill.utils.download_asset <asset id> (see ManiSkill's documentation for the id)"
    try:
        from mani_skill import ASSET_DIR
        from mani_skill.utils.assets.data import DATA_SOURCES

        rel = os.path.relpath(path, str(ASSET_DIR)).replace(os.sep, "/")
        for uid, source in DATA_SOURCES.items():
            target = str(getattr(source, "target_path", "") or "").replace(os.sep, "/")
            if target and (rel == target or rel.startswith(target.rstrip("/") + "/")):
                return f"python -m mani_skill.utils.download_asset {uid}"
    except (ImportError, AttributeError, TypeError, ValueError):
        return generic  # an asset table of another shape: the generic hint is still true
    return generic


class _Articulation:
    """What an agent's controller configs read of its SAPIEN articulation: the active joints, in SAPIEN's order."""

    def __init__(self, joint_order: tuple[str, ...]):
        self.active_joints = [SimpleNamespace(name=n) for n in joint_order]
        self.active_joints_map = {j.name: j for j in self.active_joints}

    def get_active_joints(self):
        return self.active_joints


def _bare_agent(cls, joint_order: tuple[str, ...]):
    """An instance of the agent class that was never built: no scene, no articulation.

    Some agents set the joint names their controllers use in ``__init__`` before they hand over to the base class, and
    some read the articulation's active joints. The constructor is run with the base class's ``__init__`` (which builds
    the articulation in a scene) replaced by a no-op, and ``robot`` is a stand-in that knows the joint order.
    """
    from mani_skill.agents.base_agent import BaseAgent

    agent = object.__new__(cls)
    original = BaseAgent.__init__
    BaseAgent.__init__ = lambda self, *args, **kwargs: None
    try:
        cls.__init__(agent, None, 20)
    except Exception:  # noqa: BLE001, S110  (a constructor that needs the scene: the class attributes may still do)
        pass
    finally:
        BaseAgent.__init__ = original
    agent.robot = _Articulation(joint_order)
    return agent


def _controller_leaves(cls, uid: str, joint_order: tuple[str, ...]) -> tuple[list, bool]:
    """``(parts, balance_passive_force)`` of the agent's joint-position controller, without building the agent."""
    try:
        configs = cls._controller_configs.fget(_bare_agent(cls, joint_order))
    except Exception as e:
        raise ConfigError("name", f"the controllers of maniskill/{uid} cannot be read without building it: {type(e).__name__}: {e}") from e
    for mode in POSITION_MODES:
        if mode in configs:
            chosen = configs[mode]
            break
    else:
        raise ConfigError("name", f"maniskill/{uid} has no joint-position controller ({list(POSITION_MODES)}); it has {sorted(configs)}")
    balance = bool(chosen.get("balance_passive_force", True)) if isinstance(chosen, dict) else True  # ManiSkill's default
    parts = list(chosen.values()) if isinstance(chosen, dict) else [chosen]
    return [p for p in parts if hasattr(p, "joint_names")], balance


def _servos(cls, uid: str, joint_order: tuple[str, ...]) -> tuple[tuple[Servo, ...], bool]:
    out: dict[str, Servo] = {}
    parts, balance = _controller_leaves(cls, uid, joint_order)
    for part in parts:
        names = list(part.joint_names)
        if not hasattr(part, "stiffness"):  # a passive part: its joints have no drive
            continue
        kp = np.broadcast_to(np.asarray(part.stiffness, dtype=float), (len(names),))
        kd = np.broadcast_to(np.asarray(part.damping, dtype=float), (len(names),))
        limit = np.broadcast_to(np.asarray(getattr(part, "force_limit", NO_LIMIT), dtype=float), (len(names),))
        for n, p, d, f in zip(names, kp, kd, limit):
            out.setdefault(n, Servo(n, float(p), float(d), None if not np.isfinite(f) or f >= NO_LIMIT else float(f)))
    return tuple(out.values()), balance


def _strip_geometry(root: ET.Element) -> None:
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)


def sapien_joint_order(path: str, fix_root_link: bool) -> tuple[str, ...]:
    """The active joints of a URDF in SAPIEN's order, read from SAPIEN itself in a bare physics scene.

    The URDF is loaded without its geometry: the order does not depend on it, and SAPIEN creates render materials
    while it parses visuals, which needs a Vulkan device.
    """
    import sapien
    from sapien import physx

    root = ET.parse(path).getroot()
    _strip_geometry(root)
    scene = sapien.Scene([physx.PhysxCpuSystem()])
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    with tempfile.TemporaryDirectory() as tmp:
        bare = os.path.join(tmp, "kinematics.urdf")
        with open(bare, "w") as f:
            f.write(ET.tostring(root, encoding="unicode"))
        builders = loader.parse(bare)[0]
        if len(builders) != 1:
            raise ConfigError("name", f"{os.path.basename(path)} holds {len(builders)} articulations, expected 1")
        articulation = builders[0].build()
    return tuple(j.name for j in articulation.get_active_joints())


def _friction(cls) -> tuple[tuple[str, float], ...]:
    config = getattr(cls, "urdf_config", None)
    if not isinstance(config, dict):
        return ()
    materials = config.get("_materials", {})
    out = []
    for link, props in sorted(config.get("link", {}).items()):
        material = materials.get(props.get("material"), {}) if isinstance(props, dict) else {}
        if "static_friction" in material:
            out.append((link, float(material["static_friction"])))
    return tuple(out)


@functools.cache
def facts_of(uid: str) -> AgentFacts:
    """Read agent ``uid``. ``ConfigError`` at ``name``: unknown robot, files not on disk, a class that cannot be read."""
    cls = _agent_class(uid)
    urdf, mjcf = getattr(cls, "urdf_path", None), getattr(cls, "mjcf_path", None)
    path = urdf or mjcf
    if path is None:
        raise ConfigError("name", f"maniskill/{uid} names no model file")
    path = os.path.abspath(str(path))
    if not os.path.exists(path):
        raise ConfigError("name", f"the files of maniskill/{uid} are not on disk; ManiSkill downloads them with: {_download_hint(path)}")
    kind = "urdf" if urdf else "mjcf"
    fixed = bool(getattr(cls, "fix_root_link", True))
    order = sapien_joint_order(path, fixed) if kind == "urdf" else _mjcf_joint_order(path)
    keyframes = []
    for name, key in (getattr(cls, "keyframes", None) or {}).items():
        if getattr(key, "qpos", None) is None:
            continue
        qpos = tuple(float(x) for x in np.asarray(key.qpos, dtype=float).reshape(-1))
        if len(qpos) != len(order):
            continue  # a keyframe of another variant of the model: it cannot be mapped
        pose = key.pose
        keyframes.append(Keyframe(str(name), qpos, tuple(float(x) for x in pose.p), tuple(float(x) for x in pose.q)))
    srdf = os.path.splitext(path)[0] + ".srdf"
    servos, balance = _servos(cls, uid, order)
    agent = _bare_agent(cls, order)  # attributes an agent sets in its constructor count too
    return AgentFacts(
        uid=uid,
        path=path,
        entry=os.path.basename(path),
        kind=kind,
        fix_root_link=fixed,
        disable_self_collisions=bool(getattr(cls, "disable_self_collisions", False)),
        gravity_compensated=balance,
        keyframes=tuple(keyframes),
        ee_link=getattr(agent, "ee_link_name", None) or None,
        arm_joints=tuple(getattr(agent, "arm_joint_names", None) or ()),
        servos=servos,
        joint_order=order,
        friction=_friction(cls),
        srdf=srdf if os.path.exists(srdf) else None,
    )


# --- the model file -------------------------------------------------------------------------------------------------


def _mjcf_joint_order(path: str) -> tuple[str, ...]:
    """The hinge and slide joints of an MJCF in MuJoCo's order (ManiSkill loads an MJCF in the file's order)."""
    model = mujoco.MjSpec.from_file(path).compile()
    return tuple(model.joint(j).name for j in range(model.njnt) if int(model.jnt_type[j]) in (_HINGE, _SLIDE))


def _resolve_mesh(filename: str, urdf_dir: str) -> str:
    """The absolute path of a mesh a URDF names: relative to the URDF's directory, or ``package://<pkg>/<path>``,
    which is looked up as ``<dir>/<path>`` and ``<dir>/<pkg>/<path>`` in the URDF's directory and its ancestors."""
    match = re.match(r"package:/+([^/]+)/(.*)", filename)
    if match is None:
        return os.path.normpath(os.path.join(urdf_dir, filename))
    package, rest = match.groups()
    folder = urdf_dir
    for _ in range(4):
        for candidate in (os.path.join(folder, rest), os.path.join(folder, package, rest)):
            if os.path.exists(candidate):
                return os.path.normpath(candidate)
        folder = os.path.dirname(folder)
    return os.path.normpath(os.path.join(urdf_dir, rest))


def cache_dir() -> str:
    """Where converted meshes are kept: ``$ROBOTENV_CACHE_DIR``, else the user's cache directory. Never the repository."""
    root = os.environ.get("ROBOTENV_CACHE_DIR") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "robotenv")
    return os.path.join(root, "maniskill", "meshes")


CONVERSIONS = [0]  # how many meshes this process converted (a cached one is not counted)


def _readable_mesh(path: str) -> str:
    """``path`` if MuJoCo decodes its format, else the path of an STL converted once with ``trimesh`` into the cache.
    The cached file is named after the content of the original, so a changed mesh is converted again."""
    if os.path.splitext(path)[1].lower() in MUJOCO_MESHES:
        return path
    h = hashlib.sha256()
    _digest_file(h, "", path)
    target = os.path.join(cache_dir(), h.hexdigest()[:32] + ".stl")
    if not os.path.exists(target):
        import trimesh

        try:
            mesh = trimesh.load(path, force="mesh")
            if len(mesh.faces) == 0:
                raise ValueError("no faces")
        except Exception as e:
            raise ConfigError("name", f"the mesh {os.path.basename(path)} cannot be converted for MuJoCo: {type(e).__name__}: {e}") from e
        os.makedirs(os.path.dirname(target), exist_ok=True)
        partial = f"{target}.{os.getpid()}.tmp"
        mesh.export(partial, file_type="stl")
        os.replace(partial, target)
        CONVERSIONS[0] += 1
    return target


def _give_mass_to_empty_moving_links(root: ET.Element) -> None:
    """A link that a joint moves must have mass in MuJoCo; SAPIEN accepts one without. URDFs use such links as the
    intermediate frames of a chain of joints (a floating gripper's ``root_x`` / ``root_y`` / ``root_z``). A moving link
    whose mass is missing or zero gets ``EMPTY_LINK_MASS`` and ``EMPTY_LINK_INERTIA``, small against any real link."""
    moved = {j.find("child").get("link") for j in root.findall("joint") if j.get("type") != "fixed" and j.find("child") is not None}
    for link in root.findall("link"):
        for extra in link.findall("inertial")[1:]:  # a repeated element: the first counts, as for the URDF parsers
            link.remove(extra)
        if link.get("name") not in moved:
            continue
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        if mass is not None and float(mass.get("value", 0.0)) > 0.0:
            continue
        if inertial is not None:
            link.remove(inertial)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value=repr(EMPTY_LINK_MASS))
        i = repr(EMPTY_LINK_INERTIA)
        ET.SubElement(inertial, "inertia", ixx=i, iyy=i, izz=i, ixy="0", ixz="0", iyz="0")


@functools.cache
def _urdf_document(path: str) -> tuple[str, tuple[str, ...]]:
    """``(xml, mesh files)``: the URDF as MuJoCo should read it, and the collision meshes it references (absolute)."""
    root = ET.parse(path).getroot()
    urdf_dir = os.path.dirname(path)
    meshes = []
    for link in root.iter("link"):
        for el in link.findall("visual"):
            link.remove(el)
        for collision in link.findall("collision"):
            for mesh in collision.iter("mesh"):
                resolved = _resolve_mesh(mesh.get("filename"), urdf_dir)
                if not os.path.exists(resolved):
                    raise ConfigError("name", f"{os.path.basename(path)} names a mesh that is not on disk: {mesh.get('filename')}")
                mesh.set("filename", _readable_mesh(resolved))
                meshes.append(resolved)  # the original: the converted bytes do not enter the source id
    _give_mass_to_empty_moving_links(root)
    for el in root.findall("mujoco"):
        root.remove(el)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="true", balanceinertia="true", strippath="false")
    return ET.tostring(root, encoding="unicode"), tuple(dict.fromkeys(meshes))


def _base_spec(facts: AgentFacts) -> mujoco.MjSpec:
    if facts.kind == "mjcf":
        return mujoco.MjSpec.from_file(facts.path)
    return mujoco.MjSpec.from_string(_urdf_document(facts.path)[0])


def _digest_file(h, label: str, path: str) -> None:
    h.update(label.encode() + b"\0")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(b"\0")


@functools.cache
def source_id(facts: AgentFacts) -> str:
    """The content id of the robot: the model file, the meshes and SRDF it uses, and the agent facts. Hex, 64 chars."""
    h = hashlib.sha256()
    root = os.path.dirname(facts.path)
    _digest_file(h, facts.entry, facts.path)
    files = _urdf_document(facts.path)[1] if facts.kind == "urdf" else _mjcf_files(facts.path)
    for f in sorted(files):
        _digest_file(h, os.path.relpath(f, root).replace(os.sep, "/"), f)
    if facts.srdf is not None:
        _digest_file(h, "srdf", facts.srdf)
    h.update(canonical_json(facts.hashed()).encode())
    return h.hexdigest()


def _mjcf_files(path: str) -> tuple[str, ...]:
    """The files an MJCF pulls in: includes (followed) and the ``file`` attributes of its assets, where they exist."""
    seen: dict[str, None] = {}

    def walk(xml: str) -> None:
        root = ET.parse(xml).getroot()
        base = os.path.dirname(xml)
        compiler = root.find("compiler")
        dirs = [base] + [
            os.path.join(base, compiler.get(k)) for k in ("meshdir", "texturedir", "assetdir") if compiler is not None and compiler.get(k)
        ]
        for el in root.iter():
            name = el.get("file")
            if not name:
                continue
            for d in dirs:
                candidate = os.path.normpath(os.path.join(d, name))
                if os.path.isfile(candidate) and candidate not in seen:
                    seen[candidate] = None
                    if el.tag == "include":
                        walk(candidate)
                    break

    walk(path)
    return tuple(seen)


# --- the spec the compile attaches ----------------------------------------------------------------------------------


def _disabled_pairs(srdf: str) -> list[tuple[str, str]]:
    pairs = []
    for el in ET.parse(srdf).getroot().iter("disable_collisions"):
        a, b = el.get("link1"), el.get("link2")
        if a and b and a != b:
            pairs.append((a, b))
    return pairs


def open_spec(facts: AgentFacts) -> mujoco.MjSpec:
    """A fresh ``MjSpec`` of the agent's model, with actuators, keyframes, contact excludes and link friction added."""
    spec = _base_spec(facts)
    if facts.kind == "urdf":
        # PhysX integrates a joint drive implicitly, and ManiSkill's damping is tuned for that: an explicit integrator
        # makes these servos chatter. A model states the integrator it needs in its own option (the scene's still wins).
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if not facts.fix_root_link and not any(int(j.type) == _FREE for j in spec.joints):
        root = spec.worldbody.first_body()
        if root is None:
            raise ConfigError("name", f"the model of maniskill/{facts.uid} has no root body to float")
        root.add_freejoint(name=ROOT_JOINT)
    try:
        model = spec.compile()
    except Exception as e:
        raise ConfigError("name", f"the model of maniskill/{facts.uid} does not compile in MuJoCo: {e}") from e
    joint = lambda name: mujoco.mj_name2id(model, _JOINT, name)
    bodies = {model.body(b).name for b in range(model.nbody)}
    free = [j for j in range(model.njnt) if int(model.jnt_type[j]) == _FREE]

    # ManiSkill drives an MJCF's joints with its controller's PD drives and ignores the model's own motors: a torque
    # motor on a joint of the controller is replaced by a PD servo here, keeping the motor's largest torque (gear x
    # control range) as the joint's force limit when the controller states none.
    servo_joints = {joint(s.joint) for s in facts.servos}
    motors: dict[int, float | None] = {}  # actuator id -> its largest torque
    for a in range(model.nu):
        if (
            facts.kind == "mjcf" and int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[a, 0]) in servo_joints
            and int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            and int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE)
        ):
            motors[a] = float(abs(model.actuator_gear[a, 0]) * np.abs(model.actuator_ctrlrange[a]).max()) if model.actuator_ctrllimited[a] else None
    torque = {int(model.actuator_trnid[a, 0]): t for a, t in motors.items()}
    for a in sorted(motors, reverse=True):
        spec.delete(spec.actuators[a])
    has_actuator = {
        int(model.actuator_trnid[a, 0]) for a in range(model.nu)
        if a not in motors and int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
    }
    added: list[int] = []  # joint ids of the actuators added, in actuator order
    for servo in facts.servos:
        j = joint(servo.joint)
        if j < 0:
            raise ConfigError("name", f"the controller of maniskill/{facts.uid} drives {servo.joint!r}, which the model does not have")
        if int(model.jnt_type[j]) not in (_HINGE, _SLIDE) or j in has_actuator:
            continue  # an MJCF that already drives the joint keeps its own actuator
        act = spec.add_actuator(name=f"pd_{servo.joint}", target=servo.joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gainprm[0] = servo.kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0.0, -servo.kp, -servo.kd]
        limit = servo.force_limit if servo.force_limit is not None else torque.get(j)
        if limit is not None:
            # The limit goes on the joint (its actuator-force range), not on the actuator: MuJoCo's implicit integrators
            # drop the damping term of an actuator whose own force range clamps it, and a saturated servo on a light
            # link then diverges (measured on so100, koch-v1.1 and the Allegro hand). The clamped quantity is the same.
            driven = spec.joint(servo.joint)
            driven.actfrclimited = _ON
            driven.actfrcrange = [-limit, limit]
        if model.jnt_limited[j]:
            act.ctrllimited = _ON
            act.ctrlrange = [float(x) for x in model.jnt_range[j]]
        added.append(j)

    # A link never collides with the link it is jointed to in SAPIEN. MuJoCo filters parent and child too, except where
    # the parent is welded to the world, which the base link of a fixed robot is: those pairs are excluded here.
    pairs = [(model.body(int(model.body_parentid[b])).name, model.body(b).name) for b in range(1, model.nbody) if model.body_parentid[b] > 0]
    if facts.srdf is not None:
        pairs += [(a, b) for a, b in _disabled_pairs(facts.srdf) if a in bodies and b in bodies]
    done = set()
    for a, b in pairs:
        if frozenset((a, b)) not in done:
            done.add(frozenset((a, b)))
            spec.add_exclude(bodyname1=a, bodyname2=b)
    if facts.disable_self_collisions:
        for geom in spec.geoms:  # collides with the world's default geoms (1 / 1), not with the robot's own
            geom.contype, geom.conaffinity = 2, 1
    if facts.gravity_compensated:
        for body in spec.bodies:
            if body.name != "world":
                body.gravcomp = 1.0
    for link, friction in facts.friction:
        if link in bodies:
            for geom in spec.body(link).geoms:
                geom.friction[0] = friction

    for key in facts.keyframes:
        qpos = model.qpos0.copy()
        for name, value in zip(facts.joint_order, key.qpos):
            j = joint(name)
            if j < 0:
                raise ConfigError("name", f"keyframe {key.name!r} of maniskill/{facts.uid} sets {name!r}, which the model does not have")
            qpos[model.jnt_qposadr[j]] = value
        if free and not facts.fix_root_link:  # the keyframe's root pose, in the model's world frame
            a = model.jnt_qposadr[free[0]]
            qpos[a : a + 7] = [*key.pos, *key.quat]
        kept = [a for a in range(model.nu) if a not in motors]
        ctrl = [float(model.key_ctrl[0][a]) if model.nkey else 0.0 for a in kept]  # an MJCF's own actuators keep their value
        ctrl += [float(qpos[model.jnt_qposadr[j]]) for j in added]
        spec.add_key(name=key.name, qpos=qpos.tolist(), ctrl=ctrl)
    spec.compile()  # ``MjSpec.attach`` sizes a keyframe against the last compile: the edits above must be in it
    return spec
