"""Where the GRx models are, and a content id of one. No engine import.

The models are the files of the repository ``FFTAI/Wiki-GRx-Models``. None is copied into this repository. :func:`root`
returns a checkout: the directory ``GRX_ROOT`` names, else a cache directory (``GRX_CACHE_DIR``, default
``~/.cache/robotenv/grx``) that is filled on first use by fetching the pinned commit :data:`PIN` with ``git``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from ....errors import ConfigError

URL = "https://github.com/FFTAI/Wiki-GRx-Models.git"
PIN = "7d96c758f048fe1bf92b3258864d94771ae0c093"  # master when this provider was written
HINT = "set GRX_ROOT to a checkout of https://github.com/FFTAI/Wiki-GRx-Models, or make git and the network available"
_REFERENCE = re.compile(r"""(?:filename|file)\s*=\s*["']([^"']+)["']""")
_MESHDIR = re.compile(r"""meshdir\s*=\s*["']([^"']*)["']""")


def cache_dir() -> Path:
    base = os.environ.get("GRX_CACHE_DIR") or os.path.join(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache", "robotenv", "grx")
    return Path(base)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=1800)


def _fetch(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=target.parent))
    try:
        _git("init", "-q", cwd=tmp)
        _git("remote", "add", "origin", URL, cwd=tmp)
        _git("config", "core.sparseCheckout", "true", cwd=tmp)
        (tmp / ".git" / "info" / "sparse-checkout").write_text("/GRX/\n/LICENSE\n")
        _git("fetch", "-q", "--depth", "1", "origin", PIN, cwd=tmp)
        _git("checkout", "-q", "FETCH_HEAD", cwd=tmp)
        try:
            tmp.rename(target)
        except OSError:  # another process filled it first
            if not target.is_dir():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@lru_cache(maxsize=None)
def root() -> Path:
    """The checkout that holds ``GRX/``; ``ConfigError`` at ``provider`` when there is none and none can be fetched."""
    given = os.environ.get("GRX_ROOT")
    if given:
        path = Path(given).expanduser()
        if not (path / "GRX").is_dir():
            raise ConfigError("provider", f"GRX_ROOT={given!r} has no GRX directory")
        return path
    target = cache_dir() / PIN[:12]
    if not (target / "GRX").is_dir():
        try:
            _fetch(target)
        except (OSError, subprocess.SubprocessError) as e:
            detail = getattr(e, "stderr", None) or str(e)
            raise ConfigError("provider", f"the GRx models could not be fetched ({str(detail).strip()[:200]}); {HINT}") from e
    return target


def license_text() -> str:
    return "GPL-3.0 (FFTAI/Wiki-GRx-Models, file LICENSE of the repository)"


def _referenced(entry: Path) -> list[Path]:
    text = entry.read_text()
    meshdir = _MESHDIR.search(text)
    base = entry.parent / meshdir.group(1) if meshdir and entry.suffix == ".xml" else entry.parent
    return sorted({(base / ref).resolve() for ref in _REFERENCE.findall(text) if not ref.startswith(("http", "package:"))})


def content_id(entry: Path, salt: str) -> str:
    """Hash of the model file, the mesh files it names and ``salt`` (what the provider adds to the model)."""
    h = hashlib.sha256(salt.encode())
    h.update(entry.read_bytes())
    for path in _referenced(entry):
        h.update(path.name.encode())
        h.update(path.read_bytes() if path.is_file() else b"missing")
    return h.hexdigest()[:16]
