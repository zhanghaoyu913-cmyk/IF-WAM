from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf


def sha256_file(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception as exc:
        return f"unknown: {exc}"


def repo_snapshot(repo: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(repo)
    return {
        "repo": str(p),
        "branch": _git(p, ["branch", "--show-current"]),
        "commit": _git(p, ["rev-parse", "HEAD"]),
        "dirty_files": _git(p, ["status", "--short"]).splitlines(),
    }


def environment_snapshot() -> dict[str, Any]:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    except Exception as exc:
        freeze = [f"pip freeze failed: {exc}"]
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": freeze,
    }


def write_run_artifacts(
    output_dir: str | os.PathLike[str],
    *,
    config: Any | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    repos: list[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_paths = repos or [Path(__file__).resolve().parents[3]]
    snapshots = [repo_snapshot(repo) for repo in repo_paths]
    primary_repo = Path(repo_paths[0])

    if config is not None:
        with (out / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            OmegaConf.save(config=config, f=f)
    (out / "git_commit.txt").write_text(
        "\n".join(f"{snap['repo']}: {snap['commit']}" for snap in snapshots) + "\n",
        encoding="utf-8",
    )
    (out / "git_diff.patch").write_text(_git(primary_repo, ["diff", "--binary"]) + "\n", encoding="utf-8")
    (out / "environment_snapshot.json").write_text(
        json.dumps(environment_snapshot(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "command_line": sys.argv,
        "repositories": snapshots,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    if extra:
        manifest.update(dict(extra))
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
