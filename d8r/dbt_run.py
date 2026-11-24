"""Plan and run explicit dbt model commands against a workspace target."""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from d8r.dbt import DbtError, _artifact_directory
from d8r.dbt_files import DbtFiles
from d8r.dbt_profiles import DbtTarget, DbtTargetSettings

DbtModelCommand = Literal["run", "run-with-deps", "build", "test"]
_CLI = (sys.executable, "-I", "-c", "from dbt.cli.main import cli; cli()")
_SELECTOR_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class DbtRun:
    root: Path
    verb: str
    selector: str
    target: DbtTarget
    artifact_dir: Path
    log_dir: Path

    @property
    def args(self) -> tuple[str, ...]:
        return (self.verb, "--select", self.selector, "--project-dir", str(self.root),
                "--profiles-dir", str(self.target.profiles_dir), "--profile", self.target.profile,
                "--target", self.target.name, "--target-path", str(self.artifact_dir), "--write-json",
                "--log-path", str(self.log_dir), "--log-level-file", "info", "--no-use-colors",
                "--no-send-anonymous-usage-stats", "--no-upload-to-artifacts-ingest-api")

    async def execute(self) -> int:
        # dbt owns its local log. Never echo raw macro output or database errors into the TUI.
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            *_CLI, *self.args, cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ))
        try:
            process = await asyncio.shield(spawn)
            return await process.wait()
        except asyncio.CancelledError:
            # Cancellation can race startup; reap the child in either case.
            process, = await asyncio.gather(spawn, return_exceptions=True)
            if isinstance(process, asyncio.subprocess.Process) and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            raise
        except (OSError, NotImplementedError):
            raise DbtError("Could not launch the bundled dbt Core process.") from None


def prepare_dbt_run(home: str, path: str, revision: str, settings: DbtTargetSettings,
                    command: DbtModelCommand) -> DbtRun:
    project = DbtFiles(home)
    model = project.model(path)
    if model is None:
        raise DbtError("Open a dbt model under the project's configured model-paths.")
    if project.read(path).revision != revision:
        raise DbtError(f"{path} changed on disk. Reload it before running dbt.")
    selector = f"{model.name}.v{model.version}" if model.version else model.name
    if not _SELECTOR_NAME.fullmatch(selector):
        raise DbtError(f"The name of {path} cannot be used as a literal dbt model selector.")
    target = settings.resolve(project.root, project.config)
    if target is None:
        raise DbtError("No dbt profile found. Check profile: in dbt_project.yml and Path to dbt Profiles in Workspace Settings.")
    artifact_dir = project._path(_artifact_directory(project.config, target).as_posix())
    log_dir = project._path(project.config.get("log-path", "logs"))
    if command == "run-with-deps":
        return DbtRun(project.root, "run", "+" + selector, target, artifact_dir, log_dir)
    return DbtRun(project.root, command, selector, target, artifact_dir, log_dir)
