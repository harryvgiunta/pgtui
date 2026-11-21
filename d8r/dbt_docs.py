"""Run the bundled dbt Core documentation commands on explicit user request only."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from d8r.dbt import DbtError, _relative
from d8r.dbt_files import DbtFiles

_CLI = (sys.executable, "-I", "-c", "from dbt.cli.main import cli; cli()")
_DOCS_FILES = ("index.html", "manifest.json", "catalog.json")


def _assets(project: DbtFiles) -> tuple[Path, ...]:
    assets = project.config.get("asset-paths", [])
    if not isinstance(assets, list):
        raise DbtError("asset-paths must be a list of relative directories.")
    checked = []
    for value in assets:
        path = project._path(value)
        relative = _relative(value)
        if (not relative.parts or any(marker in value for marker in ("{{", "{%", "{#"))
                or not project._visible(path / "_") or relative.parts[0] in _DOCS_FILES):
            raise DbtError("Documentation asset paths must be literal, project-local directories.")
        if path.exists():
            if not path.is_dir():
                raise DbtError(f"Not a docs asset directory: {value}")
            for directory, folders, files in path.walk(follow_symlinks=False):
                for name in folders + files:
                    member = (directory / name).relative_to(project.root).as_posix()
                    project._path(member)  # dbt's copytree follows links; refuse any in assets.
                    if name.startswith("."):
                        raise DbtError(f"Hidden files cannot be served as docs assets: {member}")
            checked.append(relative)
    return tuple(checked)


class DbtDocs:
    """Own one loopback-only dbt docs server and its disposable artifact directory."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._temp: tempfile.TemporaryDirectory | None = None
        self.port: int | None = None

    async def _spawn(self, args: tuple[str, ...], cwd: Path):
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("DBT_")
               or key.upper().startswith(("DBT_ENV_SECRET_", "DBT_ENV_CUSTOM_ENV_"))}
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            *_CLI, *args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ))
        try:
            return await asyncio.shield(spawn)
        except asyncio.CancelledError:
            self._process = await spawn  # Reap even a child created during cancellation.
            raise

    async def start(self, project: DbtFiles, port: int) -> str:
        if type(port) is not int or not 1 <= port <= 65535:
            raise DbtError("Choose a port between 1 and 65535.")
        if self._task is not None or self._process is not None:
            raise DbtError("Stop the current documentation server before starting another.")
        if project.config_error:
            raise DbtError(project.config_error)
        assets = _assets(project)
        self._task = asyncio.current_task()
        self._temp = tempfile.TemporaryDirectory(prefix="pgtui-dbt-docs-")
        workspace = Path(self._temp.name)
        generated, site = workspace / "generated", workspace / "site"
        try:
            site.mkdir()
            # An offline profile satisfies dbt's parse requirement. --no-compile and
            # --empty-catalog need neither a running warehouse nor user credentials.
            profile = {"pgtui_docs": {"target": "docs", "outputs": {"docs": {
                "type": "postgres", "host": "127.0.0.1", "port": 1, "user": "offline",
                "password": "", "dbname": "offline", "schema": "public", "threads": 1,
            }}}}
            (workspace / "profiles.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")
            # Serve only the generated site, not dbt's logs, parse caches, or profiles.
            (workspace / "dbt_project.yml").write_text(
                "name: pgtui_docs\nprofile: pgtui_docs\nconfig-version: 2\n", encoding="utf-8")
            flags = ("--profiles-dir", str(workspace), "--profile", "pgtui_docs",
                     "--log-path", str(workspace / "logs"), "--no-send-anonymous-usage-stats",
                     "--no-use-colors", "--write-json")
            project_args = (*flags, "--project-dir", str(project.root), "--target-path", str(generated))
            async with asyncio.timeout(120):
                await self._generate(("parse", *project_args, "--no-partial-parse"), project.root)
                await self._generate(("docs", "generate", *project_args, "--no-compile", "--empty-catalog"), project.root)
            if not all((generated / name).is_file() for name in _DOCS_FILES):
                raise DbtError("dbt did not generate the required documentation artifacts.")
            for name in _DOCS_FILES:
                shutil.copyfile(generated / name, site / name)
            for relative in assets:
                if (generated / relative).is_dir():
                    shutil.copytree(generated / relative, site / relative, dirs_exist_ok=True)
            self._process = await self._spawn(("docs", "serve", *flags, "--project-dir", str(workspace),
                                               "--target-path", str(site), "--host", "127.0.0.1",
                                               "--port", str(port), "--no-browser"), workspace)
            async with asyncio.timeout(20):
                while line := await self._process.stdout.readline():
                    if line.strip() == f"Serving docs at {port}".encode():
                        self.port = port
                        # Keep draining request logs while this subprocess serves.
                        self._task = asyncio.create_task(self._drain(self._process))
                        return f"http://127.0.0.1:{port}/"
            raise DbtError(f"dbt docs could not listen on port {port}. Check that the port is free.")
        except (OSError, TimeoutError) as exc:
            raise DbtError("dbt docs did not start. Check project files and the requested port.") from exc
        finally:
            if self.port is None:
                await self._cleanup()
                self._task = None

    async def _generate(self, args: tuple[str, ...], root: Path) -> None:
        process = self._process = await self._spawn(args, root)
        # Drain without retaining arbitrary project or macro output.
        while await process.stdout.read(8192):
            pass
        result = await process.wait()
        self._process = None
        if result:
            raise DbtError("dbt could not generate docs. Check the project's YAML, model references and installed packages.")


    async def _drain(self, process: asyncio.subprocess.Process) -> None:
        while await process.stdout.read(8192):
            pass
        await process.wait()
        if self._process is process:
            await self._cleanup()
            self._task = None

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        if self.port is None:
            task.cancel()  # Startup owns its cleanup.
        else:
            self._kill(self._process)  # Let the drain task reap the server and close its pipes.
        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _kill(process: asyncio.subprocess.Process | None) -> None:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    async def _cleanup(self) -> None:
        if self._process is not None:
            self._kill(self._process)
            await self._process.communicate()
            self._process = None
        self.port = None
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
