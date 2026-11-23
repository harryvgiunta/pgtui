"""Discover dbt target names without rendering profiles or retaining credentials."""
from __future__ import annotations

import ntpath
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from d8r.dbt import DbtError, _literal, _yaml

_PROFILE_LIMIT = 1024 * 1024
_TARGET_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True)
class DbtTarget:
    profile: str
    profiles_dir: Path
    name: str
    adapter: str
    database: str

    def compile_hint(self, directory: Path) -> str:
        args = ["dbt", "compile", "--profiles-dir", str(self.profiles_dir), "--profile", self.profile,
                "--target", self.name, "--target-path", directory.as_posix(), "--write-json"]
        command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
        return f"From the project home, run {command}, then refresh dbt."


@dataclass(frozen=True)
class _Output:
    adapter: str
    host: str
    port: str
    database: str


@dataclass(frozen=True)
class DbtProfile:
    name: str
    directory: Path
    default: str
    outputs: Mapping[str, _Output]

    def select(self, override: str) -> DbtTarget:
        name = override or self.default
        if not name:
            raise DbtError("The profile default target is not a literal name. Set a Workspace Default Target Override.")
        if name not in self.outputs:
            raise DbtError(f"Target {name!r} is not a valid output of profile {self.name!r}.")
        if not _TARGET_NAME.fullmatch(name) or name.endswith(".") or ntpath.isreserved(name):
            raise DbtError(f"Target {name!r} cannot be used as a local artifact directory name.")
        output = self.outputs[name]
        return DbtTarget(self.name, self.directory, name, output.adapter, output.database)

    def suggest(self, source: Mapping) -> str:
        matches = [name for name, output in self.outputs.items()
                   if output.adapter == "postgres" and output.host and output.database and output.port
                   and output.host.casefold() == str(source.get("host", "")).casefold()
                   and output.database == source.get("database") and output.port == str(source.get("port", ""))]
        return matches[0] if len(matches) == 1 else ""


@dataclass(frozen=True)
class DbtTargetSettings:
    profiles_path: str = ""
    override: str = ""

    def profile(self, root: Path, config: dict) -> DbtProfile | None:
        name = config.get("profile")
        required = bool(self.profiles_path or self.override)
        if not _literal(name) or not name.isprintable():
            if required:
                raise DbtError("Set a literal profile: name in dbt_project.yml before configuring a dbt target.")
            return None
        try:
            configured = self.profiles_path or os.environ.get("DBT_PROFILES_DIR")
            if configured:
                path = Path(configured).expanduser()
                if not path.is_absolute():
                    path = root / path
                if os.path.normcase(path.name) != "profiles.yml":
                    path /= "profiles.yml"
            else:
                path = root / "profiles.yml"
                if not path.exists():
                    path = Path.home() / ".dbt" / "profiles.yml"
            if not path.exists() and not (required or configured):
                return None
            if not path.is_file():
                raise DbtError("Choose a directory containing profiles.yml, or the profiles.yml file itself.")
            directory = path.parent.resolve()
            with path.open("rb") as handle:
                content = handle.read(_PROFILE_LIMIT + 1)
            if len(content) > _PROFILE_LIMIT:
                raise DbtError("profiles.yml exceeds the profile size limit.")
        except (OSError, RuntimeError):
            raise DbtError("Cannot read profiles.yml. Check the Path to dbt Profiles setting.") from None
        try:
            profiles = _yaml(content.decode("utf-8-sig"), path)
        except ValueError:
            # YAML errors can include the full source line, including passwords.
            raise DbtError("Invalid profiles.yml. Check its YAML syntax and duplicate keys.") from None
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            raise DbtError(f"Profile {name!r} from dbt_project.yml was not found in profiles.yml.")
        entries = profile.get("outputs")
        if not isinstance(entries, dict):
            raise DbtError(f"Profile {name!r} must have an outputs mapping in profiles.yml.")
        outputs = {}
        for key, value in entries.items():
            if not _literal(key) or not key.isprintable() or not isinstance(value, dict):
                continue
            adapter, host = value.get("type"), value.get("host")
            if not isinstance(adapter, str) or not adapter.strip():
                continue
            database = value.get("dbname", value.get("database"))
            port = value.get("port", 5432)
            outputs[key] = _Output(adapter if _literal(adapter) else "", host if _literal(host) else "",
                                   str(port) if type(port) is int or isinstance(port, str) and port.isdecimal() else "",
                                   database if _literal(database) else "")
        if not outputs:
            raise DbtError(f"Profile {name!r} has no valid outputs in profiles.yml.")
        default = profile.get("target", "default")
        return DbtProfile(name, directory, default if _literal(default) and default.isprintable() else "", outputs)

    def resolve(self, root: Path, config: dict, *, for_artifacts: bool = False) -> DbtTarget | None:
        profile = self.profile(root, config)
        if profile is None and for_artifacts and "profile" in config:
            raise DbtError("Cannot verify a dbt target. Check dbt_project.yml and Path to dbt Profiles in Workspace Settings.")
        return profile.select(self.override) if profile else None
