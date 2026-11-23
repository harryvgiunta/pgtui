"""Shared, project-confined ignore policy for the index and rename transactions."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from pathspec import GitIgnoreSpec

from d8r.dbt import DbtError, _relative
from d8r.dbt_files import DbtFiles, _FILE_LIMIT

_MAX_RULES = 8_192
_MAX_DEPTH = 128


class DbtIgnore:
    """Use project-local rules, without Git's tracked-file or global overrides."""

    def __init__(self, project: DbtFiles, drafts: Mapping[str, str] | None = None, *,
                 warn: Callable[[str], None] | None = None) -> None:
        self.project, self.drafts, self.warn = project, drafts or {}, warn
        self.controls: set[str] = set()
        self._rule_count = self._bytes = 0
        self._dbt = self._load('.dbtignore')
        self._specs = {Path('.'): self._load('.gitignore')}
        self._dirs: dict[Path, bool] = {Path('.'): False}

    def _load(self, path: str) -> GitIgnoreSpec:
        try:
            self.project.editable_path(path)
            if path in self.drafts:
                text = self.drafts[path]
                self.controls.add(path)
            else:
                file = self.project.read(path, missing=True)
                if file.revision:
                    self.controls.add(path)
                text = file.text
            lines = text.splitlines()
            self._bytes += len(text)
            self._rule_count += len(lines)
            if len(text) > _FILE_LIMIT or self._bytes > 4 * _FILE_LIMIT or self._rule_count > _MAX_RULES:
                raise DbtError('ignore rule or text limit exceeded')
            return GitIgnoreSpec.from_lines(lines)
        except (DbtError, OSError, ValueError) as exc:
            message = f'Cannot safely read or interpret ignore rules in {path!r}.'
            if self.warn is None:
                raise DbtError(message) from exc
            self.warn(message)
            return GitIgnoreSpec([])

    def _git(self, path: Path, directory: bool) -> bool:
        for parent in path.parents:
            if parent not in self._specs:
                self._specs[parent] = self._load((parent / '.gitignore').as_posix())
            relative = path.relative_to(parent).as_posix() + ('/' if directory else '')
            match = self._specs[parent].check_file(relative).include
            if match is not None:
                return match
        return False

    def _excluded_dir(self, path: Path) -> bool:
        if path not in self._dirs:
            # A negation in a child cannot reopen an excluded parent directory.
            self._dirs[path] = self._excluded_dir(path.parent) or self._git(path, True)
        return self._dirs[path]

    def git_ignored(self, relative: str) -> bool:
        path = _relative(relative)
        if len(path.parts) > _MAX_DEPTH:
            message = 'Ignore directory depth limit exceeded.'
            if self.warn is None:
                raise DbtError(message)
            self.warn(message)
            return True
        if relative.endswith('/'):
            return self._excluded_dir(path)
        return self._excluded_dir(path.parent) or self._git(path, False)

    def match_file(self, relative: str) -> bool:
        return self.git_ignored(relative) or self._dbt.match_file(relative)
