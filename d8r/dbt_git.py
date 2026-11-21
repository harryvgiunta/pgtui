"""Read-only Git baselines for the dbt editor; never stage or run repository hooks."""
from __future__ import annotations

import os
import shutil
import subprocess
from difflib import unified_diff

from d8r.dbt import DbtError
from d8r.dbt_files import DbtFiles, _FILE_LIMIT


def file_diff(before: str, after: str, left: str, right: str) -> str:
    # difflib does not terminate the final hunk line if the file lacks a newline.
    lines = unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=left, tofile=right)
    return "".join(line if line.endswith(("\n", "\r")) else line + "\n\\ No newline at end of file\n" for line in lines)


def git_diff(project: DbtFiles, relative: str, draft: str) -> str:
    disk = project.read(relative, missing=True)
    git = shutil.which("git")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1")

    def run(*args: str) -> bytes | None:
        try:
            result = subprocess.run(
                [git, *args], cwd=project.root, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DbtError("Git could not read this project's file history.") from exc
        return result.stdout if result.returncode == 0 else None

    def blob(revision: str) -> str | None:
        size = run("cat-file", "-s", revision)
        if size is None:
            return None
        if int(size) > _FILE_LIMIT:
            raise DbtError("Git baseline exceeds the 2 MiB editor limit.")
        value = run("cat-file", "blob", revision)
        try:
            return value.decode("utf-8-sig") if value is not None else None
        except UnicodeError as exc:
            raise DbtError("The Git baseline is not UTF-8 text.") from exc

    prefix = run("rev-parse", "--show-prefix") if git is not None else None
    sections = []
    if prefix is not None:
        path = prefix.decode("utf-8").rstrip("\r\n") + relative
        head, staged = blob("HEAD:" + path), blob(":" + path)
        if head is None and staged is None:
            sections.append("Untracked in Git: " + relative)
        for title, before, after, left, right in (
            ("Staged", head, staged, "HEAD", "index"),
            ("Working tree", staged, disk.text if disk.revision else None, "index", "disk"),
        ):
            if before != after:
                sections.append(title + "\n" + (file_diff(before or "", after or "", f"{left}/{relative}", f"{right}/{relative}")
                                                  or "Empty file added or removed.\n"))
    else:
        sections.append("Git is not installed." if git is None else "Project is not in a Git working tree.")
    diff = file_diff(disk.text, draft, f"disk/{relative}", f"draft/{relative}")
    if diff:
        sections.append("Editor draft (not saved)\n" + diff)
    return "\n\n".join(sections) or "No changes in Git or the editor."
