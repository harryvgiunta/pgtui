"""Edit saved dbt variables and validate invocation-only overrides."""
from __future__ import annotations

import io
from pathlib import Path

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from d8r.dbt import DbtError, _yaml
from d8r.dbt_files import DbtFile, DbtFiles

PROJECT_YAML = "dbt_project.yml"
_RUN_VARS_LIMIT = 8192  # Leave room for the remaining argv on Windows.


def _variable_mapping(text: str, label: str) -> dict:
    try:
        values = _yaml(text, Path(label))
    except (ValueError, OverflowError):
        raise DbtError(f"{label} must be a valid YAML mapping with no duplicate keys (e.g. {{lookback_days: 7}}).") from None
    if not all(isinstance(name, str) for name in values):
        raise DbtError(f"{label} variable names must be strings. Quote names that YAML treats as numbers or booleans.")
    return values


def validate_run_vars(text: str) -> str:
    """Return one argv value, preserving YAML scalars; omit the flag for no overrides."""
    if not text.strip():
        return ""
    try:
        too_large = len(text.encode("utf-8")) > _RUN_VARS_LIMIT
    except UnicodeError:
        raise DbtError("Variable overrides must be UTF-8 text.") from None
    if too_large:
        raise DbtError("Variable overrides exceed 8 KiB. Put larger values in dbt_project.yml.")
    return text if _variable_mapping(text, "Variable overrides") else ""


def _lf(text: str) -> str:
    # ruamel retains CR in comment tokens, which would double it on CRLF output.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _round_trip(text: str) -> tuple[YAML, dict]:
    editor = YAML()
    editor.preserve_quotes = True
    editor.indent(mapping=2, sequence=4, offset=2)
    return editor, editor.load(_lf(text))


def _dump(editor: YAML, value: object) -> str:
    output = io.StringIO()
    editor.dump(value, output)
    return output.getvalue()


def read_project_vars(project: DbtFiles) -> tuple[DbtFile, str]:
    before = project.read(PROJECT_YAML)
    try:
        _yaml(before.text, Path(PROJECT_YAML))
        editor, config = _round_trip(before.text)
        return before, _dump(editor, config.get("vars", {}))
    except (YAMLError, DbtError, RecursionError, TypeError, ValueError):
        raise DbtError("Cannot read variables from dbt_project.yml. Fix the project YAML in Files and reopen this screen.") from None


def save_project_vars(project: DbtFiles, before: DbtFile, text: str) -> DbtFile:
    text = text if text.strip() else "{}"
    values = _variable_mapping(text, "Project variables")
    try:
        expected = _yaml(before.text, Path(PROJECT_YAML))
        if "vars" not in expected and not values:
            return before
        editor, config = _round_trip(before.text)
        config["vars"] = editor.load(_lf(text))
        updated = _dump(editor, config)
        if "\r\n" in before.text:
            updated = updated.replace("\n", "\r\n")
        expected["vars"] = values
        # Check the same YAML semantics used by dbt and the offline renderer, including aliases.
        if yaml.safe_dump(_yaml(updated, Path(PROJECT_YAML))) != yaml.safe_dump(expected):
            raise ValueError("Project YAML changed outside the requested edit")
    except (YAMLError, DbtError, RecursionError, TypeError, ValueError):
        raise DbtError("Cannot safely replace vars without changing other YAML values. Edit dbt_project.yml directly.") from None
    return project.save(PROJECT_YAML, updated, before.revision)
