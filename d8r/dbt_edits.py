"""Model-aware dbt edits. Build reviewable file drafts, never execute dbt or SQL."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from d8r.dbt import DbtError, _call, _expression_end, _JINJA_OPEN, _yaml
from d8r.dbt_files import DbtFile, DbtFiles, DbtModel
from d8r.sql import SqlError, lex_sql, split_statements


@dataclass(frozen=True)
class DbtDraft:
    path: str
    before: DbtFile
    text: str


def add_description(model: DbtModel, before: DbtFile, description: str) -> str:
    """Round-trip YAML while preserving unrelated keys, comments and formatting conventions."""
    if before.revision or before.text:
        _yaml(before.text, Path(model.properties))
    editor = YAML()
    editor.preserve_quotes = True
    editor.indent(mapping=2, sequence=4, offset=2)
    editor.line_break = "\r\n" if "\r\n" in before.text else "\n"
    try:
        data = editor.load(before.text) if before.text else {"version": 2, "models": []}
        items = data.setdefault("models", [])
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise DbtError("Properties YAML models must be a list of mappings.")
        entries = [item for item in items if item.get("name") == model.name]
        if len(entries) > 1:
            raise DbtError(f"Multiple YAML entries describe {model.name!r}.")
        if entries:
            entry = entries[0]
        else:
            entry = {"name": model.name}
            items.append(entry)
        if model.version:
            versions = [item for item in entry.get("versions", []) if isinstance(item, dict)
                        and str(item.get("v")) == model.version]
            if len(versions) != 1:
                raise DbtError("The model's version is no longer defined in the properties file.")
            entry = versions[0]
        if entry.get("description") == description:
            return before.text
        entry["description"] = description
        output = io.StringIO()
        editor.dump(data, output)
        result = output.getvalue()
        # Shared YAML aliases must not change an unrelated model's description.
        old = _yaml(before.text, Path(model.properties)) if before.text else {"version": 2, "models": []}
        updated = _yaml(result, Path(model.properties))
        expected = json.loads(json.dumps(old, default=str))  # detach YAML aliases
        expected_items = expected.setdefault("models", [])
        expected_entry = next((item for item in expected_items if item.get("name") == model.name), None)
        if expected_entry is None:
            expected_entry = {"name": model.name}
            expected_items.append(expected_entry)
        if model.version:
            expected_entry = next(item for item in expected_entry["versions"] if str(item.get("v")) == model.version)
        expected_entry["description"] = description
        if json.loads(json.dumps(updated, default=str)) != expected:
            raise DbtError("The YAML description shares an alias or would change unrelated data. Edit the YAML manually.")
        return result
    except (YAMLError, RecursionError, TypeError, StopIteration) as exc:
        raise DbtError("Cannot safely update this YAML description. Edit the YAML manually.") from exc


def model_ref(model: DbtModel) -> str:
    version = f", version={model.version!r}" if model.version else ""
    return "{{ ref(" + repr(model.name) + version + ") }}"


def singular_test(text: str, model: DbtModel, package: str, *, dialect: str = "postgres") -> str:
    """Check the draft references this model and has the shape of one read query."""
    if not isinstance(text, str) or not text.strip() or len(text) > 32_000 or "\0" in text:
        raise DbtError("Each test needs nonempty SQL of at most 32,000 characters.")
    chunks = []
    offset = 0
    found = False
    while match := _JINJA_OPEN.search(text, offset):
        if match[0] != "{{":
            raise DbtError("A generated test must use literal ref(), source() and config() calls, not Jinja blocks.")
        end = _expression_end(text, match.end())
        kind, args, kwargs = _call(text[match.end():end])
        name_matches = args in ([model.name], [package, model.name])
        version = kwargs.get("v", kwargs.get("version", ""))
        if kind == "ref" and name_matches and str(version) == model.version:
            found = True
        chunks.extend((text[offset:match.start()], "" if kind == "config" else "dbt_relation"))
        offset = end + 2
    chunks.append(text[offset:])
    if not found:
        raise DbtError(f"Each singular data test must reference {model_ref(model)}.")
    native = "".join(chunks)
    try:
        statements = split_statements(native, dialect=dialect)
        first = next((token for token in lex_sql(native, dialect=dialect)
                      if token.kind not in {"comment", "whitespace"}), None)
    except SqlError as exc:
        raise DbtError(str(exc)) from exc
    if len(statements) != 1 or first is None or first.text.upper() not in {"SELECT", "WITH"}:
        raise DbtError("Each singular data test must be one SELECT/WITH query returning failing rows.")
    return text.rstrip().removesuffix(";").rstrip() + "\n"
