"""Plan and publish a new, credential-free dbt project for a native source."""
from __future__ import annotations

import ntpath
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from d8r.dbt import DbtError, _linked, _literal, _relative
from d8r.engine import DataSource
from d8r.sql import quote_identifier, relation_parts


@dataclass(frozen=True)
class SetupTarget:
    label: str
    source_kind: str
    profile_help: str


TARGETS = {
    "postgres": SetupTarget("PostgreSQL", "postgres-live",
                            "External dbt uses dbt-postgres and an environment-based profile template. "
                            "Choose a development schema for dbt outputs, separate from upstream tables."),
    "duckdb": SetupTarget("DuckDB · local Parquet", "parquet",
                          "External dbt uses dbt-duckdb with a local Parquet source and an in-memory profile. "
                          "The Parquet file will not be copied or changed."),
    "d1": SetupTarget("Cloudflare D1 · SQLite SQL", "d1-live",
                      "PGTUI connects to the remote D1 database. No external profile is generated: "
                      "a D1-capable dbt adapter is required. dbt-sqlite does not connect to Cloudflare D1."),
}
MATERIALIZATIONS = {"view": "Views · read current source data", "table": "Tables · rebuild to refresh the snapshot"}
CHECKS = {"summary": "Check that the summary returns one valid count",
          "nonempty": "Also require the input to contain rows"}


@dataclass(frozen=True)
class SetupInput:
    schema: str
    identifier: str
    columns: tuple[str, ...] = ()
    parquet: Path | None = None


def _identifier(value: str) -> bool:
    return _literal(value) and all(char.isprintable() for char in value)


def setup_inputs(source: DataSource) -> dict[str, SetupInput]:
    """Use catalog metadata only; never copy connection settings or sample rows."""
    if source.kind == "parquet":
        entry = source.datasets.get("data", {})
        return {"data": SetupInput("main", "data", tuple(
            col.name for col in entry.get("columns", ()) if _identifier(col.name)), Path(source.connection_key))}
    selections = {}
    for name, entry in source.datasets.items():
        if entry.get("temp"):
            continue
        parts = relation_parts(name, dialect=source.dialect)
        if not parts or not all(map(_identifier, parts)):
            continue
        if source.kind == "postgres-live" and len(parts) == 2:
            schema, identifier = parts
        elif source.kind == "d1-live" and len(parts) == 1:
            schema, identifier = "main", parts[0]
        else:
            continue
        selections[name] = SetupInput(schema, identifier, tuple(
            col.name for col in entry.get("columns", ()) if _identifier(col.name)))
    return selections


_FileID = tuple[int, int]


def _identity(info: os.stat_result) -> _FileID:
    return info.st_dev, info.st_ino


def _directory_id(path: Path) -> _FileID:
    if any(_linked(part) for part in (path, *path.parents)):
        raise DbtError("Use a real directory, not a symlink or junction.")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise DbtError(f"Not a directory: {path}")
    return _identity(info)


@dataclass(frozen=True)
class SetupDirectory:
    path: Path
    parent_id: _FileID
    existing_id: _FileID | None


def setup_directory(folder: str | Path) -> SetupDirectory:
    """Snapshot an empty or not-yet-created local directory without writing."""
    raw = str(folder).strip()
    if not raw or raw.startswith(("//", "\\\\")) or "://" in raw:
        raise DbtError("Choose an empty or new local project directory.")
    try:
        path = Path(raw).expanduser().absolute()
        if any(part == ".." or ntpath.isreserved(part) for part in path.parts[1:]):
            raise DbtError("The project directory cannot contain traversal or reserved names.")
        parent_id = _directory_id(path.parent)
        try:
            existing_id = _directory_id(path)
        except FileNotFoundError:
            existing_id = None
        if existing_id is not None and next(path.iterdir(), None) is not None:
            raise DbtError("Choose an empty directory. Setup will not merge with existing files.")
    except DbtError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise DbtError("Choose a local directory with an existing, accessible parent.") from exc
    return SetupDirectory(path, parent_id, existing_id)


def setup_name(value: str, label: str) -> str:
    name = value.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name) or ntpath.isreserved(name):
        raise DbtError(f"{label} must start with a lowercase letter and use at most 48 lowercase letters, digits or underscores.")
    return name


@dataclass(frozen=True)
class DbtScaffold:
    directory: SetupDirectory
    files: tuple[tuple[str, str], ...]

    def preview(self) -> str:
        return "\n".join(f"── {path} ──\n{text}" for path, text in self.files)


def _yaml_text(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)


def _profile(target: str, project: str) -> str:
    if target == "postgres":
        output = {
            "type": "postgres", "host": "{{ env_var('DBT_PG_HOST') }}",
            "port": "{{ env_var('DBT_PG_PORT', '5432') | int }}",
            "user": "{{ env_var('DBT_PG_USER') }}", "password": "{{ env_var('DBT_ENV_SECRET_PG_PASSWORD') }}",
            "dbname": "{{ env_var('DBT_PG_DATABASE') }}",
            "schema": f"{{{{ env_var('DBT_PG_SCHEMA', 'dbt_{project}') }}}}",
            "sslmode": "{{ env_var('DBT_PG_SSLMODE') }}", "threads": 1,
        }
    else:
        output = {"type": "duckdb", "path": ":memory:", "schema": "main", "threads": 1}
    return _yaml_text({project: {"target": "dev", "outputs": {"dev": output}}})


def _verification(target: str, project: str) -> str:
    if target == "d1":
        setup = f"""This project targets **remote Cloudflare D1** using SQLite SQL. PGTUI links it
to your D1 source and can browse and edit the files without a dbt adapter.
No D1 adapter or executable profile is provided. A local SQLite file and
`dbt-sqlite` are not substitutes for Cloudflare D1.

Only proceed with the commands below if you have a dbt adapter that explicitly
supports remote D1 and a compatible dbt Core version. Follow that adapter's
instructions to create a private profile named `{project}` with target `dev`,
point it at the actual Cloudflare account and D1 database, and supply the API
token through the adapter's secure external configuration. Check how it maps
the SQLite `main` namespace to D1 and which materializations/tests it supports.
Without such an adapter, use the PGTUI workspace for authoring; these dbt
commands cannot verify the project against D1.
"""
    else:
        adapter = "dbt-postgres" if target == "postgres" else "dbt-duckdb"
        setup = f"""Install `{adapter}` with a compatible dbt Core version in your external dbt
environment. Copy `profiles.yml.example` to `profiles.yml` in a private
profiles directory **outside this project** (or add its `{project}` entry to
your existing profile without replacing other entries). PGTUI does not
export keyring credentials to dbt.
"""
        if target == "postgres":
            setup += f"""
Set `DBT_PG_HOST`, `DBT_PG_USER`, `DBT_PG_DATABASE` and `DBT_PG_SSLMODE` to match
the PostgreSQL connection. `DBT_PG_PORT` defaults to 5432. Supply
`DBT_ENV_SECRET_PG_PASSWORD` through your external secret manager or private
process environment, not a file or chat. Use a dedicated development schema
(`DBT_PG_SCHEMA`, default `dbt_{project}`) and a role that can read the
upstream relation and create models in that schema.
"""
        else:
            setup += """
`models/sources.yml` points at the selected local Parquet file. Update its
external location if the file moves. Use `dbt build` in one invocation:
this profile's in-memory models do not persist across invocations.
"""
    return f"""# {project}

{setup.strip()}

From the project directory, replace `<profiles-directory>` with the private
profiles directory. Set **Path to dbt Profiles** in **Workspace Settings** to
that directory if it is not discovered automatically. Leave **Workspace Default
Target Override** blank to use the profile default. Review the models and choose
a development target where creating or replacing the named model relations is
safe. `debug` checks the connection; `parse` checks project structure; `build`
creates models and runs tests.

```sh
dbt debug --profiles-dir "<profiles-directory>" --profile {project} --target dev
dbt parse --profiles-dir "<profiles-directory>" --profile {project} --target dev
dbt build --profiles-dir "<profiles-directory>" --profile {project} --target dev --target-path target/dev --write-json
```

Project setup writes local files and links a PGTUI source. It does not run dbt
or execute the starter SQL.
"""


def plan_scaffold(folder: str, project: str, target: str, upstream: SetupInput, *,
                  model: str, materialization: str, check: str, key: str = "") -> DbtScaffold:
    project = setup_name(project, "Project name")
    model = setup_name(model, "Staging model name")
    if target not in TARGETS or materialization not in MATERIALIZATIONS or check not in CHECKS:
        raise DbtError("Choose a supported target, model materialization and starter check.")
    summary = f"{project}_row_count"
    if len({model, summary, upstream.identifier.casefold()}) < 3:
        raise DbtError("Use distinct names for the staging model, row count model and upstream relation.")
    if not all(map(_identifier, (upstream.schema, upstream.identifier))) or (
        key and (not _identifier(key) or key not in upstream.columns)
    ):
        raise DbtError("Choose a supported relation and key column from the source.")
    if target == "d1" and upstream.schema != "main":
        raise DbtError("A Cloudflare D1 input must belong to the linked database's main namespace.")
    table = {"name": "records", "identifier": upstream.identifier}
    if target == "duckdb":
        path = upstream.parquet
        if ((upstream.schema, upstream.identifier) != ("main", "data")
                or path is None or not path.is_absolute() or not path.is_file()
                or path.suffix.lower() not in {".parquet", ".pq"} or not _identifier(path.as_posix())
                or path.as_posix().startswith("//") or any(char in path.as_posix() for char in "*?[]")
                or _linked(path)):
            raise DbtError("Choose an existing local Parquet file without glob or Jinja markers in its path.")
        # String.Template protects literal braces; SQL quoting and $$ escape the path.
        literal = path.as_posix().replace("'", "''").replace("$", "$$")
        table["config"] = {"external_location": f"read_parquet('{literal}')", "formatter": "template"}

    config = {"name": project, "version": "1.0.0", "config-version": 2, "profile": project,
              "model-paths": ["models"], "test-paths": ["tests"], "clean-targets": ["target", "dbt_packages"],
              "models": {project: {"+materialized": materialization}}}
    sources = {"version": 2, "sources": [{"name": "raw_input", "schema": upstream.schema,
                "quoting": {"schema": True, "identifier": True}, "tables": [table]}]}
    properties = {"version": 2, "models": [
        {"name": model, "description": "Starter staging model over the selected input."},
        {"name": summary, "description": "One row with the number of staged input rows.",
         "columns": [{"name": "row_count", "description": "Number of rows in the staging model."}]},
    ]}
    files = [
        ("dbt_project.yml", _yaml_text(config)),
        (".gitignore", "/target/\n/logs/\n/dbt_packages/\nprofiles.yml\nprofiles.yaml\n"),
        ("models/sources.yml", _yaml_text(sources)),
        (f"models/{model}.sql", "select *\nfrom {{ source('raw_input', 'records') }}\n"),
        (f"models/{summary}.sql", f"select count(*) as row_count\nfrom {{{{ ref('{model}') }}}}\n"),
        ("models/properties.yml", _yaml_text(properties)),
        ("tests/valid_row_count.sql", (
            f"select count(*) as failing_count\nfrom {{{{ ref('{summary}') }}}}\n"
            "having count(*) <> 1 or count(row_count) <> 1 or min(row_count) < 0\n"
        )),
    ]
    if check == "nonempty":
        files.append(("tests/nonempty_input.sql", f"select row_count\nfrom {{{{ ref('{summary}') }}}}\nwhere row_count = 0\n"))
    if key:
        column = "staged." + quote_identifier(key)
        files.append(("tests/valid_input_key.sql", f"select {column} as failing_key\n"
                      f"from {{{{ ref('{model}') }}}} as staged\ngroup by {column}\n"
                      f"having {column} is null or count(*) > 1\n"))
    if target != "d1":
        files.append(("profiles.yml.example", _profile(target, project)))
    files.append(("README.md", _verification(target, project)))
    return DbtScaffold(setup_directory(folder), tuple(files))


def write_scaffold(plan: DbtScaffold) -> None:
    """Publish the exact reviewed files exclusively; roll back our own writes on failure."""
    root = plan.directory.path
    if setup_directory(root) != plan.directory:
        raise DbtError("The project directory changed. Review a new preview before creating it.")
    created_dirs: dict[Path, _FileID] = {}
    published: list[tuple[Path, _FileID, bytes]] = []
    directories = {root.parent: plan.directory.parent_id}

    def check_dir(path: Path) -> None:
        while path in directories:
            if _directory_id(path) != directories[path]:
                raise DbtError("The project directory changed during creation.")
            if path == root.parent:
                break
            path = path.parent

    def make_dir(path: Path) -> None:
        check_dir(path.parent)
        path.mkdir()
        directories[path] = created_dirs[path] = _directory_id(path)

    try:
        if plan.directory.existing_id is None:
            make_dir(root)
        else:
            directories[root] = plan.directory.existing_id
        check_dir(root)
        with TemporaryDirectory(dir=root, prefix=".pgtui-setup-") as temp:
            staged = []
            for index, (name, text) in enumerate(plan.files):
                destination = root / _relative(name)
                data = text.encode("utf-8")
                item = Path(temp) / str(index)
                item.write_bytes(data)
                staged.append((item, destination, data))
            for item, destination, data in staged:
                parent = root
                for name in destination.relative_to(root).parts[:-1]:
                    parent /= name
                    if parent not in directories:
                        make_dir(parent)
                check_dir(parent)
                identity = _identity(item.lstat())
                os.link(item, destination)  # Never replaces even a late collision.
                published.append((destination, identity, data))
    except (DbtError, OSError) as exc:
        leftovers = []
        for path, identity, data in reversed(published):
            try:
                check_dir(path.parent)
                info = path.lstat()
                if _identity(info) != identity or info.st_size != len(data) or path.read_bytes() != data:
                    raise DbtError("A published file was changed externally.")
                path.unlink()
            except FileNotFoundError:
                continue
            except (DbtError, OSError):
                leftovers.append(path.relative_to(root).as_posix())
        for path in reversed(created_dirs):
            try:
                check_dir(path)
                path.rmdir()
            except FileNotFoundError:
                continue
            except (DbtError, OSError):
                leftovers.append(str(path))
        suffix = (" Rollback incomplete; inspect: " + ", ".join(leftovers) if leftovers
                  else " No scaffold files were left behind.")
        raise DbtError(f"Could not create the dbt project: {exc}.{suffix}") from exc
