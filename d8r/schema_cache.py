"""Durable, source-scoped catalog metadata; runtime value pools never enter disk."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from d8r.engine.datasources import DataSource, transaction_open
from d8r.sql import ColumnDef, ForeignKey, RoutineDef, TableDef, relation_parts
from d8r.storage import _JSONStore, _record, _text


@dataclass(frozen=True)
class SchemaSnapshot:
    dialect: str
    server_version: str
    default_schema: str | None
    tables: tuple[TableDef, ...]
    routines: tuple[RoutineDef, ...]
    captured_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "routines", tuple(self.routines))

    def datasets(self) -> dict[str, dict]:
        """Rehydrate the engine's metadata shape, never rows or temporary state."""
        return {table.name: {"columns": table.columns, "doc": table.doc, "rows": None,
                             "foreign_keys": table.foreign_keys, "temp": False}
                for table in self.tables}


def _temporary(name: str, dialect: str) -> bool:
    if dialect != "postgres":
        return False
    parts = relation_parts(name, dialect=dialect)
    return bool(parts and len(parts) > 1
                and (parts[-2] == "pg_temp" or parts[-2].startswith("pg_temp_")))


def _catalog_column(column: ColumnDef, *, enum_labels: bool) -> ColumnDef:
    # PostgreSQL discovery puts pg_enum labels (also for enum domains) here.
    # Other engines' values are not catalog data and must not survive restart.
    values = column.values if enum_labels else None
    if values is not None and any(not isinstance(value, str) for value in values):
        values = None
    return ColumnDef(column.name, column.type, column.doc, values)


def capture_schema(source: DataSource) -> SchemaSnapshot | None:
    """Copy published metadata without querying, reading files, or taking a lock.

    Callers capture only after successful discovery and publish/save only while
    the source is still current. Transactional catalogs must not become durable.
    """
    if (source.con is None or getattr(source.con, "closed", False)
            or source.kind == "disconnected" or transaction_open(source)):
        return None
    datasets, routines = source.datasets, source.routines
    durable = {name: entry for name, entry in datasets.items()
               if not entry.get("temp", False) and not _temporary(name, source.dialect)
               and (source.kind != "parquet" or name == "data")}
    excluded = {relation_parts(name, dialect=source.dialect)
                for name in datasets if name not in durable}
    tables = tuple(TableDef(
        name, entry.get("doc", ""),
        tuple(_catalog_column(column, enum_labels=source.dialect == "postgres")
              for column in entry["columns"]),
        tuple(key for key in entry.get("foreign_keys", ())
              if not _temporary(key.target_table, source.dialect)
              and relation_parts(key.target_table, dialect=source.dialect) not in excluded),
    ) for name, entry in durable.items())
    native = () if source.kind == "parquet" else tuple(RoutineDef(
        routine.name, routine.parameters, routine.return_type, routine.doc,
        routine.kind, routine.variadic, routine.parameter_types,
        tuple(_catalog_column(column, enum_labels=False) for column in routine.columns),
        min_args=routine.min_args,
    ) for routine in routines if not _temporary(routine.name, source.dialect))
    return SchemaSnapshot(source.dialect, source.server_version,
                          source.default_schema or source.postgres.get("schema"),
                          tables, native, datetime.now(timezone.utc).isoformat())


def _list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _strings(value: object, label: str, *, nonempty: bool = False) -> None:
    for item in _list(value, label):
        _text(item, label, nonempty=nonempty)


def _column(value: object, dialect: str) -> None:
    column = _record(value, {"name", "type", "doc", "values"}, "catalog column")
    _text(column["name"], "column name", nonempty=True)
    _text(column["type"], "column type")
    _text(column["doc"], "column documentation")
    if column["values"] is not None:
        if dialect != "postgres":
            raise ValueError("Only PostgreSQL catalog enum labels may be saved")
        _strings(column["values"], "enum labels")


def _columns(value: object, dialect: str) -> None:
    names = set()
    for column in _list(value, "columns"):
        _column(column, dialect)
        if column["name"] in names:
            raise ValueError("Duplicate catalog column")
        names.add(column["name"])


def _snapshot_record(value: object) -> None:
    snapshot = _record(value, {"dialect", "server_version", "default_schema", "tables",
                               "routines", "captured_at"}, "schema snapshot")
    dialect = snapshot["dialect"]
    _text(dialect, "dialect")
    if dialect not in {"postgres", "duckdb", "sqlite"}:
        raise ValueError("Unsupported schema dialect")
    _text(snapshot["server_version"], "server version")
    if snapshot["default_schema"] is not None:
        _text(snapshot["default_schema"], "default schema", nonempty=True)
    _text(snapshot["captured_at"], "capture timestamp", nonempty=True)
    timestamp = datetime.fromisoformat(snapshot["captured_at"])
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("Capture timestamp must be UTC")
    names = set()
    for value in _list(snapshot["tables"], "tables"):
        table = _record(value, {"name", "doc", "columns", "foreign_keys"}, "catalog table")
        _text(table["name"], "table name", nonempty=True)
        if table["name"] in names or _temporary(table["name"], dialect):
            raise ValueError("Duplicate or temporary catalog table")
        names.add(table["name"])
        _text(table["doc"], "table documentation")
        _columns(table["columns"], dialect)
        for value in _list(table["foreign_keys"], "foreign keys"):
            key = _record(value, {"columns", "target_table", "target_columns"}, "foreign key")
            _strings(key["columns"], "foreign key columns", nonempty=True)
            _strings(key["target_columns"], "foreign key target columns", nonempty=True)
            if not key["columns"] or len(key["columns"]) != len(key["target_columns"]):
                raise ValueError("Foreign key columns must be paired")
            _text(key["target_table"], "foreign key target", nonempty=True)
            if _temporary(key["target_table"], dialect):
                raise ValueError("Temporary foreign key target")
    for value in _list(snapshot["routines"], "routines"):
        fields = {"name", "parameters", "return_type", "doc", "kind", "variadic", "parameter_types", "columns"}
        if isinstance(value, dict) and "min_args" in value:
            fields.add("min_args")
        routine = _record(value, fields, "catalog routine")
        _text(routine["name"], "routine name", nonempty=True)
        if _temporary(routine["name"], dialect):
            raise ValueError("Temporary catalog routine")
        _strings(routine["parameters"], "routine parameters")
        _strings(routine["parameter_types"], "routine parameter types")
        _text(routine["return_type"], "routine return type")
        _text(routine["doc"], "routine documentation")
        _text(routine["kind"], "routine kind")
        if routine["kind"] not in {"scalar", "aggregate", "window", "procedure", "table"}:
            raise ValueError("Invalid native routine kind")
        if type(routine["variadic"]) is not bool:
            raise ValueError("Routine variadic flag must be boolean")
        minimum = routine.get("min_args")
        if minimum is not None and (type(minimum) is not int or not 0 <= minimum <= len(routine["parameters"])):
            raise ValueError("Invalid minimum routine argument count")
        _columns(routine["columns"], dialect)


def _document(value: object) -> dict:
    document = _record(value, {"version", "snapshots"}, "schema cache")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("Unsupported schema cache version")
    if not isinstance(document["snapshots"], dict):
        raise ValueError("Schema snapshots must be an object")
    for key, snapshot in document["snapshots"].items():
        _text(key, "source identity", nonempty=True)
        _snapshot_record(snapshot)
    return document


def _encode_column(column: ColumnDef) -> dict:
    return {"name": column.name, "type": column.type, "doc": column.doc,
            "values": list(column.values) if column.values is not None else None}


def _encode(snapshot: SchemaSnapshot) -> dict:
    return {
        "dialect": snapshot.dialect, "server_version": snapshot.server_version,
        "default_schema": snapshot.default_schema, "captured_at": snapshot.captured_at,
        "tables": [{"name": table.name, "doc": table.doc,
                    "columns": [_encode_column(column) for column in table.columns],
                    "foreign_keys": [{"columns": list(key.columns), "target_table": key.target_table,
                                      "target_columns": list(key.target_columns)}
                                     for key in table.foreign_keys]} for table in snapshot.tables],
        "routines": [{"name": routine.name, "parameters": list(routine.parameters),
                      "return_type": routine.return_type, "doc": routine.doc,
                      "kind": routine.kind, "variadic": routine.variadic,
                      "parameter_types": list(routine.parameter_types),
                      "min_args": routine.min_args,
                      "columns": [_encode_column(column) for column in routine.columns]}
                     for routine in snapshot.routines],
    }


def _decode(value: dict) -> SchemaSnapshot:
    return SchemaSnapshot(
        value["dialect"], value["server_version"], value["default_schema"],
        tuple(TableDef(table["name"], table["doc"],
                       tuple(ColumnDef(**column) for column in table["columns"]),
                       tuple(ForeignKey(**key) for key in table["foreign_keys"]))
              for table in value["tables"]),
        tuple(RoutineDef(**{**routine, "columns": tuple(ColumnDef(**column)
                                                        for column in routine["columns"])})
              for routine in value["routines"]),
        value["captured_at"],
    )


class SchemaCacheStore(_JSONStore):
    """Decode once; failed or stale writes leave the last durable snapshot intact."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__(data_dir, "schema_cache", {"version": 1, "snapshots": {}}, _document)
        self.snapshots: dict[str, SchemaSnapshot] = {
            key: _decode(value) for key, value in self.document["snapshots"].items()
        }

    def save_snapshot(self, key: str, snapshot: SchemaSnapshot) -> None:
        document = {"version": 1, "snapshots": {**self.document["snapshots"], key: _encode(snapshot)}}
        self._save(document)
        self.snapshots[key] = snapshot
