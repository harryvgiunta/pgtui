"""Durable, source-scoped catalog metadata; runtime value pools never enter disk."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from d8r.engine.datasources import DataSource, transaction_open
from d8r.sql import CatalogObject, ColumnDef, ForeignKey, IndexDef, RoutineDef, TableDef, relation_parts
from d8r.storage import _JSONStore, _record, _text


@dataclass(frozen=True)
class SchemaSnapshot:
    dialect: str
    server_version: str
    default_schema: str | None
    tables: tuple[TableDef, ...]
    routines: tuple[RoutineDef, ...]
    captured_at: str
    objects: tuple[CatalogObject, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "routines", tuple(self.routines))
        object.__setattr__(self, "objects", tuple(self.objects))

    def datasets(self) -> dict[str, dict]:
        """Rehydrate the engine's metadata shape, never rows or temporary state."""
        datasets = {}
        for table in self.tables:
            entry = {"columns": table.columns, "doc": table.doc, "rows": None,
                     "foreign_keys": table.foreign_keys, "temp": False}
            if table.kind != "table":
                entry["kind"] = table.kind
            if table.indexes:
                entry["indexes"] = table.indexes
            if table.partition_parent is not None:
                entry["partition_parent"] = table.partition_parent
            datasets[table.name] = entry
        return datasets


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
    datasets, routines, objects = source.datasets, source.routines, source.objects
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
        kind=entry.get("kind", "table"), indexes=tuple(entry.get("indexes", ())),
        partition_parent=entry.get("partition_parent") if entry.get("partition_parent") in durable else None,
    ) for name, entry in durable.items())
    native = () if source.kind == "parquet" else tuple(RoutineDef(
        routine.name, routine.parameters, routine.return_type, routine.doc,
        routine.kind, routine.variadic, routine.parameter_types,
        tuple(_catalog_column(column, enum_labels=False) for column in routine.columns),
        min_args=routine.min_args,
    ) for routine in routines if not _temporary(routine.name, source.dialect))
    extra = tuple(CatalogObject(obj.name, obj.kind, obj.detail, obj.doc,
                                tuple(_catalog_column(column, enum_labels=False) for column in obj.columns))
                  for obj in objects if source.dialect == "postgres" and not _temporary(obj.name, "postgres"))
    return SchemaSnapshot(source.dialect, source.server_version,
                          source.default_schema or source.postgres.get("schema"),
                          tables, native, datetime.now(timezone.utc).isoformat(), extra)


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


_RELATION_FIELDS = {"kind", "indexes", "partition_parent"}
_INDEX_FLAGS = {"unique", "primary", "valid", "partial", "immediate"}


def _index_record(value: object) -> dict:
    index = _record(value, {"name", "method", "keys", "included"} | _INDEX_FLAGS, "catalog index")
    _text(index["name"], "index name", nonempty=True)
    if _temporary(index["name"], "postgres"):
        raise ValueError("Temporary catalog index")
    _text(index["method"], "index method", nonempty=True)
    if not _list(index["keys"], "index keys"):
        raise ValueError("An index must have keys")
    for key in index["keys"]:
        if key is not None:
            _text(key, "index key", nonempty=True)
    _strings(index["included"], "included columns", nonempty=True)
    if any(type(index[flag]) is not bool for flag in _INDEX_FLAGS):
        raise ValueError("Index flags must be boolean")
    return index


def _optional_record(value: object, required: set[str], optional: set[str], label: str) -> dict:
    return _record(value, required | (optional.intersection(value) if isinstance(value, dict) else set()), label)


def _snapshot_record(value: object) -> None:
    snapshot = _optional_record(value, {"dialect", "server_version", "default_schema", "tables",
                                        "routines", "captured_at"}, {"objects"}, "schema snapshot")
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
    names, partitions = {}, []
    for value in _list(snapshot["tables"], "tables"):
        table = _optional_record(value, {"name", "doc", "columns", "foreign_keys"},
                                 _RELATION_FIELDS if dialect == "postgres" else set(), "catalog table")
        _text(table["name"], "table name", nonempty=True)
        if table["name"] in names or _temporary(table["name"], dialect):
            raise ValueError("Duplicate or temporary catalog table")
        kind = table.get("kind", "table")
        if kind not in ("table", "partitioned table", "view", "materialized view", "foreign table"):
            raise ValueError("Invalid relation kind")
        names[table["name"]] = kind
        parent = table.get("partition_parent")
        if parent is not None:
            _text(parent, "partition parent", nonempty=True)
            partitions.append((table["name"], parent))
        indexes = set()
        for index in _list(table.get("indexes", []), "indexes"):
            name = _index_record(index)["name"]
            if name in indexes:
                raise ValueError("Duplicate catalog index")
            indexes.add(name)
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
    for child, parent in partitions:
        if child == parent or names.get(parent) != "partitioned table":
            raise ValueError("Partition parent must be a cached partitioned table")
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
    objects = _list(snapshot.get("objects", []), "catalog objects")
    if objects and dialect != "postgres":
        raise ValueError("Only PostgreSQL extra catalog objects may be saved")
    identities = set()
    for value in objects:
        obj = _record(value, {"name", "kind", "doc", "detail", "columns"}, "catalog object")
        _text(obj["name"], "object name", nonempty=True)
        if obj["kind"] not in ("sequence", "extension", "collation", "composite", "domain"):
            raise ValueError("Invalid catalog object kind")
        identity = (obj["kind"], obj["name"])
        if identity in identities or _temporary(obj["name"], dialect):
            raise ValueError("Duplicate or temporary catalog object")
        identities.add(identity)
        _text(obj["detail"], "object detail")
        _text(obj["doc"], "object documentation")
        _columns(obj["columns"], "duckdb")  # Extra object fields never carry values.


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


def _encode_table(table: TableDef) -> dict:
    encoded = {"name": table.name, "doc": table.doc, "columns": [_encode_column(c) for c in table.columns],
               "foreign_keys": [{"columns": list(key.columns), "target_table": key.target_table,
                                 "target_columns": list(key.target_columns)} for key in table.foreign_keys]}
    if table.kind != "table":
        encoded["kind"] = table.kind
    if table.partition_parent is not None:
        encoded["partition_parent"] = table.partition_parent
    if table.indexes:
        encoded["indexes"] = [{"name": index.name, "method": index.method, "keys": list(index.keys),
                               "included": list(index.included),
                               **{flag: getattr(index, flag) for flag in sorted(_INDEX_FLAGS)}}
                              for index in table.indexes]
    return encoded


def _encode(snapshot: SchemaSnapshot) -> dict:
    return {
        "dialect": snapshot.dialect, "server_version": snapshot.server_version,
        "default_schema": snapshot.default_schema, "captured_at": snapshot.captured_at,
        "tables": [_encode_table(table) for table in snapshot.tables],
        "routines": [{"name": routine.name, "parameters": list(routine.parameters),
                      "return_type": routine.return_type, "doc": routine.doc,
                      "kind": routine.kind, "variadic": routine.variadic,
                      "parameter_types": list(routine.parameter_types),
                      "min_args": routine.min_args,
                      "columns": [_encode_column(column) for column in routine.columns]}
                     for routine in snapshot.routines],
        "objects": [{"name": obj.name, "kind": obj.kind, "detail": obj.detail, "doc": obj.doc,
                     "columns": [_encode_column(column) for column in obj.columns]} for obj in snapshot.objects],
    }


def _decode_columns(record: dict) -> tuple[ColumnDef, ...]:
    return tuple(ColumnDef(**column) for column in record["columns"])


def _decode(value: dict) -> SchemaSnapshot:
    return SchemaSnapshot(
        value["dialect"], value["server_version"], value["default_schema"],
        tuple(TableDef(table["name"], table["doc"],
                       _decode_columns(table), tuple(ForeignKey(**key) for key in table["foreign_keys"]),
                       table.get("kind", "table"), tuple(IndexDef(**index) for index in table.get("indexes", ())),
                       table.get("partition_parent"))
              for table in value["tables"]),
        tuple(RoutineDef(**{**routine, "columns": _decode_columns(routine)})
              for routine in value["routines"]),
        value["captured_at"],
        tuple(CatalogObject(**{**obj, "columns": _decode_columns(obj)}) for obj in value.get("objects", ())),
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
