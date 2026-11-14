"""A single local Parquet file, lazily exposed as the DuckDB view ``data``."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

from .datasources import DataSource, EngineError, _safe_error, refresh_source_schema


def add_parquet_source(path, display="") -> DataSource:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise EngineError("Select an existing local Parquet file")
    if raw.startswith(("\\\\", "//")) or "://" in raw or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw) and not re.match(r"^[A-Za-z]:[\\/]", raw):
        raise EngineError("Parquet sources must be local files, not URLs or network shares")
    if any(char in raw for char in "*?[]"):
        raise EngineError("Select one Parquet file, not a glob")
    try:
        resolved = Path(raw).expanduser().resolve(strict=True)
        if str(resolved).startswith(("\\\\", "//")) or not resolved.is_file() or resolved.suffix.lower() not in {".parquet", ".pq"}:
            raise EngineError("Select an existing .parquet or .pq local file")
    except (OSError, RuntimeError, ValueError):
        raise EngineError("Select an existing .parquet or .pq local file") from None
    con = None
    try:
        import duckdb
        con = duckdb.connect(":memory:", config={
            "allow_unsigned_extensions": "false", "allow_community_extensions": "false",
            "allow_persistent_secrets": "false", "autoinstall_known_extensions": "false",
            "threads": "4", "memory_limit": "512MB", "temp_directory": "",
        })
        # Parquet is a bundled extension; never install or fetch extensions.
        con.execute("LOAD parquet")
        con.execute("SET autoload_known_extensions = false")
        literal = "'" + str(resolved).replace("'", "''") + "'"
        con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet({literal})")
        # An allow-list alone is not read-only: execute_sql also rejects all
        # COPY/EXPORT/ATTACH operations, and AI queries use a read-only transaction.
        con.execute(f"SET allowed_paths = [{literal}]")
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        identity = hashlib.sha256(os.path.normcase(str(resolved)).encode()).hexdigest()[:20]
        source = DataSource(f"parquet:{identity}", display or resolved.name,
                            "Local Parquet · lazy data view", "parquet", "duckdb",
                            con=con, connection_key=str(resolved),
                            default_schema="main", server_version=duckdb.__version__)
        refresh_source_schema(source)
        return source
    except Exception as exc:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        if isinstance(exc, EngineError):
            raise
        raise _safe_error(exc, "Parquet connection") from None
