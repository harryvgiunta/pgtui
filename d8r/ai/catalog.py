"""Share bounded structural index metadata, never definitions or predicates."""
from __future__ import annotations

import json

from d8r.sql import IndexDef


def index_metadata(indexes: tuple[IndexDef, ...]) -> dict:
    records = []
    remaining = 2048
    for index in indexes[:12]:
        record = {"name": index.name, "method": index.method, "keys": list(index.keys),
                  "included": list(index.included), "unique": index.unique, "primary": index.primary,
                  "partial": index.partial, "valid": index.valid, "immediate": index.immediate}
        size = len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 2
        if size <= remaining:
            records.append(record)
            remaining -= size
    return {"indexes": records, "indexes_truncated": len(records) < len(indexes)}
