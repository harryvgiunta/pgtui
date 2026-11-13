"""Cloudflare REST contract harness backed by real SQLite, used only in tests."""
from __future__ import annotations

import json
import sqlite3
from threading import RLock

import httpx

ACCOUNT_ID = "a" * 32
DATABASE_ID = "11111111-2222-4333-8444-555555555555"
API_TOKEN = "d1-test-secret"
DATABASE_NAME = "test-db"


class D1Harness:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            "CREATE TABLE records(id INTEGER PRIMARY KEY, label TEXT);"
            "INSERT INTO records VALUES(1, 'alpha'), (2, 'beta'), (3, NULL);"
        )
        self.lock = RLock()
        self.requests: list[httpx.Request] = []

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handle), timeout=30)

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.requests.append(request)
            if (request.url.host != "api.cloudflare.com"
                    or request.headers.get("authorization") != f"Bearer {API_TOKEN}"):
                return httpx.Response(403, json={"success": False, "errors": [{"code": 10000, "message": "Unauthorized"}], "result": []})
            base = f"/client/v4/accounts/{ACCOUNT_ID}/d1/database"
            if request.method == "GET" and request.url.path == base:
                matches = request.url.params.get("name", DATABASE_NAME) == DATABASE_NAME
                rows = [{"uuid": DATABASE_ID, "name": DATABASE_NAME}] if matches else []
                return httpx.Response(200, json={"success": True, "errors": [], "result": rows,
                    "result_info": {"page": 1, "per_page": 1000, "count": len(rows), "total_count": len(rows), "total_pages": 1}})
            if (request.method != "POST" or request.url.path not in {f"{base}/{DATABASE_ID}/query", f"{base}/{DATABASE_ID}/raw"}):
                return httpx.Response(404, json={"success": False, "errors": [{"code": 10001, "message": "Unknown database"}], "result": []})
            body = json.loads(request.content)
            if "batch" in body:
                queries = body["batch"]
            else:
                queries = [body]
            results = []
            try:
                for query in queries:
                    before_changes = self.connection.total_changes
                    before_schema = self.connection.execute("PRAGMA schema_version").fetchone()[0]
                    cursor = self.connection.execute(query["sql"], query.get("params", []))
                    columns = [column[0] for column in cursor.description] if cursor.description else []
                    rows = [[list(value) if isinstance(value, bytes) else value for value in row] for row in cursor.fetchall()]
                    changes = self.connection.total_changes - before_changes
                    changed = changes > 0 or self.connection.execute("PRAGMA schema_version").fetchone()[0] != before_schema
                    raw = {"columns": columns, "rows": rows}
                    data = raw if request.url.path.endswith("/raw") else [dict(zip(columns, row)) for row in rows]
                    results.append({"success": True, "results": data,
                                    "meta": {"duration": 0.1, "changes": changes, "changed_db": changed,
                                             "rows_read": len(rows), "rows_written": changes}})
            except sqlite3.Error as exc:
                return httpx.Response(200, json={"success": False, "errors": [{"code": 7500, "message": str(exc)}], "result": []})
            return httpx.Response(200, json={"success": True, "errors": [], "result": results})

    def close(self):
        self.connection.close()
