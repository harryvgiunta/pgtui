"""Offline dbt exchange contracts; every project and artifact is temporary."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from d8r.dbt import DbtError, DbtProject


def project(tmp_path: Path, *, model_path: str = "models", extra: dict | None = None) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / model_path).mkdir(parents=True)
    config = {"name": "demo", "version": "1.0", "model-paths": [model_path]}
    config.update(extra or {})
    (root / "dbt_project.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return root


def put(root: Path, path: str, text: str) -> Path:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file


def source(root: Path, *, model_path: str = "models") -> None:
    put(root, f"{model_path}/sources.yml", """version: 2
sources:
  - name: landing
    schema: raw
    tables:
      - name: orders
        identifier: orders
""")


def manifest(root: Path, nodes: dict, sources: dict | None = None, *, target: str = "target") -> None:
    put(root, f"{target}/manifest.json", json.dumps({"nodes": nodes, "sources": sources or {}}))


def node(name="upstream", relation='"analytics"."orders"', **kwargs) -> dict:
    return {"name": name, "package_name": "demo", "resource_type": "model",
            "relation_name": relation, "original_file_path": f"models/{name}.sql", **kwargs}


def test_custom_model_paths_roundtrip_preserves_strings_comments_and_ctes(tmp_path):
    root = project(tmp_path, model_path="transformations/core")
    source(root, model_path="transformations/core")
    original = """-- raw.orders must not change here
WITH orders AS (
  SELECT id, 'raw.orders' AS note FROM "raw"."orders"
)
SELECT orders.id, 'from raw.orders' FROM orders
"""
    exported = DbtProject(root).export_model(original, "daily", "table", "Daily orders")
    assert exported.sql_path == root / "transformations/core/daily.sql"
    assert exported.sql.count("{{ source('landing', 'orders') }}") == 1
    assert "-- raw.orders must not change here" in exported.sql
    assert "FROM orders" in exported.sql
    imported = DbtProject(root).import_model("transformations/core/daily.sql")
    assert imported.sql == "\n" + original
    assert imported.name == "daily"
    assert imported.materialized == "table"
    assert imported.description == "Daily orders"
    assert not imported.used_compiled


def test_manifest_alias_package_and_version_are_authoritative(tmp_path):
    root = project(tmp_path)
    manifest(root, {"model.pkg.actual.v2": node("actual", '"warehouse"."custom"."alias"',
             package_name="pkg", version=2, latest_version=2)})
    exported = DbtProject(root).export_model('SELECT * FROM "warehouse"."custom"."alias"', "consumer")
    assert "ref('pkg', 'actual', version=2)" in exported.sql
    assert DbtProject(root).import_model("models/consumer.sql").sql.endswith('FROM "warehouse"."custom"."alias"')
    with pytest.raises(DbtError, match="No authoritative"):
        DbtProject(root).to_dbt("SELECT * FROM custom.alias")


def test_missing_or_ambiguous_mapping_never_guesses(tmp_path):
    root = project(tmp_path)
    manifest(root, {"one": node("one"), "two": node("two")})
    with pytest.raises(DbtError, match="Ambiguous"):
        DbtProject(root).export_model("select * from analytics.orders", "out")
    assert not (root / "models/out.sql").exists()
    with pytest.raises(DbtError, match="No authoritative"):
        DbtProject(root).to_dbt("select * from orders")


def test_model_yaml_alias_schema_do_not_guess_generated_relation(tmp_path):
    root = project(tmp_path)
    put(root, "models/upstream.yml", "models:\n  - name: upstream\n    config: {schema: custom, alias: aliased}\n")
    put(root, "models/upstream.sql", "select 1")
    with pytest.raises(DbtError, match="No authoritative"):
        DbtProject(root).to_dbt("select * from custom.aliased")


def test_source_quoting_overrides_are_respected(tmp_path):
    root = project(tmp_path)
    put(root, "models/sources.yml", """sources:
  - name: landing
    schema: Raw
    quoting: {schema: false}
    tables:
      - name: orders
        identifier: Orders
        quoting: {identifier: false}
""")
    assert "source('landing', 'orders')" in DbtProject(root).to_dbt("select * from raw.orders")


def test_nested_cte_scope_does_not_hide_outer_physical_relation(tmp_path):
    root = project(tmp_path)
    manifest(root, {"orders": node(relation="orders")})
    sql = "SELECT * FROM (WITH orders AS (SELECT 1) SELECT * FROM orders) x JOIN orders o ON true"
    result = DbtProject(root).to_dbt(sql)
    assert "SELECT * FROM orders)" in result
    assert "JOIN {{ ref('demo', 'upstream') }} o" in result


def test_nonrecursive_cte_self_name_and_recursive_cte(tmp_path):
    root = project(tmp_path)
    manifest(root, {"orders": node(relation="orders")})
    sql = "WITH orders AS (SELECT * FROM orders) SELECT * FROM orders"
    result = DbtProject(root).to_dbt(sql)
    assert result.count("{{ ref(") == 1
    recursive = "WITH RECURSIVE orders AS (SELECT 1 UNION ALL SELECT * FROM orders) SELECT * FROM orders"
    assert DbtProject(root).to_dbt(recursive) == recursive


def test_native_functions_comma_relations_and_comments(tmp_path):
    root = project(tmp_path)
    source(root)
    sql = "SELECT 'raw.orders', $$ raw.orders $$ FROM generate_series(1, 2) n, raw /* keep */ .orders o"
    result = DbtProject(root).to_dbt(sql)
    assert "generate_series(1, 2)" in result
    assert "$$ raw.orders $$" in result
    assert "/* keep */" in result
    assert result.count("{{ source(") == 1
    assert "source(" in DbtProject(root).to_dbt("TABLE raw.orders")


@pytest.mark.parametrize("dialect,relation", [("sqlite", "[raw].[orders]"), ("duckdb", '"raw"."orders"')])
def test_native_dialects(tmp_path, dialect, relation):
    root = project(tmp_path)
    source(root)
    assert "source('landing', 'orders')" in DbtProject(root, dialect=dialect).to_dbt(f"select * from {relation}")


def test_advanced_jinja_requires_real_compiled_artifact(tmp_path):
    root = project(tmp_path)
    raw = "{% if is_incremental() %}select 1{% else %}select 2{% endif %}"
    put(root, "models/advanced.sql", raw)
    with pytest.raises(DbtError, match="No compiled artifact"):
        DbtProject(root).import_model("models/advanced.sql")
    manifest(root, {"advanced": node("advanced", raw_code=raw, compiled_code="select 2", config={"materialized": "incremental"})})
    imported = DbtProject(root).import_model("models/advanced.sql")
    assert imported.sql == "select 2"
    assert imported.used_compiled
    assert imported.materialized == "incremental"


def test_compiled_file_uses_custom_target_and_refuses_stale(tmp_path):
    root = project(tmp_path, extra={"target-path": "artifacts"})
    model = put(root, "models/advanced.sql", "select {{ custom_macro() }}")
    artifact = put(root, "artifacts/compiled/demo/models/advanced.sql", "select 42")
    os.utime(artifact, ns=(model.stat().st_mtime_ns + 1, model.stat().st_mtime_ns + 1))
    assert DbtProject(root).import_model("models/advanced.sql").sql == "select 42"
    os.utime(artifact, ns=(1, 1))
    with pytest.raises(DbtError, match="predates"):
        DbtProject(root).import_model("models/advanced.sql")


def test_stale_manifest_cannot_supply_compiled_sql(tmp_path):
    root = project(tmp_path)
    put(root, "models/advanced.sql", "select {{ new_macro() }}")
    manifest(root, {"advanced": node("advanced", raw_code="select {{ old_macro() }}", compiled_code="select 1")})
    with pytest.raises(DbtError, match="stale"):
        DbtProject(root).import_model("models/advanced.sql")


def test_ephemeral_ref_requires_compiled_sql(tmp_path):
    root = project(tmp_path)
    put(root, "models/consumer.sql", "select * from {{ ref('temporary') }}")
    manifest(root, {"temporary": node("temporary", config={"materialized": "ephemeral"})})
    with pytest.raises(DbtError, match="compiled artifact"):
        DbtProject(root).import_model("models/consumer.sql")


def test_dynamic_calls_are_never_evaluated(tmp_path):
    root = project(tmp_path)
    marker = tmp_path / "must-not-exist"
    put(root, "models/unsafe.sql", "{{ __import__('pathlib').Path(" + repr(str(marker)) + ").touch() }}")
    with pytest.raises(DbtError, match="compiled artifact"):
        DbtProject(root).import_model("models/unsafe.sql")
    assert not marker.exists()


def test_literal_config_precedence_and_roundtrip(tmp_path):
    root = project(tmp_path, extra={"models": {"+materialized": "view", "demo": {"+tags": ["project"], "current": {"+materialized": "table"}}}})
    put(root, "models/schema.yml", "models:\n  - name: current\n    description: Original\n    config: {tags: [yaml], materialized: table}\n")
    put(root, "models/current.sql", "{{ config(materialized='incremental', unique_key='id', enabled=true, meta={'team': 'data'}) }}\nselect 1 id")
    model = DbtProject(root).import_model("models/current.sql")
    assert model.materialized == "incremental"
    assert model.config["tags"] == ["yaml"]
    assert model.config["enabled"] is True
    DbtProject(root).export_model(model.sql, "copy", model.materialized, model.description, config=model.config)
    restored = DbtProject(root).import_model("models/copy.sql")
    assert restored.config["unique_key"] == "id"
    assert restored.config["meta"] == {"team": "data"}


def test_no_clobber_and_unrelated_yaml_semantics(tmp_path):
    root = project(tmp_path)
    put(root, "models/current.sql", "select 1")
    data = {"version": 2, "models": [
        {"name": "current", "description": "old", "columns": [{"name": "id", "data_tests": ["unique"]}], "config": {"tags": ["keep"]}},
        {"name": "other", "description": "untouched", "meta": {"nested": [1, 2]}},
    ], "sources": [{"name": "external", "tables": [{"name": "data"}]}]}
    path = put(root, "models/schema.yml", yaml.safe_dump(data))
    original = path.read_bytes()
    with pytest.raises(DbtError, match="overwrite"):
        DbtProject(root).export_model("select 2", "current", "table", "new")
    assert path.read_bytes() == original
    assert (root / "models/current.sql").read_text() == "select 1"
    DbtProject(root).export_model("select 2", "current", "table", "new", overwrite=True)
    after = yaml.safe_load(path.read_text())
    assert after["sources"] == data["sources"]
    assert after["models"][1] == data["models"][1]
    assert after["models"][0]["columns"] == data["models"][0]["columns"]
    assert after["models"][0]["config"] == {"tags": ["keep"], "materialized": "table"}


def test_yaml_alias_updates_do_not_mutate_other_models(tmp_path):
    root = project(tmp_path)
    put(root, "models/current.sql", "select 1")
    path = put(root, "models/schema.yml", """version: 2
models:
  - name: current
    config: &shared {materialized: view, tags: [keep]}
  - name: other
    config: *shared
""")
    DbtProject(root).export_model("select 2", "current", "table", overwrite=True)
    models = yaml.safe_load(path.read_text())["models"]
    assert models[0]["config"]["materialized"] == "table"
    assert models[1]["config"]["materialized"] == "view"


@pytest.mark.parametrize("bad_path", ["../outside", "/outside", "C:/outside", "models/../../outside", "models\\outside"])
def test_model_paths_refuse_traversal_and_absolute_paths(tmp_path, bad_path):
    root = project(tmp_path)
    put(root, "dbt_project.yml", yaml.safe_dump({"name": "demo", "model-paths": [bad_path]}))
    with pytest.raises(DbtError):
        DbtProject(root)


def test_profiles_are_never_opened(tmp_path, monkeypatch):
    root = project(tmp_path)
    profiles = put(root, "profiles.yml", "secret: do-not-read")
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path != profiles
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    DbtProject(root)
    put(root, "dbt_project.yml", "name: demo\nmodel-paths: ['.']\n")
    with pytest.raises(DbtError, match="profiles"):
        DbtProject(root)


def test_symlink_models_and_export_destinations_refused(tmp_path):
    root = project(tmp_path)
    outside = put(tmp_path, "outside.sql", "select 99")
    try:
        (root / "models/linked.sql").symlink_to(outside)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(DbtError, match="Symlink"):
        DbtProject(root)
    assert outside.read_text() == "select 99"


def test_duplicate_yaml_keys_refused_without_changes(tmp_path):
    root = project(tmp_path)
    path = put(root, "models/schema.yml", "models: []\nmodels: [{name: hidden}]\n")
    before = path.read_bytes()
    with pytest.raises(DbtError, match="Duplicate YAML"):
        DbtProject(root)
    assert path.read_bytes() == before


def test_multiple_model_metadata_entries_refused(tmp_path):
    root = project(tmp_path)
    put(root, "models/current.sql", "select 1")
    put(root, "models/a.yml", "models: [{name: current}]\n")
    put(root, "models/b.yml", "models: [{name: current}]\n")
    with pytest.raises(DbtError, match="multiple YAML"):
        DbtProject(root).export_model("select 2", "current", overwrite=True)


def test_modal_import_is_manual_undoable_and_cancel_preserves_editor(tmp_path, parquet_source):
    from textual.widgets import Button, Input, Select
    from d8r.tui.app import D8RApp
    from d8r.tui.dbt import DBTScreen
    from d8r.tui.session import Session
    root = project(tmp_path)
    put(root, "models/simple.sql", "select 42 AS answer")
    async def scenario():
        app = D8RApp(Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state"))
        async with app.run_test(size=(140, 55)) as pilot:
            app.editor.load_text("select 1")
            screen = DBTScreen(app)
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#dbt-project-dir", Input).value = str(root)
            screen.query_one("#dbt-open", Button).press()
            await pilot.pause()
            screen.query_one("#dbt-model", Select).value = "models/simple.sql"
            screen.query_one("#dbt-load", Button).press()
            await pilot.pause()
            assert app.editor.text == "select 1"
            screen.query_one("#dbt-apply", Button).press()
            await pilot.pause()
            assert app.editor.text == "select 42 AS answer"
            app.editor.action_undo()
            assert app.editor.text == "select 1"
            cancel = DBTScreen(app)
            app.push_screen(cancel)
            await pilot.pause()
            cancel.action_cancel()
            await pilot.pause()
            assert app.editor.text == "select 1"
    asyncio.run(scenario())


def test_modal_refuses_changed_editor(tmp_path, parquet_source):
    from textual.widgets import Button, Input, Select
    from d8r.tui.app import D8RApp
    from d8r.tui.dbt import DBTScreen
    from d8r.tui.session import Session
    root = project(tmp_path)
    put(root, "models/simple.sql", "select 42")
    async def scenario():
        app = D8RApp(Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state"))
        async with app.run_test(size=(140, 55)) as pilot:
            app.editor.load_text("select 1")
            screen = DBTScreen(app)
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#dbt-project-dir", Input).value = str(root)
            screen.query_one("#dbt-open", Button).press()
            await pilot.pause()
            screen.query_one("#dbt-model", Select).value = "models/simple.sql"
            screen.query_one("#dbt-load", Button).press()
            await pilot.pause()
            app.editor.load_text("select 9")
            screen.query_one("#dbt-apply", Button).press()
            await pilot.pause()
            assert app.editor.text == "select 9"
            assert app.screen is screen
    asyncio.run(scenario())


def test_modal_exports_reviewed_sql_without_applying(tmp_path, parquet_source):
    from textual.widgets import Button, Input, TextArea
    from d8r.tui.app import D8RApp
    from d8r.tui.dbt import DBTScreen
    from d8r.tui.session import Session
    root = project(tmp_path)
    async def scenario():
        app = D8RApp(Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state"))
        async with app.run_test(size=(140, 55)) as pilot:
            app.editor.load_text("select 1")
            screen = DBTScreen(app)
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#dbt-project-dir", Input).value = str(root)
            screen.query_one("#dbt-open", Button).press()
            await pilot.pause()
            screen.query_one("#dbt-name", Input).value = "reviewed"
            screen.query_one("#dbt-description", TextArea).load_text("Reviewed model")
            screen.query_one("#dbt-sql", TextArea).load_text("select 7 AS reviewed")
            screen.query_one("#dbt-export", Button).press()
            await pilot.pause()
            assert app.editor.text == "select 1"
            imported = DbtProject(root).import_model("models/reviewed.sql")
            assert imported.sql.strip() == "select 7 AS reviewed"
            assert imported.description == "Reviewed model"
    asyncio.run(scenario())
