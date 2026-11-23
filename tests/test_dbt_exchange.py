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
    file.write_text(text, encoding="utf-8", newline="")
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
    converted = DbtProject(root).to_dbt(original)
    put(root, "transformations/core/daily.sql", "{{ config(materialized='table') }}\n" + converted)
    put(root, "transformations/core/models.yml", "models: [{name: daily, description: Daily orders}]\n")
    assert converted.count("{{ source('landing', 'orders') }}") == 1
    assert "-- raw.orders must not change here" in converted
    assert "FROM orders" in converted
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
    converted = DbtProject(root).to_dbt('SELECT * FROM "warehouse"."custom"."alias"')
    assert "ref('pkg', 'actual', version=2)" in converted
    put(root, "models/consumer.sql", converted)
    assert DbtProject(root).import_model("models/consumer.sql").sql.endswith('FROM "warehouse"."custom"."alias"')
    with pytest.raises(DbtError, match="No authoritative"):
        DbtProject(root).to_dbt("SELECT * FROM custom.alias")


def test_query_conversion_resolves_active_database_and_preserves_cte_scope(tmp_path):
    root = project(tmp_path)
    source(root)
    manifest(root, {
        "local": node("recent", '"warehouse"."analytics"."orders"'),
        "other": node("remote", '"other_db"."analytics"."orders"'),
    })
    dbt = DbtProject(root)
    sql = "WITH orders AS (SELECT * FROM orders) SELECT * FROM orders JOIN analytics.orders ON true"
    _, jinja = dbt.query_views(sql, database="warehouse", schema="raw")
    assert jinja == (
        "WITH orders AS (SELECT * FROM {{ source('landing', 'orders') }}) "
        "SELECT * FROM orders JOIN {{ ref('demo', 'recent') }} ON true"
    )
    native, _ = dbt.query_views(jinja, database="warehouse", schema="raw")
    assert native == (
        'WITH orders AS (SELECT * FROM "raw"."orders") '
        'SELECT * FROM orders JOIN "warehouse"."analytics"."orders" ON true'
    )
    with pytest.raises(DbtError, match="another database"):
        dbt.query_views("select * from {{ ref('remote') }}", database="warehouse", schema="raw")


def test_query_conversion_leaves_sql_json_and_literal_config_intact(tmp_path):
    root = project(tmp_path)
    source(root)
    dbt = DbtProject(root)
    sql = """\nSELECT '{"nested": {"value": 1}}' FROM raw.orders -- keep }}\n"""
    call = "{{ source('landing', 'orders') }}"
    _, jinja = dbt.query_views(sql, database="warehouse")
    assert jinja == sql.replace("raw.orders", call)
    model = "{{ config(meta={'nested': {'marker': '}}'}}) }}" + jinja
    assert dbt.query_views(model, database="warehouse") == (jinja.replace(call, '"raw"."orders"'), model)


@pytest.mark.parametrize("sql", [
    "SELECT '{{ config() }}' FROM raw.orders",
    "SELECT $tag${% text$tag$ FROM raw.orders",
    "-- {{ ref('example') }}\nSELECT * FROM raw.orders",
])
def test_query_conversion_refuses_ambiguous_quoted_template_markers(tmp_path, sql):
    root = project(tmp_path)
    source(root)
    with pytest.raises(DbtError, match="ambiguous"):
        DbtProject(root).query_views(sql, database="warehouse")


def test_query_conversion_refuses_source_call_shared_by_multiple_relations(tmp_path):
    root = project(tmp_path)
    manifest(root, {}, {
        "source.demo.landing.orders": node("orders", "raw.orders", source_name="landing", resource_type="source"),
        "source.pkg.landing.orders": node("orders", "other.orders", source_name="landing", resource_type="source", package_name="pkg"),
    })
    with pytest.raises(DbtError, match="unambiguously"):
        DbtProject(root).query_views("SELECT * FROM raw.orders", database="warehouse")


def test_editor_toggle_preserves_crlf_and_is_undoable(tmp_path):
    from d8r.engine import DataSource
    from d8r.tui.app import D8RApp
    from d8r.tui.session import Session

    root = project(tmp_path)
    source(root)
    sql = 'SELECT * FROM raw -- keep\r\n  .orders AS "o";\r\n'
    target = DataSource("offline", "PostgreSQL", "", "disconnected", "postgres")

    async def scenario():
        app = D8RApp(Session({target.id: target}, data_dir=tmp_path / "state"))
        async with app.run_test(size=(140, 50)) as pilot:
            assert app.set_dbt_enabled(True)
            assert app.set_dbt_project_dir(str(root))
            app.editor.load_text(sql)
            await pilot.pause()
            await pilot.click("#dbt-toggle")
            jinja = app.editor.text
            assert "{{ source('landing', 'orders') }} -- keep\r\n" in jinja
            assert jinja.endswith(' AS "o";\r\n')
            app.editor.action_undo()
            assert app.editor.text == sql
            await pilot.pause(0.25)  # Textual briefly debounces a pressed button.
            await pilot.click("#dbt-toggle")
            assert app.editor.text == jinja
            await pilot.pause(0.25)
            await pilot.click("#dbt-toggle")
            assert app.editor.text == sql

    asyncio.run(scenario())


def test_missing_or_ambiguous_mapping_never_guesses(tmp_path):
    root = project(tmp_path)
    manifest(root, {"one": node("one"), "two": node("two")})
    with pytest.raises(DbtError, match="Ambiguous"):
        DbtProject(root).to_dbt("select * from analytics.orders")
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
    assert DbtProject(root).render_query(raw, database="", path="models/advanced.sql") == "select 2"
    with pytest.raises(DbtError, match="manifest is stale"):
        DbtProject(root).render_query(raw + "\n-- unsaved edit", database="", path="models/advanced.sql")


def test_jinja_loop_budget_covers_iterations_that_emit_nothing(tmp_path):
    dbt = DbtProject(project(tmp_path))
    draft = "{% for x in range(200) %}{% for y in range(200) %}{% endfor %}{% endfor %}select 1"
    with pytest.raises(DbtError, match="Jinja render limit exceeded"):
        dbt.render_query(draft, database="")


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


def test_literal_config_precedence(tmp_path):
    root = project(tmp_path, extra={"models": {"+materialized": "view", "demo": {"+tags": ["project"], "current": {"+materialized": "table"}}}})
    put(root, "models/schema.yml", "models:\n  - name: current\n    description: Original\n    config: {tags: [yaml], materialized: table}\n")
    put(root, "models/current.sql", "{{ config(materialized='incremental', unique_key='id', enabled=true, meta={'team': 'data'}) }}\nselect 1 id")
    model = DbtProject(root).import_model("models/current.sql")
    assert model.materialized == "incremental"
    assert model.config["tags"] == ["yaml"]
    assert model.config["enabled"] is True
    assert model.config["unique_key"] == "id"
    assert model.config["meta"] == {"team": "data"}


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


def test_symlink_models_refused(tmp_path):
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
        DbtProject(root).import_model("models/current.sql")


def test_modal_saves_project_home_without_editing_query(tmp_path, parquet_source):
    from textual.widgets import Button, Input
    from d8r.tui.app import D8RApp
    from d8r.tui.dbt import DBTScreen
    from d8r.tui.session import Session
    root = project(tmp_path)
    put(root, "models/simple.sql", "select 42 AS answer")
    async def scenario():
        app = D8RApp(Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state"))
        async with app.run_test(size=(140, 55)) as pilot:
            assert app.set_dbt_enabled(True)
            app.editor.load_text("select 1")
            screen = DBTScreen(app)
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#dbt-project-dir", Input).value = str(root)
            screen.query_one("#dbt-open", Button).press()
            await pilot.pause()
            assert app.editor.text == "select 1"
        restored = Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state")
        assert restored.dbt_project_dir == str(root.resolve())
    asyncio.run(scenario())


