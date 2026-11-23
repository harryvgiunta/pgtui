"""End-to-end dbt draft, navigation, Git and documentation boundaries."""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess

import httpx
import pytest
from textual.widgets import Button, Input, OptionList, Switch, TabbedContent, TextArea, Tree

from d8r.ai import client
from d8r.ai.client import AIConfig, AIEvent
from d8r.dbt import DbtError
from d8r.dbt_analysis import analyze_file
from d8r.dbt_files import DbtFiles
from d8r.dbt_index import DbtIndex
from d8r.dbt_rename import apply_rename, plan_rename
from d8r.engine import DataSource
from d8r.sql import SchemaContext
from d8r.tui.app import D8RApp
from d8r.tui.dbt import DbtExplorer
from d8r.tui.session import Session
from tests.test_dbt_exchange import project, put, source


def workspace(tmp_path):
    offline = DataSource("offline", "PostgreSQL", "", "disconnected", "postgres")
    return Session({offline.id: offline}, data_dir=tmp_path / "workspace")


def tree_node(node, data):
    if node.data == data:
        return node
    return next((match for child in node.children if (match := tree_node(child, data)) is not None), None)


async def open_home(app, pilot, root):
    assert app.set_dbt_enabled(True)
    await pilot.pause()
    tabs = app.query_one("#explorer-tabs", TabbedContent)
    await pilot.click(tabs.get_tab("tab-dbt"))
    await pilot.click("#dbt-home")
    app.screen.query_one("#dbt-project-dir", Input).value = str(root)
    await pilot.click("#dbt-open")
    await pilot.pause()
    await app.workers.wait_for_complete()
    return app.query_one("#dbt-tree", Tree)


async def select(tree, pilot, data):
    node = tree_node(tree.root, data)
    assert node is not None
    parent = node.parent
    while parent is not None:
        parent.expand()
        parent = parent.parent
    await pilot.pause()
    tree.select_node(node)
    await pilot.pause()


async def follow(app, pilot, line, column):
    await pilot.pause()  # Wait for the previous page's editor and toolbar layout to settle.
    editor = app.editor
    await pilot.click(editor, offset=(editor.gutter.left + editor.gutter_width + column, editor.gutter.top + line), control=True)
    await pilot.pause()


def test_explorer_links_edit_real_yaml_and_restore_historical_drafts(tmp_path):
    root = project(tmp_path, model_path="transform/core")
    source(root, model_path="transform/core")
    properties = put(root, "transform/core/properties.yaml", "# Keep my comment\nversion: 2\nmodels:\n  - name: downstream\n    description: Initial\n")
    sql = "select * from {{ source('landing', 'orders') }}\njoin {{ ref('upstream') }} on true\n"
    model = put(root, "transform/core/downstream.sql", sql)
    upstream = put(root, "transform/core/upstream.sql", "select 1 as id\n")
    test_sql = "select * from {{ ref('downstream') }} where id is null\n"
    data_test = put(root, "tests/downstream_has_id.sql", test_sql)
    put(root, "transform/core/incomplete.yml", "models: [\n")
    put(root, "profiles.yml", "secret: excluded\n")
    put(root, "target/compiled/demo/generated.sql", "select 1\n")

    async def scenario():
        app = D8RApp(workspace(tmp_path))
        async with app.run_test(size=(145, 55)) as pilot:
            tree = await open_home(app, pilot, root)
            assert tree_node(tree.root, ("file", "profiles.yml")) is None
            assert tree_node(tree.root, ("file", "target/compiled/demo/generated.sql")) is None
            await select(tree, pilot, ("file", "transform/core/incomplete.yml"))
            assert app.editor.file_format == "yaml" and app.editor.text == "models: [\n"
            await select(tree, pilot, ("file", "tests/downstream_has_id.sql"))
            edited_test = test_sql.replace("id is null", "id is null or id < 0")
            app.editor.load_text(edited_test)
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert data_test.read_text() == edited_test
            await select(tree, pilot, ("file", "transform/core/downstream.sql"))
            assert app.editor.text == sql
            model_page = app._document_identity
            await follow(app, pilot, 0, sql.index("landing"))
            assert app.editor.file_format == "yaml" and app._dbt_file["path"].endswith("sources.yml")
            app.open_page(model_page)
            await follow(app, pilot, 1, sql.splitlines()[1].index("upstream"))
            assert app.editor.text == upstream.read_text(), str(app.query_one("#status").content)
            app.open_page(model_page)
            await pilot.click("#page-details-toggle")
            assert app._dbt_file["path"] == "transform/core/properties.yaml"
            assert app.editor.cursor_location[0] == 3
            app.editor.move_cursor((app.editor.document.line_count - 1, 0))
            await pilot.press("tab")
            assert app.editor.text.endswith("\n  ")
            app.editor.action_undo()
            app.editor.load_text(app.editor.text.replace("Initial", "Edited from PGTUI"))
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert "description: Edited from PGTUI" in properties.read_text()
            assert properties.read_text().startswith("# Keep my comment")
            app.open_page(model_page)
            assert app._document_identity == model_page
            app.editor.load_text(sql + "-- autosaved on page switch\n")
            await pilot.press("ctrl+n")
            assert model.read_text() == sql + "-- autosaved on page switch\n"
        restored = D8RApp(workspace(tmp_path))
        async with restored.run_test(size=(145, 55)) as pilot:
            tabs = restored.query_one("#explorer-tabs", TabbedContent)
            await pilot.click(tabs.get_tab("tab-pages"))
            pages = restored.query_one("#pages-list", OptionList)
            pages.highlighted = pages.get_option_index(model_page)
            pages.focus()
            await pilot.press("enter")
            assert restored.editor.text == model.read_text()

    asyncio.run(scenario())


def test_mixed_switch_normalizes_and_does_not_save_a_sql_preview(tmp_path):
    root = project(tmp_path)
    source(root)
    sql = "{{ config(materialized='table') }}\nselect * from {{ source('landing', 'orders') }} a join raw.orders b on true\n"
    model = put(root, "models/current.sql", sql)

    async def scenario():
        app = D8RApp(workspace(tmp_path))
        async with app.run_test(size=(145, 55)) as pilot:
            tree = await open_home(app, pilot, root)
            await select(tree, pilot, ("file", "models/current.sql"))
            switch = app.query_one("#dbt-toggle", Switch)
            assert switch.value
            await pilot.click(switch)
            assert not switch.value and "{{" not in app.editor.text
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert model.read_text() == sql
            await pilot.click(switch)
            assert switch.value and app.editor.text.count("{{ source('landing', 'orders') }}") == 2
            assert app.editor.text.startswith("{{ config(materialized='table') }}")
            assert model.read_text() == sql
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert model.read_text() == sql  # The toggle alone is not an edit to the file.
            app.editor.insert("-- edited in dbt\n", (0, 0))
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert model.read_text().startswith("-- edited in dbt\n")
            assert model.read_text().count("{{ source('landing', 'orders') }}") == 2
            saved = model.read_bytes()
            await pilot.click(switch)
            app.editor.insert("-- edited in SQL preview\n", (0, 0))
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert model.read_bytes() == saved
            await pilot.click(switch)
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert "-- edited in SQL preview" in model.read_text()
            assert model.read_text().count("{{ source('landing', 'orders') }}") == 2

    asyncio.run(scenario())


def test_git_diff_and_external_edit_do_not_overwrite_disk(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is not installed")
    root = project(tmp_path)
    model = put(root, "models/current.sql", "select 1\n")

    def run(*args):
        subprocess.run([git, *args], cwd=root, check=True, capture_output=True)

    run("init", "-q")
    run("add", "--", "models/current.sql")
    run("-c", "user.name=Test", "-c", "user.email=dbt@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "--no-verify", "-qm", "baseline")
    model.write_text("select 2\n", newline="\n")
    run("add", "--", "models/current.sql")
    model.write_text("select 3\n", newline="\n")

    async def scenario():
        app = D8RApp(workspace(tmp_path))
        async with app.run_test(size=(145, 55)) as pilot:
            tree = await open_home(app, pilot, root)
            await select(tree, pilot, ("file", "models/current.sql"))
            model.write_text("select 5\n", newline="\n")  # External edit after opening the model.
            app.editor.load_text("select 4\n")
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert model.read_text() == "select 5\n"
            assert app.editor.text == "select 4\n"
            await pilot.click("#dbt-file-diff")
            await app.workers.wait_for_complete()
            await pilot.pause()
            diff = app.screen.query_one("#dbt-diff-text", TextArea).text
            assert "-select 1\n+select 2" in diff
            assert "-select 2\n+select 5" in diff
            assert "-select 5\n+select 4" in diff
            await pilot.press("escape")
            await select(tree, pilot, ("action", "reload"))
            assert app.editor.text == "select 5\n"
            app.editor.action_undo()
            await pilot.pause(app._WORKSPACE_SAVE_INTERVAL + 0.1)
            assert model.read_text() == "select 4\n"

    asyncio.run(scenario())


def test_assistant_applies_ten_tests_that_real_dbt_documents(tmp_path, monkeypatch):
    root = project(tmp_path, extra={"test-paths": ["checks"]})
    original = "select " + ", ".join(f"{i} as col_{i}" for i in range(10)) + ", 42 as manual_limit"
    model = put(root, "models/orders.sql", original)
    metadata = put(root, "models/properties.yml", "# an authored comment\nversion: 2\nmodels:\n  - name: orders\n    config: {tags: [keep]}\n")
    names = [f"orders_col_{i}_required" for i in range(10)]
    edited = "-- Reviewed projection\n" + original + "\n"
    sql_blocks = [edited] + [f"select * from {{{{ ref('orders') }}}} where col_{i} is null\n" for i in range(10)]
    reply = "\n".join(f"```sql\n{body}```" for body in sql_blocks)
    reply += "\n```json\n" + json.dumps({"kind": "dbt", "tests": names, "edit_model": True,
                                        "description": "Synthetic order fields"}) + "\n```"

    async def provider(config, messages, tools, callback, *, max_output_tokens):
        assert all(tool["function"]["name"] == "ask_user" for tool in tools)
        messages.append({"role": "assistant", "content": reply})
        yield AIEvent("text", reply)

    monkeypatch.setattr(client, "run_turn", provider)

    async def scenario():
        session = workspace(tmp_path)
        session.update_settings(ai=AIConfig(base_url="http://127.0.0.1:9999/v1", model="synthetic"),
                                remember_ai_key=False, ai_auto_accept=True)
        app = D8RApp(session)
        async with app.run_test(size=(155, 60)) as pilot:
            tree = await open_home(app, pilot, root)
            tests = tree_node(tree.root, ("group", "tests"))
            models = tree_node(tree.root, "models")
            assert tree.root.children[tree.root.children.index(models) + 1] is tests
            await select(tree, pilot, ("file", "models/orders.sql"))
            model_page = app._document_identity
            await select(tree, pilot, ("action", "test"))
            app.screen.query_one("#dbt-test-name", Input).value = "orders_nonnegative"
            test_sql = app.screen.query_one("#dbt-test-sql", TextArea)
            test_sql.focus()
            test_sql.move_cursor((2, 6))
            await pilot.press("m", "a", "n", "ctrl+space", "tab", "space", "less_than_sign", "space", "0")
            assert "where manual_limit < 0" in test_sql.text
            await pilot.click("#dbt-test-create")
            manual_test = root / "checks/orders_nonnegative.sql"
            assert manual_test.read_text() == "select *\nfrom {{ ref('orders') }}\nwhere manual_limit < 0\n"
            app.open_page(model_page)
            await app.workers.wait_for_complete()
            tests = tree_node(tree.root, ("group", "tests"))
            assert tree_node(tests, ("file", "checks/orders_nonnegative.sql")) is not None
            await select(tree, pilot, ("file", "checks/orders_nonnegative.sql"))
            assert app.editor.text == manual_test.read_text()
            app.open_page(model_page)
            await pilot.click("#open-ai")
            await pilot.pause()
            app.query_one("#ai-input", TextArea).load_text("Make me 10 tests for this model, describe it, and add a review comment")
            await pilot.click("#ai-send")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert sorted(path.stem for path in (root / "checks").glob("*.sql")) == [manual_test.stem]
            assert "Synthetic order fields" not in metadata.read_text()
            assert model.read_text() == original
            assert not app.query_one("#ai-apply", Button).disabled
            await pilot.click("#ai-apply")
            assert app.editor.text == edited
            assert sorted(path.stem for path in (root / "checks").glob("*.sql")) == sorted([*names, manual_test.stem])
            tabs = app.query_one("#explorer-tabs", TabbedContent)
            await pilot.click(tabs.get_tab("tab-dbt"))
            await app.workers.wait_for_complete()
            tests = tree_node(tree.root, ("group", "tests"))
            assert all(tree_node(tests, ("file", f"checks/{name}.sql")) is not None for name in names)
            await select(tree, pilot, ("file", f"checks/{names[0]}.sql"))
            assert app.editor.text == (root / "checks" / f"{names[0]}.sql").read_text()
            assert "Synthetic order fields" in metadata.read_text()
            assert metadata.read_text().startswith("# an authored comment")
            assert "tags: [keep]" in metadata.read_text()
            assert model.read_text() == edited
            assert not session.history
            with socket.socket() as port_check:
                port_check.bind(("127.0.0.1", 0))
                port = port_check.getsockname()[1]
            await select(tree, pilot, ("action", "docs"))
            app.screen.query_one("#dbt-docs-port", Input).value = str(port)
            await pilot.click("#dbt-docs-start")
            await app.workers.wait_for_complete()
            docs = app.query_one(DbtExplorer).docs
            assert docs.port == port
            url = f"http://127.0.0.1:{port}/"
            async with httpx.AsyncClient(trust_env=False) as browser:
                assert (await browser.get(url)).status_code == 200
                nodes = (await browser.get(url + "manifest.json")).json()["nodes"]
                assert nodes["model.demo.orders"]["description"] == "Synthetic order fields"
                for name in [*names, manual_test.stem]:
                    assert nodes[f"test.demo.{name}"]["depends_on"]["nodes"] == ["model.demo.orders"]
                assert (await browser.get(url + "profiles.yml")).status_code == 404
                await select(tree, pilot, ("action", "stop-docs"))
                await app.workers.wait_for_complete()
                with pytest.raises(httpx.ConnectError):
                    await browser.get(url)

    asyncio.run(scenario())


def test_paths_exclude_credentials_and_refuse_external_changes(tmp_path):
    root = project(tmp_path)
    path = put(root, "models/utf8.yml", "\ufeffversion: 2\r\n")
    put(root, "models/profiles.yml", "secret: no\n")
    files = DbtFiles(root)
    with pytest.raises(DbtError, match="profiles"):
        files.read("models/profiles.yml")
    baseline = files.read("models/utf8.yml")
    files.save("models/utf8.yml", baseline.text + "models: []\r\n", baseline.revision)
    assert path.read_bytes() == b"\xef\xbb\xbfversion: 2\r\nmodels: []\r\n"
    with pytest.raises(DbtError, match="changed on disk"):
        files.save("models/utf8.yml", "version: 2", baseline.revision)


def test_rename_refuses_new_project_reference_after_review(tmp_path):
    home = project(tmp_path)
    model = put(home, "models/base.sql", "select 1 as id\n")
    ref = "select * from {{ ref('base') }}\n"
    report = put(home, "models/report.sql", ref)
    files = DbtFiles(home)
    plan = plan_rename(files, "models/base.sql", "upstream")

    # A concurrent addition could otherwise keep a reference to the old name
    # even if every file in the diff still has its original revision.
    late = put(home, "tests/new_check.sql", ref)
    with pytest.raises(DbtError, match="no longer safe"):
        apply_rename(files, plan)
    assert model.read_text() == "select 1 as id\n"
    assert report.read_text() == late.read_text() == ref
    assert not (home / "models/upstream.sql").exists()


def test_star_reveals_partial_source_without_rewriting_projection(tmp_path):
    home = project(tmp_path)
    put(home, "models/sources.yml", """version: 2
sources:
  - name: landing
    tables:
      - name: orders
        columns: [{name: id, data_type: bigint}]
""")
    path = "models/report.sql"
    text = "select o.* from {{ source('landing', 'orders') }} o\n"
    put(home, path, text)
    index = DbtIndex(DbtFiles(home), {})
    result = analyze_file(index, path, text, "postgres", SchemaContext())

    # YAML documents one field; it is not evidence that the warehouse relation
    # has only one field. Expanding this projection could drop data.
    star, = result.stars
    assert text[star.start:star.end] == "o.*"
    assert any("id" in label and "bigint" in label.lower() for label in star.columns)
    assert not star.replacement
