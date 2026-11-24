import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from jinja2 import Environment

from d8r import dbt_run as runtime
from d8r.dbt import DbtError
from d8r.dbt_files import file_revision
from d8r.dbt_profiles import DbtTargetSettings


_MODEL = "models/report.sql"
_SQL = "select 1\n"
_CURRENT = "00d8a188-2308-48cb-934a-59246a07bea7"
_OLD = "a16be188-2408-48cb-934a-59246a07bea7"
_MANIFEST_SCHEMA = "https://schemas.getdbt.com/dbt/manifest/v12.json"
_RESULTS_SCHEMA = "https://schemas.getdbt.com/dbt/run-results/v6.json"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "dbt_project.yml").write_text("name: demo\nprofile: demo\n", encoding="utf-8")
    (tmp_path / "profiles.yml").write_text(
        "demo:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n", encoding="utf-8"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / _MODEL).write_bytes(_SQL.encode())
    return tmp_path


def _plan(project, command, *, model=True, **options):
    return runtime.prepare_dbt_run(str(project / "models"), _MODEL if model else "",
                                   file_revision(_SQL) if model else "", DbtTargetSettings(str(project / "profiles.yml")),
                                   command, **options)


def _event(events, name, code, *, invocation=_CURRENT, **data):
    events.feed(json.dumps({"info": {"name": name, "code": code, "invocation_id": invocation,
                                     "msg": "do not display: some-secret"}, "data": data}).encode())


def _artifacts(run, *, invocation=_CURRENT, raw=_SQL.strip(), extra=()):
    run.artifact_dir.mkdir(parents=True, exist_ok=True)
    model = {"unique_id": "model.demo.report", "name": "report", "resource_type": "model", "package_name": "demo",
             "original_file_path": _MODEL.replace("/", "\\"), "raw_code": raw, "compiled_code": "select 1 as compiled",
             "compiled": True}
    nodes = {model["unique_id"]: model}
    rows = [{"unique_id": model["unique_id"], "status": "success"}]
    for node, row in extra:
        nodes[node["unique_id"]] = node
        rows.append({"unique_id": node["unique_id"], **row})
    manifest = {"metadata": {"invocation_id": invocation, "dbt_schema_version": _MANIFEST_SCHEMA,
                             "project_name": "demo", "adapter_type": "postgres"}, "nodes": nodes}
    results = {"metadata": {"invocation_id": invocation, "dbt_schema_version": _RESULTS_SCHEMA},
               "args": {"profile": "demo", "target": "dev", "profiles_dir": str(run.target.profiles_dir), "which": run.verb},
               "results": rows}
    for name, data in (("manifest.json", manifest), ("run_results.json", results)):
        (run.artifact_dir / name).write_text(json.dumps(data), encoding="utf-8")


def test_runtime_override_with_spaces_and_no_silent_fallback(project):
    environment = project / "custom env"
    if os.name == "nt":
        executable = environment / "Scripts" / "dbt.exe"
        executable.parent.mkdir(parents=True)
        shutil.copyfile(sys.executable, executable)
        batch = environment / "run dbt.cmd"
        batch.write_text("@echo off\n", encoding="utf-8")
        with pytest.raises(DbtError, match="override"):
            runtime.find_dbt_runtime(project, str(batch))
    else:
        executable = environment / "bin" / "dbt"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        executable.chmod(0o755)
    assert runtime.find_dbt_runtime(project, "custom env").argv == (str(executable),)
    assert runtime.find_dbt_runtime(project, str(executable.relative_to(project))).argv == (str(executable),)
    with pytest.raises(DbtError, match="override"):
        runtime.find_dbt_runtime(project, "missing env/dbt")


def test_named_selector_and_state_are_separate_argv(project):
    (project / "selectors.yml").write_text("selectors:\n- name: daily\n  definition: tag:daily\n", encoding="utf-8")
    state = project / "state snapshot"
    state.mkdir()
    with pytest.raises(DbtError, match="manifest.json"):
        _plan(project, "build", model=False, named_selector="daily", state_path=state.name, defer=True)
    (state / "manifest.json").write_text(json.dumps({"metadata": {"dbt_schema_version": _MANIFEST_SCHEMA}, "nodes": {}}),
                                        encoding="utf-8")
    run = _plan(project, "build", model=False, named_selector="daily", state_path=state.name, defer=True)
    assert run.root == project
    assert run.args[run.args.index("--selector") + 1] == "daily"
    assert run.args[run.args.index("--state") + 1] == str(state)
    assert "--select" not in run.args
    assert "--defer" in run.args
    run.artifact_dir.mkdir(parents=True)
    with pytest.raises(DbtError, match="different directories"):
        _plan(project, "build", model=False, named_selector="daily", state_path=str(run.artifact_dir), defer=True)


@pytest.mark.parametrize("case", ["untouched", "same-invocation-spoof", "stale-source", "valid"])
def test_compile_only_returns_current_artifacts(project, monkeypatch, case):
    run = _plan(project, "compile")
    _artifacts(run, invocation=_CURRENT if case == "same-invocation-spoof" else _OLD)

    async def execute(plan, args, events):
        _event(events, "MainReportVersion", "A001")
        if case in {"stale-source", "valid"}:
            _artifacts(plan, raw="select 9" if case == "stale-source" else _SQL.strip())
        return 0

    monkeypatch.setattr(runtime, "_execute", execute)
    result = asyncio.run(run.execute())
    if case != "valid":
        assert not result.compiled_sql
        assert not result.artifact_revision
    else:
        assert result.compiled_sql == "select 1 as compiled"
        assert runtime.dbt_artifact_revision(run) == result.artifact_revision
        assert runtime.dbt_artifact_current(run, result.artifact_revision)
        (project / "profiles.yml").unlink()
        assert not runtime.dbt_artifact_current(run, result.artifact_revision)
        assert runtime.dbt_artifact_revision(run) == ()


def test_build_failure_links_to_the_yaml_test_without_echoing_error(project, monkeypatch):
    properties = "models/schema.yml"
    (project / properties).write_text("models:\n- name: report\n  columns:\n  - name: id\n    data_tests:\n    - not_null\n",
                                      encoding="utf-8")
    run = _plan(project, "build")

    async def execute(plan, args, events):
        _event(events, "MainReportVersion", "A001")
        node = {"unique_id": "test.demo.not_null_report_id.b12d", "resource_type": "test", "package_name": "demo",
                "original_file_path": properties, "attached_node": "model.demo.report", "column_name": "id",
                "test_metadata": {"name": "not_null"}}
        _artifacts(plan, extra=[(node, {"status": "fail", "failures": 3, "message": "password=some-secret"})])
        return 1

    monkeypatch.setattr(runtime, "_execute", execute)
    result = asyncio.run(run.execute())
    assert [(row.path, row.row) for row in result.nodes] == [(_MODEL, None), (properties, 5)]
    assert "3 failures reported" in result.nodes[-1].reason
    assert "some-secret" not in repr(result)


def test_preview_is_literal_and_empty_rows_are_a_verified_result(project, monkeypatch):
    sql = "SELECT '{{ password }}', '{% endraw %}', '💡', '\\x27' -- comment\r\n"
    run = _plan(project, "show", model=False, preview_sql=sql)
    progress = []

    async def execute(plan, args, events):
        template = args[args.index("--inline") + 1]
        assert Environment().from_string(template).render(password="wrong") == sql
        assert Path(args[args.index("--target-path") + 1]) != run.artifact_dir
        assert Path(args[args.index("--log-path") + 1]) != run.log_dir
        assert "--no-write-json" in args
        _event(events, "MainReportVersion", "A001")
        _event(events, "LogStartLine", "Q011", description="some-secret", index="spoofed", total=99)
        _event(events, "ShowNode", "Q041", invocation=_OLD, is_inline=True, node_name="inline_query",
               output_format="json", preview='[{"key":"some-secret"}]')
        _event(events, "ShowNode", "Q041", is_inline=True, node_name="inline_query", output_format="json", preview="[]")
        return 0

    monkeypatch.setattr(runtime, "_execute", execute)
    result = asyncio.run(run.execute(progress.append))
    assert result.preview_available and result.preview == ()
    assert progress == ["dbt Core started."]
    assert "some-secret" not in repr(result)


def test_cancellation_during_spawn_still_reaps_the_child(project, monkeypatch):
    run = _plan(project, "debug", model=False)

    async def cancel():
        starting, release = asyncio.Event(), asyncio.Event()

        class Child:
            stdout = asyncio.StreamReader()
            returncode = None
            killed = False
            reaped = False

            def kill(self):
                self.killed = True
                self.stdout.feed_eof()

            async def wait(self):
                self.reaped = True
                return -1

        child = Child()

        async def spawn(*argv, **kwargs):
            assert argv[:len(run.runtime.argv)] == run.runtime.argv
            assert kwargs["cwd"] == project
            starting.set()
            await release.wait()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(runtime._execute(run, run.args, runtime._Events("debug", 50, None)))
        await starting.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.killed and child.reaped

    asyncio.run(cancel())


def test_event_stream_discards_oversized_and_incomplete_lines(monkeypatch):
    monkeypatch.setattr(runtime, "_EVENT_LIMIT", 8)

    async def lines():
        stream = asyncio.StreamReader()
        stream.feed_data(b"too-long-line\nvalid\npartial")
        stream.feed_eof()
        return [line async for line in runtime._lines(stream)]

    assert asyncio.run(lines()) == [b"valid"]
