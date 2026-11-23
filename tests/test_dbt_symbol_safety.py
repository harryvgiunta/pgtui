"""Regression cases where a static refactor could silently change the wrong binding."""
from __future__ import annotations

import os

import pytest

from d8r.dbt import DbtError
from d8r.dbt_analysis import analyze_file
from d8r.dbt_column_rename import plan_column_rename
from d8r.dbt_files import DbtFiles
from d8r.dbt_index import DbtIndex
from d8r.dbt_rename import apply_rename, plan_rename
from d8r.sql import SchemaContext
from tests.test_dbt_exchange import project, put


def test_column_rename_wont_capture_a_cross_join_binding(tmp_path):
    home = project(tmp_path)
    declaration = 'select 1 as id\n'
    put(home, 'models/base.sql', declaration)
    put(home, 'models/other.sql', 'select 2 as new_id\n')
    consumer = "select id from {{ ref('base') }} cross join {{ ref('other') }}\n"
    path = put(home, 'models/report.sql', consumer)
    with pytest.raises(DbtError, match='binding of .id. could change'):
        plan_column_rename(DbtFiles(home), 'models/base.sql', declaration.index('id'), 'new_id')
    assert path.read_text() == consumer


def test_column_rename_rolls_back_if_a_consumer_cannot_publish(tmp_path, monkeypatch):
    home = project(tmp_path)
    declaration = '\ufeffselect 1 as id\r\n'
    base = put(home, 'models/base.sql', declaration)
    report = put(home, 'models/report.sql', "select id from {{ ref('base') }}\r\n")
    files = DbtFiles(home)
    text = files.read('models/base.sql').text
    plan = plan_column_rename(files, 'models/base.sql', text.index('id'), 'new_id')
    original_report = report.read_bytes()
    replace = os.replace

    def failed_consumer(source, destination):
        if destination == report:
            raise PermissionError('test: consumer is locked')
        replace(source, destination)

    monkeypatch.setattr(os, 'replace', failed_consumer)
    with pytest.raises(DbtError, match='All published changes were rolled back'):
        apply_rename(files, plan)
    assert base.read_bytes() == declaration.encode('utf-8')
    assert report.read_bytes() == original_report


def test_rename_preview_stales_when_nested_gitignore_changes(tmp_path):
    home = project(tmp_path)
    put(home, 'models/base.sql', 'select 1 as id\n')
    nested = "select * from {{ ref('base') }}\n"
    ignored = put(home, 'models/staging/omit.sql', nested)
    kept = put(home, 'models/staging/keep.sql', nested)
    put(home, '.gitignore', '/models/staging/*.sql\n')
    control = put(home, 'models/staging/.gitignore', '!keep.sql\n')
    files = DbtFiles(home)
    index = DbtIndex(files, {})
    assert not index.for_path('models/staging/omit.sql')
    assert index.for_path('models/staging/keep.sql')[0].name == 'keep'
    plan = plan_rename(files, 'models/base.sql', 'renamed')
    assert {edit.path for edit in plan.edits} == {'models/base.sql', 'models/staging/keep.sql'}
    control.write_text('!keep.sql\n!omit.sql\n')
    with pytest.raises(DbtError, match='no longer safe'):
        apply_rename(files, plan)
    assert ignored.read_text() == kept.read_text() == nested


def test_schema_warnings_only_for_complete_inputs(tmp_path):
    home = project(tmp_path)
    put(home, 'models/base.sql', 'select 1 as id\n')
    put(home, 'models/sources.yml', '''sources:
  - name: landing
    tables:
      - name: incomplete
        columns: [{name: id}]
''')
    path = 'models/report.sql'
    text = "\ufeffselect b.missing, p.unknown\r\nfrom {{ ref('base') }} b join {{ source('landing', 'incomplete') }} p on true\r\n"
    put(home, path, text)
    files = DbtFiles(home)
    result = analyze_file(DbtIndex(files, {}), path, files.read(path).text, 'postgres', SchemaContext())
    warning, = result.diagnostics
    assert warning.severity == 'warning'
    assert files.read(path).text[warning.start:warning.end] == 'missing'
