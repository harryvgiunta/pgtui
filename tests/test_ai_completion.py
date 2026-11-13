"""Inline AI proposals are tool-free, bounded, private and never partially published."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps
import json

import httpx
import pytest

from d8r.ai import client
from d8r.ai.client import AIConfig, AIError
from d8r.ai.sql_completion import AICompletion, CompletionRequest, complete_sql
from d8r.sql import ColumnDef, FnDef, SchemaContext, TableDef
from d8r.tui.session import Session
from tests.test_ai_client import completed, delta, event, stream_response


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


@pytest.fixture
def request_context():
    text = 'SELECT r.la FROM records r'
    cursor = len('SELECT r.la')
    schema = SchemaContext(
        tables=(TableDef('records', columns=(ColumnDef('id', 'integer'),
            ColumnDef('label', 'text', values=('never-send-cached-row',)))),),
        fns=(FnDef('saved', ('n',), "SELECT 'never-send-function-body'", 'Local query'),),
    )
    return CompletionRequest(text, cursor, len('SELECT r.'), cursor, 'sqlite', '3.45.3', schema)


@pytest.fixture
def provider(monkeypatch):
    requests = []
    def install(response):
        def handle(request):
            requests.append(request)
            return response(request) if callable(response) else response
        monkeypatch.setattr(client, 'create_client', lambda config: httpx.AsyncClient(
            transport=httpx.MockTransport(handle), timeout=config.timeout))
        return requests
    return install


CONFIG = AIConfig('https://provider.invalid/v1', 'completion-model', 'completion-private-key', max_attempts=5)


@run_async
async def test_inline_transport_sends_bounded_sql_context_without_values_or_tools(request_context, provider):
    requests = provider(stream_response(completed(json.dumps({'suggestions': [
        {'insert': 'label', 'detail': 'Field from records', 'cursor_back': 0},
    ]}))))
    result = await complete_sql(CONFIG, request_context)
    assert result == (AICompletion('label', 'Field from records', 0),)
    assert len(requests) == 1
    wire = requests[0]
    assert str(wire.url) == 'https://provider.invalid/v1/chat/completions'
    assert wire.headers['authorization'] == 'Bearer completion-private-key'
    body = wire.content.decode()
    assert 'completion-private-key' not in body
    assert 'never-send-cached-row' not in body and 'never-send-function-body' not in body
    payload = json.loads(body)
    assert 'tools' not in payload
    data = json.loads(payload['messages'][-1]['content'])
    assert data['sql_before'] == 'SELECT r.'
    assert data['replace'] == 'la' and data['sql_after'] == ' FROM records r'
    assert data['tables'][0]['columns'][1] == {'name': 'label', 'type': 'text'}
    assert request_context.document == 'SELECT r.la FROM records r'


@run_async
async def test_schema_permission_does_not_send_hidden_names(request_context, provider):
    hidden = SchemaContext(tables=(TableDef('unshared_table', columns=(ColumnDef('private_field','text'),)),),
                           fns=(FnDef('unshared_function', body='SELECT 1'),))
    request = replace(request_context, schema=hidden, allow_schema=False)
    requests = provider(stream_response(completed('{"suggestions":[]}')))
    assert await complete_sql(CONFIG, request) == ()
    body = requests[0].content.decode()
    assert 'unshared_table' not in body and 'private_field' not in body and 'unshared_function' not in body
    data = json.loads(json.loads(body)['messages'][-1]['content'])
    assert data['schema_shared'] is False and 'tables' not in data


@pytest.mark.parametrize('answer', [
    '{"suggestions":[{"insert":"label","start":0}]}',
    '{"suggestions":[{"insert":"label","cursor_back":true}]}',
    '{"suggestions":[{"insert":"label","cursor_back":6}]}',
    '{"suggestions":[{"insert":"label","cursor_back":-1}]}',
    '{"suggestions":[{"insert":"label","detail":"secret\\u001b[2J"}]}',
    '{"suggestions":[{"insert":"\\u0000"}]}',
    '{"suggestions":[{"insert":"label"},{"insert":"bad","cursor_back":9}]}',
    '{"suggestions":[],"suggestions":[{"insert":"label"}]}',
    '{"suggestions":[{"insert":"label","cursor_back":NaN}]}',
    '{"suggestions":[{"insert":"   "}]}',
    json.dumps({'suggestions':[{'insert':'x'}]*4}),
    json.dumps({'suggestions':[{'insert':'x'*2049}]}),
])
@run_async
async def test_invalid_completion_never_returns_a_partial_edit(request_context, provider, answer):
    provider(stream_response(completed(answer)))
    with pytest.raises(AIError):
        await complete_sql(CONFIG, request_context)


@run_async
async def test_complete_duplicates_are_deduplicated_and_multiline_caret_is_bounded(request_context, provider):
    value = {'suggestions':[
        {'insert':'SUM(\n)', 'cursor_back':1},
        {'insert':'SUM(\n)', 'cursor_back':1},
        {'insert':'COUNT(*)', 'detail':'Aggregate'},
    ]}
    provider(stream_response(completed('```json\n'+json.dumps(value)+'\n```')))
    result = await complete_sql(CONFIG, request_context)
    assert result == (AICompletion('SUM(\n)', '', 1), AICompletion('COUNT(*)','Aggregate',0))


@run_async
async def test_partial_response_never_becomes_a_suggestion(request_context, provider):
    provider(stream_response(event(delta('{"suggestions":[{"insert":"label"}]}'))))
    with pytest.raises(AIError):
        await complete_sql(CONFIG, request_context)


@run_async
async def test_completion_never_retries_or_exposes_provider_errors(request_context, provider):
    requests = provider(httpx.Response(503, text=CONFIG.api_key))
    with pytest.raises(AIError) as failure:
        await complete_sql(CONFIG, request_context)
    assert CONFIG.api_key not in str(failure.value)
    assert len(requests) == 1


@run_async
async def test_cancellation_closes_provider_stream(request_context, provider):
    started = asyncio.Event()
    closed = asyncio.Event()
    class Pending(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b''
        async def aclose(self):
            closed.set()
    provider(httpx.Response(200, headers={'Content-Type':'text/event-stream'}, stream=Pending()))
    task = asyncio.create_task(complete_sql(CONFIG, request_context))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


@run_async
async def test_completion_deadline_bounds_idle_stream(request_context, provider):
    class Pending(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b''
    provider(httpx.Response(200, headers={'Content-Type':'text/event-stream'}, stream=Pending()))
    with pytest.raises(AIError):
        await asyncio.wait_for(complete_sql(replace(CONFIG, timeout=0.03), request_context), 2)


@run_async
async def test_bad_target_is_rejected_before_network(request_context, monkeypatch):
    def forbidden(config):
        pytest.fail('Invalid completion target reached the provider')
    monkeypatch.setattr(client, 'create_client', forbidden)
    with pytest.raises(AIError):
        await complete_sql(CONFIG, replace(request_context, end=len(request_context.document)+1))


def test_ai_completion_opt_in_persists_without_enabling_execution_or_autoapply():
    session = Session()
    assert not session.ai_completion
    session.update_settings(ai_completion=True)
    restored = Session()
    assert restored.ai_completion
    assert not restored.ai_auto_accept and not restored.ai_test_queries and not restored.ai_allow_sample_data
    before = session.settings_path.read_bytes()
    with pytest.raises(ValueError):
        session.update_settings(ai_completion='yes')
    assert session.settings_path.read_bytes() == before
    session.update_settings(ai_completion=False)
    assert not Session().ai_completion


@run_async
async def test_oversized_schema_drops_whole_records_without_inventing_names(request_context, provider):
    from d8r.ai.sql_completion import MAX_REQUEST_BYTES
    schema = SchemaContext(tables=tuple(TableDef('table_' + str(index), columns=tuple(
        ColumnDef('field_' + str(column) + '_' + 'n' * 160, 'text') for column in range(80)))
        for index in range(24)))
    requests = provider(stream_response(completed('{"suggestions":[]}')))
    await complete_sql(CONFIG, replace(request_context, schema=schema))
    data_text = json.loads(requests[0].content)['messages'][-1]['content']
    assert len(data_text.encode('utf-8')) <= MAX_REQUEST_BYTES
    data = json.loads(data_text)
    assert data['tables_truncated']
    for table in data['tables']:
        assert all(column['name'].endswith('n' * 160) for column in table['columns'])


@run_async
async def test_provider_echoed_key_is_not_exposed_as_an_edit(request_context, provider):
    provider(stream_response(completed(json.dumps({'suggestions':[{'insert':CONFIG.api_key}]}))))
    with pytest.raises(AIError) as failure:
        await complete_sql(CONFIG, request_context)
    assert CONFIG.api_key not in str(failure.value)
