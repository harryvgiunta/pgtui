# PGTUI — repository guide

## Product and boundaries

PGTUI is the PostgreSQL-first Textual terminal IDE forked from anyql/D8R.
The user explicitly replaced the former query DSL and expression compiler with
**native PostgreSQL SQL**, retaining **native DuckDB SQL only for local Parquet**.
The user subsequently authorized **Cloudflare D1 with native SQLite SQL**. These
are the only three source types. Do not restore the old DSL, compiler, generic
adapters, compatibility matrices, demos or a local SQLite-file source.

The internal Python package is `d8r`; the public command, project and storage
namespace are `pgtui`. Preserve original MIT attribution. The source checkout at
`C:/tmp/anyql` is independent and must not be modified by work in this fork.

- The TUI is the application: no browser, HTTP application server or login system.
- PostgreSQL uses real psycopg connections. Native SQL includes DDL, DML, explicit
  transactions, temporary objects and server routines; never translate it.
- Local Parquet uses an in-memory DuckDB with a lazy view named `data`. Accept one
  existing local file only. Do not copy/change it or enable unrestricted external
  access. DuckDB cannot become an alternate network/database adapter.
- Cloudflare D1 uses its official REST API, never a local database fallback.
  Tokens are supplied in the masked source form; no token enters public source
  metadata. Native SQL is unchanged, apart from explicit local function expansion.
  Session transaction commands, temporary objects and attached databases are refused.
- Startup restores local drafts and reconnects only the selected saved target after
  Mount. It never executes SQL, applies AI, saves a function or resumes an AI worker.
- Blank startup is a disconnected PostgreSQL placeholder, not a fake dataset.

## Architecture

- `d8r/sql.py`: immutable schema/function metadata, lossless SQL lexer, statement
  boundaries and local function expansion. This is not a server SQL parser or
  schema/type validator. Keep native PostgreSQL, DuckDB and D1 SQLite syntax untouched
  except explicit local template expansion. Source-bound lexical rules cover comments,
  quoting, dollar bodies, rule actions, BEGIN ATOMIC routines and SQLite trigger bodies.
- `d8r/completion/`: completion-only tolerant scope analysis, immutable proposal
  models, native syntax/signature aids and bounded candidate ranking. Never reuse
  this inferred model to execute or validate runtime SQL. Respect active statement,
  CTE/derived/local-function fields, alias shadowing and legal correlation; unknown
  metadata must remain unknown. Native catalogs are authoritative for routines,
  except native grammar expressions which do not appear in server catalogs.
  Popup and explorer edits share these scope rules and physical table identities;
  do not reintroduce Session's retired global completion lists or alias helpers.
- `d8r/engine/datasources.py`: PostgreSQL connections, safe errors and catalog-only
  metadata, including native types, paired composite foreign keys and routine
  signatures/output columns. Discovery never scans user rows or executes routines.
  Session schema capture must not acquire the execution lock or wait on database I/O.
- `d8r/engine/parquet.py`: validated canonical local file identity and lazy DuckDB
  view, locked external-access configuration.
- `d8r/engine/d1api.py`: fixed-origin Cloudflare HTTP client, canonical database
  identity, metadata-only schema/FKs, bounded responses and read-only EXPLAIN guards.
  No automatic retries; cancellation/timeout cannot guarantee remote rollback.
- `d8r/engine/execute.py`: one native statement, bounded Python result buffers,
  deadlines, cancellation, read-only tools, quoted registered-table sampling.
- `d8r/tui/session.py`: source-scoped functions/history/workspace/schema/cache,
  native Run and offline expanded-SQL preview. Successful DDL refreshes native
  metadata in the worker; metadata/autosave failure must not relabel executed SQL
  as failed. Earlier successful statements are not implicitly rolled back.
- `d8r/tui/app.py` and widgets: retained workspace/pages/results/functions/AI,
  consuming Session rather than reimplementing SQL semantics. Completion uses
  cached metadata only; no synchronous I/O per keystroke.
- `d8r/ai/`: bounded OpenAI-compatible client, native SQL context and guarded tools.
  `ai/sql_completion.py` is a separate, tool-free inline proposal request using the
  same configured transport. It sends no sampled/cache values or unrelated pages.
- `d8r/storage.py`, `d8r/credentials.py`: atomic durable JSON plus OS-native keyring.

There is no independent compilation dialect: the source determines SQL. Session
`compile()` returns a RunOutcome containing expanded native SQL without execution.
Offline lexical success must never be described as complete server validation.
Default rows caps fetched preview rows, not an injected LIMIT. Buffer at most
10,000 visible rows and report truncation without inventing a full result count.
Normal PostgreSQL cursors may buffer additional driver data; do not claim scan or
memory bounds stronger than the exercised behavior.

## Functions

Keep the local reusable function layer. Definitions are source-scoped FnDef
records (name, ordered params, SQL body, description). Bodies are one native
SELECT/WITH/VALUES/TABLE query. `@parameter` tokens occupy expression positions,
not identifiers. Relation calls expand into lateral subqueries on PostgreSQL/DuckDB
and ordinary subqueries on D1. Do not emulate correlation or inject LATERAL on D1.
Parenthesize arguments and preserve aliases. Quoted text and comments are opaque.
Expression precedence, recursion/depth/count bounds and case-insensitive name
collisions are consumer-visible contracts.

These templates are not server-installed functions, do not emulate bound application
parameters, and must not shadow schema-qualified native functions. PostgreSQL supports
parameter binding; the local layer supplies reusable parameterized relations without
server deployment. Native CREATE FUNCTION remains ordinary PostgreSQL SQL. Local
WITH ORDINALITY calls are explicitly unsupported, not silently approximated.

Save persists a definition. Function Run preview saves and runs its call without
history. AI form Apply modifies only a draft; it never executes or persists a definition.

## AI and credentials

Preserve streaming, cancellation/steering, questions, agent/chat roster, source and
request-generation guards, immutable definition baselines, manual Apply, optional
auto-accept, and explicit authorized workspace function saves. Proposals use SQL
fences; explanatory text never becomes an applicable proposal. The assistant knows
PostgreSQL target 18.7 and the actual source.server_version, or the actual DuckDB/D1
SQLite version for the selected source. No cross-dialect translation tools.

Sample data, test-query execution and auto-accept are opt-in. PostgreSQL/DuckDB reads
use native read-only transactions. D1 prepares the exact query using server EXPLAIN,
rejects persistent writes/unsafe operations and incomplete plans, then runs a bounded
outer SELECT. Ephemeral sorter/CTE writes must not be confused with persistent writes.
All tool reads have row/deadline bounds; prefix regexes alone do not prove read-only.
PostgreSQL tools additionally reject known unsafe file/network/control functions,
but privileged installed functions/extensions remain outside a true sandbox.
DuckDB external access stays locked to the selected Parquet file. Refuse AI reads
inside user transactions. Apply/Save never execute SQL. Cancellation retains shared
connection ownership until real background work finishes; late results cannot
publish into replaced sources or new requests.

Inline completion is always manually accepted, independent of chat auto-accept.
F4/Ctrl+Shift+Space explicitly requests it; persisted `ai_completion` is off by
default and opts into a 750ms real-edit debounce. Restores, programmatic document
loads, caret-only moves and startup never initiate network traffic. Keep the local
path offline and responsive even when the connection is busy. Bound requests,
responses and latency; model output cannot choose its replacement span or invoke
tools. Respect schema-sharing permission; current SQL can contain sensitive literals.

Before displaying or accepting AI, match request generation, document/caret/load
revision, source object, schema/functions/parameters, provider/settings and stable
page/function identity. Escape/blur/exit/retarget revoke results even if a provider
ignores cancellation. Never accept a hidden AI preview. Maintain absolute edit spans,
quote/suffix preservation, existing-call parenthesis behavior and single-step undo.
Signature help and scrollable details must not steal focus; Alt+Up/Down scroll them.
Keep arbitrary model/driver errors and provider credentials out of status surfaces.

Configuration is isolated under `~/.pgtui` or `PGTUI_DATA_DIR`; explicit Session
`data_dir` wins. Do not load/migrate original anyql files. PostgreSQL passwords,
D1 tokens and AI keys persist only in the OS-native keyring; JSON holds references.
Remember-off remains session-only across later preference/function saves. Never
log credentials or raw secret-bearing driver exceptions. Locked/unavailable vaults
preserve metadata and block missing-secret use, with no plaintext fallback.
Current version-2 PostgreSQL-only memory files load with an empty D1 profile list;
this additive change must not rewrite files on load or lose existing credential refs.
Remember-off D1 workspaces retain their public account/UUID identity and reopen the
D1 token form after restart, never a PostgreSQL form or an automatic network request.

Pages, function drafts, histories, chats and connection metadata are plaintext and
source-scoped. Runtime rows/transactions/temp objects never persist. Preserve atomic
replacement, stale-writer detection, failed-save in-memory drafts and corrupt-file
protection. Tests must isolate both data home and the credential backend.

## PostgreSQL runtime

Latest stable target at cutover: **18.7**, announced 2026-11-12; 19 RC2 is not stable.
`postgres.Dockerfile` builds the official checksummed 18.7 source on a digest-pinned
official runtime because the new official version tags were not yet published.
Keep the source version/checksum, engine constant and AI guidance consistent.
Rebuild the configured server modules into a clean staging prefix and replace the
runtime module directories; never retain older PostgreSQL extension binaries.
The final container explicitly listens on all container interfaces; Compose exposes
only `127.0.0.1:55434`. PostgreSQL 18's volume mount is `/var/lib/postgresql`.
No seed schema/data. `.env.postgres` is private; build context excludes secrets.

## Commands and verification

- `uv sync --locked`
- `uv run pgtui`
- `uv run pytest -q`
- `uv build`
- `docker compose --env-file .env.postgres up -d --build` (optional local database)

Use the project `.venv`, dependency declarations in `pyproject.toml` and `uv.lock`.
Do not add alternate dependency records, linters, formatters, browser automation
or a server. Real UI verification uses Textual `App.run_test()`/Pilot against real
native engine sources. PostgreSQL tests accept `PGTUI_TEST_POSTGRES_DSN` for an
isolated test database. Generate tiny temporary Parquet files at runtime; never
bundle production data, synthetic demo databases or SQL examples.
D1 tests use the actual HTTP adapter with a test-only Cloudflare envelope harness
backed by temporary SQLite. This is not live Cloudflare verification and must never
be a production connection fallback. Never harvest original application credentials
to run tests; live account access requires supplied authorization.
Live inline-provider smokes require authorization and synthetic context only. The
user authorized the existing yolo endpoint for this task; do not persist its key
or send private database contents as a test. Internal completion quality/latency
measurements are not DataGrip/Cursor comparisons; never claim unmeasured superiority.

Preserve keyboard-first behavior, source identity isolation, draft restoration,
results selection/copy/export, page switching, pane resizing and function editing.
Keybindings are defined by widget Binding lists; derive UI descriptions from them.
Test behavior, transitions and boundaries rather than prompt wording or source text.
