# PGTUI

A keyboard-first PostgreSQL SQL workspace in the terminal, forked from anyql/D8R,
with Cloudflare D1 support. DuckDB is retained only for querying a user-selected
local Parquet file. No other backend adapters, bundled datasets, sample queries,
expression compiler, or SQL dialect translator are included.

## Run

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --locked
uv run pgtui
```

A new workspace starts blank and disconnected. **Ctrl+O** opens the source form:

- **PostgreSQL**: host, port, database, user, password, schema, SSL mode.
- **Cloudflare D1**: account ID, database name or UUID, and a masked API token.
  Names resolve to stable database UUIDs through Cloudflare's official API.
- **Local Parquet**: one existing local `.parquet` or `.pq` file, exposed as `data`
  through DuckDB's lazy reader. URLs, network shares and globs are not accepted.

Test and Cancel save nothing. Add registers the connection. Remember stores
PostgreSQL passwords and D1 API tokens in the native OS keyring; turning it off
keeps credentials session-only. Metadata discovery reads catalogs, not table row counts. Startup
reconnects only the selected saved target, without executing or changing its draft.
Missing credentials leave the target disconnected.

## Native SQL

The editor sends PostgreSQL SQL directly through psycopg, D1 SQLite SQL through
Cloudflare's REST API, or DuckDB SQL to the selected Parquet source. The source
determines the language; there is no independent dialect selector. PostgreSQL accepts
native queries, writes, DDL, transactions, temporary tables and server routines.
Statements execute in document order and stop at the first failure. PostgreSQL and
DuckDB support explicit transactions; D1 does not keep transactions across HTTP
requests. A failed statement does not undo earlier committed work.

**Ctrl+K** previews SQL after expanding local saved functions, without connecting
or executing. Offline checks cover lexical boundaries and template expansion only;
the database remains the authority for SQL grammar, names and types. Semicolons
inside strings, comments, dollar-quoted routine bodies, SQL-standard `BEGIN ATOMIC`
bodies and SQLite trigger bodies are not document boundaries. SQLite bracket and
backtick identifiers retain their native meaning.

Default rows controls the **fetched preview**, not an injected SQL LIMIT. Its initial
value is 50; zero uses the 10,000-row preview cap. Authored LIMIT clauses retain
native semantics. Truncated results report more rows available, never an invented
full count. CSV export uses only the captured buffer. Fetch limits do not bound
scan cost; ordinary psycopg cursors may buffer more data internally. Execution has
a 30-second deadline. PostgreSQL AI reads use server-side cursors.

Parquet connections are in-memory and restricted to the selected file. The file
is never modified. External writes, attachments, extension loading and configuration
bypasses are disabled; in-memory tables and native transactions remain available.
Parquet execution uses a 512 MiB DuckDB memory budget with disk spill disabled.

### Cloudflare D1

D1 uses the official [REST raw query API](https://developers.cloudflare.com/api/resources/d1/subresources/database/methods/raw/)
and [native SQLite SQL](https://developers.cloudflare.com/d1/sql-api/sql-statements/).
The source header and AI context show the SQLite version reported by D1. Use a
token scoped to the account/database with **D1 Read** for reads; **D1 Write** is
required for manual writes. Supplying a UUID avoids database-name listing.

Tables, views, generated columns and foreign keys populate the cached schema.
There is no local SQLite snapshot adapter or local fallback. Session transaction
commands, temporary objects and attached databases are refused. D1 also does not
support LATERAL joins or server CREATE FUNCTION; local saved SQL templates remain
available. Each statement is a separate request. Timeout/cancellation can discard
the response but cannot guarantee remote rollback; writes are never retried
automatically. Check the database before manually retrying an uncertain write.

Normal Run sends authored SQL unchanged. Its visible-row cap does not limit the
remote response; the client caps response bodies at 50 MB and operations at 30
seconds. Use an authored LIMIT when needed. Token-free workspace identities retain
drafts after Remember-off use; reconnect prompts for a token without redirecting
the workspace to PostgreSQL.


## Saved SQL functions

The reusable function layer is retained as **source-scoped SQL query templates**.
A definition has a name, ordered parameter names, a SQL body, and a description.
Its body is one SELECT, WITH, VALUES or TABLE query. Parameters use `@name` in
expression positions. A call in FROM or JOIN expands into a lateral subquery on
PostgreSQL/DuckDB or an ordinary derived table on D1. D1 templates cannot reference
outer relations through correlated arguments; no correlation translation is attempted.
Arguments are parenthesized SQL expressions and explicit aliases are preserved.
Quoted text, comments, quoted identifiers and dollar bodies are never substituted.
Cycles, arity mismatches, identifier substitution and excessive expansion are refused.
`WITH ORDINALITY` is not supported on local template calls; schema-qualified native
server function calls are passed through unchanged.

These are local templates, not server-installed routines and not a substitute for
parameter binding on untrusted application input. PostgreSQL supports bound query
parameters; this layer supplies reusable parameterized relations without requiring
server deployment. Native CREATE FUNCTION statements remain ordinary SQL.

The Functions explorer opens the retained editor. Save persists a definition;
Run preview saves and executes its call without adding query history. Apply from
AI in the function form edits the draft only.

## IntelliSense

Local completion is source-scoped and entirely offline. It analyzes the active SQL
statement and query scope rather than listing every catalog column. It understands
aliases and shadowing, nested queries, CTE projections and explicit CTE column lists,
derived tables, local query-function outputs, and legal correlated outer references.
Incomplete quoted identifiers and unrelated unfinished SQL no longer disable the
whole list. Unknown derived names/types stay unknown rather than being invented.

- Prefix, word-boundary and abbreviation matches lead; typo/subsequence matches
  are fallbacks. Relevant types/scope lead source-specific successful-query history.
- Qualified names stay in their scope. Ambiguous columns are qualified; table
  alternatives can insert short collision-free aliases without duplicating an
  alias already present. Mid-token edits preserve the surrounding SQL and undo once.
  Explorer insertions use the same caret scope, rather than an alias from an earlier
  statement; schema-qualified table selections preserve their namespace.
- Declared foreign keys supply JOIN predicates in both directions, including all
  composite-key pairs. USING offers common columns. Projection stars can expand
  into explicit fields without confusing COUNT(*) or multiplication with a wildcard.
- INSERT/UPDATE column lists, output aliases in ordering/grouping, native casts,
  window functions and SQL snippets are contextual rather than a global word list.
- PostgreSQL and DuckDB routine catalogs provide real signatures, return types and
  documentation; D1 uses native SQLite aids. Active-argument help prioritizes useful
  fields without pretending to be a complete SQL type checker. Native catalogs are
  captured on connection and successful schema changes, never while typing.
- PostgreSQL enum/domain labels come from catalog metadata, without reading rows.
  Other predicate values use already-populated in-memory value caches. Completion
  never fetches rows; expand a column in the explorer to inspect/cache values explicitly.

The popup shows kind, type, provenance and documentation. Long details and AI edit
previews scroll with Alt+Up/Down while keyboard focus stays in the editor. Local
completion remains available during a running query. These are editor suggestions,
not server grammar validation or a guarantee that a query will run.

### AI completion

F4 (or Ctrl+Shift+Space) requests a proposal through the existing configured
OpenAI-compatible provider. Settings → AI provider retains **Import yolo key**;
import fills the form, and explicit Save chooses whether to remember it. There is
no hidden provider fallback or automatic key import.

**AI completion while typing** is separately opt-in under AI settings. It waits
750 ms after a real edit, not a caret move, page restore or startup. Local suggestions
do not wait for AI. The request sends a bounded current SQL window and permitted
cached schema/signatures—no database reads, tool calls, query history, unrelated
pages or separate row-value payload. SQL literals can themselves contain sensitive
data. Disabling schema sharing also removes schema context from inline requests.

At most three bounded edits are returned, with a 15-second total deadline and no
automatic retry. Edits, caret movement, source/schema/provider/parameter changes,
page/function switches, Escape and app exit invalidate pending results. Inspect the
preview, then Tab or Enter to accept; an invisible preview cannot be accepted.
Suggestions never apply automatically—even if chat Auto accept is enabled—and never
execute SQL or save a function. For larger replacements, use the existing Ask AI chat.

## Workspace and keys

Schema, Pages, Functions, Results, SQL preview, History and AI/Agents retain their
existing workspace layout. Local IntelliSense is offline; optional inline AI follows
the explicit controls above. Pages,
functions, history and chats are isolated by connection identity. Pane resizing,
result selection, clipboard copying and CSV export remain available.

| Keys | Action |
|---|---|
| Ctrl+Enter / F5 | Run native SQL |
| Ctrl+K / F6 | Preview expanded SQL, execute nothing |
| Ctrl+O | Add PostgreSQL, Cloudflare D1 or local Parquet |
| Ctrl+, | Settings |
| Ctrl+J | AI agent/chat roster |
| Ctrl+R in function mode | Save and run function preview |
| Ctrl+Space | Open local SQL completion |
| F4 / Ctrl+Shift+Space | Request AI completion |
| Alt+Up / Alt+Down | Scroll completion documentation/preview |
| Tab / Enter with completion open | Accept completion |
| Escape in the editor | Dismiss/cancel completion; press again to reopen local offers |
| Ctrl+Q | Return from function mode; otherwise quit |

Backslash entries are **workspace actions only**: settings, functions, pages, pane
visibility and AI. They are never query clauses. The query language is SQL.

## PostgreSQL release and local service

The target is **PostgreSQL 18.7**, the latest stable release announced on
[November 12, 2026](https://www.postgresql.org/about/news/postgresql-187-1712-1616-1520-1425-and-19-rc-2-released-3442/).
PostgreSQL 19 RC2 is not selected. The workspace and AI context report the actual
connected server version rather than assuming every remote server is 18.7.

`compose.yaml` provides an optional, isolated development service on
`127.0.0.1:55434`, with database/user `pgtui` and a dedicated named volume. Set
`PGTUI_POSTGRES_PASSWORD` in a private `.env.postgres`, then run:

```powershell
docker compose --env-file .env.postgres up -d --build
```

The official 18.7 Docker tags were not available at cutover. `postgres.Dockerfile`
builds the official 18.7 source tarball, verifies its published SHA-256, and retains
the digest-pinned official PostgreSQL runtime/entrypoint. No password is baked into
the image and no sample schema is installed. The build context excludes application
files and secrets. Use a restricted database role and verified TLS for remote
production targets; the development role is not a production security policy.

The server and configured extension modules are rebuilt into a clean staging prefix;
the final image does not inherit older PostgreSQL module binaries.

## AI assistance and privacy

Configure an OpenAI-compatible provider in Settings. AI is optional. The assistant
receives native PostgreSQL/D1 SQLite/DuckDB guidance, the actual engine version, the current
draft, and permitted schema/history/function context. Complete proposals use SQL
fences. Manual Apply is required unless Auto accept AI updates is enabled; neither
path executes SQL. Explicitly requested workspace function saves can persist through
the guarded save tool without opening the function form.

Sample Data, Test AI Queries and Auto accept are off by default. PostgreSQL/DuckDB
checks use a bounded read-only transaction; D1 checks inspect server EXPLAIN bytecode
for persistent writes and unsafe operations before executing a bounded outer SELECT.
Malformed or incomplete plans are refused. Checks do not alter drafts/results/history
and refuse an existing user transaction. They are not a sandbox for privileged
installed PostgreSQL functions or extensions; use a least-privileged role. Read
limits are not query-cost guarantees, and D1 checks do not create a remote session.
Streaming, questions, steering, cancellation, source/definition guards and saved
chat history remain available. No app HTTP server or browser is involved.

State is independent of the original application: `~/.pgtui`, overridden by
`PGTUI_DATA_DIR`. PostgreSQL passwords, D1 tokens and AI keys use the native OS keyring;
JSON stores references only. Preferences, source metadata, SQL drafts, functions,
history and chats remain sensitive plaintext. Protect the data directory and
backups. Old anyql configuration is not imported or rewritten. Invalid files and
stale concurrent writers block unsafe saves rather than overwriting data.

## Development

`uv run pytest -q` runs the suite with generated temporary Parquet data and an
isolated credential backend. Set `PGTUI_TEST_POSTGRES_DSN` to an isolated PostgreSQL
database to include live engine checks. No bundled data fixtures or example databases ship.
D1 tests exercise the native HTTP adapter against a Cloudflare-protocol harness
backed by temporary SQLite, not a live Cloudflare account. No tokens are required.
The internal Python package retains its historical `d8r` name; the installed
application command and storage namespace are `pgtui`.

## License

[MIT](LICENSE). Original copyright attribution retained.
