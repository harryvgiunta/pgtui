# PGTUI

A keyboard-first SQL workspace for people who already live in a terminal multiplexer.
Keep your data-engineering work in a pane alongside tools such as **oh-my-pi** or
**OpenCode**, instead of opening a separate desktop database client. Whether your
multiplexer is Herdr or something else, the goal is to stay in your terminal.

This is an **early-alpha, AI-built project**—"AI slop," affectionately. It has rough
edges, and people trying it, reporting bugs, and contributing are very welcome.

**Free forever. Open source. MIT-licensed.** There is no paid PGTUI tier. AI is
optional; ordinary SQL editing and execution do not require an AI provider.
External AI providers and hosted database services may have their own charges.

The current scope is **PostgreSQL, local Parquet through DuckDB, and Cloudflare D1**:
the three sources the maintainer uses daily. No custom query language or SQL
translation layer, and no browser application.

Already available:
- Native SQL editing with alias completion, foreign-key JOIN suggestions, and full
  GROUP BY proposals from known projections.
- Schema browsing, source-scoped query pages and reusable SQL functions, results
  selection, and CSV export.
- Optional AI chat and inline SQL proposals, with explicit acceptance controls.

Forked from anyql/D8R; original MIT attribution is retained.

**Want to help?** Try it on your workflow, [report a bug](https://github.com/harryvgiunta/pgtui/issues),
or read the [contributing guide](CONTRIBUTING.md). Small, focused improvements count.

## Run

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/harryvgiunta/pgtui.git
cd pgtui
uv sync --locked
uv run pgtui
```

If the global `pgtui` command was installed with `uv tool`, close running instances
before updating it from this checkout:

```sh
uv tool install --force --editable .
```

The editable installation keeps the global command on the checkout's code. Older
installed builds can reject workspace fields written by a newer checkout; update
the launcher before treating `workspace.json` as corrupt or moving it aside.

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
Missing credentials leave the target disconnected. Use **Cancel connection** or
**F8** to abandon a pending connection; a late connection cannot run the queued SQL.

Run it with a restricted database role while trying the alpha. Manual Run sends
native SQL with your role's permissions, including writes and DDL; it is not a
read-only sandbox. Do not paste credentials or private data into public bug reports.

## Native SQL

The editor sends PostgreSQL SQL directly through psycopg, D1 SQLite SQL through
Cloudflare's REST API, or DuckDB SQL to the selected Parquet source. The source
determines the language; there is no independent dialect selector. PostgreSQL accepts
native queries, writes, DDL, transactions, temporary tables and server routines.
Statements execute in document order and stop at the first failure. PostgreSQL and
DuckDB support explicit transactions; D1 does not keep transactions across HTTP
requests. A failed statement does not undo earlier committed work.

SQL failures include actionable hints for syntax, ambiguous references, missing
objects, grouping, and type/function mismatches. PostgreSQL diagnostics retain a
validated SQLSTATE and character position when available. Hints explain repairs
without displaying raw driver messages, server DETAIL/HINT, or secret-bearing SQL.

**Ctrl+K** previews SQL after expanding local saved functions, without connecting
or executing. Offline checks cover lexical boundaries and template expansion only;
the database remains the authority for SQL grammar, names and types. Semicolons
inside strings, comments, dollar-quoted routine bodies, SQL-standard `BEGIN ATOMIC`
bodies and SQLite trigger bodies are not document boundaries. SQLite bracket and
backtick identifiers retain their native meaning.

**Run SQL** (**Ctrl+Enter** or **F5**) pages reads by default on PostgreSQL,
DuckDB and D1, including reads in a multi-statement script. **Default rows**
is the page size: 50 initially, at most 10,000 (zero uses 10,000). The editor
and history retain your SQL. No `COUNT(*)` is added. PostgreSQL fetches one page
plus one lookahead row through a forward-only cursor, then closes it.
Writes (including modifying CTEs and `RETURNING`) and locking statements run
once, natively. Reads inside your own transaction stay in that transaction and
do not offer navigation. **Shift+F5** requires one read-only statement. Use
**Ctrl+Shift+F5** for an explicit native run, e.g. for a side-effecting function;
a rejected paged read is never automatically retried as a native command.

Before sending SQL on an idle PostgreSQL connection, Run checks that the server
still answers. If the connection has expired, the same Run reconnects to the saved
target and submits the captured SQL once. A lost transaction is not resumed on a
fresh connection. If the connection breaks *after* SQL was sent, check its outcome
before retrying: PGTUI does not replay it.

**Cancel query** or **F8** requests a driver interrupt and stops the remaining
statements in the batch. It does not undo earlier commits or guarantee rollback
on D1. The connection stays busy until the driver finishes.

The pager below each unpinned read result shows **Previous · Page N · Next**.
Click a button or press **Ctrl+Left / Ctrl+Right** with the grid focused. Each
page request re-reads only that statement, read-only, at the
requested position; it does not rerun the script, modify the editor or add history.
Authored `LIMIT` and `OFFSET` still apply. Pages are separate reads, not a frozen
snapshot: use a unique `ORDER BY` for repeatable order; concurrent changes can
still shift rows. Paging bounds the rows fetched, not the cost of sorting,
aggregating or skipping a deep offset. The query deadline is 30 seconds.

**Export → Full query result (rerun)** writes every row returned by the result
tab’s original read-only statement to CSV, starting at the first row regardless
of the displayed page. Your SQL’s `LIMIT` and `OFFSET` still apply. PostgreSQL
and DuckDB fetch the fresh result in batches; D1 reads separate bounded pages,
so concurrent changes can shift rows during export. Full export requires a
pageable result on its original connection, outside a user transaction. Choose
**All buffered rows** or **Selected rows** to export the captured data without
running SQL again. Clipboard copying still uses only the displayed buffer.

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

D1’s raw query API has no cursor. Run checks each read's native `EXPLAIN`
program and requests one bounded page using an outer `LIMIT`/`OFFSET`. Only an
explicit native run bypasses paging for a regular `SELECT`; the raw API may
then return the full result. The 50 MB response and 30-second operation limits
still apply. Token-free workspace identities retain drafts after
Remember-off use; reconnect prompts for a token without redirecting to PostgreSQL.


## Saved SQL functions

The reusable function layer is retained as **source-scoped SQL query templates**.
A definition has a name, ordered parameters, a SQL body, and a description.
Its body is one SELECT, WITH, VALUES or TABLE query. Plain parameters use `@name`
in expression positions. Declare `name:relation` for a qualified schema/table name,
or `name:identifier` for one schema, table or column identifier. Typed arguments
must be SQL identifiers, not string literals or arbitrary SQL; expansion quotes
them safely. Separate identifier parameters can form `@schema.@table`.
A call in FROM or JOIN expands into a lateral subquery on
PostgreSQL/DuckDB or an ordinary derived table on D1. D1 templates cannot reference
outer relations through correlated arguments; no correlation translation is attempted.
Expression arguments are parenthesized and explicit aliases are preserved.
Quoted text, comments, quoted identifiers and dollar bodies are never substituted.
Cycles, arity/type mismatches, untyped identifier substitution and excessive expansion are refused.
`WITH ORDINALITY` is not supported on local template calls; schema-qualified native
server function calls are passed through unchanged.

These are local templates, not server-installed routines and not a substitute for
parameter binding on untrusted application input. PostgreSQL supports bound query
parameters; this layer supplies reusable parameterized relations without requiring
server deployment. Native CREATE FUNCTION statements remain ordinary SQL.

`\fn` (or `\functions`) opens the searchable Functions pane without replacing the editor.
Type to filter names, signatures or descriptions. Down enters the list; Up from its
first row (or an empty list) returns to search without clearing the filter. Enter
inserts the selected call at the original caret. For a new call at the start of a
statement, the picker creates `SELECT * FROM name();`, reusing an existing semicolon.
Inside FROM/JOIN it inserts only the relation call. Scalar-expression and argument
positions are left unchanged rather than inserting an invalid nested call.
Required arguments leave the caret inside the parentheses. Edit selected, New and Use
current query page as function are explicit management actions. Save persists a
definition; Run preview saves and executes its call without adding query history.
The preview's Arguments field takes only the argument list, not the function name or
a complete SQL statement. AI Apply changes the form draft only.

Reconnecting to the same saved source while the picker is open preserves its insertion
target. Changing or reloading the target document, moving its caret, or switching its
page, function draft or source requires reopening Functions before inserting.

In AI chat, typing `\fn` opens a searchable picker for the current source's saved
functions. Filter, press Down, then Enter to replace just the command with `name()`;
the caret lands inside the parentheses when arguments are required. Surrounding
message text and the SQL/function editors stay unchanged. Escape cancels the picker
and returns focus to the query editor without changing the chat draft. A restored
or pasted command does not open the picker by itself;
Enter or Ctrl+Enter activates it instead of sending. Choosing a function only edits
the local chat draft: it never sends a provider request, runs SQL or saves a definition.

### Data-quality library

Functions → **Install quality templates** stores a source-local native SQL library.
Installation never executes SQL and never runs automatically. Identical installations
are idempotent; a conflicting function name cancels the entire install without overwriting it.
Choose a template through the same searchable picker, provide its arguments, inspect
expanded SQL with Ctrl+K, then explicitly Run. In a hand-written query use
`SELECT * FROM name(...)`, not a bare call or a scalar SELECT expression: these are
local relation templates, not server-installed functions.

The `dq_` library covers row counts/deltas; NULL missingness and completeness;
predicate-based validity; distinct, excess duplicate and singleton metrics; min/max/
sum/average and range checks; string lengths, blanks and LIKE patterns; single-column
referential integrity; freshness in seconds; and visible column/schema metadata.
PostgreSQL and DuckDB additionally provide native regex and population/sample variance
and standard deviation. D1 SQLite does not pretend that extension-only functions exist.

Metrics return a `value` column, except `dq_schema_columns`, which returns names, types
and positions. Empty percentage denominators return NULL; duplicate metrics exclude
NULL; blank strings are distinct from missing values. Timestamp, regex, LIKE and type
spellings follow the selected engine. Metadata checks use string-valued schema/table
names and cannot distinguish an absent table from one the role cannot see.

These independently authored queries draw on [Soda's quality-check categories](https://github.com/sodadata/soda-core).
They are not a Soda contract runner or anomaly-detection service. Computation stays
inside PostgreSQL, D1 SQLite or the local DuckDB engine; no full dataset is fetched
into Python, and no server-installed routines or Soda dependency are required.

## dbt workspace

### Create a project with the setup assistant

Choose **Settings → Workspace Settings → New dbt Project with AI**. You do not
need to enable dbt first. Pick PostgreSQL, DuckDB for local Parquet, or remote
Cloudflare D1 with SQLite SQL. Enter a project name and an empty directory, or a
new directory under an existing parent. Reuse a connected source or open the
secure connection form, then choose the upstream relation.

The configured AI provider can recommend views or tables and starter tests. It
receives only the target type and your choices about refreshes and expected
rows—not project names, paths, connection details, or data. You can choose the
settings yourself if AI is unavailable. Enter passwords, tokens, and keys only
in the existing secure forms; the **Remember** options use your OS keyring.

Review the full contents of the source definitions, starter models, data tests,
and profile guidance. **Create & Link** writes the reviewed files, enables dbt
for the selected source, and opens the project's `README.md`. Setup does not
overwrite files or run dbt. PostgreSQL and DuckDB include a credential-free
profile template and instructions for external `dbt debug`, `dbt parse`, and
`dbt build`. D1 includes conditional verification instructions, not a profile:
you need an adapter that explicitly supports remote Cloudflare D1. A local
SQLite database or `dbt-sqlite` is not a substitute.

### Open and edit an existing project

In **Settings → Workspace Settings**, turn on **Enable dbt**, then choose
**Open & Save Project Home** and select the directory containing `dbt_project.yml`.
You can also change the directory from **Project Home** in the dbt explorer.
New source workspaces default to off; existing configured projects stay enabled.
Both settings are per data source. Turning dbt off hides its tools and stops
project-file autosave. Saved file drafts remain readable in **Pages**; ordinary
SQL pages keep their native IntelliSense and execution tools.

- Edits to project files, including SQL models, data tests, and YAML, autosave to
  disk after a short pause or when you switch pages. **Pages** remembers opened files
  and any unsaved drafts. Removing a page never deletes its project file.
- On a SQL model, **Show Page Details** opens its properties YAML at the model's
  entry. If no properties file exists, it opens a new YAML draft alongside the model.
- **Ctrl+click** a literal `source()` or `ref()` in the editor to open the local
  source YAML or model. **F12** follows the reference at the caret when a terminal
  does not forward modified clicks.
- **Shift+F12** (`\dbt-references`) lists statically resolved usages of the
  `ref()`, `source()`, or column at the caret. On a model output name it follows
  that column downstream. **Enter** opens the selected usage at its location.
- The file editor's **Git Diff** shows staged, working-tree, and unsaved changes.
  If a file changed externally, autosave keeps your draft without overwriting
  the file. **Actions → Reload File from Disk** loads the new baseline and leaves
  the previous editor text available via Undo.

### Lineage and editor feedback

- Select **dbt → Lineage** (or `\dbt-lineage`) for the open file's upstream
  dependencies, downstream consumers, and known column origins and uses. Expand
  branches to trace indirect dependencies; **Enter** opens a local resource.
  The graph uses current project files and drafts. An existing manifest can
  supply package metadata and dependencies that static analysis cannot recover.
  Use **Refresh** in the Files tab to pick up external project edits.
- **Problems** lists positioned syntax errors and warnings for the open file.
  The **Lineage** tree also lists them in a **File diagnostics** branch. Both
  panels show project index notes separately from file diagnostics. **Enter**
  on a file diagnostic jumps to the affected text. Long lists are paginated
  rather than capped; the existing draft, token, and scope limits still apply.
- Local warnings include missing or ambiguous columns in known inputs, output
  types that differ from model properties, and suspicious operands. `JOIN`
  checks distinguish different known key types from a key whose type cannot be
  verified. They never infer a mismatch from missing metadata alone.
- Argument counts are checked against matching cached overloads, including
  defaults and variadic arguments where the catalog exposes them. Selected
  built-in signatures are used when catalog entries are unavailable.
- PostgreSQL and DuckDB flag ungrouped columns. D1 uses SQLite affinity and
  coercion rules and warns when an ungrouped value may come from any row in its
  group. SQLite's single `min()`/`max()` extremum idiom is left alone. These
  draft checks neither validate against the warehouse nor run dbt.
- The model editor suggests `ref()`, `source()`, and macro names while typing
  Jinja. **Ctrl+Space** requests suggestions; **Tab** accepts the selection.
- Hover over a relation, column, or `SELECT *`, or press **F9** at the caret,
  to inspect known schema and types. **Shift+F9** expands `*` in the draft only
  when the complete projection is known. YAML can document a partial schema;
  unknown columns and types are not invented.

### Rename a model or output column

Open the model and press **F2** or choose **Rename Model**. **Preview Rename**
lists the file move and local reference and metadata edits, including supported
properties and selectors. **Apply Rename** rechecks the project before writing.
Versioned models, unresolved dynamic `ref()` calls, and unsaved dbt buffers that
could affect the result block the rename rather than risking a partial refactor.
Rename does not alter package files, run dbt, or change warehouse relations.

For a SQL model's output column, put the caret on its name and press **Shift+F2**
(`\dbt-rename-column`). Preview shows the definition, resolved downstream uses,
and supported YAML property edits without moving the file. Inherited output
names propagate through downstream projections; explicit aliases keep their
own names. Ambiguous bindings, name collisions, and unsupported column-bearing
metadata block the operation rather than leaving a partial rename.

Indexing and both rename commands honor `.dbtignore` and project-local
`.gitignore` rules (including nested `.gitignore` files). The rules apply
regardless of Git tracking status. Apply rechecks ignore rules and the input
inventory before writing.

### SQL ↔ dbt switch

Turn the editor switch to **SQL** to resolve literal `ref()`, `source()`, and `config()`
calls, or to **dbt** to normalize a mixed draft: resolve it to SQL first, then turn
mapped relations into dbt calls. `\dbt` flips the switch via the command palette.
Switching views is undoable and does not itself run or save the file. Model
mappings use the project's manifest; literal sources also work directly from YAML.
Dynamic Jinja and unmapped relations report an error instead of guessing. Edits to
a templated model in the native SQL preview autosave after switching back to dbt.

### Assistant skills

With your AI provider configured, open a model and choose **Ask AI**. The assistant
uses three bundled skills: [Edit Model](d8r/ai/skills/dbt-edit-model/SKILL.md),
[Create Data Test](d8r/ai/skills/dbt-create-data-test/SKILL.md), and
[Add Documentation](d8r/ai/skills/dbt-add-documentation/SKILL.md). For example, ask
“Make me 10 tests for this model.” Review the diff and **Apply dbt drafts** to save
the changes to the project. The assistant does not run models, tests, or dbt.

### Explorer actions

**Create Data Test** opens a manual editor with local SQL and model-column
IntelliSense (**Ctrl+Space**), without a database connection. Manual and AI-created
tests are saved as `.sql` files under the project's `test-paths` (`tests/` by default).
Expand **tests** below the model folders in the dbt explorer and open a file to
see the SQL that returns failing rows. Tests declared in properties YAML remain
in those YAML files.

**Generate & Serve Documentation** takes a port and uses bundled dbt Core
to build the documentation from saved files and serve it at `http://127.0.0.1:<port>/`.
This uses an offline PostgreSQL profile: lineage, descriptions, and tests are
included, but there is no warehouse catalog or query execution. Projects with dbt
package dependencies need their packages already installed. **Stop Documentation
Server** or exiting PGTUI stops the server. Only the documentation artifacts and
configured docs assets are served, not profiles, logs, or the project directory.


## IntelliSense

Local SQL analysis and ranking are source-scoped and offline, using cached metadata.
Bounded predicate-value reads are the explicit exception described below. Completion analyzes the active SQL
statement and query scope rather than listing every catalog column. It understands
aliases and shadowing, nested queries, CTE projections and explicit CTE column lists,
derived tables, local query-function outputs, and legal correlated outer references.
Incomplete quoted identifiers and unrelated unfinished SQL no longer disable the
whole list. Unknown derived names/types stay unknown rather than being invented.

The last successful catalog snapshot is saved per connection identity in
`schema_cache.json`. After disconnect or restart, table-first completion, columns,
foreign keys, PostgreSQL enum labels and native routine signatures use that snapshot
without contacting the database. The source header, completion details and permitted
AI schema context mark it **cached, with its capture time; it may be stale**.
Reconnect replaces the snapshot with freshly discovered metadata, including an
empty catalog if permissions or objects changed. No SQL is executed to restore it.
Use **Schema → Refresh schema** to reload tables, columns and routines after an
external change. This updates the explorer and completion without running the draft.

Only committed catalog metadata is saved. Temporary PostgreSQL objects and routines,
transactions, result rows, distinct-value pools and credentials are excluded. For
local Parquet, only the selected file's `data` schema survives restart, not in-memory
tables/macros. Cache-save errors preserve live metadata and never relabel a successful
connection or SQL statement as failed. A first successful connection is required:
older builds did not write a durable snapshot that could be recovered retroactively.

- After a table or alias, `WHERE`, `JOIN` and `LEFT JOIN` lead the list. After a
  joined table, expect `ON`/`USING` instead (except for cross and natural joins).
  Table and column slots still prioritize matching names; typo/subsequence matches
  are fallbacks. `WITH` declarations offer `AS (` after the name and `SELECT` at
  the start of the body or main query.
- In a relation slot or at the start of a PostgreSQL statement, a matching schema
  prefix leads partial table-name matches. Accept the schema to insert `schema.`
  and narrow suggestions to that schema. Accepting a table at the start of a
  statement inserts `SELECT  FROM "schema"."table" AS alias` with the caret
  between SELECT and FROM.
- **Esc** in the query editor pauses SQL suggestions across typing, newlines and
  page changes. Backslash commands still work. Press **Esc** again to resume if
  IntelliSense is enabled in Settings. Neither keypress requests AI completion.
- Column suggestions use their visible relation alias/qualifier, not just ambiguous
  columns. SQL-required bare target/USING columns and projection aliases remain bare.
  FROM/JOIN table acceptance inserts a short collision-free alias, preserving an
  existing alias and avoiding illegal aliases on DML/DDL targets. Mid-token edits undo once.
  Explorer insertions use the same caret scope and preserve schema-qualified names.
  Typing an alias prefix offers the relation and its qualified fields; a dot narrows
  completion to that relation without duplicating the dot or surrounding quotes.
- Declared foreign keys supply JOIN predicates while choosing a joined table and
  entering ON, in both directions and with every composite-key pair. USING offers
  common columns. Projection stars expand without confusing COUNT(*) or multiplication.
- GROUP BY offers the full nonaggregate projection, expanding known `*`/`alias.*`
  fields and retaining complex expressions and unaggregated dependencies of mixed
  aggregate/window expressions. Already-grouped expressions, aliases and ordinals
  are deduplicated. Unresolved stars are not advertised as a complete grouping list.
- INSERT/UPDATE column lists, output aliases in ordering/grouping, native casts,
  window functions and SQL snippets are contextual rather than a global word list.
- PostgreSQL and DuckDB routine catalogs provide real signatures, return types and
  documentation; D1 uses native SQLite aids. Active-argument help prioritizes useful
  fields without pretending to be a complete SQL type checker. Native catalogs are
  captured on connection and successful schema changes, never while typing.
- Saved-function `relation` arguments offer tables and CTEs without adding aliases.
  Subsequent `identifier` arguments offer columns from the supplied preceding
  relations, not the function's output or unrelated query sources. Acceptance inserts
  a single identifier, preserving required quotes; explorer insertion follows the
  same rule. Unknown or ambiguous input relations do not supply guessed columns.
  These suggestions use cached metadata only.
- PostgreSQL enum/domain labels come from catalog metadata, without reading rows.
  For other WHERE values, a real predicate edit or Ctrl+Space requests a bounded,
  read-only `SELECT DISTINCT ... LIMIT ...` in the background if the column is not
  cached. The value cache setting and per-column limit control these reads; values
  remain memory-only and are reused offline within the same session, including
  through an offline source handoff. Restart starts with no cached row values.
  Reconnect, successful SQL execution, settings changes and Clear cache invalidate
  value pools. Startup, document loads and caret-only movement never fetch values.
  Busy connections do not block local suggestions. Pausing IntelliSense, blur, and
  retargeting prevent late reads from reopening a popup; typing within the same
  predicate uses its current literal. Explorer expansion shares the same cache and in-flight read.
  Limits bound returned values, not scan cost.

The popup shows kind, type, provenance and documentation. Long details and AI edit
previews scroll with Alt+Up/Down while keyboard focus stays in the editor. Local
completion remains available during a running query. Typing refreshes are coalesced;
catalog-name decoding, source snapshots and popup layout are reused. Tab flushes
pending local work before accepting an edit. These are editor suggestions, not server
grammar validation or a guarantee that a query will run.

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
page/function switches, Escape, pausing IntelliSense and app exit invalidate pending
results. Inspect the preview, then use the configured accept key (Enter by default)
or Tab. An invisible preview cannot be accepted.
Suggestions never apply automatically—even if chat Auto accept is enabled—and never
execute SQL or save a function. For larger replacements, use the existing Ask AI chat.

## Workspace and keys

Schema, Pages, Functions, Results, SQL preview, History and Conversations retain the
workspace's resizable layout. Metadata IntelliSense remains offline; predicate
value reads and optional inline AI follow the controls above. Pages,
functions, history and chats are isolated by connection identity. Pane resizing,
result selection, clipboard copying and CSV export remain available.
Pages stay in creation order (oldest first), including when opened, edited or
filtered. Up from the first row or an empty list focuses search; Down returns to the list.
Query batches open **Results 1**, **Results 2**, and so on, one tab per statement.
The next run replaces unpinned results. Pin on Results 1 keeps a separate snapshot;
Pin on another result keeps that tab. Pinned tabs are marked ◆.
Drag the vertical dividers to resize Explorer or Conversations. Drag the horizontal
line beside **Query results**, or the title itself, to resize the editor and results.
**Close** or **Ctrl+W** closes the selected result tab, including **Results 1**.
Other tabs stay open, and the next run can show fresh results. **[x]** or **Escape**
with results focused hides the whole output window without discarding its tabs.
If the grid has a selection, Escape clears it first. Use `\results` to reopen the window.
After a successful run, each result tab shows its statement’s row count and timing.
The footer shows `Ready` for a single result or the number of results for a batch.
Hover a button to see its bound keyboard shortcut. New Query preserves the current
draft; connection forms retain field focus and selection across edits and connection work.

In **Settings → Keybindings**, select one of the four IntelliSense actions, press
the new shortcut, then choose **Save**. Changes apply immediately and are restored
at startup. **Default** stages the original key for the selected action. The same
key can be used in different contexts, but overlapping shortcuts are rejected.
Typing keys such as Space can navigate or accept when suggestions are visible;
otherwise they type normally. The IntelliSense toggle requires a non-typing key
because it also works with the suggestion list closed.
Tab remains an alternate accept key. Esc still dismisses inline AI after the
IntelliSense toggle is rebound.

In query and function SQL, **Format SQL** indents CTE bodies, lays out clauses and
join conditions, and puts long SELECT lists on separate lines, including inside
CTEs and subqueries. It formats the selection, or the whole draft if nothing is
selected, in one undo step. Table names have a separate syntax color. **Find**
selects literal matches; Enter / Shift+Enter moves forward / back and wraps.
Fold arrows appear alongside line numbers for multiline sections. Click an arrow
or press **Shift+F7** on its line to collapse or expand that section. **Fold/Expand
All** (**F7**) expands any folds, or collapses the sections when all are expanded.
Folds never leave the editor or change the SQL. Selection endpoints stay visible;
copying or running a selection still includes any folded lines it spans. Find opens
folds that hide a match.

| Keys | Action |
|---|---|
| Ctrl+Enter / F5 | Run SQL; reads are paged by default |
| F8 | Cancel the pending connection or interrupt the running query |
| Ctrl+F in SQL | Find in the current draft; Esc closes Find |
| Ctrl+Shift+F in SQL | Format the selection or whole draft |
| F7 in SQL | Collapse or expand all sections |
| Shift+F7 in SQL | Toggle the section at the caret's line |
| Shift+F5 | Run one guarded read-only statement |
| Ctrl+Shift+F5 | Explicit native execution instead of guarded paging |
| Ctrl+Right / Ctrl+Left in a result grid | Fetch next / previous page |
| Ctrl+K / F6 | Preview expanded SQL, execute nothing |
| Ctrl+N | New query; new conversation when conversation tabs have focus |
| Ctrl+O | Add PostgreSQL, Cloudflare D1 or local Parquet |
| Ctrl+, | Settings |
| Ctrl+J | Focus recent conversation tabs |
| Ctrl+R in function mode | Save and run function preview |
| Ctrl+Space | Open local SQL completion |
| F4 / Ctrl+Shift+Space | Request AI completion |
| Ctrl+A in a text editor or chat composer | Select the entire draft |
| Alt/Option+Left / Right in a text editor or chat composer | Jump by word |
| Alt/Option+Up / Down | Jump by page; scroll completion details instead when they overflow |
| Up Arrow / Down Arrow with completion open (defaults) | Navigate suggestions |
| Tab / Enter with completion open (Enter configurable) | Accept completion |
| Esc in the query editor (default toggle) | Pause or resume IntelliSense without changing the saved setting |
| Escape in Explorer (Schema, Pages or Functions) | Hide Explorer and return to the query editor at the same cursor position |
| Escape in the function editor (default bindings) | Return to the query and previous Explorer tab even with suggestions open; preserve the function draft |
| Escape in chat | Hide Conversations and return to the query editor without discarding the chat draft |
| Ctrl+Q | Return from function mode; otherwise quit |

Backslash entries are **editor actions**, available mid-query as well as at the start:
settings, functions, dbt, pages, pane visibility and AI. Only the command span is
removed; surrounding SQL stays intact. Commands inside SQL strings/comments are not
interpreted. These actions are never query clauses. The query language is SQL.

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

### Optional Pagila test data

For manual testing, load the DVD-rental dataset from
[Pagila v3.1.0](https://github.com/devrimgunduz/pagila/tree/fef9675714cfba1756df4719b5e36075a7ddf90e)
into a separate `pagila` database. This pinned dataset needs no third-party
extensions; the server remains PostgreSQL 18.7. Seeding is explicit, never part of
application startup or the default container image.

On a fresh service, create a `postgres NOLOGIN` role for the dump's object ownership,
then create `pagila` owned by `pgtui`. Download `pagila-schema.sql` and
`pagila-data.sql` from the pinned revision into a temporary directory outside the
checkout. Copy them into the container's `/tmp` directory with
`docker compose --env-file .env.postgres cp`, then import schema and data together:

```powershell
docker compose --env-file .env.postgres exec -T postgres psql -X -U pgtui -d pagila -v ON_ERROR_STOP=1 --single-transaction -f /tmp/pagila-schema.sql -f /tmp/pagila-data.sql
```

Do not also import `pagila-insert-data.sql`; it contains the same data.
Run `ANALYZE` after import.

Connect from PGTUI with host `127.0.0.1`, port `55434`, database `pagila`, user
`pgtui`, schema `public`, SSL mode `disable`, and the password from `.env.postgres`.
The dataset includes 1,000 films, 200 actors, 599 customers, 16,044 rentals,
16,049 payments, views, foreign keys and server routines. Data stays in the
`pgtui_postgres-data` volume across container stops; removing that volume deletes it.

## AI assistance and privacy

Configure OpenAI-compatible providers in **Settings → AI providers → Providers**.
AI is optional. Keep named profiles for internal gateways, hosted providers and
local endpoints. Select a profile to edit it, or choose **New provider**. **Save & use**
persists and activates the profile for chat, inline completion and page details;
Cancel discards form edits. Delete provider requires a second confirmation click.
Switching providers revokes pending responses and unapplied proposals.

Enter the base URL (including `/v1` and any gateway path), API key, and optional
request headers. The headers field is free text: one `Name: value` per line, such
as `x-ntnx-harness-device-id: your-device-token`. Header text is visible while editing.
The API-key field takes the token only and sends `Authorization: Bearer …`.
For a custom Authorization header, leave API key blank. Duplicate header names,
control characters and HTTP routing/framing overrides are rejected.

Then choose **Populate models** to issue `GET /models` using that draft's URL, key
and headers, and select an ID from the dropdown. A full `/chat/completions` URL is
also accepted; discovery uses the corresponding `/models` path. Discovery sends
no SQL, follows no redirects, makes one bounded request, and never runs on startup
or while typing. Cancel or edit the connection fields to discard an in-flight
lookup. Manual model-ID entry remains available when discovery is unsupported.
Listing a model does not prove Chat Completions streaming or tool-call support.

Save stores profile names, URLs, selected model IDs, limits and Remember choices
locally. Remember stores **both API keys and header text only in the OS keyring**;
unchecked credentials remain session-only, including across provider switches and
later preference/function saves. JSON contains only credential references.
Limits can be saved before configuring an endpoint. Locked or unavailable saved
credentials block requests; metadata-only edits preserve their references rather
than silently deleting them. Re-enter missing credentials, or unlock the vault and
restart. Existing single-provider settings load as a Default profile without
rewriting the file until an explicit settings save; secrets never fall back to plaintext.

The assistant receives native PostgreSQL/D1 SQLite/DuckDB guidance, the actual engine version, the current
draft, and permitted schema/history/function context. Complete proposals use SQL
fences. Manual Apply is required unless Auto accept AI updates is enabled; neither
path executes SQL. Explicitly requested workspace function saves can persist through
the guarded save tool without opening the function form.

Schema discovery returns table names and column counts, not the full catalog of
columns. The `schema` tool takes `query` to find tables or `table` to retrieve
typed columns. With `table` set, `query` filters column names. A `next_offset`
can be passed back as `offset` with the same filters to read the next page.
Chat and SQL repair requests send `max_tokens: 8192`. Logs distinguish a provider
stopping at its length limit from PGTUI stopping an oversized response stream.
Neither can make an incomplete response applicable. Inline completion retains
its separate, smaller response budget.

For data questions, chat gets a short list of saved function names for the current
source and is instructed to check their definitions before writing equivalent SQL.
It can search beyond that list by name or description. For example, with the quality
templates installed, “how many extra email entries are duplicates?” calls for
`dq_duplicate_count`; “which emails repeat?” needs a function or query that returns
the addresses instead. If no saved function fits, or you ask for standalone SQL,
chat is instructed to write the query directly. No template installation is needed
for the fallback.

**Test AI Queries is on by default.** Before a chat query draft can be applied,
PGTUI checks the exact final SQL on its source, expanding saved functions first.
It checks every page of a multi-page proposal, even if the model calls no tools.
By default it uses native `EXPLAIN` without `ANALYZE`: a pass means the database
could plan the query, not that it ran successfully. If SQL fails a lexical or
database syntax/type check, PGTUI sends the model the check error and gives it
up to **two repair attempts**. Each corrected draft is checked again. Repairs
cannot call tools or change pages that already passed. Connection and permission
failures are not reasons to rewrite SQL.

If no repair passes, **Apply anyway** copies the last complete query draft into
the editor for manual editing. It never runs it. Failed checks never auto-apply;
invalid proposal formats, stale targets and function-save guards still block Apply.
Turning Test AI Queries off keeps validation offline and disables repairs. Saved
off settings are preserved. Function drafts and inline completions are not run.

**Run read-only test queries** opts into bounded execution instead of planning.
Earlier settings that explicitly enabled execution retain that permission.
Sample Data and auto-accept remain off by default. PostgreSQL/DuckDB checks
use read-only transactions and refuse an existing user transaction. D1 checks
inspect server EXPLAIN bytecode and reject incomplete or unsafe programs; only
execution tests go on to run a bounded outer SELECT. Checks do not update the
editor's results or Run history. Timeouts and row caps do not guarantee cheap
queries or sandbox privileged functions; use a least-privileged database role.

For PostgreSQL, Test AI Queries also enables `explain_query`. It expands saved local
functions and inspects one read-only query using fixed
`EXPLAIN (FORMAT JSON, ANALYZE FALSE)` options. Caller-supplied EXPLAIN/options,
multiple statements, mutations and known unsafe functions are refused. The tool
shares read-only transaction, cancellation and source/permission guards with query
checks, with a deadline of the configured timeout or 30 seconds, whichever is lower.
Plans exceeding 64,000 characters are withheld in full rather than returned as a
partial tree. The driver can buffer the full JSON cell before this output limit;
it is not a peak-memory or database-work guarantee.

Plan scans, joins, row estimates and costs are not measured execution results or
proof that a query will run successfully. ANALYZE stays disabled, but PostgreSQL
planning can evaluate immutable functions. Plans can expose schema details and
SQL literals even when Allow Sample Data is off; no result rows are sampled.
Plan inspection never changes drafts, results, history or query-execution receipts.

Streaming, questions, steering, cancellation, source/definition guards and saved
chat history remain available. No app HTTP server or browser is involved.

Recent conversations appear as closable tabs with status-colored dots. Hover over a
tab to read its full title, target, and status. Ctrl+J focuses the strip; arrows
navigate, Enter selects, and Delete/Ctrl+W closes a tab. Closing or switching tabs retains its draft, history and background request;
History reopens closed chats. Only explicit permanent deletion removes history and
cancels the request. Conversations remain source-scoped.

While the assistant is working, a wave of the theme’s accent color moves across its
status saying. The words follow the actual stage (connection, generation, schema,
history, functions, or validation), not a timer. Validation means offline lexical/template
checks, not server validation. Retry, cancellation, and error messages stay still.

### Future AI — roadmap only

- Dreaming: identify the queries used most often.
- Surface the queries taking the longest.
- Suggest indexes from relevant query evidence.
- Review potentially unused indexes with usage-history/reset caveats.

These are not background jobs or implemented index-management features. No indexes
are created or dropped automatically.

### Local state

State is independent of the original application: `~/.pgtui`, overridden by
`PGTUI_DATA_DIR`. PostgreSQL passwords, D1 tokens and AI keys use the native OS keyring;
JSON stores credential references only. Preferences, source metadata (including cached
catalog names, types, documentation and enum labels), SQL drafts, functions, history
and chats remain sensitive plaintext. Protect the data directory and
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
