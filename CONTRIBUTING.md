# Contributing to PGTUI

PGTUI is an early-alpha terminal SQL workspace, built with substantial AI help.
It is free, MIT-licensed, and will stay free. Users, bug reports, documentation fixes,
and focused code contributions are welcome. You do not need to contribute code to help.

## Useful places to start

- Try your normal PostgreSQL, local Parquet, or Cloudflare D1 workflow and report
  where typing, completion, navigation, forms, or results handling gets in the way.
- Improve SQL completion against real query shapes: aliases, joins, CTEs, grouping,
  quoted identifiers, and incomplete SQL.
- Improve keyboard accessibility, terminal layouts, installation instructions,
  and understandable, credential-safe error hints.
- Add a regression for a real user-visible bug that existing tests do not cover.

Check [open issues](https://github.com/harryvgiunta/pgtui/issues) before starting.
For a large change or a new backend, open an issue to discuss it first. The current
scope is deliberately PostgreSQL, local Parquet through DuckDB, and Cloudflare D1.
Do not restore the old query DSL, generic database adapters, or a local SQLite-file source.

## Reporting a bug

Include:

1. What you expected and what happened.
2. Your OS, terminal/multiplexer, Python version, and PGTUI commit.
3. Source type and database version, if relevant.
4. The smallest reproducible steps, including caret position for completion bugs.
5. Synthetic SQL/schema or a redacted screenshot when useful.

Never attach connection passwords, D1 tokens, AI API keys, connection strings with
secrets, private query results, or your entire PGTUI state directory. SQL, page drafts,
history, and chats can contain sensitive information even when credentials are masked.
Use a restricted database role: manual SQL can modify data and schema.

## Local setup

Install Python 3.13+ and [uv](https://docs.astral.sh/uv/), then clone the repository:

```sh
git clone https://github.com/harryvgiunta/pgtui.git
cd pgtui
uv sync --locked
uv run pgtui
```

For a pull request, fork the repository on GitHub and push a branch to your fork.
The commands above also work for trying the application without contributing code.

## Verification

Run the smallest relevant tests while working, then include the commands and results
in your pull request:

```sh
uv run pytest -q
uv build
```

Tests isolate local state and credentials and generate temporary Parquet files.
Set `PGTUI_TEST_POSTGRES_DSN` to an **isolated test database** to include live
PostgreSQL checks; otherwise those checks are skipped. D1 tests use the real HTTP
adapter against a test-only Cloudflare-protocol harness backed by temporary SQLite,
not a live Cloudflare account. Live database or AI-provider checks require your own
explicitly authorized test setup; never use production data or someone else's credentials.

UI changes should exercise the actual Textual surface through `App.run_test()`/Pilot.
Keep regression tests deterministic and focused on behavior, not implementation text.

## Pull requests

- Keep changes focused and explain the user-visible problem and solution.
- Preserve native SQL, source isolation, keyboard behavior, credential safety,
  cancellation, and explicit AI acceptance. Local completion must not query a
  database or provider on each keystroke.
- Use the existing `pyproject.toml`/`uv.lock` dependency setup and repository patterns.
- Update relevant documentation when behavior changes. Include a screenshot or
  short recording for visible UI changes, with all private information removed.
- AI-assisted contributions are welcome. Review what you submit, understand the
  behavior, and report what you actually verified; generated code is not proof.
- Preserve the original MIT copyright and license notices.

The internal Python package is `d8r`; the public application and command are `pgtui`.
