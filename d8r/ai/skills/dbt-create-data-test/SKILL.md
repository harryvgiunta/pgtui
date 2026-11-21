---
name: dbt-create-data-test
description: Propose new named singular dbt data tests that return failing rows for the open model.
---

# Create Data Test

1. Use this skill for an open model with a `model_ref`, whether the model is written in SQL or Python. Use the model source and defined columns in `model_properties` as evidence for each assertion. Do not invent columns or business rules, or assume omitted snapshot content. Propose the requested number of distinct, meaningful tests if the evidence supports it; ask for genuinely missing information instead of duplicating assertions.
2. Write each test as one `SELECT` or `WITH` query in the active SQL dialect. Use the exact `model_ref` from the snapshot. Use only literal `ref()`, `source()`, and `config()` calls in Jinja, not Jinja blocks. Return failing rows rather than a scalar count of failures. Each query can be at most 32,000 characters.
3. Set `"tests": ["name", ...]` in the shared dbt proposal metadata, listing at most 30 distinct new test names, unique ignoring case. Prefer short names prefixed with the model name. Use identifiers of at most 128 characters, with only ASCII letters, digits, and underscores, starting with a letter or underscore. Do not supply paths or extensions: PGTUI creates the `.sql` files under the project's configured `test-paths`.
4. Add one complete fenced `sql` block per name in the same order, after any model replacement block.
