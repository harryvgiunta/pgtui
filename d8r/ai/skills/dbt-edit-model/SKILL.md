---
name: dbt-edit-model
description: Propose a full SQL replacement for the open dbt model when the user asks to edit it and the entire SQL snapshot is available.
---

# Edit Model

1. Use this skill only for an open `.sql` model with a `model_ref` and `document_truncated` set to `false`. Do not propose to replace a Python model or reconstruct omitted SQL.
2. Edit the snapshot's `document` in the active dialect. Preserve existing `ref()` and `source()` calls, Jinja configuration, and behavior unrelated to the requested change.
3. Set `"edit_model": true` in the shared dbt proposal metadata. Put the complete, nonempty replacement in the first fenced `sql` block, before any test SQL. Do not return a patch or use placeholders for unchanged code.
