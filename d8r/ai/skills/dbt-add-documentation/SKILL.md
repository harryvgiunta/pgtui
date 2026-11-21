---
name: dbt-add-documentation
description: Propose adding or revising the open dbt model's description in its properties YAML.
---

# Add Documentation

1. Use this skill for an open model with a `model_ref`, whether the model is written in SQL or Python. Use the available model source, `model_properties`, and user-supplied context to describe what the model represents. Do not invent facts about its inputs, columns, grain, or business purpose.
2. Set `"description"` to the proposed description string in the shared dbt proposal metadata. PGTUI patches only the description for the selected model or model version in properties YAML. Do not provide replacement YAML, choose a path, or propose changes to column descriptions, tags, or other properties.
3. This skill adds no SQL block; any SQL blocks in the same response belong to model editing or data tests.
