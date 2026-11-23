"""Tool-free assistant turn for the structured dbt setup interview."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace

from d8r.ai import client
from d8r.dbt_setup import CHECKS, MATERIALIZATIONS, TARGETS


WORKLOADS = {"explore": "Explore current input", "refresh": "Rebuild snapshots periodically"}

_GUIDE = """Guide the user through the models and tests step of a dbt setup interview.
They are building a staging model and a row-count model. Recommend views for reading
current data, or tables for a snapshot refreshed by dbt builds. The summary check
makes no assumptions about upstream rows. Recommend the additional nonempty check
only if the user explicitly expects rows. Cloudflare D1 means remote D1 with SQLite
SQL; do not assume there is a compatible adapter or recommend a local substitute.
Return only a JSON object with exactly two fields: "materialization" ("view" or
"table") and "check" ("summary" or "nonempty"). Do not ask for credentials or
return SQL, files, commands, profile settings, paths or prose."""


@dataclass(frozen=True)
class SetupAdvice:
    materialization: str
    check: str


async def recommend_setup(config: client.AIConfig, target: str, workload: str, expects_rows: bool) -> SetupAdvice:
    """Send only categorical answers, not paths, names, metadata or credentials."""
    config.validate()
    if target not in TARGETS or workload not in WORKLOADS or type(expects_rows) is not bool:
        raise ValueError("Invalid dbt setup interview state.")
    messages = [{"role": "system", "content": _GUIDE}, {"role": "user", "content": json.dumps({
        "target": target, "workload": workload, "expects_rows": expects_rows,
    })}]
    bounded = replace(config, max_attempts=1, max_tool_rounds=1, max_tool_calls=1,
                      timeout=min(config.timeout, 15.0))

    async def no_tools(name, arguments):
        raise client.AIError("dbt setup guidance cannot invoke tools.")

    size = 0
    async with asyncio.timeout(bounded.timeout):
        async for event in client.run_turn(bounded, messages, [], no_tools):
            if event.kind == "text":
                size += len(event.text)
                if size > 1024:
                    raise client.AIError("The dbt setup response was too large.")
    try:
        text = messages[-1]["content"].strip()
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0] == "```json" and lines[-1] == "```":
            text = "\n".join(lines[1:-1])
        result = json.loads(text)
        if (messages[-1]["role"] == "assistant" and isinstance(result, dict)
                and result.keys() == {"materialization", "check"}):
            materialization, check = result["materialization"], result["check"]
            if (isinstance(materialization, str) and materialization in MATERIALIZATIONS
                    and isinstance(check, str) and check in CHECKS and (check != "nonempty" or expects_rows)):
                return SetupAdvice(materialization, check)
    except (TypeError, ValueError, RecursionError):
        pass
    raise client.AIError("The provider did not return supported dbt setup choices.")
