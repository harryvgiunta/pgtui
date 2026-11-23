"""Explicit, offline Jinja rendering without a dbt adapter or template loader."""
from __future__ import annotations

from jinja2 import StrictUndefined, TemplateRuntimeError, nodes, pass_context
from jinja2.sandbox import ImmutableSandboxedEnvironment

_MAX_TEMPLATE = 128_000
_MAX_SQL = 2 * 1024 * 1024
_MAX_ITERATIONS = 20_000


class _RenderLimit(TemplateRuntimeError):
    def __init__(self) -> None:
        super().__init__("Jinja render limit exceeded")


class _SqlEnvironment(ImmutableSandboxedEnvironment):
    intercepted_binops = frozenset({"*", "**"})

    def _parse(self, source, name, filename):
        tree = super()._parse(source, name, filename)
        # Bound the work even in nested loops that don't emit any SQL.
        for loop in tree.find_all(nodes.For):
            loop.iter = nodes.Filter(loop.iter, "_pgtui_iter", [], [], None, None, lineno=loop.lineno)
        return tree

    def call_binop(self, context, operator, left, right):
        if operator == "*":
            sequence, count = (right, left) if isinstance(left, int) else (left, right)
            if isinstance(sequence, (str, list, tuple)) and isinstance(count, int):
                if len(sequence) * count > _MAX_SQL:
                    raise _RenderLimit()
        elif isinstance(right, (int, float)) and abs(right) > 1000:
            raise _RenderLimit()
        return super().call_binop(context, operator, left, right)


def render_template(template: str, **context) -> str:
    if len(template) > _MAX_TEMPLATE:
        raise _RenderLimit()
    environment = _SqlEnvironment(
        undefined=StrictUndefined, keep_trailing_newline=True,
        newline_sequence="\r\n" if "\r\n" in template else "\n",
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
    )
    remaining = _MAX_ITERATIONS

    @pass_context
    def bounded_iter(_context, values):
        def iterate():
            nonlocal remaining
            for value in values:
                remaining -= 1
                if remaining < 0:
                    raise _RenderLimit()
                yield value
        return iterate()

    environment.filters["_pgtui_iter"] = bounded_iter
    chunks = []
    size = 0
    for chunk in environment.from_string(template).generate(context):
        size += len(chunk)
        if size > _MAX_SQL:
            raise _RenderLimit()
        chunks.append(chunk)
    return "".join(chunks)
