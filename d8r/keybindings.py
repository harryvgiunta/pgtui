"""Persistent key IDs and one-chord validation for the IntelliSense keymap."""
from __future__ import annotations

import re

from textual.keys import KEY_ALIASES, key_to_character

DEFAULT_COMPLETION_KEYS = {
    "completion_toggle": "escape",
    "completion_previous": "up",
    "completion_next": "down",
    "completion_accept": "enter",
}

_KEY_CHORD = re.compile(r"(?:(?:ctrl|alt|shift|super|meta|hyper)\+){0,6}[a-zA-Z0-9_]{1,48}")
_ALIASES = {alias: key for key, aliases in KEY_ALIASES.items() for alias in aliases}
_KEY_LABELS = {"escape": "Esc", "up": "Up Arrow", "down": "Down Arrow"}


def key_identity(key: str) -> str:
    """Compare terminal aliases without changing the key actually recorded."""
    return _ALIASES.get(key, key)


def key_label(key: str) -> str:
    return _KEY_LABELS.get(key, "+".join(part.replace("_", " ").title() for part in key.split("+")))


def is_typing_key(key: str) -> bool:
    character = key_to_character(key)
    return character is not None and character.isprintable()


def validate_completion_key(key: object, *, action: str) -> None:
    if not isinstance(key, str) or not _KEY_CHORD.fullmatch(key):
        raise ValueError("Choose one key or modified shortcut, such as Ctrl+G.")
    if action == "completion_toggle" and is_typing_key(key):
        raise ValueError("Use a non-typing shortcut to toggle IntelliSense, even when the suggestion list is closed.")
    if key_identity(key) == "escape" and action != "completion_toggle":
        raise ValueError("Esc is reserved for the IntelliSense toggle or cancelling an AI suggestion.")


def completion_keymap(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value.keys() <= DEFAULT_COMPLETION_KEYS.keys():
        raise ValueError("Invalid IntelliSense shortcut IDs.")
    keys = DEFAULT_COMPLETION_KEYS | value
    for action, key in keys.items():
        validate_completion_key(key, action=action)
    if len({key_identity(key) for key in keys.values()}) != len(keys):
        raise ValueError("Choose a different key for each IntelliSense action.")
    return keys
