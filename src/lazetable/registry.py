"""Registration helpers for lazetable runtime extensions."""

from __future__ import annotations

from typing import Any

from .core import DEFAULT_PAYLOAD_MANAGER, DEFAULT_ROW_VIEW_MANAGER


def register_payload_type(obj: Any) -> Any:
    """Decorator: register a payload type instance or class-derived instance."""

    instance = obj() if isinstance(obj, type) else obj
    DEFAULT_PAYLOAD_MANAGER.register(instance)
    return obj


def register_parser(obj: Any) -> Any:
    """Backward-compatible alias for payload-type registration."""

    return register_payload_type(obj)


def register_row_view(obj: Any) -> Any:
    """Decorator: register a row-view instance or class-derived instance."""

    instance = obj() if isinstance(obj, type) else obj
    DEFAULT_ROW_VIEW_MANAGER.register(instance)
    return obj
