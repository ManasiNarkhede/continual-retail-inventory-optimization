"""Helper factories for Avalanche plugin composition."""

from __future__ import annotations

from typing import Any


def build_plugin_config(plugin_names: list[str]) -> list[str]:
    """Return normalized plugin names from configuration.

    Args:
        plugin_names: Plugin names defined in YAML.

    Returns:
        Normalized plugin name list.
    """
    return [str(name).strip() for name in plugin_names]


def plugin_summary(plugin_names: list[str]) -> dict[str, Any]:
    """Generate a lightweight plugin summary dictionary.

    Args:
        plugin_names: Configured plugin names.

    Returns:
        Summary metadata for logging.
    """
    normalized = build_plugin_config(plugin_names)
    return {"count": len(normalized), "plugins": normalized}
