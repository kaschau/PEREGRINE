"""What a run does alongside the stepping, by the name the config gives it."""

from . import (
    catalyst,
    nanCheck,
    report,
    trace,
    writer,
)  # noqa: F401  (registers the plugins)
from ..misc import subclassWhere
from .base import BasePlugin

__all__ = ["BasePlugin", "getPlugins"]


def getPlugins(config, solver):
    """Makes every plugin the config names, by name, for this solver."""
    return {
        name: subclassWhere(BasePlugin, name=name)(solver, sect)
        for name, sect in config["plugins"].items()
    }
