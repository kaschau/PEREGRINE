"""What a run does alongside the stepping, by the name the config gives it."""

from . import (
    catalyst,
    nanCheck,
    report,
    trace,
    viscousSponge,
    writer,
)  # noqa: F401  (registers the plugins)
from ..misc import subclassWhere
from .base import BasePlugin
from .cadence import Cadence

__all__ = ["BasePlugin", "Cadence", "getPlugins"]


def getPlugins(config):
    """Makes every plugin the config names, by name, from its section."""
    return {
        name: subclassWhere(BasePlugin, name=name)(sect)
        for name, sect in config["plugins"].items()
    }
