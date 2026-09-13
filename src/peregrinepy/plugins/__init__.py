"""What a run does alongside the stepping, by the name the config gives it."""

from ..misc import subclassWhere
from . import (
    catalyst,
    nanCheck,
    report,
    trace,
    writer,
)  # noqa: F401  (registers the plugins)
from .base import BasePlugin

__all__ = ["BasePlugin", "getPlugin", "pluginsOf"]


def getPlugin(name):
    """The class for a plugin name, which is also the check that it is one."""
    return subclassWhere(BasePlugin, name=name)


def pluginsOf(solver, config):
    """Every plugin the config names, made for this case."""
    return {
        name: getPlugin(name)(solver, sect) for name, sect in config["plugins"].items()
    }
