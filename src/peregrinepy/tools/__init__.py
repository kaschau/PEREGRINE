"""The peregrine command's tools, one module each: its name, a help line,
what it takes on the command line, and what it does. Every input and
output is an explicit file."""

from . import (
    analyze,
    channel,
    config,
    gridpro2pg,
    icem2pg,
    interpolate,
    partition,
    rotate,
    run,
    verify,
)

tools = (
    run,
    config,
    partition,
    verify,
    analyze,
    interpolate,
    gridpro2pg,
    icem2pg,
    channel,
    rotate,
)

__all__ = ["tools"]
