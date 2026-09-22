"""A case's gas, the physical constants it is expressed in (SI with kmol), and
the one word every model's requirements are written with."""

from functools import cache
from pathlib import Path

import yaml

# J/kmol/K
Ru = 8314.46261815324
# J/K
kB = 1.380649e-23
# 1/kmol
avogadro = 6.02214076e26
# F/m
epsilon0 = 8.854187812773345e-12
# m/s, exact
speedOfLight = 299792458.0
# C.m, as cantera defines it; the rounded 3.33564e-30 is 3e-7 off
debye = 1.0e-21 / speedOfLight
# K, the standard state every formation enthalpy and entropy is at
Tref = 298.15


class anyOf(tuple):
    """A requirement satisfied by whichever of its members is present."""


@cache
def database(name):
    """A file from the PEREGRINE database, read once."""
    with open(Path(__file__).parent / "database" / f"{name}.yaml") as f:
        # libyaml where the build has it, several times faster
        return yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


from .diffusionModel import BaseSpeciesDiffusionModel  # noqa: E402
from .eosModel import BaseEosModel  # noqa: E402
from .mixture import Mixture, ReactingMixture  # noqa: E402
from .species import Species  # noqa: E402
from .transportModel import BaseTransportModel  # noqa: E402

__all__ = [
    "database",
    "BaseEosModel",
    "BaseSpeciesDiffusionModel",
    "BaseTransportModel",
    "Mixture",
    "Species",
    "Ru",
    "avogadro",
    "debye",
    "epsilon0",
    "kB",
    "Tref",
]


def getMixture(config, root=None):
    """Makes the case's mixture from its mixture section: reacting when the
    chemistry section names a source."""
    kind = ReactingMixture if config["chemistry"]["source"] else Mixture
    return kind(config["mixture"], root)
