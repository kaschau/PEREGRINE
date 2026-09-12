"""What the kernels know about the species, as device arrays filled from a
Mixture. Every temperature dependence is one polynomial in ln T, ascending
coefficients, zero padded to the widest species of that quantity."""

import numpy as np

from ..abi import DeviceArray
from ..mixture import Ru


def _padded(polys):
    """Ragged ascending-coefficient lists as one zero-padded array."""
    width = max(len(p) for p in polys)
    out = np.zeros((len(polys), width))
    for row, p in zip(out, polys):
        row[: len(p)] = p
    return out


def _device(array):
    array = np.atleast_1d(np.asarray(array, dtype=np.float64))
    out = DeviceArray(array.shape)
    out.set(array)
    return out


class thtrdat:
    """Every quantity the mixture put on its species, one device array each;
    a quantity no model provided is zeros, and no kernel of that case reads it."""

    scalars = (
        "MW",
        "hRef",
        "cp0",
        "mu0",
        "kappa0",
        "lewis",
        "Tcrit",
        "pcrit",
        "Vcrit",
        "acentric",
        "redDipole",
    )
    polys = ("cpPoly", "hPoly", "sPoly", "muPoly", "kappaPoly", "chungA", "chungB")

    def __init__(self, mixture):
        sp = list(mixture.species.values())
        self.ns = mixture.ns
        self.Ru = Ru
        self.speciesNames = mixture.speciesNames
        for name in self.scalars:
            setattr(self, name, _device([s.get(name, 0.0) for s in sp]))
        for name in self.polys:
            setattr(self, name, _device(_padded([s.get(name, [0.0]) for s in sp])))
        # every pair's fit, padded to one width
        rows = [s.get("dij", [[0.0]] * self.ns) for s in sp]
        width = max(len(p) for row in rows for p in row)
        dij = np.zeros((self.ns, self.ns, width))
        for i, row in enumerate(rows):
            for j, p in enumerate(row):
                dij[i, j, : len(p)] = p
        self.dij = _device(dij)
