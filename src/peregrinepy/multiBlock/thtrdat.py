"""What the kernels know about the species, as arrays on the solver's backend
filled from a Mixture. Every temperature dependence is one polynomial in
ln T, ascending coefficients, zero padded to the widest species of that
quantity."""

import numpy as np

from ..mixture import Ru


# To be removed with composition.
def _padded(polys):
    """Ragged ascending-coefficient lists as one zero-padded array."""
    width = max(len(p) for p in polys)
    out = np.zeros((len(polys), width))
    for row, p in zip(out, polys):
        row[: len(p)] = p
    return out


class thtrdat:
    """Every quantity the mixture put on its species, one array each; a
    quantity no model provided is zeros, and no kernel of that case reads it."""

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

    def __init__(self, mixture, backend):
        sp = list(mixture.species.values())
        self.ns = mixture.ns
        self.Ru = Ru
        self.speciesNames = mixture.speciesNames
        values = {name: [s.get(name, 0.0) for s in sp] for name in self.scalars}
        values |= {
            name: _padded([s.get(name, [0.0]) for s in sp]) for name in self.polys
        }
        # every pair's fit, padded to one width
        rows = [s.get("dij", [[0.0]] * self.ns) for s in sp]
        width = max(len(p) for row in rows for p in row)
        dij = np.zeros((self.ns, self.ns, width))
        for i, row in enumerate(rows):
            for j, p in enumerate(row):
                dij[i, j, : len(p)] = p
        values["dij"] = dij
        for name, array in values.items():
            array = np.atleast_1d(np.asarray(array, np.float64))
            kept = backend.allocate(array.shape, name=name)
            kept.set(array)
            setattr(self, name, kept)
