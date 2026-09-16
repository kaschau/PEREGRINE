"""A case's gas: its species and its reactions, collected and completed."""

from pathlib import Path

import numpy as np

from .cantera import CanteraParser
from ..misc import subclassWhere
from . import Ru, anyOf
from .diffusionModel import BaseSpeciesDiffusionModel
from .eosModel import BaseEosModel
from .species import Species
from .transportModel import BaseTransportModel


class Mixture:
    """The gas a case is solving, from its mcPhysics section of the config."""

    def __init__(self, configSect, root=None):
        self.configSect = configSect
        self.eos = subclassWhere(BaseEosModel, name=configSect["eos"])(configSect)
        trans, diffusion = configSect["trans"], configSect["diffusion"]
        self.trans = (
            subclassWhere(BaseTransportModel, name=trans)(configSect) if trans else None
        )
        self.diffusion = (
            subclassWhere(BaseSpeciesDiffusionModel, name=diffusion)(configSect)
            if trans
            else None
        )
        self._checkCombination()

        usersp, self.reactions = self.readMixture(configSect["mixture"], root)
        self.species = Species.build(usersp, self.models)
        self.speciesNames = list(self.species)
        self.ns = len(self.speciesNames)

        for model in self.models:
            if model is not None:
                model.check(self.species)

        # each model stores what it works out on the species, eos first
        self.eos.populateSpeciesData(self.species)
        for model in (self.trans, self.diffusion):
            if model is not None:
                model.populateSpeciesData(self.species)

    # what the kernels read of a species: one value each, or one polynomial
    # in ln T each, ascending coefficients; a quantity no model provided is
    # zeros, and no kernel of that case reads it
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
    polynomials = (
        "cpPoly",
        "hPoly",
        "sPoly",
        "muPoly",
        "kappaPoly",
        "chungA",
        "chungB",
    )

    def tables(self):
        """The species data as the jit bakes it: name -> array. A scalar is
        (ns,). A polynomial table is ragged, every row its own length, as
        (offsets, coefs): offsets[r] .. offsets[r + 1] are row r's
        coefficients in coefs. The rows are the species, and for dij the
        unordered pairs (i <= j) in numpy's triu_indices order, which
        species.hpp's pairIndex reproduces."""
        sp = list(self.species.values())
        tables = {"Ru": np.array(Ru)}
        for name in self.scalars:
            tables[name] = np.array([s.get(name, 0.0) for s in sp], dtype=np.float64)
        # a kernel multiplies by the reciprocal, never divides by MW; Wilke's
        # rule wants MW^(-1/4) per species and, per ordered pair, its constant
        # 1 / sqrt(8 (1 + MW_n / MW_m)), row-major
        MW = tables["MW"]
        tables["MWinv"] = 1.0 / MW
        tables["MWqInv"] = MW**-0.25
        tables["wilke"] = (
            1.0 / np.sqrt(8.0 * (1.0 + MW[:, None] / MW[None, :]))
        ).ravel()
        for name in self.polynomials:
            tables[name] = self._ragged([s.get(name, [0.0]) for s in sp])
        i, j = np.triu_indices(self.ns)
        pairs = [sp[a]["dij"][b] if "dij" in sp[a] else [0.0] for a, b in zip(i, j)]
        tables["dij"] = self._ragged(pairs)
        return tables

    @staticmethod
    def _ragged(rows):
        offsets = np.zeros(len(rows) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(r) for r in rows])
        coefs = np.concatenate([np.asarray(r, dtype=np.float64) for r in rows])
        return offsets, coefs

    @property
    def models(self):
        """The case's choices, in the order they are checked."""
        return (self.eos, self.trans, self.diffusion)

    def _checkCombination(self):
        """Reject a selection whose models do not work alongside each other."""
        selected = [m.name for m in self.models if m is not None]
        for model in self.models:
            for need in model.dependsOn if model is not None else ():
                names = need if isinstance(need, anyOf) else (need,)
                if not any(n in selected for n in names):
                    raise ValueError(
                        f"{model.name} needs {'|'.join(names)}; "
                        f"the case has {', '.join(selected)}"
                    )

    def __repr__(self):
        models = [self.eos.name]
        if self.trans is not None:
            models.append(f"{self.trans.name}/{self.diffusion.name}")
        return (
            f"<Mixture {self.ns} species, {len(self.reactions)} reactions, "
            f"{' + '.join(models)}>"
        )

    @staticmethod
    def readMixture(mixture, root):
        """({species: data}, reactions) from a dict, a list of library names,
        or a Cantera file found in the case's input dir, cwd or the database."""
        if isinstance(mixture, dict):
            return dict(mixture), []
        if isinstance(mixture, (list, tuple)):
            return {sp: {} for sp in mixture}, []

        here = Path(__file__).parent
        locs = [Path(mixture)] if root is None else [Path(root) / mixture]
        locs += [
            Path(mixture),
            here / "database" / "mechanisms" / mixture,
        ]
        for loc in locs:
            if loc.is_file():
                parsed = CanteraParser(str(loc))
                return parsed.species, parsed.reactions
        raise FileNotFoundError(
            f"Cannot find the mixture {mixture!r}. Tried "
            + ", ".join(str(p) for p in locs)
        )
