"""A case's gas: its species, collected and completed, and for a reacting
case its reactions."""

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
    """The gas a case is solving, from its mixture section of the config:
    its species, with what every model works out on them."""

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
        # how the species' viscosities mix, a switch the jit bakes by name
        self.mixingRule = configSect["mixingRule"]
        self._checkCombination()

        usersp, self.reactions = self.readMixture(configSect, root)
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

    def speciesData(self):
        """Gives the species data as the jit bakes it: name -> array. A scalar is
        (ns,). A polynomial table is (rows, terms), every row padded with
        leading zeros to the table's longest, so a kernel walks it
        contiguously with one trip count; the leading zeros cost Horner
        nothing and change no bit. The rows are the species, and for dij
        the unordered pairs (i <= j) in numpy's triu_indices order, which
        species.hpp's pairIndex reproduces."""
        sp = list(self.species.values())
        tables = {"Ru": np.array(Ru)}
        for name in self.scalars:
            tables[name] = np.array([s.get(name, 0.0) for s in sp], dtype=np.float64)
        # a kernel multiplies by the reciprocal, never divides by MW; Wilke's
        # rule wants MW^(-1/4) per species and, per ordered pair, its constant
        # 1 / sqrt(8 (1 + MW_n / MW_m)), row-major; Herning's wants sqrt(MW)
        MW = tables["MW"]
        tables["MWinv"] = 1.0 / MW
        tables["MWqInv"] = MW**-0.25
        tables["sqrtMW"] = np.sqrt(MW)
        tables["wilkePair"] = (
            1.0 / np.sqrt(8.0 * (1.0 + MW[:, None] / MW[None, :]))
        ).ravel()
        for name in self.polynomials:
            tables[name] = self._padded([s.get(name, [0.0]) for s in sp])
        # g / (Ru T) less hRef / T: h and s are in the one basis, so their
        # difference is one polynomial
        tables["gPoly"] = self._padded(
            [self._difference(s.get("hPoly", [0.0]), s.get("sPoly", [0.0])) for s in sp]
        )
        i, j = np.triu_indices(self.ns)
        pairs = [sp[a]["dij"][b] if "dij" in sp[a] else [0.0] for a, b in zip(i, j)]
        tables["dij"] = self._padded(pairs)
        return tables

    @staticmethod
    def _difference(a, b):
        """a - b of two ascending polynomials of any lengths."""
        n = max(len(a), len(b))
        out = np.zeros(n)
        out[: len(a)] += a
        out[: len(b)] -= b
        return out

    @staticmethod
    def _padded(rows, dtype=np.float64):
        """Rows of coefficients, ascending in the power, as one (rows, terms)
        array: the high powers a row lacks are zero."""
        terms = max((len(r) for r in rows), default=1)
        table = np.zeros((len(rows), max(terms, 1)), dtype=dtype)
        for k, r in enumerate(rows):
            table[k, : len(r)] = r
        return table

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

    def tables(self):
        """Gives what the jit bakes of this mixture, by the header that
        declares the accessors: the species data."""
        return {"species": self.speciesData()}

    def __repr__(self):
        models = [self.eos.name]
        if self.trans is not None:
            models.append(f"{self.trans.name}/{self.diffusion.name}")
        return f"<{type(self).__name__} {self.ns} species, {' + '.join(models)}>"

    @staticmethod
    def readMixture(configSect, root):
        """({species: data}, reactions) from the section's species: a dict, a
        list of library names, or a Cantera file found in the case's input
        dir, cwd or the database."""
        mixture = configSect["species"]
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


class ReactingMixture(Mixture):
    """A mixture with the reactions among its species, as the chemistry
    kernels are composed from them."""

    # what the kernels read of a reaction, by its kind
    reactionTypes = ("elementary", "three-body", "lindemann", "troe")

    def __init__(self, configSect, root=None):
        super().__init__(configSect, root)
        if not self.reactions:
            raise ValueError(f"{configSect['species']} has no reactions to react by")

    def tables(self):
        return {**super().tables(), "reactions": self.reactionData()}

    def __repr__(self):
        return (
            super()
            .__repr__()
            .replace(" species,", f" species, {len(self.reactions)} reactions,")
        )

    def reactionData(self):
        """Gives the reactions as the jit bakes them: name -> array over the
        reactions, in the log-space form the rate kernels evaluate. A
        reaction with no rate (A = 0) is left out. Per reaction: its type
        and reversibility, ln A, b and Ea/Ru of its rate and, for a falloff,
        of k0/kinf; Troe's centering terms resolved to +-exp(c0 + cT T +
        cTinv/T); the forward exponents, the net stoichiometry (and times
        MW), the reverse orders of a reversible reaction and the third-body
        efficiencies' deviations from the default as (species, value) rows
        padded with (0, 0), which cost a rate nothing.
        Mechanism-wide: the concentration floor of an absent species and
        the largest gain from a rate of progress to a production rate."""
        names = self.speciesNames
        MW = np.array([self.species[n]["MW"] for n in names])
        index = {n: k for k, n in enumerate(names)}
        reactions = [r for r in self.reactions if r["rate"][0] != 0.0]
        for r in reactions:
            if r["rate"][0] < 0 or r.get("lowRate", (1.0,))[0] <= 0:
                raise ValueError(
                    f"{r['equation']}: a rate is evaluated in log space, A is positive"
                )
        nr = len(reactions)
        tables = {
            "nr": np.array(nr),
            "logPrefRu": np.array(np.log(101325.0 / Ru)),
            "type": np.array(
                [self.reactionTypes.index(r["type"]) for r in reactions], np.int32
            ),
            "reversible": np.array([int(r["reversible"]) for r in reactions], np.int32),
            "defaultEfficiency": np.array([r["defaultEfficiency"] for r in reactions]),
        }
        rate = np.array([r["rate"] for r in reactions]).reshape(nr, 3)
        tables["logA"], tables["b"], tables["EaR"] = (
            np.log(rate[:, 0]),
            rate[:, 1],
            rate[:, 2],
        )
        # a falloff's pressure ratio k0 / kinf is its own Arrhenius rate
        low = np.array([r.get("lowRate", (1.0, 0.0, 0.0)) for r in reactions]).reshape(
            nr, 3
        )
        tables["logPrA"] = np.log(low[:, 0]) - tables["logA"]
        tables["prB"], tables["prEaR"] = low[:, 1] - rate[:, 1], low[:, 2] - rate[:, 2]
        fcent = [self._fcentTerms(r.get("troe")) for r in reactions]
        tables["fcentTerms"] = np.array([len(t) for t in fcent], np.int32)
        for k, name in enumerate(("fcentSign", "fcentC0", "fcentCT", "fcentCTinv")):
            tables[name] = self._padded([[t[k] for t in terms] for terms in fcent])
        # the (species, value) rows
        fwd = [
            [(index[sp], v) for sp, v in r["fwd"].items() if v != 0] for r in reactions
        ]
        net = []
        for r in reactions:
            nu = {}
            for sp, v in r["reactants"].items():
                nu[sp] = nu.get(sp, 0.0) - v
            for sp, v in r["products"].items():
                nu[sp] = nu.get(sp, 0.0) + v
            net.append([(index[sp], v) for sp, v in nu.items() if v != 0])
        # the reverse orders of a reversible reaction, d ln(reverse) / d ln c:
        # the forward orders plus the net stoichiometry, its products
        rev = []
        for r, f, n in zip(reactions, fwd, net):
            order = {}
            for k, v in f + n if r["reversible"] else []:
                order[k] = order.get(k, 0.0) + v
            rev.append([(k, v) for k, v in order.items() if v != 0])
        eff = [
            [
                (index[sp], e - r["defaultEfficiency"])
                for sp, e in r["efficiencies"].items()
                if e != r["defaultEfficiency"]
            ]
            for r in reactions
        ]
        tables["fwdSpecies"] = self._padded(
            [[k for k, _ in row] for row in fwd], np.int32
        )
        tables["fwdExponent"] = self._padded([[v for _, v in row] for row in fwd])
        tables["netSpecies"] = self._padded(
            [[k for k, _ in row] for row in net], np.int32
        )
        tables["netNu"] = self._padded([[v for _, v in row] for row in net])
        tables["netNuMW"] = self._padded([[v * MW[k] for k, v in row] for row in net])
        tables["revSpecies"] = self._padded(
            [[k for k, _ in row] for row in rev], np.int32
        )
        tables["revExponent"] = self._padded([[v for _, v in row] for row in rev])
        tables["nuTotal"] = np.array([sum(v for _, v in row) for row in net])
        tables["effSpecies"] = self._padded(
            [[k for k, _ in row] for row in eff], np.int32
        )
        tables["effDeviation"] = self._padded([[v for _, v in row] for row in eff])
        # the worst amplification from one rate of progress to a production rate
        gain = np.zeros(self.ns)
        for row in net:
            for k, v in row:
                gain[k] += abs(v)
        tables["maxOmegaGain"] = np.array(max(1.0, (gain * MW).max()) if nr else 1.0)
        tables["logCFloor"] = np.array(self._logCFloor(reactions, net, fwd, index))
        return tables

    @staticmethod
    def _fcentTerms(troe):
        """Troe's Fcent = (1 - a) exp(-T/T3) + a exp(-T/T1) [+ exp(-T2/T)]
        as (sign, c0, cT, cTinv) rows, each +-exp(c0 + cT T + cTinv / T),
        without the terms a mechanism disables by a sentinel: one that is
        zero, or below rounding, at every gas temperature."""
        if troe is None:
            return []
        alpha, T3, T1, T2 = troe
        tmin, tmax = 100.0, 20000.0
        candidates = []
        if alpha != 1.0 and T3 != 0:
            candidates.append(
                (np.sign(1.0 - alpha), np.log(abs(1.0 - alpha)), -1.0 / T3, 0.0)
            )
        if alpha != 0.0 and T1 != 0:
            candidates.append((np.sign(alpha), np.log(abs(alpha)), -1.0 / T1, 0.0))
        if T2:
            candidates.append((1.0, 0.0, 0.0, -T2))
        terms = []
        for sign, c0, cT, cTinv in candidates:
            if (cT < 0 and -cT * tmin > 745) or (cTinv < 0 and -cTinv / tmax > 745):
                continue
            cT = 0.0 if abs(cT) * tmax < 1e-14 else cT
            cTinv = 0.0 if abs(cTinv) / tmin < 1e-14 else cTinv
            terms.append((sign, c0, cT, cTinv))
        return terms

    def gibbs(self, T):
        """Gives every species' g / (Ru T) at the temperatures :T:, from the
        refit h and s the eos works with: (species, len(T))."""
        u, Tinv = np.log(T), 1.0 / np.asarray(T, dtype=np.float64)
        gPoly, hRef = self.speciesData()["gPoly"], self.speciesData()["hRef"]
        return np.array(
            [np.polyval(np.flip(g), u) + h * Tinv for g, h in zip(gPoly, hRef)]
        )

    def _logCFloor(self, reactions, net, fwd, index):
        """The log concentration an absent species is floored at: deep
        enough that every direction of every reaction it takes part in is
        suppressed below e^-45 at any state over the case's temperatures,
        with ln c <= 12 for the other factors, and no shallower than the
        smallest normal float; PyFR's derivation."""
        logCMax, margin = 12.0, 45.0
        Tlow, Thigh = self.configSect.get("Trange") or (200.0, 5000.0)
        T = np.geomspace(max(Tlow, 100.0), Thigh, 256)
        logT, gbs = np.log(T), self.gibbs(T)
        required = 90.0
        for r, netRow, fwdRow in zip(reactions, net, fwd):
            A, b, EaR = r["rate"]
            logkf = np.log(A) + b * logT - EaR / T
            aux = {"three-body": logCMax + 2.0, "lindemann": 1.0, "troe": 1.0}.get(
                r["type"], 0.0
            )
            top = logkf.max() + aux
            for k, vk in fwdRow:
                if vk <= 0:
                    continue
                other = sum(v * logCMax for n, v in fwdRow if n != k and v > 0)
                required = max(required, (top + other + margin) / vk)
            if r["reversible"]:
                affinity = sum(v * gbs[n] for n, v in netRow)
                logkr = (
                    logkf
                    + affinity
                    + sum(v for _, v in netRow) * np.log(Ru * T / 101325.0)
                )
                rev = [
                    (index[sp], float(v)) for sp, v in r["products"].items() if v > 0
                ]
                top = logkr.max() + aux
                for k, vk in rev:
                    other = sum(v * logCMax for n, v in rev if n != k)
                    required = max(required, (top + other + margin) / vk)
        return -float(required)
