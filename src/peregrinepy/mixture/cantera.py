"""A Cantera yaml mechanism read without cantera: species in PEREGRINE's
units and reactions as plain records. Ported from the PyFR reader."""

import re

import yaml

from . import Ru, avogadro, debye, kB
from .species import Species


# YAML 1.1 reads NO/YES/ON/OFF as booleans, which breaks the species 'NO'
class _canteraLoader(yaml.SafeLoader):
    pass


_canteraLoader.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


class CanteraParser:
    """`species` as {name: {property: value}}, what a case could have written by
    hand, and `reactions` as a list of records."""

    def __init__(self, path, species=None):
        """The phase's species, or the given subset in that order."""
        with open(path) as f:
            data = yaml.load(f, Loader=_canteraLoader)
        if species is None:
            if "phases" not in data:
                raise ValueError(f"{path} has no phases section; give the species list")
            species = list(data["phases"][0]["species"])
        lookup = {sp["name"]: sp for sp in data.get("species", [])}
        # the file's own elements (custom or overriding weights) win
        weights = dict(Species.atomicWeights)
        for el in data.get("elements", []):
            weights[el["symbol"]] = float(el["atomic-weight"])
        self.species = {}
        for name in species:
            if name not in lookup:
                raise KeyError(f"Species '{name}' not found in {path}")
            self.species[name] = self._parseSpecies(lookup[name], weights)

        self.reactions = []
        if data.get("reactions"):
            units = data.get("units", {})
            # to J/kmol, m and kmol
            eaFactor = {
                "cal/mol": 4184.0,
                "kcal/mol": 4184000.0,
                "J/mol": 1000.0,
                "kJ/mol": 1e6,
                "J/kmol": 1.0,
                "K": Ru,
            }[units.get("activation-energy", "cal/mol")]
            lengthFac = {"cm": 1e-2, "m": 1.0}[units.get("length", "cm")]
            quantityFac = {"mol": 1e-3, "kmol": 1.0, "molec": 1.0 / avogadro}[
                units.get("quantity", "mol")
            ]
            for raw in data["reactions"]:
                rxn = self._parseReaction(raw, eaFactor, lengthFac, quantityFac)
                unknown = (
                    set(rxn["reactants"])
                    | set(rxn["products"])
                    | set(rxn["efficiencies"])
                )
                unknown -= set(self.species)
                if unknown:
                    raise KeyError(
                        f"Reaction '{rxn['equation']}' uses species not in the phase: {sorted(unknown)}"
                    )
                self.reactions.append(rxn)

    @staticmethod
    def _bool(sect, key, default=False):
        v = sect.get(key, default)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "on")
        return bool(v)

    @staticmethod
    def _parseSide(side):
        species, hasM = {}, False
        side = re.sub(r"\(\s*\+[^)]+\)", "", side).strip()
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            if term == "M":
                if hasM:
                    raise ValueError(
                        "Multiple generic third-body colliders 'M' are not supported"
                    )
                hasM = True
                continue
            m = re.match(r"^(\d+\.?\d*)\s+(.+)$", term)
            coeff, name = (float(m.group(1)), m.group(2).strip()) if m else (1.0, term)
            species[name] = species.get(name, 0.0) + coeff
        return species, hasM

    @staticmethod
    def _parseEquation(equation):
        equation = equation.split("#")[0].strip()
        colliders = {c.strip() for c in re.findall(r"\(\s*\+([^)]+)\)", equation)}
        if len(colliders) > 1:
            raise ValueError(f"Inconsistent falloff colliders in '{equation}'")
        collider = colliders.pop() if colliders else None
        if "<=>" in equation:
            lhs, rhs = equation.split("<=>")
            reversible = True
        elif "=>" in equation:
            lhs, rhs = equation.split("=>")
            reversible = False
        else:
            raise ValueError(f"Cannot parse equation: '{equation}'")
        reactants, lhsM = CanteraParser._parseSide(lhs)
        products, rhsM = CanteraParser._parseSide(rhs)
        if lhsM != rhsM:
            raise ValueError(f"Third body M must appear on both sides: '{equation}'")
        return reactants, products, reversible, lhsM, collider

    @staticmethod
    def _rate(sect, eaFactor, order, lengthFac, quantityFac):
        """(A, b, Ea/Ru) in SI: A in (m^3/kmol)^(order-1)/s, Ea/Ru in K."""
        A = float(sect["A"]) * (lengthFac**3 / quantityFac) ** (order - 1)
        return A, float(sect["b"]), float(sect["Ea"]) * eaFactor / Ru

    @staticmethod
    def _parseReaction(raw, eaFactor, lengthFac, quantityFac):
        equation = raw["equation"]
        reactants, products, reversible, hasM, collider = CanteraParser._parseEquation(
            equation
        )
        rtypeRaw = raw.get("type", None)
        effs = {k: float(v) for k, v in raw.get("efficiencies", {}).items()}
        defaultEff = float(raw.get("default-efficiency", 1.0))

        if rtypeRaw == "Arrhenius":
            rtypeRaw = "elementary"
        elif rtypeRaw == "three-body-Arrhenius":
            rtypeRaw = "three-body"
        elif rtypeRaw in ("Lindemann", "Troe"):
            rtypeRaw = "falloff"

        hasTroe = "Troe" in raw
        if rtypeRaw == "falloff":
            rtype = "troe" if hasTroe else "lindemann"
            if collider is None:
                raise ValueError(
                    f"Falloff reaction requires a (+M) or (+species) collider: '{equation}'"
                )
            if collider != "M":
                # explicit collider: zero default efficiency, unit collider efficiency
                if "default-efficiency" in raw and defaultEff != 0.0:
                    raise ValueError(
                        f"Invalid default efficiency for explicit collider: '{equation}'"
                    )
                if effs and (len(effs) != 1 or collider not in effs):
                    raise ValueError(
                        f"Incompatible third-body collider definitions: '{equation}'"
                    )
                defaultEff = 0.0
                effs = effs or {collider: 1.0}
        elif rtypeRaw == "three-body" or (rtypeRaw is None and hasM):
            rtype = "three-body"
            if collider is not None:
                raise ValueError(
                    f"'(+{collider})' collider requires type: falloff: '{equation}'"
                )
            if not hasM:
                # explicit-species collider: one unit removed from each side
                if len(effs) != 1:
                    raise ValueError(
                        f"Third-body definition requires a single-species efficiency: '{equation}'"
                    )
                (sp,) = effs
                for side in (reactants, products):
                    if side.get(sp, 0) < 1:
                        raise ValueError(
                            f"Third-body collider '{sp}' must appear on both sides: '{equation}'"
                        )
                    if side[sp] == 1:
                        del side[sp]
                    else:
                        side[sp] -= 1
                defaultEff = 0.0
        elif rtypeRaw in (None, "elementary"):
            rtype = "elementary"
            if hasM or collider is not None:
                raise ValueError(f"Elementary reaction with a third body: '{equation}'")
            if "efficiencies" in raw or "default-efficiency" in raw:
                raise ValueError(
                    f"Efficiencies on a reaction without third bodies: '{equation}'"
                )
        else:
            raise ValueError(f"Unsupported reaction type '{rtypeRaw}': '{equation}'")

        orders = {sp: float(v) for sp, v in raw.get("orders", {}).items()}
        if orders:
            if reversible:
                raise ValueError(
                    f"Reaction orders may only be given for irreversible reactions: '{equation}'"
                )
            if any(sp not in reactants for sp in orders) and not CanteraParser._bool(
                raw, "nonreactant-orders"
            ):
                raise ValueError(
                    f"Reaction order specified for non-reactant species: '{equation}'"
                )
            if any(v < 0 for v in orders.values()) and not CanteraParser._bool(
                raw, "negative-orders"
            ):
                raise ValueError(f"Negative reaction order specified: '{equation}'")
        # forward concentration exponents: orders override stoichiometry
        fwd = {sp: coeff for sp, coeff in reactants.items()}
        fwd.update(orders)
        order = sum(fwd.values()) + (1 if rtype == "three-body" else 0)

        rxn = {
            "equation": equation,
            "type": rtype,
            "reversible": reversible,
            "duplicate": CanteraParser._bool(raw, "duplicate"),
            "reactants": reactants,
            "products": products,
            "fwd": fwd,
            "efficiencies": effs,
            "defaultEfficiency": defaultEff,
        }
        if rtype in ("lindemann", "troe"):
            rxn["rate"] = CanteraParser._rate(
                raw["high-P-rate-constant"], eaFactor, order, lengthFac, quantityFac
            )
            rxn["lowRate"] = CanteraParser._rate(
                raw["low-P-rate-constant"], eaFactor, order + 1, lengthFac, quantityFac
            )
        else:
            rxn["rate"] = CanteraParser._rate(
                raw["rate-constant"], eaFactor, order, lengthFac, quantityFac
            )
        if hasTroe:
            troe = raw["Troe"]
            rxn["troe"] = [
                float(troe["A"]),
                float(troe["T3"]),
                float(troe["T1"]),
                float(troe.get("T2", 0.0)),
            ]
        return rxn

    @staticmethod
    def _parseSpecies(raw, weights):
        """A species in PEREGRINE's species-data convention (SI)."""
        comp = {k: float(v) for k, v in raw["composition"].items()}
        MW = Species.molecularWeight(comp, weights)
        sp = {"comp": comp, "MW": MW}
        if "thermo" in raw and raw["thermo"]["model"] == "constant-cp":
            # J/(kg K) from a molar value with its unit, e.g. '12345.0 J/kmol/K'
            v, unit = str(raw["thermo"]["cp0"]).split()
            sp["cp0"] = float(v) * {"J/kmol/K": 1.0, "J/mol/K": 1e3}[unit] / MW
        elif "thermo" in raw:
            thermo = raw["thermo"]
            if thermo["model"].upper() != "NASA7":
                raise ValueError(
                    f"Unsupported thermo model {thermo['model']} for {raw['name']}"
                )
            ranges = [float(r) for r in thermo["temperature-ranges"]]
            data = [list(map(float, c)) for c in thermo["data"]]
            # a single fit serves both ranges
            low, high = (data[0], data[0]) if len(data) == 1 else data[:2]
            Tmid = ranges[1] if len(data) > 1 else ranges[-1]
            # source data with the range it is good for; it gets refit to the case's
            sp["NASA7"] = {
                "Trange": [ranges[0], Tmid, ranges[-1]],
                "high": high,
                "low": low,
            }
        if "transport" in raw:
            tr = raw["transport"]
            sp["geometry"] = tr["geometry"]
            sp["well"] = float(tr["well-depth"]) * kB
            sp["diam"] = float(tr["diameter"]) * 1e-10
            sp["dipole"] = float(tr.get("dipole", 0.0)) * debye
            sp["polarize"] = float(tr.get("polarizability", 0.0)) * 1e-30
            sp["zrot"] = float(tr.get("rotational-relaxation", 0.0))
            # absent is unstated, not zero; a zero would shadow the library's value
            if "acentric-factor" in tr:
                sp["acentric"] = float(tr["acentric-factor"])
        return sp
