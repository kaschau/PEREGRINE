"""The equations of state a case can pick. Each names a caloric model, and
every one puts its four polynomials in ln T on each species."""

from . import Ru, kB
from .baseModel import BaseModel
from .caloricModel import ConstantCpModel, TempDepCpModel


class BaseEosModel(BaseModel):
    """One equation of state, with the caloric model it gets cp from."""

    def __init__(self, cfgsect, caloricModel):
        super().__init__(cfgsect)
        self.caloric = caloricModel(cfgsect)
        self.fromSpecies += self.caloric.needs
        # ordered: each reads what the one before stored
        self.provides.update(cpPoly="cpPoly", hPoly="hPoly", hRef="hRef", sPoly="sPoly")

    def cpPoly(self, species):
        return [self.caloric.cpPoly(sp) for sp in species.values()]

    def hPoly(self, species):
        return [self.caloric.hPoly(sp) for sp in species.values()]

    def hRef(self, species):
        return [self.caloric.hRef(sp) for sp in species.values()]

    def sPoly(self, species):
        return [self.caloric.sPoly(sp) for sp in species.values()]


class CpgModel(BaseEosModel):
    """Calorically perfect."""

    name = "cpg"

    def __init__(self, cfgsect):
        super().__init__(cfgsect, ConstantCpModel)


class TpgModel(BaseEosModel):
    """Thermally perfect."""

    name = "tpg"

    def __init__(self, cfgsect):
        super().__init__(cfgsect, TempDepCpModel)


class RealGasModel(BaseEosModel):
    """Cubic: thermally perfect as the ideal reference, plus a critical point."""

    name = "realGas"
    critical = ("Tcrit", "pcrit", "Vcrit", "acentric")

    def __init__(self, cfgsect):
        super().__init__(cfgsect, TempDepCpModel)
        self.fromSpecies += self.critical
        self.derivable[self.critical] = ("well", "diam")
        self.provides["criticalPoint"] = self.critical

    def criticalPoint(self, species):
        """The measured critical point, or one from the Lennard-Jones parameters
        where a species has none: Tee, Gotoh and Stewart's (1966) relations
        run backwards, and Pitzer's Zc at zero acentric factor."""
        out = {key: [] for key in self.critical}
        for sp in species.values():
            if sp.missing(self.critical):
                Tc = 1.2593 * sp["well"] / kB
                VcMolar = (sp["diam"] * 1e10 / 0.809) ** 3 * 1e-3  # cm3/mol -> m3/kmol
                got = {
                    "Tcrit": Tc,
                    "pcrit": 0.291 * Ru * Tc / VcMolar,  # Zc = 0.291 - 0.08 w at w = 0
                    "Vcrit": VcMolar / sp["MW"],
                    "acentric": 0.0,
                }
            else:
                got = {key: sp[key] for key in self.critical}
            for key in self.critical:
                out[key].append(got[key])
        return tuple(out[key] for key in self.critical)
