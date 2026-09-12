"""A case's gas: its species and its reactions, collected and completed."""

from pathlib import Path

from .cantera import CanteraParser
from ..misc import subclassWhere
from . import anyOf
from .diffusionModel import BaseSpeciesDiffusionModel
from .eosModel import BaseEosModel
from .species import Species
from .transportModel import BaseTransportModel


class Mixture:
    """The gas a case is solving, from its mcPhysics section of the config."""

    def __init__(self, cfgsect, root=None):
        self.cfgsect = cfgsect
        self.eos = subclassWhere(BaseEosModel, name=cfgsect["eos"])(cfgsect)
        trans, diffusion = cfgsect["trans"], cfgsect["diffusion"]
        self.trans = (
            subclassWhere(BaseTransportModel, name=trans)(cfgsect) if trans else None
        )
        self.diffusion = (
            subclassWhere(BaseSpeciesDiffusionModel, name=diffusion)(cfgsect)
            if trans
            else None
        )
        self._checkCombination()

        usersp, self.reactions = self.readMixture(cfgsect["mixture"], root)
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

    @property
    def transportKernel(self):
        """The one kernel the transport and species diffusion choices pick
        between them."""
        pair = (self.trans.name, self.diffusion.name)
        kernel = {
            ("kineticTheory", "binary"): "kineticTheory",
            ("kineticTheory", "lewis"): "kineticTheoryUnityLewis",
            ("chungDenseGas", "lewis"): "chungDenseGasUnityLewis",
            ("constantProps", "lewis"): "constantProps",
        }.get(pair)
        if kernel is None:
            raise ValueError(
                f"no transport kernel for {pair[0]} with {pair[1]} diffusion"
            )
        return kernel

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
