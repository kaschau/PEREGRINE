"""What every property model has in common: what it needs from the species and
what it works out and stores back."""

import numpy as np

from . import anyOf


class BaseModel:
    """One property model. The base declares the empties; each model adds
    what it needs and provides on top of what its base does."""

    # what the config calls it, on the class so subclassWhere can find it
    name = None

    def __init__(self, cfgsect):
        # the case's mcPhysics section: what every fit is made over and to
        self.cfgsect = cfgsect
        # what it reads off each species; everything needs MW
        self.fromSpecies = ("MW",)
        # what the case must state per species: modelling choices the library never carries
        self.requiredInput = ()
        # which of fromSpecies may be absent if what derives them is present
        self.derivable = {}
        # what it works out, in order: {method: the quantity, or quantities, it returns}
        self.provides = {}
        # models this one only works alongside, by name or anyOf names
        self.dependsOn = ()

    def check(self, species):
        """Raise if this species set cannot serve the model, naming every shortfall."""
        soft = [key for keys in self.derivable for key in keys]
        hard = [key for key in self.fromSpecies if key not in soft]
        missing = self._missingFor(hard, species)
        if missing:
            raise ValueError(f"{self.name} needs {self._shortfall(missing)}")
        for derived, inputs in self.derivable.items():
            # a species with the set does not need the inputs
            short = [
                n
                for n, sp in species.items()
                if sp.missing(derived) and sp.missing(inputs)
            ]
            if short:
                raise ValueError(
                    f"{self.name} needs {', '.join(derived)}, "
                    f"or {', '.join(inputs)} to derive them, "
                    f"for {self._listed(short)}"
                )
        absent = self._missingFor(self.requiredInput, species)
        if absent:
            raise ValueError(f"{self.name} needs {self._shortfall(absent)}")

    def populateSpeciesData(self, species):
        """Work out everything this model provides and store it on each species."""
        for method, quantities in self.provides.items():
            returned = getattr(self, method)(species)
            if isinstance(quantities, str):
                quantities, returned = (quantities,), (returned,)
            for quantity, values in zip(quantities, returned):
                for sp, value in zip(species.values(), values):
                    sp[quantity] = value

    @staticmethod
    def collect(key, species):
        """One property of every species in mechanism order, as an array."""
        prop = [sp[key] for sp in species.values()]
        return prop if isinstance(prop[0], str) else np.array(prop, dtype=np.float64)

    @staticmethod
    def _missingFor(keys, species):
        """{species: [keys it cannot answer for]}"""
        missing = {}
        for name, sp in species.items():
            absent = sp.missing(keys)
            if absent:
                missing[name] = absent
        return missing

    def _shortfall(self, missing):
        """'cp0 for N2, O2' -- species grouped by what they lack."""
        grouped = {}
        for name, keys in missing.items():
            grouped.setdefault(tuple(keys), []).append(name)
        return "; ".join(
            f"{', '.join('|'.join(k) if isinstance(k, anyOf) else k for k in keys)} "
            f"for {self._listed(names)}"
            for keys, names in grouped.items()
        )

    @staticmethod
    def _listed(names, most=6):
        if len(names) <= most:
            return ", ".join(names)
        return ", ".join(names[:most]) + f" and {len(names) - most} more"
