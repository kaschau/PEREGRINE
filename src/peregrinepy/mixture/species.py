"""One species: what it was given by the case, the library and the models,
in that order of precedence, and enough about itself to be written into a header."""

from . import anyOf, database


class Species:
    """What is known about one species."""

    # kg/kmol, Cantera's Elements.cpp
    atomicWeights = {
        "H": 1.008,
        "He": 4.002602,
        "Li": 6.94,
        "Be": 9.0121831,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998403163,
        "Ne": 20.1797,
        "Na": 22.98976928,
        "Mg": 24.305,
        "Al": 26.9815384,
        "Si": 28.085,
        "P": 30.973761998,
        "S": 32.06,
        "Cl": 35.45,
        "Ar": 39.95,
        "K": 39.0983,
        "Ca": 40.078,
        "Sc": 44.955908,
        "Ti": 47.867,
        "V": 50.9415,
        "Cr": 51.9961,
        "Mn": 54.938043,
        "Fe": 55.845,
        "Co": 58.933194,
        "Ni": 58.6934,
        "Cu": 63.546,
        "Zn": 65.38,
        "Ga": 69.723,
        "Ge": 72.630,
        "As": 74.921595,
        "Se": 78.971,
        "Br": 79.904,
        "Kr": 83.798,
    }
    # rotational degrees of freedom over two, by molecule shape
    rotDOF = {"atom": 0.0, "linear": 1.0, "nonlinear": 1.5}

    def __init__(self, name, data=None):
        self.name = name
        self.data = {}
        self.supplement(data)

    def __repr__(self):
        return f"<Species {self.name}: {', '.join(sorted(self.data))}>"

    def __contains__(self, key):
        return self.has(key)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        # the one thing a species can answer without being told
        if key == "MW" and "comp" in self.data:
            return self.molecularWeight(self.data["comp"])
        raise KeyError(f"species {self.name} has no {key}")

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def supplement(self, data):
        """Fill what we lack from `data`; what we already have stands."""
        for key, value in (data or {}).items():
            if value is not None:
                self.data.setdefault(key, value)
        return self

    def has(self, key):
        if key in self.data:
            return True
        return key == "MW" and "comp" in self.data

    def missing(self, keys):
        """Which of `keys` this species cannot answer for."""
        return [
            key
            for key in keys
            if not (
                any(map(self.has, key)) if isinstance(key, anyOf) else self.has(key)
            )
        ]

    @staticmethod
    def molecularWeight(composition, weights=None):
        """kg/kmol for a composition like {"C": 1, "H": 4}."""
        weights = Species.atomicWeights if weights is None else weights
        return sum(weights[e] * n for e, n in composition.items())

    @staticmethod
    def referenceData():
        """PEREGRINE's curated, cited data for every species it knows."""
        return database("speciesLibrary")

    @classmethod
    def build(cls, usersp, models, reference=None):
        """The case's species in mechanism order, supplemented by the case and
        then the reference data; nothing overrides."""
        reference = cls.referenceData() if reference is None else reference
        sameAs = {}
        for model in models:
            if model is not None:
                # keys that are one quantity in different forms; having one is having it
                for group in (k for k in model.fromSpecies if isinstance(k, anyOf)):
                    sameAs.update({key: group for key in group})

        built = {}
        for name, data in usersp.items():
            sp = cls(name)
            sp.supplement(data)
            sp.supplement(
                {
                    key: value
                    for key, value in (reference.get(name) or {}).items()
                    if key not in sameAs or sp.missing([sameAs[key]])
                }
            )
            built[name] = sp
        return built
