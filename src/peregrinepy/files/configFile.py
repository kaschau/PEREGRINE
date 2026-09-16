from ..misc import frozenDict

"""

This module holds a defines a dictionary version
of a standard PEREGRINE config file.

"""


class pgConfigError(Exception):
    def __init__(self, setting1, setting2, altMessage=""):
        message = f"\n\n*****\nInvalid PEREGRINE settings: {str(setting1)} and {str(setting2)}. "
        super().__init__(message + altMessage + "\n*****\n\n")


class configFile(frozenDict):
    def __init__(self):
        self["simulation"] = frozenDict({"niter": 1})
        # the uniform state a case starts from when it does not restart
        self["initialConditions"] = frozenDict(
            {
                "p": 101325.0,
                "u": 0.0,
                "v": 0.0,
                "w": 0.0,
                "T": 300.0,
                # mass fraction by species name; the rest are zero, and the
                # last species takes the remainder
                "Y": {},
            }
        )

        self["timeIntegration"] = frozenDict(
            {
                "integrator": "rk3",
                # dual time's pseudo time scheme, any Runge-Kutta integrator,
                # and how many pseudo steps each physical step gets
                "pseudoIntegrator": "rk3",
                "subIterations": 20,
                # how each step is sized: fixed at dt, or cfl up to maxDt
                "controller": "fixed",
                "dt": 1e-3,
                "maxDt": 1e-3,
                "maxCFL": 0.1,
            }
        )

        self["RHS"] = frozenDict(
            {
                "shockHandling": None,
                "primaryAdvFlux": "KEPaEC",
                "secondaryAdvFlux": None,
                "switchAdvFlux": None,
                "diffusion": False,
                "subgrid": None,
                # items of one block per launch tile
                "tileSize": 128,
            }
        )

        self["mcPhysics"] = frozenDict(
            {
                # a Cantera mechanism file, or a list of species from the library
                "mixture": None,
                "eos": "cpg",
                # none, like RHS diffusion: a viscous case picks one
                "trans": None,
                "diffusion": "lewis",
                "chemistry": False,
                "nChemSubSteps": 1,
                # what every temperature-dependent property is refit over and
                # to: the lowest degree within the tolerance, or the best at the
                # cap, which is seven terms, the count the source data has, and
                # what a polynomial in ln T stays well conditioned at
                "Trange": None,
                "reFitTol": 1e-3,
                "reFitMaxDegree": 6,
            }
        )

        self["viscousSponge"] = frozenDict(
            {
                "spongeON": False,
                "origin": [0.0, 0.0, 0.0],
                "ending": [1.0, 0.0, 0.0],
                "multiplier": 5.0,
            },
        )

        # What each boundary reads, by the name the grid gives it. Which
        # faces carry a name is the grid's business; what they read is the
        # case's, so the names here are the user's and not frozen.
        self["bcValues"] = {}
        # What runs alongside the stepping, by plugin name, each with its own
        # options and how often it acts. The names are the user's and not frozen.
        self["plugins"] = {}

        for key in self.keys():
            if isinstance(self[key], frozenDict):
                self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()

    @classmethod
    def fromDict(cls, given):
        """A config with :given: over the defaults, section by section."""
        config = cls()
        for section, entries in (given or {}).items():
            for key, value in (entries or {}).items():
                config[section][key] = value
        config.validateConfig()
        return config

    def toDict(self):
        """The config as plain dicts, which is what yaml writes."""
        return {
            section: dict(entries) if isinstance(entries, dict) else entries
            for section, entries in self.items()
        }

    def validateConfig(self):
        """What the file's values have to be; whether they make a step is the
        step graph's to say."""
        self["timeIntegration"]["dt"] = float(self["timeIntegration"]["dt"])
        self["mcPhysics"]["nChemSubSteps"] = max(1, self["mcPhysics"]["nChemSubSteps"])
        tile = self["RHS"]["tileSize"]
        if not isinstance(tile, int) or tile < 1:
            raise pgConfigError(
                "RHS", "tileSize", f"{tile!r} is not a positive integer."
            )
        sub = self["timeIntegration"]["subIterations"]
        if not isinstance(sub, int) or sub < 1:
            raise pgConfigError(
                "timeIntegration",
                "subIterations",
                f"{sub!r} is not a positive integer.",
            )
