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
        self["io"] = frozenDict(
            {
                "gridDir": "./Grid",
                "inputDir": "./Input",
                "resultsDir": "./Results",
                "niterOut": 10,
                "niterPrint": 1,
            }
        )
        self["simulation"] = frozenDict(
            {
                "niter": 1,
                # which result to restart from; None starts from the initial conditions
                "restartFrom": None,
                "checkNan": False,
            }
        )
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
                "dt": 1e-3,
                "variableTimeStep": False,
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
                # what every temperature-dependent property is refit over and to
                "Trange": None,
                "reFitTol": 1e-3,
                "reFitMaxDegree": 8,
            }
        )

        self["coprocess"] = frozenDict(
            {
                "catalyst": False,
                "catalystFile": "./Input/coproc.py",
                "trace": False,
                "niterTrace": 1,
            },
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

        for key in self.keys():
            if isinstance(self[key], frozenDict):
                self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()

    def validateConfig(self):
        """What the file's values have to be; whether they make a step is the
        step graph's to say."""
        self["timeIntegration"]["dt"] = float(self["timeIntegration"]["dt"])
        self["mcPhysics"]["nChemSubSteps"] = max(1, self["mcPhysics"]["nChemSubSteps"])
