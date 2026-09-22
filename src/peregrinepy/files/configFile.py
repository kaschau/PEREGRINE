from ..misc import frozenDict


class pgConfigError(Exception):
    def __init__(self, setting1, setting2, altMessage=""):
        message = f"\n\n*****\nInvalid PEREGRINE settings: {str(setting1)} and {str(setting2)}. "
        super().__init__(message + altMessage + "\n*****\n\n")


class configFile(frozenDict):
    """The case's settings by section, over these defaults; every key is
    described in docs/config.md."""

    def __init__(self):
        self["simulation"] = frozenDict(
            {
                "simulator": "navierStokes",
                "precision": "double",
                "niter": 1,
                "controller": "fixed",
                "dt": 1e-3,
                "maxDt": 1e-3,
                "maxCFL": 0.1,
            }
        )
        self["mixture"] = frozenDict(
            {
                "species": None,
                "eos": "cpg",
                "trans": None,
                "diffusion": "lewis",
                "mixingRule": "wilke",
                "Trange": None,
                "reFitTol": 1e-3,
                "reFitMaxDegree": 6,
            }
        )
        self["chemistry"] = frozenDict(
            {"source": None, "maxSubSteps": 200, "entropyBisections": 12}
        )
        self["initialConditions"] = frozenDict(
            {"p": 101325.0, "u": 0.0, "v": 0.0, "w": 0.0, "T": 300.0, "Y": {}}
        )
        self["timeIntegration"] = frozenDict(
            {
                "integrator": "rk3",
                "pseudoIntegrator": "rk3",
                "subIterations": 20,
                "lowMach": True,
                "chemistryJacobian": None,
                "pseudoCFL": 1.5,
                "pseudoVNN": 0.1,
            }
        )
        self["RHS"] = frozenDict(
            {
                "primaryAdvFlux": "KEPaEC",
                "secondaryAdvFlux": None,
                "switchAdvFlux": None,
                "switchValues": {},
            }
        )
        # a run reads the section of the backend the runtime was built for
        for name in ("serial", "openmp"):
            self[f"backend-{name}"] = frozenDict(
                {"tileSize": 128, "tileElements": 1024}
            )
        self["backend-cuda"] = frozenDict(
            {
                "tileSize": 128,
                "tileElements": 1024,
                "launchThreads": 256,
                "launchWaves": 1,
            }
        )
        self["backend-hip"] = frozenDict(
            {
                "tileSize": 128,
                "tileElements": 1024,
                "launchThreads": 256,
                "launchWaves": 2,
            }
        )
        self["haloExchange"] = frozenDict({"kind": "hostStaged"})
        # the names under these two are the case's own, so they are not frozen
        self["bcValues"] = {}
        self["plugins"] = {}

        for key in self.keys():
            if isinstance(self[key], frozenDict):
                self[key]._freeze()
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
        self["simulation"]["dt"] = float(self["simulation"]["dt"])
        for section in (k for k in self if k.startswith("backend-")):
            launch = self[section]
            for key in launch:
                if not isinstance(launch[key], int) or launch[key] < 1:
                    raise pgConfigError(
                        section, key, f"{launch[key]!r} is not a positive integer."
                    )
            # a device's cell team is its tile, within the launch bound
            if (
                "launchThreads" in launch
                and launch["tileSize"] > launch["launchThreads"]
            ):
                raise pgConfigError(section, "tileSize", "is at most launchThreads.")
        sub = self["timeIntegration"]["subIterations"]
        if not isinstance(sub, int) or sub < 1:
            raise pgConfigError(
                "timeIntegration",
                "subIterations",
                f"{sub!r} is not a positive integer.",
            )
