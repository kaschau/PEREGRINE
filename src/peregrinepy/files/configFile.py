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
        # What is simulated: the simulator, by the name of its class, in what
        # precision, for how many steps, and how each step is sized: fixed at
        # dt, or cfl up to maxDt
        self["simulation"] = frozenDict(
            {
                "simulator": "navierStokes",
                # what every array and kernel value is: double, or single
                "precision": "double",
                "niter": 1,
                "controller": "fixed",
                "dt": 1e-3,
                "maxDt": 1e-3,
                "maxCFL": 0.1,
            }
        )
        # The gas: its species, its equation of state, its transport and
        # diffusion models, and how their fits are made
        self["mixture"] = frozenDict(
            {
                # a Cantera mechanism file, or a list of species from the library
                "species": None,
                "eos": "cpg",
                # none: a viscous simulator picks one
                "trans": None,
                "diffusion": "lewis",
                # how the species' viscosities mix: wilke or herning
                "mixingRule": "wilke",
                # what every temperature-dependent property is refit over and
                # to: the lowest degree within the tolerance, or the best at the
                # cap, which is seven terms, the count the source data has, and
                # what a polynomial in ln T stays well conditioned at
                "Trange": None,
                "reFitTol": 1e-3,
                "reFitMaxDegree": 6,
            }
        )
        # Finite-rate chemistry: no source, the production rates as the
        # source, or substepped to the fastest species' bound, up to so many
        # substeps, each capped where the mixture's entropy stops rising
        # along it, located by so many bisections (0: no cap)
        self["chemistry"] = frozenDict(
            {"source": None, "maxSubSteps": 200, "entropyBisections": 12}
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

        # The time integrator, and what is its own
        self["timeIntegration"] = frozenDict(
            {
                "integrator": "rk3",
                # dual time's pseudo time scheme, any Runge-Kutta integrator,
                # and how many pseudo steps each physical step gets
                "pseudoIntegrator": "rk3",
                "subIterations": 20,
                # dual time's per-cell pseudo system: the low-Mach (Weiss and
                # Smith) preconditioner, and the chemistry source's Jacobian
                # in it, none or each species' own entry with its temperature's
                "lowMach": True,
                "chemistryJacobian": None,
            }
        )

        self["RHS"] = frozenDict(
            {
                "primaryAdvFlux": "KEPaEC",
                # shock capturing: a secondary flux blended in by the
                # switch's weight, and what the switch takes
                "secondaryAdvFlux": None,
                "switchAdvFlux": None,
                "switchValues": {},
            }
        )

        # How the kernels are launched, a section per backend the runtime
        # may have been built for; a run reads the one it was. tileSize and
        # tileElements are the items of one block a team does: cells for a
        # launch over cells or faces, elements for one whose item is one
        # element (a copy, a launch over cells and components). On a device
        # a cell team is its tile, and the kernels are compiled with a launch
        # bound of launchThreads with a floor of launchWaves resident waves,
        # which is the register budget: a floor of 2 measured best on an
        # MI100 (the floor of 4 an unbounded launch implies spilled the
        # viscous flux and transport); on CUDA 1 keeps nvcc's budget until a
        # card measures otherwise. The host is one thread per team.
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

        # How a halo exchange's messages travel: staged through pinned host
        # memory, or straight from the device buffers when the MPI is
        # GPU-aware, which the installation knows and the runtime cannot
        self["haloExchange"] = frozenDict({"kind": "hostStaged"})

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
