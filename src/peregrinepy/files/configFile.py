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
                "restartDir": "./Restart",
                "archiveDir": "./Archive",
                "animateArchive": True,
                "animateRestart": False,
                "lumpIO": True,
                "niterArchive": 1000000000,
                "niterRestart": 10,
                "niterPrint": 1,
                "saveExtraVars": [],
            }
        )
        self["simulation"] = frozenDict(
            {
                "niter": 1,
                "restartFrom": 0,
                "checkNan": False,
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
                "primaryAdvFlux": "KEEPes",
                "secondaryAdvFlux": None,
                "switchAdvFlux": None,
                "diffusion": False,
                "subgrid": None,
            }
        )

        self["thermochem"] = frozenDict(
            {
                "spdata": ["Air"],
                "eos": "cpg",
                "trans": "constantProps",
                "chemistry": False,
                "mechanism": None,
                "nChemSubSteps": 1,
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

        for key in self.keys():
            self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()

    def validateConfig(self):
        #######################################################################
        # Sanity checks go here
        #######################################################################

        # ---------------------------------------------------------------------#
        # io checks
        # ---------------------------------------------------------------------#
        if self["io"]["saveExtraVars"] is None:
            self["io"]["saveExtraVars"] = []

        # ---------------------------------------------------------------------#
        # timeIntegration Checks
        # ---------------------------------------------------------------------#
        ti = self["timeIntegration"]["integrator"]
        eos = self["thermochem"]["eos"]

        self["timeIntegration"]["dt"] = float(self["timeIntegration"]["dt"])
        if ti == "dualTime" and eos not in [
            "cpg",
            "tpg",
        ]:
            raise pgConfigError(ti, eos, "Only cpg and tpg currently supported.")

        if ti == "dualTime" and self["timeIntegration"]["variableTimeStep"]:
            raise pgConfigError(
                "dualTime",
                "variableTimeStep",
                "Only fixed time step currently supported.",
            )

        # ---------------------------------------------------------------------#
        # RHS checks
        # ---------------------------------------------------------------------#
        primaryAdvFlux = self["RHS"]["primaryAdvFlux"]
        secondaryAdvFlux = self["RHS"]["secondaryAdvFlux"]
        switch = self["RHS"]["switchAdvFlux"]
        shock = self["RHS"]["shockHandling"]
        if primaryAdvFlux is None:
            raise pgConfigError(
                "primaryAdvFlux", primaryAdvFlux, "primaryAdvFlux cannot be none."
            )

        if shock == "artificialDissipation" and secondaryAdvFlux not in [
            "scalarDissipation"
        ]:
            raise pgConfigError(shock, secondaryAdvFlux)
        if shock == "hybrid" and secondaryAdvFlux in ["scalarDissipation"]:
            raise pgConfigError(shock, secondaryAdvFlux)

        if shock is not None:
            if secondaryAdvFlux is None:
                raise pgConfigError(
                    shock,
                    secondaryAdvFlux,
                    "\nYou set a shock handlind without a secondary adv flux.",
                )

        if switch is None:
            if secondaryAdvFlux is not None:
                raise pgConfigError(
                    switch,
                    secondaryAdvFlux,
                    "\nYou set a secondaryAdvFlux without setting switchAdvFlux.",
                )
        else:
            if secondaryAdvFlux is None:
                raise pgConfigError(
                    switch,
                    secondaryAdvFlux,
                    "\nYou set an flux switching option without a secondary flux.",
                )

        # ---------------------------------------------------------------------#
        # viscousSponge check
        # ---------------------------------------------------------------------#
        if self["viscousSponge"]["spongeON"]:
            if not self["RHS"]["diffusion"]:
                raise pgConfigError(
                    True,
                    False,
                    "\nYou turned on viscousSponge without solving for diffusion.",
                )

        # ---------------------------------------------------------------------#
        # thermochem checks
        # ---------------------------------------------------------------------#
        self["thermochem"]["nChemSubSteps"] = max(
            1, self["thermochem"]["nChemSubSteps"]
        )
