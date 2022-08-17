# -*- coding: utf-8 -*-

""" config.py

Authors:

Kyle Schau

This module holds a python dictionary version
of a standard PEREGRINE config file.

"""

from ..misc import frozenDict


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
                "niterArchive": 1e10,
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
                "integrator": "rk4",
                "dt": 1e-3,
                "variableTimeStep": False,
                "maxDt": 1e-3,
                "maxCFL": 0.1,
            }
        )

        self["RHS"] = frozenDict(
            {
                "shockHandling": None,
                "primaryAdvFlux": "secondOrderKEEP",
                "secondaryAdvFlux": None,
                "switchAdvFlux": None,
                "diffusion": False,
                "diffOrder": 2,
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

        for key in self.keys():
            self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()

    def validateConfig(self):
        ###################################################################################
        # Sanity checks go here:
        ###################################################################################

        # Simulation checks

        # Time Integration Checks
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

        # RHS checks
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
