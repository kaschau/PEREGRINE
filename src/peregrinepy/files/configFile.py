# -*- coding: utf-8 -*-

""" config.py

Authors:

Kyle Schau

This module holds a python dictionary version
of a standard PEREGRINE config file.

"""

from ..misc import frozenDict


class configFile(frozenDict):
    def __init__(self):
        self["io"] = frozenDict(
            {
                "gridDir": "./Grid",
                "inputDir": "./Input",
                "restartDir": "./Restart",
                "archiveDir": "./Archive",
            }
        )
        self["simulation"] = frozenDict(
            {
                "niter": 1,
                "dt": 1e-3,
                "restartFrom": 0,
                "animateArchive": True,
                "animateRestart": False,
                "niterArchive": 1e10,
                "niterRestart": 10,
                "niterPrint": 1,
                "variableTimeStep": False,
                "maxCFL": 0.1,
                "checkNan": False,
            }
        )

        self["solver"] = frozenDict({"timeIntegration": "rk4"})

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
                "trans": "kineticTheory",
                "chemistry": False,
                "mechanism": None,
            }
        )

        self["Catalyst"] = frozenDict({"coprocess": False, "cpFile": "./coproc.py"})

        for key in self.keys():
            self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()
