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
                "restartFrom": 0,
                "animateArchive": True,
                "animateRestart": False,
                "niterArchive": 1e10,
                "niterRestart": 10,
                "niterPrint": 1,
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
                "trans": "kineticTheory",
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
