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
            {"griddir": "./Grid", "inputdir": "./Input", "outputdir": "./Output"}
        )
        self["simulation"] = frozenDict(
            {
                "niter": 1,
                "dt": 1e-3,
                "restartFrom": 0,
                "animate": True,
                "niterout": 10,
                "niterprint": 1,
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
            }
        )

        self["thermochem"] = frozenDict(
            {
                "spdata": "pure-debug.yaml",
                "eos": "cpg",
                "trans": "kineticTheory",
                "chemistry": False,
                "mechanism": None,
            }
        )

        self["Kokkos"] = frozenDict({"Space": "Default"})

        for key in self.keys():
            self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()
