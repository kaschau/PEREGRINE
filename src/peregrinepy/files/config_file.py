# -*- coding: utf-8 -*-

""" config.py

Authors:

Kyle Schau

This module holds a python dictionary version
of a standard PEREGRINE config file.

"""

from ..misc import FrozenDict


class config_file(FrozenDict):
    def __init__(self):
        self["io"] = FrozenDict(
            {"griddir": "./Grid", "inputdir": "./Input", "outputdir": "./Output"}
        )
        self["simulation"] = FrozenDict(
            {
                "niter": 1,
                "dt": 1e-3,
                "restart_from": 0,
                "animate": True,
                "niterout": 10,
                "niterprint": 1,
            }
        )

        self["solver"] = FrozenDict({"time_integration": "rk4"})

        self["RHS"] = FrozenDict(
            {
                "primaryAdvFlux": "centralEuler",
                "secondaryAdvFlux": None,
                "switchAdvFlux": None,
                "diffusion": True,
                "diffFlux": "centralVisc",
            }
        )

        self["thermochem"] = FrozenDict(
            {
                "spdata": "pure-debug.yaml",
                "eos": "cpg",
                "trans": "kineticTheory",
                "chemistry": False,
                "mechanism": None,
            }
        )

        self["Kokkos"] = FrozenDict({"Space": "Default"})

        for key in self.keys():
            self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()
