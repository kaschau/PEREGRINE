from pathlib import Path

import yaml
from kokkos import deep_copy

from ..compute.thermo import thtrdat_
from ..misc import createViewMirrorArray, frozenDict
from .completeSpecies import completeSpecies
from .findUserSpData import findUserSpData


class thtrdat(thtrdat_):
    def __init__(self, config):
        thtrdat_.__init__(self)

        self.array = frozenDict(
            {
                "MW": None,
                "cp0": None,
                "NASA7": None,
                "muPoly": None,
                "kappaPoly": None,
                "DijPoly": None,
                "mu0": None,
                "kappa0": None,
                "Tcrit": None,
                "pcrit": None,
                "Vcrit": None,
                "acentric": None,
                "chungA": None,
                "chungB": None,
                "redDipole": None,
            }
        )
        self.mirror = frozenDict(
            {
                "MW": None,
                "cp0": None,
                "NASA7": None,
                "muPoly": None,
                "kappaPoly": None,
                "DijPoly": None,
                "mu0": None,
                "kappa0": None,
                "Tcrit": None,
                "pcrit": None,
                "Vcrit": None,
                "acentric": None,
                "chungA": None,
                "chungB": None,
                "redDipole": None,
            }
        )
        self.array._freeze()
        self.mirror._freeze()

        # Get user species data, the spData from the config file is just a list of strings,
        # then we assume we just want to grab those species, otherwise we use the thtr.yaml file.
        if type(config["thermochem"]["spdata"]) == list:
            usersp = {}
            for key in config["thermochem"]["spdata"]:
                usersp[key] = None
        else:
            usersp = findUserSpData(config)
        # Now we get the reference data to fill in missing information not provided by the user
        relpath = str(Path(__file__).parent)
        with open(f"{relpath}/database/speciesLibrary.yaml", "r") as f:
            refsp = yaml.load(f, Loader=yaml.SafeLoader)

        ns = len(usersp.keys())
        self.ns = ns
        # Ru = refsp["Ru"]
        # self.Ru = Ru

        # Species names string
        speciesNames = list(usersp.keys())
        self.speciesNames = speciesNames

        # Species MW
        self.array["MW"] = completeSpecies("MW", usersp, refsp)
        shape = [ns]
        createViewMirrorArray(self, ["MW"], shape)

        ########################################
        # Set thermodynamic properties
        ########################################
        # Set either constant cp or NASA7 polynomial coefficients
        if config["thermochem"]["eos"] == "cpg":
            # Values for constant Cp
            # J/(kg.K)
            self.array["cp0"] = completeSpecies("cp0", usersp, refsp)
            shape = [ns]
            createViewMirrorArray(self, ["cp0"], shape)

        elif config["thermochem"]["eos"] in ["tpg", "cubic"]:
            self.array["NASA7"] = completeSpecies("NASA7", usersp, refsp)
            shape = [ns, 15]
            createViewMirrorArray(self, ["NASA7"], shape)
        else:
            raise KeyError(
                f'PEREGRINE ERROR: Unknown EOS {config["thermochem"]["eos"]}'
            )

        # Extra properties for cubic eos
        if config["thermochem"]["eos"] == "cubic":
            for var in ["Tcrit", "pcrit", "Vcrit", "acentric"]:
                self.array[var] = completeSpecies(var, usersp, refsp)
                shape = [ns]
                createViewMirrorArray(self, [var], shape)

        ################################
        # Set transport properties
        ################################

        if config["RHS"]["diffusion"]:
            if "kineticTheory" in config["thermochem"]["trans"]:
                from .kineticTheoryPoly import kineticTheoryPoly

                (
                    muPoly,
                    kappaPoly,
                    DijPoly,
                ) = kineticTheoryPoly(usersp, refsp, config["thermochem"]["eos"])

                self.array["muPoly"] = muPoly
                shape = [ns, 5]
                createViewMirrorArray(self, ["muPoly"], shape)

                self.array["kappaPoly"] = kappaPoly
                shape = [ns, 5]
                createViewMirrorArray(self, ["kappaPoly"], shape)

                self.array["DijPoly"] = DijPoly
                shape = [int(ns * (ns + 1) / 2), 5]
                createViewMirrorArray(self, ["DijPoly"], shape)

            elif config["thermochem"]["trans"] == "constantProps":

                self.array["mu0"] = completeSpecies("mu0", usersp, refsp)
                shape = [ns]
                createViewMirrorArray(self, ["mu0"], shape)

                self.array["kappa0"] = completeSpecies("kappa0", usersp, refsp)
                shape = [ns]
                createViewMirrorArray(self, ["kappa0"], shape)

            elif "chungDenseGas" in config["thermochem"]["trans"]:
                from .chungDenseGas import chungDenseGas

                (chungA, chungB, redDipole) = chungDenseGas(usersp, refsp)
                self.array["chungA"] = chungA
                shape = [ns, 10]
                createViewMirrorArray(self, ["chungA"], shape)
                self.array["chungB"] = chungB
                shape = [ns, 7]
                createViewMirrorArray(self, ["chungB"], shape)
                self.array["redDipole"] = redDipole
                shape = [ns]
                createViewMirrorArray(self, ["redDipole"], shape)

            else:
                raise KeyError(
                    f'PEREGRINE ERROR: Unknown TRANS {config["thermochem"]["trans"]}'
                )

    def updateDeviceView(self, vars):
        if type(vars) == str:
            vars = [vars]
        for var in vars:
            deep_copy(getattr(self, var), self.mirror[var])

    def updateHostView(self, vars):
        if type(vars) == str:
            vars = [vars]
        for var in vars:
            deep_copy(self.mirror[var], getattr(self, var))
