import kokkos
import yaml
from pathlib import Path
from ..compute.thermo import thtrdat_
from .completeSpecies import completeSpecies
from .findUserSpData import findUserSpData
from ..misc import frozenDict, createViewMirrorArray


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
            }
        )
        self.array._freeze()
        self.mirror._freeze()

        # Determine what kokkos space we are living in
        space = config["Kokkos"]["Space"]

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
        Ru = refsp["Ru"]
        self.Ru = Ru

        # Species names string
        speciesNames = list(usersp.keys())
        self.speciesNames = speciesNames

        # Species MW
        self.array["MW"] = completeSpecies("MW", usersp, refsp)
        shape = [ns]
        createViewMirrorArray(self, ["MW"], shape, space)

        ########################################
        # Set thermodynamic properties
        ########################################
        # Set either constant cp or NASA7 polynomial coefficients
        if config["thermochem"]["eos"] == "cpg":
            # Values for constant Cp
            # J/(kg.K)
            self.array["cp0"] = completeSpecies("cp0", usersp, refsp)
            shape = [ns]
            createViewMirrorArray(self, ["cp0"], shape, space)

        elif config["thermochem"]["eos"] == "tpg":
            self.array["NASA7"] = completeSpecies("NASA7", usersp, refsp)
            shape = [ns, 15]
            createViewMirrorArray(self, ["NASA7"], shape, space)
        else:
            raise KeyError(
                f'PEREGRINE ERROR: Unknown EOS {config["thermochem"]["eos"]}'
            )

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
                createViewMirrorArray(self, ["muPoly"], shape, space)

                self.array["kappaPoly"] = kappaPoly
                shape = [ns, 5]
                createViewMirrorArray(self, ["kappaPoly"], shape, space)

                self.array["DijPoly"] = DijPoly
                shape = [int(ns * (ns + 1) / 2), 5]
                createViewMirrorArray(self, ["DijPoly"], shape, space)

            elif config["thermochem"]["trans"] == "constantProps":

                self.array["mu0"] = completeSpecies("mu0", usersp, refsp)
                shape = [ns]
                createViewMirrorArray(self, ["mu0"], shape, space)

                self.array["kappa0"] = completeSpecies("kappa0", usersp, refsp)
                shape = [ns]
                createViewMirrorArray(self, ["kappa0"], shape, space)
