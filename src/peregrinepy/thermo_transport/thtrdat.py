import kokkos
import numpy as np
import yaml
from pathlib import Path
from ..compute.thermo import thtrdat_
from .completeSpecies import completeSpecies
from .findUserSpData import findUserSpData


class thtrdat(thtrdat_):
    def __init__(self, config):
        thtrdat_.__init__(self)

        self.array = {
            "MW": None,
            "cp0": None,
            "NASA7": None,
            "muPoly": None,
            "kappaPoly": None,
            "DijPoly": None,
            "mu0": None,
            "kappa0": None,
        }
        self.mirror = {
            "MW": None,
            "cp0": None,
            "NASA7": None,
            "muPoly": None,
            "kappaPoly": None,
            "DijPoly": None,
            "mu0": None,
            "kappa0": None,
        }

        # Determine what kokkos space we are living in
        if config["Kokkos"]["Space"] in ["OpenMP", "Serial", "Default"]:
            space = kokkos.HostSpace
        elif config["Kokkos"]["Space"] in ["Cuda"]:
            space = kokkos.CudaSpace
        else:
            raise ValueError("What space?")

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
        MW = completeSpecies("MW", usersp, refsp)
        self.MW = kokkos.array(
            "MW",
            shape=MW.shape,
            dtype=kokkos.double,
            space=space,
            dynamic=False,
        )
        self.mirror["MW"] = kokkos.create_mirror_view(self.MW)
        self.array["MW"] = np.array(self.mirror["MW"], copy=False)
        self.array["MW"][:] = MW[:]
        kokkos.deep_copy(self.MW, self.mirror["MW"])

        ########################################
        # Set thermodynamic properties
        ########################################
        # Set either constant cp or NASA7 polynomial coefficients
        if config["thermochem"]["eos"] == "cpg":
            # Values for constant Cp
            # J/(kg.K)
            cp0 = completeSpecies("cp0", usersp, refsp)
            self.cp0 = kokkos.array(
                "cp0",
                shape=cp0.shape,
                dtype=kokkos.double,
                space=space,
                dynamic=False,
            )
            self.mirror["cp0"] = kokkos.create_mirror_view(self.cp0)
            self.array["cp0"] = np.array(self.mirror["cp0"], copy=False)
            self.array["cp0"][:] = cp0[:]
            kokkos.deep_copy(self.cp0, self.mirror["cp0"])

        elif config["thermochem"]["eos"] == "tpg":
            NASA7 = completeSpecies("NASA7", usersp, refsp)
            self.NASA7 = kokkos.array(
                "NASA7",
                shape=NASA7.shape,
                dtype=kokkos.double,
                space=space,
                dynamic=False,
            )
            self.mirror["NASA7"] = kokkos.create_mirror_view(self.NASA7)
            self.array["NASA7"] = np.array(self.mirror["NASA7"], copy=False)
            self.array["NASA7"][:] = NASA7[:]
            kokkos.deep_copy(self.NASA7, self.mirror["NASA7"])
            self.array["NASA7"] = completeSpecies("NASA7", usersp, refsp)
        else:
            raise KeyError(
                f'PEREGRINE ERROR: Unknown EOS {config["thermochem"]["eos"]}'
            )

        ################################
        # Set transport properties
        ################################

        if config["RHS"]["diffusion"]:
            if config["thermochem"]["trans"] == "kineticTheory":
                from .kineticTheoryPoly import kineticTheoryPoly

                (
                    muPoly,
                    kappaPoly,
                    DijPoly,
                ) = kineticTheoryPoly(usersp, refsp, config["thermochem"]["eos"])

                self.muPoly = kokkos.array(
                    "muPoly",
                    shape=muPoly.shape,
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.mirror["muPoly"] = kokkos.create_mirror_view(self.muPoly)
                self.array["muPoly"] = np.array(self.mirror["muPoly"], copy=False)
                self.array["muPoly"][:] = muPoly[:]
                kokkos.deep_copy(self.muPoly, self.mirror["muPoly"])
                self.array["muPoly"] = completeSpecies("muPoly", usersp, refsp)

                self.kappaPoly = kokkos.array(
                    "kappaPoly",
                    shape=kappaPoly.shape,
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.mirror["kappaPoly"] = kokkos.create_mirror_view(self.kappaPoly)
                self.array["kappaPoly"] = np.array(self.mirror["kappaPoly"], copy=False)
                self.array["kappaPoly"][:] = kappaPoly[:]
                kokkos.deep_copy(self.kappaPoly, self.mirror["kappaPoly"])
                self.array["kappaPoly"] = completeSpecies("kappaPoly", usersp, refsp)

                self.DijPoly = kokkos.array(
                    "DijPoly",
                    shape=DijPoly.shape,
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.mirror["DijPoly"] = kokkos.create_mirror_view(self.DijPoly)
                self.array["DijPoly"] = np.array(self.mirror["DijPoly"], copy=False)
                self.array["DijPoly"][:] = DijPoly[:]
                kokkos.deep_copy(self.DijPoly, self.mirror["DijPoly"])
                self.array["DijPoly"] = completeSpecies("DijPoly", usersp, refsp)

            elif config["thermochem"]["trans"] == "constantProps":

                mu0 = completeSpecies("mu0", usersp, refsp)
                self.mu0 = kokkos.array(
                    "mu0",
                    shape=mu0.shape,
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.mirror["mu0"] = kokkos.create_mirror_view(self.mu0)
                self.array["mu0"] = np.array(self.mirror["mu0"], copy=False)
                self.array["mu0"][:] = mu0[:]
                kokkos.deep_copy(self.mu0, self.mirror["mu0"])
                self.array["mu0"] = completeSpecies("mu0", usersp, refsp)

                kappa0 = completeSpecies("kappa0", usersp, refsp)
                self.kappa0 = kokkos.array(
                    "kappa0",
                    shape=kappa0.shape,
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.mirror["kappa0"] = kokkos.create_mirror_view(self.kappa0)
                self.array["kappa0"] = np.array(self.mirror["kappa0"], copy=False)
                self.array["kappa0"][:] = kappa0[:]
                kokkos.deep_copy(self.kappa0, self.mirror["kappa0"])
                self.array["kappa0"] = completeSpecies("kappa0", usersp, refsp)
