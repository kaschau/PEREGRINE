import yaml
from pathlib import Path
import kokkos
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
        self.array["MW"] = completeSpecies("MW", usersp, refsp)
        self.MW = kokkos.array(
            self.array["MW"],
            dtype=kokkos.double,
            space=space,
            dynamic=False,
        )

        ########################################
        # Set thermodynamic properties
        ########################################
        # Set either constant cp or NASA7 polynomial coefficients
        if config["thermochem"]["eos"] == "cpg":
            # Values for constant Cp
            # J/(kg.K)
            self.array["cp0"] = completeSpecies("cp0", usersp, refsp)
            self.cp0 = kokkos.array(
                self.array["cp0"],
                dtype=kokkos.double,
                space=space,
                dynamic=False,
            )
        elif config["thermochem"]["eos"] == "tpg":
            self.array["NASA7"] = completeSpecies("NASA7", usersp, refsp)
            self.NASA7 = kokkos.array(
                self.array["NASA7"],
                dtype=kokkos.double,
                space=space,
                dynamic=False,
            )
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
                    self.array["muPoly"],
                    self.array["kappaPoly"],
                    self.array["DijPoly"],
                ) = kineticTheoryPoly(usersp, refsp, config["thermochem"]["eos"])

                self.muPoly = kokkos.array(
                    self.array["muPoly"],
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.kappaPoly = kokkos.array(
                    self.array["kappaPoly"],
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
                self.DijPoly = kokkos.array(
                    self.array["DijPoly"],
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
            elif config["thermochem"]["trans"] == "constantProps":

                self.array["mu0"] = completeSpecies("mu0", usersp, refsp)
                self.mu0 = kokkos.array(
                    self.array["mu0"],
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )

                self.array["kappa0"] = completeSpecies("kappa0", usersp, refsp)
                self.kappa0 = kokkos.array(
                    self.array["kappa0"],
                    dtype=kokkos.double,
                    space=space,
                    dynamic=False,
                )
