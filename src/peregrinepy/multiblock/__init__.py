from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from ..integrators import get_integrator
from ..thermo_transport import thtrdat, get_eos, get_trans
from peregrinepy import compute


def null(*args):
    pass


class pgConfigError(Exception):
    def __init__(self, option):
        message = f'Unknown PEREGRINE config options {option}.'
        super().__init__(message)


#########################################
# Consistify
#########################################
def set_consistify(cls, config):
    cls.eos = get_eos(config["thermochem"]["eos"])
    if config["RHS"]["diffusion"]:
        cls.trans = get_trans(config["thermochem"]["trans"])
        cls.dqdxyz = compute.utils.dq2FD
    else:
        cls.trans = null
        cls.dqdxyz = null
    # Switching function between non-diss and diss adv fluxes
    cls.switch = null


#########################################
# RHS
#########################################
def set_RHS(cls, config):
    # Non dissipative advective fluxes
    if config["RHS"]["nonDissAdvFlux"] == "centralEuler":
        cls.nonDissAdvFlux = compute.advFlux.centralEuler
    elif config["RHS"]["nonDissAdvFlux"] is None:
        cls.nonDissAdvFlux = null
    else:
        raise pgConfigError(config["RHS"]["nonDissAdvFlux"])

    # Dissipative advective fluxes
    if config["RHS"]["dissAdvFlux"] == "upwind":
        cls.dissAdvFlux = compute.advFlux.upwind
    elif config["RHS"]["dissAdvFlux"] is None:
        cls.dissAdvFlux = null
    else:
        raise pgConfigError(config["RHS"]["nonDissAdvFlux"])

    # Diffusive fluxes
    if config["RHS"]["diffusion"]:
        if config["RHS"]["diffFlux"] == "centralVisc":
            cls.diffFlux = compute.diffFlux.centralVisc
        else:
            raise pgConfigError(config["RHS"]["diffFlux"])
    else:
        cls.diffFlux = null

    # Chemical source terms
    if config["thermochem"]["chemistry"]:
        if config["thermochem"]["mechanism"] == "chem_CH4_O2_Stanford_Skeletal":
            cls.expchem = compute.chemistry.chem_CH4_O2_Stanford_Skeletal
        elif config["thermochem"]["mechanism"] == "chem_GRI30":
            cls.expchem = compute.chemistry.chem_GRI30
        else:
            raise pgConfigError(config["thermochem"]["chemistry"])
        # If we are using an implicit chemistry integration
        #  we need to set it here and set the explicit
        #  module to null so it is not called in RHS
        if config["solver"]["time_integration"] in ["strang"]:
            cls.impchem = cls.expchem
            cls.expchem = null

    else:
        cls.expchem = null
        cls.impchem = null


def generate_multiblock_solver(nblks, config, myblocks=None):

    # Get the time integrator from config file
    ti = config["solver"]["time_integration"]
    tic = get_integrator(ti)
    name = "solver" + ti
    mbsolver = type(name, (solver, tic), dict(name=name))

    # Get the number of species from the spdata file
    spdat = thtrdat(config)
    spn = spdat.species_names
    # Instantiate the combined mbsolver+timeint object
    cls = mbsolver(nblks, spn)

    # In parallel we need to overwrite the generated block numbers
    if myblocks is not None:
        assert (
            len(myblocks) == nblks
        ), "You passed a quantity of block number assignments that != nblks"
        for blk, nblki in zip(cls, myblocks):
            blk.nblki = nblki

    # Set the config file on
    cls.config = config
    # Set the thtrdat object on
    cls.thtrdat = spdat

    # Set the compute routines for consistify
    set_consistify(cls, config)

    # Set the compute routines for the RHS
    set_RHS(cls, config)

    return cls
