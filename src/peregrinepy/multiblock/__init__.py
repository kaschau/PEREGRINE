from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from ..integrators import get_integrator
from ..thermo_transport import thtrdat
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
    eos = config["thermochem"]["eos"]
    try:
        cls.eos = getattr(compute.thermo, eos)
    except AttributeError:
        raise pgConfigError(eos)
    if config["RHS"]["diffusion"]:
        trans = config["thermochem"]["trans"]
        try:
            cls.trans = getattr(compute.transport, trans)
        except AttributeError:
            raise pgConfigError(trans)
        cls.dqdxyz = compute.utils.dq2FD
    else:
        cls.trans = null
        cls.dqdxyz = null

    # Switching function between primary and secondary advective fluxes
    cls.switch = null


#########################################
# RHS
#########################################
def set_RHS(cls, config):
    # Primary advective fluxes
    primary = config["RHS"]["primaryAdvFlux"]
    if primary is None:
        raise ValueError("Primary advective flux cannot be None")
    else:
        try:
            cls.primaryAdvFlux = getattr(compute.advFlux, primary)
        except AttributeError:
            raise pgConfigError(primary)

    # Secondary advective fluxes
    secondary = config["RHS"]["secondaryAdvFlux"]
    if secondary is None:
        cls.secondaryAdvFlux = null
    else:
        try:
            cls.secondaryAdvFLux = getattr(compute.advFlux, secondary)
        except AttributeError:
            raise pgConfigError(secondary)

    # Diffusive fluxes
    if config["RHS"]["diffusion"]:
        diff = config["RHS"]["diffFlux"]
        try:
            cls.primaryAdvFlux = getattr(compute.diffFlux, diff)
        except AttributeError:
            raise pgConfigError(diff)
    else:
        cls.diffFlux = null

    # Chemical source terms
    if config["thermochem"]["chemistry"]:
        mech = config["thermochem"]["mechanism"]
        try:
            cls.expchem = getattr(compute.chemistry, mech)
        except AttributeError:
            raise pgConfigError(diff)
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
