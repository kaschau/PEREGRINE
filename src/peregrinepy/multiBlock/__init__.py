from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from ..integrators import getIntegrator
from ..thermo_transport import thtrdat
from peregrinepy import compute

from ..misc import null


class pgConfigError(Exception):
    def __init__(self, option):
        message = f"Unknown PEREGRINE config option {option}."
        super().__init__(message)


#########################################
# Consistify
#########################################
def setConsistify(cls, config):

    # EOS
    eos = config["thermochem"]["eos"]
    try:
        cls.eos = getattr(compute.thermo, eos)
    except AttributeError:
        raise pgConfigError(eos)

    # Diffusion, transport properties, spatial derivatives.
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

    # If we aren't using a secondary flux function, we rely on the
    #  initialization of the switch array "phi" = 0.0 and then
    #  just never change it.
    if config["RHS"]["secondaryAdvFlux"] is None:
        cls.switch = null
    else:
        switch = config["RHS"]["switchAdvFlux"]
        try:
            cls.switch = getattr(compute.switches, switch)
        except AttributeError:
            raise pgConfigError(switch)


#########################################
# RHS
#########################################
def setRHS(cls, config):
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
            cls.secondaryAdvFlux = getattr(compute.advFlux, secondary)
        except AttributeError:
            raise pgConfigError(secondary)

    # Diffusive fluxes
    if config["RHS"]["diffusion"]:
        diff = config["RHS"]["diffFlux"]
        try:
            cls.diffFlux = getattr(compute.diffFlux, diff)
        except AttributeError:
            raise pgConfigError(diff)
    else:
        cls.diffFlux = null

    # Chemical source terms
    if config["thermochem"]["chemistry"]:
        mech = config["thermochem"]["mechanism"]
        try:
            cls.expChem = getattr(compute.chemistry, mech)
        except AttributeError:
            raise pgConfigError(diff)
        # If we are using an implicit chemistry integration
        #  we need to set it here and set the explicit
        #  module to null so it is not called in RHS
        if config["solver"]["timeIntegration"] in ["strang"]:
            cls.impChem = cls.expChem
            cls.expChem = null
    else:
        cls.expChem = null
        cls.impChem = null


def generateMultiBlockSolver(nblks, config, myblocks=None):

    # Get the time integrator from config file
    ti = config["solver"]["timeIntegration"]
    tic = getIntegrator(ti)
    name = "solver" + ti
    mbsolver = type(name, (solver, tic), dict(name=name))

    # Get the number of species from the spdata file
    spdat = thtrdat(config)
    spn = spdat.speciesNames

    # Get the number of ghost layers
    if config["RHS"]["primaryAdvFlux"] == "fourthOrderKEEP":
        ng = 2
    else:
        ng = 1
    # Instantiate the combined mbsolver+timeint object
    cls = mbsolver(nblks, spn, ng=ng)

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
    setConsistify(cls, config)

    # Set the compute routines for the RHS
    setRHS(cls, config)

    return cls
