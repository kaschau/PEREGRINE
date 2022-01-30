from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from ..integrators import getIntegrator
from ..thermo_transport import thtrdat, findUserSpData
from peregrinepy import compute

from ..misc import null


class pgConfigError(Exception):
    def __init__(self, setting, option, altMessage=""):
        message = f"Unknown PEREGRINE config {setting} option: {option}. "
        super().__init__(message + altMessage)


#########################################
# Consistify
#########################################
def setConsistify(cls, config):

    # EOS
    eos = config["thermochem"]["eos"]
    try:
        cls.eos = getattr(compute.thermo, eos)
    except AttributeError:
        raise pgConfigError("eos", eos)

    # Diffusion, transport properties, spatial derivatives.
    if config["RHS"]["diffusion"]:
        trans = config["thermochem"]["trans"]
        try:
            cls.trans = getattr(compute.transport, trans)
        except AttributeError:
            raise pgConfigError("trans", trans)

        dqO = config["RHS"]["diffOrder"]
        try:
            cls.dqdxyz = getattr(compute.utils, f"dq{dqO}FD")
        except AttributeError:
            raise pgConfigError("diffOrder", f"dq{dqO}FD")
        cls.commList += ["dqdx", "dqdy", "dqdz"]
    else:
        cls.trans = null
        cls.dqdxyz = null

    # Switching function between primary and secondary advective fluxes
    #  If we aren't using a secondary flux function, we rely on the
    #  initialization of the switch array "phi" = 0.0 and then
    #  just never change it.
    if config["RHS"]["switchAdvFlux"] is None:
        cls.switch = null
    else:
        switch = config["RHS"]["switchAdvFlux"]
        try:
            cls.switch = getattr(compute.switches, switch)
        except AttributeError:
            raise pgConfigError("switchAdvFlux", switch)

        # Just as a check, we will make sure we didnt accidentially
        #  set a switch without setting a secondary flux
        if config["RHS"]["secondaryAdvFlux"] is None:
            raise pgConfigError(
                switch,
                "You set a advective flux switching option without a secondary flux.",
            )
        cls.commList += ["phi"]


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
            raise pgConfigError("primaryAdvFlux", primary)
    # How to apply primary flux
    shock = config["RHS"]["shockHandling"]
    if shock is None or shock == "artificialDissipation":
        cls.applyPrimaryAdvFlux = compute.utils.applyFlux
    elif shock == "hybrid":
        cls.applyPrimaryAdvFlux = compute.utils.applyHybridFlux
    else:
        raise pgConfigError("shockHandling", shock)

    # Secondary advective fluxes
    secondary = config["RHS"]["secondaryAdvFlux"]
    if secondary is None:
        cls.secondaryAdvFlux = null
    else:
        try:
            cls.secondaryAdvFlux = getattr(compute.advFlux, secondary)
        except AttributeError:
            raise pgConfigError("secondaryAdvFlux", secondary)
    # How to apply secondary flux
    if shock is None:
        cls.applySecondaryAdvFlux = null
    elif shock == "artificialDissipation":
        cls.applySecondaryAdvFlux = compute.utils.applyDissipationFlux
    elif shock == "hybrid":
        cls.applySecondaryAdvFlux = compute.utils.applyHybridFlux

    # Diffusive fluxes
    if config["RHS"]["diffusion"]:
        cls.diffFlux = compute.diffFlux.diffusiveFlux
        cls.applyDiffFlux = compute.utils.applyFlux
    else:
        cls.diffFlux = null
        cls.applyDiffFlux = null

    # Chemical source terms
    if config["thermochem"]["chemistry"]:
        mech = config["thermochem"]["mechanism"]
        if config["solver"]["timeIntegration"] in ["rk1", "rk3", "rk4"]:
            try:
                cls.expChem = getattr(compute.chemistry, mech)
                cls.impChem = null
            except AttributeError:
                raise pgConfigError("mechanism", mech)
        # If we are using an implicit chemistry integration
        #  we need to set it here and set the explicit
        #  module to null so it is not called in RHS
        elif config["solver"]["timeIntegration"] in ["strang"]:
            try:
                cls.expChem = null
                cls.impChem = getattr(compute.chemistry, mech)
            except AttributeError:
                raise pgConfigError("mechanism", mech)
    else:
        cls.expChem = null
        cls.impChem = null


def howManyNG(config):
    ng = 1

    # First check advective term order
    if config["RHS"]["primaryAdvFlux"] == "fourthOrderKEEP":
        ng = max(ng, 2)

    # Check seconary advective term order
    if config["RHS"]["secondaryAdvFlux"] == "jamesonDissipation":
        ng = max(ng, 2)

    # Now check diffusion term order
    dO = config["RHS"]["diffOrder"]
    if dO == 2:
        ng = max(ng, 1)
    elif dO == 4:
        ng = max(ng, 2)

    return ng


def generateMultiBlockSolver(nblks, config, myblocks=None):

    # Get the time integrator from config file
    ti = config["solver"]["timeIntegration"]
    tic = getIntegrator(ti)
    name = "solver" + ti
    # Merge the time integration class with the multiblock solver class
    mbsolver = type(name, (solver, tic), dict(name=name))

    # Get the species names from the spdata file
    spn = list(findUserSpData(config).keys())

    # Get the number of ghost layers
    ng = howManyNG(config)
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
    cls.thtrdat = thtrdat(config)

    # Set the compute routines for consistify
    setConsistify(cls, config)

    # Set the compute routines for the RHS
    setRHS(cls, config)

    return cls
