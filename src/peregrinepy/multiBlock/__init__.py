from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from ..integrators import getIntegrator
from ..thermoTransport import thtrdat, findUserSpData
from peregrinepy import compute

from ..misc import null

"""
This is the mother ship of a peregrine case. The function buildSolver
sets all the attributes of the solver class that we need to run a case, checks
that our config file makes sense, etc.
"""


class pgConfigError(Exception):
    def __init__(self, setting, option, altMessage=""):
        message = f"Unknown PEREGRINE config {setting} option: {option}. "
        super().__init__(message + altMessage)


#########################################
# Consistify
#########################################
def setConsistify(cls, config):
    """
    Sets the funcitons for the consistify routine calls.
    see PEREGRINE/src/peregrinepy/consistify.py
    """

    # EOS
    eos = config["mcPhysics"]["eos"]
    try:
        cls.eos = getattr(compute.thermo, eos)
    except AttributeError:
        raise pgConfigError("eos", eos)

    # Transport properties
    if config["RHS"]["diffusion"]:
        trans = config["mcPhysics"]["trans"]
        try:
            cls.trans = getattr(compute.transport, trans)
        except AttributeError:
            raise pgConfigError("trans", trans)

    else:
        cls.trans = null

    # Switching function between primary and secondary advective fluxes
    #  If we aren't using a secondary flux function, we rely on the
    #  initialization of the switch array "phi" = 0.0 and then
    #  just never change it.
    switch = config["RHS"]["switchAdvFlux"]
    if switch is None:
        cls.switch = null
    else:
        try:
            cls.switch = getattr(compute.switches, switch)
        except AttributeError:
            raise pgConfigError("switchAdvFlux", switch)

        cls.phiComm = True


#########################################
# RHS
#########################################
def setRHS(cls, config):
    """
    Set the functions for the right hand side calls,
    see PEREGRINE/src/peregrinepy/RHS.py
    """

    # Primary advective fluxes
    primary = config["RHS"]["primaryAdvFlux"]
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
        assert (
            shock is not None
        ), "*** You set a secondary flux without a shock handler!"
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

    # spatial derivatives, subgrid mode, diffusive fluxes
    if config["RHS"]["diffusion"]:
        cls.dqdxyz = getattr(compute.utils, "dq2FD")

        # Subgrid models
        if config["RHS"]["subgrid"] is not None:
            sgs = config["RHS"]["subgrid"]
            try:
                cls.sgs = getattr(compute.subgrid, sgs)
            except AttributeError:
                raise pgConfigError("sgs", sgs)
        else:
            cls.sgs = null
        cls.diffFlux = compute.diffFlux.alphaDampingFlux
        cls.applyDiffFlux = compute.utils.applyFlux

    else:
        cls.dqdxyz = null
        cls.sgs = null
        cls.diffFlux = null
        cls.applyDiffFlux = null

    if config["viscousSponge"]["spongeON"]:
        cls.viscousSponge = compute.utils.viscousSponge
    else:
        cls.viscousSponge = null

    # Chemical source terms
    if config["mcPhysics"]["chemistry"]:
        mech = config["mcPhysics"]["mixture"]
        if cls.stepType in ["explicit", "dualTime"]:
            try:
                cls.expChem = getattr(compute.chemistry, mech)
                cls.impChem = null
            except AttributeError:
                raise pgConfigError("mechanism", mech)
        # If we are using an implicit chemistry integration
        #  we need to set it here and set the explicit
        #  module to null so it is not called in RHS
        elif cls.stepType == "split":
            try:
                cls.expChem = null
                cls.impChem = getattr(compute.chemistry, mech)
                if config["mcPhysics"]["nChemSubSteps"] > 1:
                    import warnings

                    warnings.warn(
                        "WARNING: nChemSubSteps > 1 with implicit chemistry. This has no effect.",
                        RuntimeWarning,
                    )
                    config["mcPhysics"]["nChemSubSteps"] = 1
            except AttributeError:
                raise pgConfigError("mechanism", mech)
    else:
        cls.expChem = null
        cls.impChem = null


def howManyNG(config):
    """
    Determine how many ghost layers does this case need.
    """
    advFluxNG = {
        "KEEP": 1,
        "KEEPpe": 1,
        "KEPaEC": 1,
        "centralDifference": 1,
        "fourthOrderKEEP": 2,
        "hllc": 1,
        "rusanov": 1,
        "muscl2hllc": 2,
        "muscl2rusanov": 2,
        "scalarDissipation": 2,
        None: 1,
    }

    subgridNG = {
        "smagorinsky": 1,
        None: 1,
    }

    ng = 1

    # First check primary advective flux
    pAdv = config["RHS"]["primaryAdvFlux"]
    ng = max(ng, advFluxNG[pAdv])

    # Check seconary advective flux
    sAdv = config["RHS"]["secondaryAdvFlux"]
    ng = max(ng, advFluxNG[sAdv])

    sub = config["RHS"]["subgrid"]
    ng = max(ng, subgridNG[sub])

    return ng


def buildSolver(config, nblks=None, myblocks=None):
    """A runnable solver multiBlock: the time integrator merged in, the species
    data attached, and the consistify and RHS methods set from what the config
    asks for. Unlike grid.fromGrid this does not fill a container, it builds
    and wires one. :myblocks: are the block numbers this rank is responsible
    for, which is also how many blocks it holds."""
    if myblocks is not None:
        nblks = len(myblocks)

    # merge the time integration class with the multiBlock solver class
    ti = config["timeIntegration"]["integrator"]
    name = "solver" + ti
    mbsolver = type(name, (solver, getIntegrator(ti)), dict(name=name))

    spn = list(findUserSpData(config).keys())
    cls = mbsolver(nblks, spn, ng=howManyNG(config), config=config)

    # in parallel the blocks are numbered by the partition, not by order
    if myblocks is not None:
        for blk, nblki in zip(cls, myblocks):
            blk.nblki = blk.baseNblki = nblki

    cls.thtrdat = thtrdat(config)
    setConsistify(cls, config)
    setRHS(cls, config)

    return cls
