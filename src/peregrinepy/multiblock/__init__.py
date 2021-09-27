from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from ..integrators import get_integrator
from ..thermo_transport import thtrdat, get_eos, get_trans
from peregrinepy import compute


def dummy(a=None, b=None, c=None):
    pass


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

    #########################################
    # Consistify
    #########################################
    cls.eos = get_eos(config["thermochem"]["eos"])
    cls.trans = get_trans(config["thermochem"]["trans"])
    if config["diffusion"]:
        cls.dqdxyz = compute.utils.dq2FD
    else:
        cls.dqdxyz = dummy
    # Switching function between non-diss and diss adv fluxes
    cls.switch = dummy

    #########################################
    # RHS
    #########################################
    # Non dissipative advective fluxes
    cls.nonDissAdvFlx = compute.advFlx.centeredEuler
    # Dissipative advective fluxes
    cls.DissAdvFlx = dummy

    # Diffusive fluxes fluxes
    if config["diffusion"]:
        cls.diffFlx = compute.diffFlx.centeredVisc
    else:
        cls.diffFlx = dummy

    # Chemical source terms
    if config["thermochem"]["chemistry"]:
        if config["thermochem"]["mechanism"] == "chem_CH4_O2_Stanford_Skeletal":
            cls.chem = compute.chemistry.chem_CH4_O2_Stanford_Skeletal
        elif config["thermochem"]["mechanism"] == "chem_GRI30":
            cls.chem = compute.chemistry.chem_GRI30
        else:
            raise ValueError("What mechanism?")
    else:
        cls.chem = dummy

    return cls
