from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from .topology_block import topology_block
from .grid_block import grid_block
from .restart_block import restart_block
from .solver_block import solver_block

from ..integrators import get_integrator
from ..thermo_transport import thtrdat, get_eos

from pathlib import Path

def generate_multiblock_solver(nblks, config, myblocks=None):

    #Get the time integrator from config file
    ti = config['solver']['time_integration']
    tic = get_integrator(ti)
    name = 'solver'+ti
    mbsolver = type(name, (solver,tic), dict(name=name))

    #Get the number of species from the spdata file
    spdat = thtrdat(config)
    spn = spdat.species_names
    #Instantiate the combined mbsolver+timeint object
    cls = mbsolver(nblks,spn)

    #In parallel we need to overwrite the generated block numbers
    if myblocks != None:
        assert (len(myblocks)==nblks),'You passed a quantity of block number assignments that != nblks'
        for blk,nblki in zip(cls,myblocks):
            blk.nblki = nblki

    #TODO: Do this better in the future
    #Stick the config file on
    cls.config = config
    #Stick the thtrdat object on
    cls.thtrdat = spdat
    #Stick the equation of state on
    cls.eos = get_eos(config['thermochem']['eos'])

    #Stick the chemistry mechanism on
    if config['thermochem']['chemistry']:
        if config['thermochem']['mechanism'] == 'chem_CH4_O2_Stanford_Skeletal':
            from ..compute.chemistry import chem_CH4_O2_Stanford_Skeletal
            cls.chem = chem_CH4_O2_Stanford_Skeletal
        elif config['thermochem']['mechanism'] == 'chem_GRI30':
            from ..compute.chemistry import chem_GRI30
            cls.chem = chem_GRI30
        else:
            raise ValueError("What mechanism?")

    return cls
