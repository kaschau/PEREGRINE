from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from .topology_block import topology_block
from .grid_block import grid_block
from .restart_block import restart_block
from .solver_block import solver_block

from ..integrators import rk1,rk4
from ..thermo import thermdat

import cantera as ct
from pathlib import Path

def generate_multiblock_solver(nblks, config, myblocks=None):

    #Get the time integrator from config file
    ti = config['solver']['time_integration']
    if ti == 'rk1':
        tic = rk1
    elif ti == 'rk4':
        tic = rk4

    name = 'solver'+ti
    mbsolver = type(name, (solver,tic), dict(name=name))

    #Get the number of species from the ct file
    relpath = str(Path(__file__).parent)+'/../thermo/'
    ct.add_directory(relpath)
    gas = ct.Solution(config['thermochem']['ctfile'])
    spn = gas.species_names

    #Instantiate the combined mbsolver+timeint object
    cls = mbsolver(nblks,spn)

    #In parallel we need to overwrite the generated block numbers
    if myblocks != None:
        assert (len(myblocks)==nblks),'You passed a quantity of block number assignments that != nblks'
        for blk,nblki in zip(cls,myblocks):
            blk.nblki = nblki

    #TODO: Do this better in the future
    #Stick the config file one
    cls.config = config

    #Stick the thermdat object on
    cls.thermdat = thermdat(config)

    #Stick the equation of state on
    if config['thermochem']['eos'] == 'cpg':
        from ..compute import cpg
        cls.eos = cpg
    elif config['thermochem']['eos'] == 'tpg':
        from ..compute import tpg
        cls.eos = tpg


    return cls
