from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver

from .topology_block import topology_block
from .grid_block import grid_block
from .restart_block import restart_block
from .solver_block import solver_block

from ..integrators import rk1,rk4

def generate_multiblock_solver(nblki, config):
    ti = config['solver']['time_integration']
    if ti == 'rk1':
        tic = rk1
    elif ti == 'rk4':
        tic = rk4

    name = 'solver'+ti
    mbsolver = type(name, (solver,tic), dict(name=name))

    return mbsolver(nblki,config)
