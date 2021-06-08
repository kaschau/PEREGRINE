from ..block import block
from ..readers import read_grid

def init_grid(compBlocks,config):

    read_grid(compBlocks,config)

    return compBlocks
