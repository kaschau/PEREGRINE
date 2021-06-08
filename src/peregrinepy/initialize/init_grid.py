from ..block import block
from ..readers import read_grid

def init_grid(myBlocks,config):

    myCompBlocks = []
    for nblki in myBlocks:
        b = block(nblki)
        myCompBlocks.append(b)

    read_grid(myCompBlocks,config)

    return myCompBlocks
