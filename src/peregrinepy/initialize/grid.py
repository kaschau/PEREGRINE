from ..block import block
from ..readers import read_grid

def grid(myBlocks):

    myCompBlocks = []
    for nblki in myBlocks:
        b = block(nblki)
        myCompBlocks.append(b)

    read_grid(myCompBlocks)

    return myCompBlocks
