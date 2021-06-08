
from mpi4py import MPI

from ..multiblock import multiblock
from ..readers import read_blocks4procs

def init_multiblock(config):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        blocks4procs = read_blocks4procs(config)
        sendranks = [i+1 for i in range(len(blocks4procs[1::]))]
        for send,line in zip(sendranks,blocks4procs[1::]):
            comm.send(line, dest=send, tag=1111)
        myblocks = blocks4procs[0]
    else:
        myblocks = comm.recv(source=0, tag=1111)

    print(myblocks)
    compBlocks = multiblock(len(myblocks))

    for i,nblki in enumerate(myblocks):
        compBlocks[i].nblki = nblki


    return compBlocks
