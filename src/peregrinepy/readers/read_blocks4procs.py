
from mpi4py import MPI


def read_blocks4procs(config):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        with open(f"{config['io']['inputdir']}/blocks4procs.inp") as f:
            lines = [i for i in f.readlines() if not i.strip().startswith('#')]
            sendranks = [i+1 for i in range(len(lines[1::]))]
            for send,line in zip(sendranks,lines[1::]):
                comm.send(line, dest=send, tag=1111)
            line = lines[0]
    else:
        line = comm.recv(source=0, tag=1111)

    myblocks = [int(i) for i in line.split(',')]

    return myblocks
