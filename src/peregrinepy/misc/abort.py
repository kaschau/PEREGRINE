from mpi4py import MPI
import kokkos


def abort(mb):

    if mb.config["Catalyst"]["coprocess"]:
        mb.coproc.finalize()

    kokkos.finalize()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    comm.Barrier()
    if rank == 0:
        comm.Abort()