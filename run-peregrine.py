#!/usr/bin/env python
import mpi4py.rc
mpi4py.rc.initialize = False

import kokkos

import peregrinepy as pgpy
import time

import sys
def simulate(config_file_path):
    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()

    comm,rank,size = pgpy.mpicomm.mpiutils.get_comm_rank_size()
    # Ensure MPI is suitably cleaned up
    pgpy.mpicomm.mpiutils.register_finalize_handler()

    config = pgpy.initialize.init_config(config_file_path)
    #set here for now
    config['RunTime']['ngls'] = 2
    mb = pgpy.initialize.init_multiblock(config)
    pgpy.initialize.init_grid(mb,config)

    #ts = time.time()
    #for b in myCompBlocks:
    #    pgpy.compute.add3D(b,1.0)
    #    pgpy.compute.add3D(b,1.0)
    #    pgpy.compute.add3D(b,1.0)
    #print(time.time()-ts, 'took this many seconds')
    #if rank == 1:
    print(mb[0].array['x'])
    print(mb[0].connectivity)
    #return CompBlocks

    # Finalise MPI
    MPI.Finalize()

if __name__ == "__main__":
    config_file_path = sys.argv[1]
    try:
        kokkos.initialize()
        simulate(config_file_path)
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
