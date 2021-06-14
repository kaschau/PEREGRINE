#!/usr/bin/env python
import mpi4py.rc
mpi4py.rc.initialize = False

import kokkos

import peregrinepy as pg
import time

import sys

def simulate(config_file_path):
    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()

    comm,rank,size = pg.mpicomm.mpiutils.get_comm_rank_size()
    # Ensure MPI is suitably cleaned up
    pg.mpicomm.mpiutils.register_finalize_handler()

    config = pg.initialize.init_config(config_file_path)
    #set here for now
    config['RunTime']['ngls'] = 2
    mb = pg.initialize.init_multiblock(config)

    pg.mpicomm.blockcomm.communicate(mb,['x','y','z'])

    if rank == 0:
        print('blk0')
        print('x')
        print(mb[0].array['x'][-2::,:,:])
        print('y')
        print(mb[0].array['y'][-2::,:,:])
        print('z')
        print(mb[0].array['z'][-2::,:,:])
    else:
        print('blk1')
        print('x')
        print(mb[0].array['x'][0:2])
        print('y')
        print(mb[0].array['y'][0:2])
        print('z')
        print(mb[0].array['z'][0:2])

    #ts = time.time()
    #for b in mb:
    #    pg.compute.add3D(b,1.0)
    #    pg.compute.add3D(b,1.0)
    #    pg.compute.add3D(b,1.0)
    #print(time.time()-ts, 'took this many seconds')
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
