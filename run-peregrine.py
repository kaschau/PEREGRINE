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

    mb = pg.initialize.init_multiblock(config)

    pg.grid.generate_halo(mb,config)
    pg.mpicomm.blockcomm.communicate(mb,['x','y','z'])

    pg.initialize.init_arrays(mb,config)

    for blk in mb:
        pg.compute.metrics(blk)

    # init flow
    for blk in mb:
        # Cons
        blk.array['Q'][:,:,:,0] = 1.2
        blk.array['Q'][:,:,:,1] = 1.2
        blk.array['Q'][:,:,:,2] = 0.0
        blk.array['Q'][:,:,:,3] = 0.0

        # Prim
        blk.array['q'][:,:,:,0] = 101325.0
        blk.array['q'][:,:,:,1] = blk.array['Q'][:,:,:,1]/blk.array['Q'][:,:,:,0]
        blk.array['q'][:,:,:,2] = blk.array['Q'][:,:,:,2]/blk.array['Q'][:,:,:,0]
        blk.array['q'][:,:,:,3] = blk.array['Q'][:,:,:,3]/blk.array['Q'][:,:,:,0]

        # Cons Energy
        blk.array['Q'][:,:,:,4] = blk.array['q'][:,:,:,0]/(1.4-1)+0.5*blk.array['Q'][:,:,:,0]*mb.np.sqrt(blk.array['q'][:,:,:,1]**2+blk.array['q'][:,:,:,2]**2+blk.array['q'][:,:,:,3]**2)


    for blk in mb:
        pg.compute.advective(blk)

    print('blk0')
    print('iF rho')
    print(mb[0].array['iF'][1,:,:,0])
    print('iF rhoU')
    print(mb[0].array['iF'][1,:,:,1])
    print('iF rhoV')
    print(mb[0].array['iF'][1,:,:,2])
    print('iF rhoW')
    print(mb[0].array['iF'][1,:,:,3])
    print('iF rhoE')
    print(mb[0].array['iF'][1,:,:,4])

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
