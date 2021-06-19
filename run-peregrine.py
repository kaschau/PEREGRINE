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
        # Prim
        blk.array['q'][:,:,:,0] = 101325.0
        blk.array['q'][:,:,:,1] = mb.np.random.random(blk.array['q'][:,:,:,1].shape)
        blk.array['q'][:,:,:,2] = 0.0
        blk.array['q'][:,:,:,3] = 0.0
        blk.array['q'][:,:,:,4] = 300.0

        blk.array['Q'][:,:,:,0] = 1.2
        blk.array['Q'][:,:,:,1] = 1.0
        blk.array['Q'][:,:,:,2] = 0.0
        blk.array['Q'][:,:,:,3] = 0.0

    for blk in mb:
        pg.compute.total_energy(blk)


    for blk in mb:
        pg.compute.advective(blk)
        pg.compute.apply_flux(blk)


    print('blk0')
    print('dQ rho')
    print(mb[0].array['dQ'][2,:,:,0])
    print('dQ rhoU')
    print(mb[0].array['dQ'][2,:,:,1])
    print('dQ rhoV')
    print(mb[0].array['dQ'][2,:,:,2])
    print('dQ rhoW')
    print(mb[0].array['dQ'][2,:,:,3])
    print('dQ rhoE')
    print(mb[0].array['dQ'][2,:,:,4])

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
