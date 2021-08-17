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

    config = pg.mpicomm.mpiread_config(config_file_path)

    mb = pg.bootstrap_case(config)

    # init flow
    for blk in mb:
        # Prim
        blk.array['q'][:,:,:,0] = 101325.0
        blk.array['q'][:,:,:,1] = 10.0 #mb.np.random.random(blk.array['q'][1:-1,1:-1,1:-1,1].shape)
        #blk.array['q'][1:-1,1:-1,1:-1,2] = 0.0
        #blk.array['q'][1:-1,1:-1,1:-1,3] = 0.0
        blk.array['q'][:,:,:,4] = 300 #np.random.random(blk.array['q'][1:-1,1:-1,1:-1,1].shape)*300
        blk.array['q'][3:6,:,:,4] = 350 #np.random.random(blk.array['q'][1:-1,1:-1,1:-1,1].shape)*300

        #Get Density
        mb.eos(blk,mb.thermdat,'0','prims')

        #Zero out prims
        blk.array['q'][1:-1,1:-1,1:-1,0] = 0.0
        blk.array['q'][1:-1,1:-1,1:-1,1] = 0.0
        blk.array['q'][1:-1,1:-1,1:-1,4] = 0.0

    pg.consistify(mb)
    pg.writers.write_restart(mb,path=config['io']['outputdir'],grid_path='../Grid')

    config['simulation']['niter'] = 10000
    config['simulation']['dt'] = 1e-6

    for niter in range(config['simulation']['niter']):
        print(mb.nrt,mb.tme)

        mb.step(config['simulation']['dt'])

        if mb.nrt%config['simulation']['niterout'] == 0:
            pg.writers.write_restart(mb,config['io']['outputdir'],grid_path='../Grid')

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
