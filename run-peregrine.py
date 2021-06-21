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

    pg.compute.metrics(mb)
    # init flow
    for blk in mb:
        # Prim
        blk.array['q'][1:-1,1:-1,1:-1,0] = 101325.0
        blk.array['q'][1:-1,1:-1,1:-1,1] = 10.0 #mb.np.random.random(blk.array['q'][1:-1,1:-1,1:-1,1].shape)
        #blk.array['q'][1:-1,1:-1,1:-1,2] = 0.0
        #blk.array['q'][1:-1,1:-1,1:-1,3] = 0.0
        blk.array['q'][1:-1,1:-1,1:-1,4] = 300 #np.random.random(blk.array['q'][1:-1,1:-1,1:-1,1].shape)*300
        blk.array['q'][3:6,1:-1,1:-1,4] = 600 #np.random.random(blk.array['q'][1:-1,1:-1,1:-1,1].shape)*300

        #blk.array['Q'][:,:,:,0] = 1.2
        #blk.array['Q'][:,:,:,1] = 1.2
        #blk.array['Q'][:,:,:,2] = 0.0
        #blk.array['Q'][:,:,:,3] = 0.0

        #Get Density
        pg.compute.EOS_ideal(blk,'0','PT')
        #Get momentum
        pg.compute.momentum(blk,'0','u')
        #Get total energy
        pg.compute.calEOS_perfect(blk,'0','PT')

        #Zero out prims
        blk.array['q'][1:-1,1:-1,1:-1,0] = 0.0
        blk.array['q'][1:-1,1:-1,1:-1,1] = 0.0
        blk.array['q'][1:-1,1:-1,1:-1,4] = 0.0

    pg.consistify(mb,config)
    pg.writers.write_restart(mb,config['io']['outputdir'])

    #ts = time.time()
    for i in range(10):
        print(i)
        pg.rk1.step(mb,0.01,config)
        pg.writers.write_restart(mb,config['io']['outputdir'])

    #print(time.time()-ts, 'took this many seconds')



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
