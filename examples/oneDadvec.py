#!/usr/bin/env python

import mpi4py.rc
mpi4py.rc.initialize = False

import kokkos

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt
import time

import sys

def simulate():
    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()
    comm,rank,size = pg.mpicomm.mpiutils.get_comm_rank_size()
    # Ensure MPI is suitably cleaned up
    pg.mpicomm.mpiutils.register_finalize_handler()

    config = pg.files.config_file()
    mb = pg.multiblock.solver(1,config)
    pg.grid.create.multiblock_cube(mb,
                                   mb_dimensions=[1,1,1],
                                   dimensions_perblock=[41,2,2],
                                   lengths=[1,0.01,0.01])
    blk = mb[0]

    blk.connectivity['1']['bc'] = 'b1'
    blk.connectivity['1']['neighbor'] = 0
    blk.connectivity['1']['orientation'] = '123'
    blk.connectivity['1']['comm_rank'] = 0

    blk.connectivity['2']['bc'] = 'b1'
    blk.connectivity['2']['neighbor'] = 0
    blk.connectivity['2']['orientation'] = '123'
    blk.connectivity['2']['comm_rank'] = 0

    pg.grid.generate_halo(mb,config)
    ccshape = [blk.ni+2,blk.nj+2,blk.nk+2]
    for name in ('x','y','z'):
        setattr(blk,name, kokkos.array(blk.array[name], dtype=kokkos.double, space=kokkos.HostSpace, dynamic=False))

    pg.mpicomm.blockcomm.set_block_communication(mb,config)
    blk.init_koarrays(config)

    pg.compute.metrics(mb)

    R=281.4583333333333
    blk.array['q'][:,:,:,0] = 1.0
    blk.array['q'][:,:,:,1] = 1.0
    initial_rho = 2.0 + np.sin(2*np.pi*blk.array['xc'][1:-1,0,0])
    initial_T = 1.0/(R*initial_rho)
    blk.array['q'][1:-1,1,1,4] = initial_T

    #Get Density
    pg.compute.EOS_ideal(blk,'0','PT')
    #Get momentum
    pg.compute.momentum(blk,'0','u')
    #Get total energy
    pg.compute.calEOS_perfect(blk,'0','PT')

    pg.consistify(mb,config)

    dt = 0.1 * 0.025
    niterout = 1100
    while mb.tme < 11.0:

        if mb.nrt%niterout == 0:
            #Compute primatives from conserved Q
            fig, ax1 = plt.subplots()
            ax1.set_title(f'{mb.tme}')
            ax1.set_xlabel(r'x')
            x = blk.array['xc'][1:-1,1,1]
            rho = blk.array['Q'][1:-1,1,1,0]
            p = blk.array['q'][1:-1,1,1,0]
            u = blk.array['q'][1:-1,1,1,1]
            ax1.plot(x,rho, color='g',label='rho',linewidth=0.5)
            ax1.plot(x,p, color='r',label='p',linewidth=0.5)
            ax1.plot(x,u, color='k',label='u',linewidth=0.5)
            ax1.scatter(x,initial_rho, marker='o',facecolor='w',edgecolor='b',label='exact',linewidth=0.5)
            ax1.legend()
            plt.savefig(f'{mb.nrt:04d}.png',dpi=400)
            plt.close()

        pg.rk4.step(mb,dt,config)

    # Finalise MPI
    MPI.Finalize()

if __name__ == "__main__":
    try:
        kokkos.initialize()
        simulate()
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)