#!/usr/bin/env python
'''

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 1 for the KEEP scheme (blue line)


'''

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
    config['RHS']['diffusion'] = False
    mb = pg.multiblock.generate_multiblock_solver(1,config)
    therm = pg.thermo.thermdat(config)
    pg.grid.create.multiblock_cube(mb,
                                   mb_dimensions=[1,1,1],
                                   dimensions_perblock=[41,2,2],
                                   lengths=[1,0.01,0.01])
    mb.init_solver_arrays(config)

    blk = mb[0]
    for face in blk.faces:
        face.connectivity['bctype'] = 'adiabatic_slip_wall'

    blk.get_face_conn(1)['bctype'] = 'b1'
    blk.get_face_conn(1)['neighbor'] = 0
    blk.get_face_conn(1)['orientation'] = '123'
    blk.get_face(1).comm_rank = 0

    blk.get_face_conn(2)['bctype'] = 'b1'
    blk.get_face_conn(2)['neighbor'] = 0
    blk.get_face_conn(2)['orientation'] = '123'
    blk.get_face(2).comm_rank = 0

    pg.mpicomm.blockcomm.set_block_communication(mb)

    mb.unify_solver_grid()

    mb.compute_metrics()

    R=281.4583333333333
    blk.array['q'][:,:,:,0] = 1.0
    blk.array['q'][:,:,:,1] = 1.0
    initial_rho = 2.0 + np.sin(2*np.pi*blk.array['xc'][1:-1,0,0])
    initial_T = 1.0/(R*initial_rho)
    blk.array['q'][1:-1,1,1,4] = initial_T

    #Update cons
    mb.eos(blk, mb.thermdat, 0 ,'prims')
    pg.consistify(mb)

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

        mb.step(dt)

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
