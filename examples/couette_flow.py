 #!/usr/bin/env python
'''

Poisuielle Flow with top wall moving at 5m/s

'''

import mpi4py.rc
mpi4py.rc.initialize = False

import kokkos

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

def simulate():
    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()
    comm,rank,size = pg.mpicomm.mpiutils.get_comm_rank_size()
    # Ensure MPI is suitably cleaned up
    pg.mpicomm.mpiutils.register_finalize_handler()

    config = pg.files.config_file()
    config['simulation']['dt'] = 1e-5
    config['simulation']['niter'] = 500000
    config['simulation']['niterout'] = 10000
    config['simulation']['niterprint'] = 1000
    config['RHS']['diffusion'] = True

    mb = pg.multiblock.generate_multiblock_solver(1,config)
    pg.grid.create.multiblock_cube(mb,
                                   mb_dimensions=[1,1,1],
                                   dimensions_perblock=[5,40,2],
                                   lengths=[0.01,0.1,0.01])

    mb.init_solver_arrays(config)

    blk = mb[0]

    #face 1
    blk.get_face_conn(1)['bcfam'] = None
    blk.get_face_conn(1)['bctype'] = 'b1'
    blk.get_face_conn(1)['neighbor'] = 0
    blk.get_face_conn(1)['orientation'] = '123'
    blk.get_face(1).comm_rank = 0
    #face 2
    blk.get_face_conn(2)['bcfam'] = None
    blk.get_face_conn(2)['bctype'] = 'b1'
    blk.get_face_conn(2)['neighbor'] = 0
    blk.get_face_conn(2)['orientation'] = '123'
    blk.get_face(2).comm_rank = 0
    #face 3
    blk.get_face_conn(3)['bcfam'] = None
    blk.get_face_conn(3)['bctype'] = 'adiabatic_noslip_wall'
    blk.get_face_conn(3)['neighbor'] = None
    blk.get_face_conn(3)['orientation'] = None
    blk.get_face(3).comm_rank = 0
    #face 4 isoT moving wall
    blk.get_face_conn(4)['bcfam'] = 'whoosh'
    blk.get_face_conn(4)['bctype'] = 'adiabatic_moving_wall'
    blk.get_face_conn(4)['neighbor'] = None
    blk.get_face_conn(4)['orientation'] = None
    blk.get_face(4).comm_rank = 0

    for face in [5,6]:
        blk.get_face_conn(face)['bcfam'] = None
        blk.get_face_conn(face)['bctype'] = 'adiabatic_slip_wall'
        blk.get_face(face).comm_rank = 0

    blk.get_face(4).bc = {'bctype':'adiabatic_moving_wall',
                          'values':{'u':5.0,
                                    'v':0.0,
                                    'w':0.0}}

    pg.grid.generate_halo(mb)

    pg.mpicomm.blockcomm.set_block_communication(mb)

    pg.grid.unify_solver_grid(mb)

    pg.compute.metrics(mb)

    pg.writers.write_grid(mb,config['io']['griddir'])

    blk.array['q'][1:-1,1:-1,1,0] = 101325.0
    blk.array['q'][:,:,:,1:4] = 0.0
    blk.array['q'][1:-1,1:-1,1,4] = 300.0

    mb.eos(blk, mb.thermdat, 0, 'prims')
    pg.consistify(mb)

    pg.writers.write_restart(mb,config['io']['outputdir'],grid_path='../Grid')

    for niter in range(config['simulation']['niter']):

        if mb.nrt%config['simulation']['niterprint'] == 0:
            print(mb.nrt,mb.tme)

        mb.step(config['simulation']['dt'])

        if mb.nrt%config['simulation']['niterout'] == 0:
            pg.writers.write_restart(mb,config['io']['outputdir'],grid_path='../Grid')

    # Finalise MPI
    MPI.Finalize()

if __name__ == "__main__":
    try:
        from os import mkdir
        mkdir('./Grid')
        mkdir('./Input')
        mkdir('./Output')
    except FileExistsError:
        pass
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
