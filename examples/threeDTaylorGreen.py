 #!/usr/bin/env python
'''

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 2 for the KEEP scheme (blue line)


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
    config['RHS']['diffusion'] = False

    mb = pg.multiblock.generate_multiblock_solver(1,config)
    pg.grid.create.multiblock_cube(mb,
                                   mb_dimensions=[1,1,1],
                                   dimensions_perblock=[65,65,65],
                                   lengths=[2*np.pi for _ in range(3)])

    mb.init_solver_arrays(config)

    blk = mb[0]

    for face in blk.faces:
        face.connectivity['bctype'] = 'b1'
        face.connectivity['neighbor'] = 0
        face.connectivity['orientation'] = '123'
        face.comm_rank = 0

    pg.grid.generate_halo(mb)

    pg.mpicomm.blockcomm.set_block_communication(mb)

    pg.grid.unify_solver_grid(mb)

    pg.compute.metrics(mb)

    pg.writers.write_grid(mb,config['io']['griddir'])

    R=281.4583333333333
    cp = 1000.0
    cv = cp-R
    M0 = 0.4
    rho0 = 1.0
    gamma = cp/(cp-R)
    blk.array['q'][:,:,:,0] = 1/gamma + (rho0*M0**2/16.0)*(np.cos(2*blk.array['xc'])+np.cos(2*blk.array['yc']))*(np.cos(2*blk.array['zc']+2.0))
    blk.array['q'][:,:,:,1] = M0*np.sin(blk.array['xc'])*np.cos(blk.array['yc'])*np.cos(blk.array['zc'])
    blk.array['q'][:,:,:,2] = -M0*np.cos(blk.array['xc'])*np.sin(blk.array['yc'])*np.cos(blk.array['zc'])
    blk.array['q'][:,:,:,4] = blk.array['q'][:,:,:,0]/(R*rho0)

    mb.eos(blk, mb.thermdat, 0, 'prims')
    pg.consistify(mb)

    dt = 0.1 * 2*np.pi/64
    niterout = 5000
    ke = []
    e = []
    s = []
    t = []
    while mb.tme*M0 < 120:

        if mb.nrt%niterout == 0:
            pg.writers.write_restart(mb,config['io']['outputdir'],'../Grid')

        if mb.nrt%50 == 0:
            print(mb.tme*M0)
            rke = np.sum(0.5*blk.array['Q'][1:-1,1:-1,1:-1,0]*
                           (blk.array['q'][1:-1,1:-1,1:-1,1]**2+
                            blk.array['q'][1:-1,1:-1,1:-1,2]**2+
                            blk.array['q'][1:-1,1:-1,1:-1,3]**2  ))
            re  = np.sum(    blk.array['Q'][1:-1,1:-1,1:-1,0]*
                           (blk.array['q'][1:-1,1:-1,1:-1,4]*cv  ))

            rE  = np.sum(    blk.array['Q'][1:-1,1:-1,1:-1,4]     )
            rS  = np.sum(    blk.array['Q'][1:-1,1:-1,1:-1,0]*
                    np.log10(blk.array['q'][1:-1,1:-1,1:-1,0]*
                             blk.array['Q'][1:-1,1:-1,1:-1,0]**(-gamma)))

            ke.append(rke)
            e.append(re)
            s.append(rS)
            t.append(mb.tme*M0)

        mb.step(dt)

    plt.plot(t,ke/ke[0])
    plt.ylim([0,2.4])
    plt.savefig('ke.png')
    plt.clf()
    plt.plot(t,(-(s-s[0]))/s[0])
    plt.ylim([-3e-2,1e-2])
    plt.savefig('entropy.png')
    plt.clf()
    plt.plot(t,e/e[0])
    plt.savefig('e.png')
    plt.clf()

    # Finalise MPI
    MPI.Finalize()

if __name__ == "__main__":
    try:
        from os import mkdir
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
