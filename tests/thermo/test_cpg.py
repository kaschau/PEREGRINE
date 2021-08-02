import mpi4py.rc
mpi4py.rc.initialize = False

import kokkos

import peregrinepy as pg
import numpy as np
import cantera as ct

import sys

#np.random.seed(111)

##################################################################################
##### Test for all positive i aligned orientations
##################################################################################

def test_thermo(ctfile):

    # Import but do not initialise MPI
    from mpi4py import MPI

    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=100  , high=1000)
    Y = np.random.uniform(low=0.0  , high=1.0,size=gas.n_species)
    Y = Y/np.sum(Y)

    gas.TPY = T,p,Y

    # Manually initialise MPI
    MPI.Init()
    comm,rank,size = pg.mpicomm.mpiutils.get_comm_rank_size()
    # Ensure MPI is suitably cleaned up
    pg.mpicomm.mpiutils.register_finalize_handler()

    config = pg.files.config_file()
    config['thermochem']['ctfile'] = ctfile

    mb = pg.multiblock.generate_multiblock_solver(1,config)
    therm = pg.thermo.thermdat(config)
    pg.grid.create.multiblock_cube(mb,
                                   mb_dimensions=[1,1,1],
                                   dimensions_perblock=[2,2,2],
                                   lengths=[1,1,1])
    mb.init_solver_arrays(config)

    blk = mb[0]

    pg.grid.generate_halo(mb,config)

    pg.mpicomm.blockcomm.set_block_communication(mb,config)

    pg.compute.metrics(mb)

    blk.array['q'][:,:,:,0] = p
    blk.array['q'][:,:,:,4] = T
    blk.array['q'][:,:,:,5::] = Y[0:-1]

    #Update cons
    pg.compute.cpg(blk,mb.thermdat,'0','prims')

    #test the properties
    pgcons = blk.array['Q'][1,1,1]
    pgprim = blk.array['q'][1,1,1]
    pgthrm = blk.array['qh'][1,1,1]
    print(f'    {"Cantera":<11}  | {"PEREGRINE":<11}')
    print(f'p:  {gas.P:10.5f}  | {pgprim[0]:10.5f}')
    print(f'T:  {gas.T:10.5f}  | {pgprim[4]:10.5f}')
    for i,n in enumerate(gas.species_names[0:-1]):
        print(f'{n}:   {gas.Y[i]:>10.5f}  | {pgprim[5+i]:>10.5f}')
    print(f'{gas.species_names[-1]}:   {gas.Y[-1]:>10.5f}  | {1.0-np.sum(pgprim[5::]):>10.5f}')
    print(f'rho:{gas.density:>10.5f}  | {pgcons[0]:>10.5f}')
    print(f'E:  {gas.int_energy_mass:>10.5f}  | {pgcons[4]/pgcons[0]:>10.5f}')
    print(f'gamma:{gas.cp/gas.cv:>10.5f}  | {pgthrm[0]:>10.5f}')
    print(f'cp:{gas.cp:>10.5f}  | {pgthrm[1]:>10.5f}')
    print(f'h:{gas.enthalpy_mass:>10.5f}  | {pgthrm[2]/pgcons[0]:>10.5f}')

    # Finalise MPI
    MPI.Finalize()

if __name__ == "__main__":
    try:
        kokkos.initialize()
        test_thermo('dummy_ct.yaml')
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
