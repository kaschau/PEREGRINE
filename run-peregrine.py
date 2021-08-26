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
    import numpy as np
    np.seterr(all='raise')

    # Manually initialise MPI
    MPI.Init()

    comm,rank,size = pg.mpicomm.mpiutils.get_comm_rank_size()
    # Ensure MPI is suitably cleaned up
    pg.mpicomm.mpiutils.register_finalize_handler()

    config = pg.mpicomm.mpiread_config(config_file_path)

    mb = pg.bootstrap_case(config)

    pg.writers.parallel_writer.parallel_write_restart(mb,path=config['io']['outputdir'])

    for niter in range(config['simulation']['niter']):

        if mb.nrt%config['simulation']['niterprint'] == 0:
            if rank == 0:
                print(mb.nrt,mb.tme)

        mb.step(config['simulation']['dt'])

        if mb.nrt%config['simulation']['niterout'] == 0:
            pg.writers.parallel_writer.parallel_write_restart(mb,config['io']['outputdir'])

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
