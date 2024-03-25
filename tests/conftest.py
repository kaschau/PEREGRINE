from peregrinepy.compute import pgkokkos
import pytest

import mpi4py.rc

mpi4py.rc.finalize = False
mpi4py.rc.initialize = False

from mpi4py import MPI


@pytest.fixture(scope="session")
def my_setup(request):
    MPI.Init()
    pgkokkos.initialize()

    def fin():
        pgkokkos.finalize()
        MPI.Finalize()

    request.addfinalizer(fin)
