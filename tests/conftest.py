import mpi4py.rc

mpi4py.rc.finalize = False
mpi4py.rc.initialize = False
import kokkos
import pytest
from mpi4py import MPI


@pytest.fixture(scope="session")
def my_setup(request):
    kokkos.initialize()
    MPI.Init()

    def fin():
        kokkos.finalize()
        MPI.Finalize()

    request.addfinalizer(fin)
