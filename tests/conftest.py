from peregrinepy.compute import pgkokkos
import pytest


@pytest.fixture(scope="session")
def my_setup(request):
    pgkokkos.initialize()
    request.addfinalizer(pgkokkos.finalize)
