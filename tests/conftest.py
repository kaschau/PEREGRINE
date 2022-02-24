import pytest
import kokkos


@pytest.fixture(scope="session")
def my_setup(request):
    kokkos.initialize()

    def fin():
        kokkos.finalize()

    request.addfinalizer(fin)
