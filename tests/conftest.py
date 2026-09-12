from peregrinepy import abi
import pytest


@pytest.fixture(scope="session")
def my_setup(request):
    abi.initialize()
    request.addfinalizer(abi.finalize)
