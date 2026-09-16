from peregrinepy.backend import abi
import pytest


@pytest.fixture(scope="session")
def my_setup(request):
    abi.lib.initialize()
    request.addfinalizer(abi.lib.finalize)
