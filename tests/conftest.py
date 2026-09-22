import subprocess
import sys
from pathlib import Path

import pytest
from peregrinepy.backend import abi


@pytest.fixture(scope="session")
def my_setup(request):
    abi.lib.initialize()
    request.addfinalizer(abi.lib.finalize)


def testDirectories(path):
    """The directories under :path: holding tests, when it holds more than
    one; none when it is a file or a single directory of tests."""
    if not path.is_dir():
        return []
    below = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob("test_*.py")))
    return below if len(below) > 1 else []


def pytest_cmdline_main(config):
    """On macOS the suite runs one directory per process. Every kernel
    library carries a Kokkos thread-local, dyld gives each such library a
    pthread key, a process has 512 of them, and the whole suite loads more
    than that: past the cap the next load aborts without a word. A run
    naming one directory, or files, is left alone."""
    if sys.platform != "darwin":
        return None
    invocation = list(config.invocation_params.args)
    paths = [Path(a) for a in config.args]
    batches = {p: testDirectories(p) for p in paths}
    if not any(batches.values()):
        return None
    options = [a for a in invocation if a not in config.args]
    failed = []
    for path in paths:
        for batch in batches[path] or [path]:
            print(f"\n==== {batch} in its own process", flush=True)
            code = subprocess.call(
                [sys.executable, "-m", "pytest", *options, str(batch)]
            )
            if code not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
                failed.append(str(batch))
    print(
        (
            "\n==== every directory passed"
            if not failed
            else f"\n==== failed: {', '.join(failed)}"
        ),
        flush=True,
    )
    return int(bool(failed))
