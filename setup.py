#!/usr/bin/env python

import re

from setuptools import setup, find_packages

# Get version
vfile = open("src/peregrinepy/_version.py").read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)


# Hard dependencies
install_requires = [
    "h5py >= 2.6",
    "mpi4py >= 3.0",
    "numpy >= 1.20",
    "lxml >= 4.6",
    "pyyaml >= 6.0",
    "pymetis >= 2023.1",
]

long_description = """
peregrinepy is the python encapsulation of PEREGRINE. A multi-block,
multi-physics solver for advection-diffusion problems. The structure
of the code is designed such that peregrinepy can be a light weight
pre/post processing aid, as well as the driver that calls compute
kernels insitu. All physics are solved for in C++ using the kokkos
model for portability between any architecture.
"""

setup(
    name="peregrinepy",
    version=version,
    author="Kyle Schau",
    author_email="ksachau89@gmail.com",
    description="A hybrid Python/C++ CFD Code",
    long_description=long_description,
    install_requires=install_requires,
    # tell setuptools to look for any packages under 'src'
    packages=find_packages("src"),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={"": "src"},
    # Testing folder
    python_requires=">=3.8",
    test_suite="tests",
    zip_safe=False,
)
