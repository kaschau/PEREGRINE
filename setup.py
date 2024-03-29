#!/usr/bin/env python

import os
import re
import sys
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion
from setuptools.command.build_ext import build_ext


# Get version
vfile = open("src/peregrinepy/_version.py").read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = (
            os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            + "/peregrinepy"
        )
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug"  # if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        print()  # Add an empty line for cleaner output


# Hard dependencies
install_requires = [
    "h5py >= 2.6",
    "mpi4py >= 3.0",
    "numpy >= 1.20",
    "scipy >= 1.5",
    "lxml >= 4.6",
    "pyyaml >= 6.0",
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
    # add an extension module named 'python_cpp_example' to the package
    # 'python_cpp_example'
    ext_modules=[CMakeExtension("peregrinepy")],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    # Testing folder
    python_requires=">=3.8",
    test_suite="tests",
    zip_safe=False,
)
