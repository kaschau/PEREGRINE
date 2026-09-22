#!/usr/bin/env python
"""

run with 2 mpi processes!!!!!!

"""

from mpi4py import MPI  # noqa: F401

import peregrinepy as pg
import os

fname = """
# script-version: 2.0
# Catalyst state generated using paraview version 5.10.0

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XDMF Reader'
# qxmf = XDMFReader(registrationName="input")
qxmf = TrivialProducer(registrationName="input")
qxmf.CellArrayStatus = ["Air", "T", "Velocity", "p", "rho"]

# create a new 'Slice'
slice1 = Slice(registrationName="Slice1", Input=qxmf)
slice1.SliceType = "Plane"
slice1.HyperTreeGridSlicer = "Plane"
slice1.Triangulatetheslice = 0
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.1, 0.01, 0.0005]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.1, 0.01, 0.0005]

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
vTM1 = CreateExtractor("VTM", slice1, registrationName="VTM1")
# trace defaults for the extractor.
vTM1.Trigger = "TimeStep"

# init the 'TimeStep' selected for 'Trigger'
vTM1.Trigger.Frequency = 10

# init the 'VTM' selected for 'Writer'
vTM1.Writer.FileName = "Slice1_{timestep:06d}.vtm"

# ----------------------------------------------------------------
# restore active source
SetActiveSource(vTM1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst



options = catalyst.Options()
options.ExtractsOutputDirectory = "./"
options.GlobalTrigger = "TimeStep"
options.CatalystLiveTrigger = "TimeStep"

# init the 'TimeStep' selected for 'GlobalTrigger'
options.GlobalTrigger.Frequency = 10

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from paraview.simple import SaveExtractsUsingCatalystOptions

    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
"""

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


def simulate():
    """A channel of two blocks, fed at one end and let out the other, with
    Catalyst watching every step."""
    config = pg.files.configFile()
    config["simulation"]["simulator"] = "navierStokes"
    config["mixture"]["species"] = air
    config["mixture"]["trans"] = "constantProps"
    config["plugins"]["catalyst"] = {"script": "tempcoproc.py"}
    config["initialConditions"]["u"] = 10.0
    config["bcValues"]["inlet"] = {
        "bcType": "constantVelocitySubsonicInlet",
        "u": 10.0,
        "v": 0.0,
        "w": 0.0,
        "T": 300.0,
    }
    config["bcValues"]["exit"] = {
        "bcType": "constantPressureSubsonicExit",
        "p": 101325.0,
    }
    config["bcValues"]["walls"] = {"bcType": "adiabaticNoSlipWall"}
    config.validateConfig()

    comm, rank, size = pg.misc.getCommRankSize()
    if rank == 0:
        with open("tempcoproc.py", "w") as f:
            f.write(fname)
    mesh = pg.mesher.CubeMesher(
        mbDims=[2, 1, 1],
        dimsPerBlock=[100, 40, 2],
        lengths=[0.2, 0.02, 0.001],
        boundaryNames={1: "inlet", 2: "exit", 3: "walls", 4: "walls"},
    )
    mb = pg.multiBlock.solver(config, mesh)

    if rank == 0:
        print(mb)
    dt = 1.44e-6
    catalyst = mb.plugins["catalyst"]
    catalyst(mb)
    bar = pg.misc.Progress(100)
    while mb.nrt < 100:
        bar.at(mb.nrt)
        mb.integrator.step(dt)
        catalyst(mb)

    catalyst.finalize(mb)
    if rank == 0:
        os.remove("./tempcoproc.py")


if __name__ == "__main__":
    try:
        pg.backend.abi.lib.initialize()
        simulate()
        pg.backend.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
