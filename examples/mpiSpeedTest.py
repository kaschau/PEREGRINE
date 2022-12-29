#!/usr/bin/env python
"""

A time trial test for testing and stuff.


"""

import peregrinepy as pg
import numpy as np
from time import perf_counter
import subprocess
import os

np.seterr(all="raise")


def simulate():

    config = pg.files.configFile()
    config["RHS"]["diffusion"] = True
    config["RHS"]["shockHandling"] = "artificialDissipation"
    config["RHS"]["switchAdvFlux"] = "jamesonPressure"
    config["RHS"]["secondaryAdvFlux"] = "scalarDissipation"
    config["thermochem"]["spdata"] = ["Air"]

    config["io"]["niterRestart"] = 1000000
    config["io"]["niterPrint"] = 1000000

    config["simulation"]["niter"] = 100

    config["timeIntegration"]["integrator"] = rk3
    config["timeIntegration"]["variableTimeStep"] = True
    config.validateConfig()

    ni = 30
    nbi = 10
    mb = pg.multiBlock.restart(nbi**3, config["thermochem"]["spdata"])
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[nbi, nbi, nbi],
        dimsPerBlock=[ni, ni, ni],
        lengths=[1, 1, 1],
        periodic=[True, True, True],
    )

    for blk in mb:
        blk.array["q"][:, :, :, 0] = 101325.0
        blk.array["q"][:, :, :, 4] = 300.0

    # Create the case structure
    os.mkdir("./Grid")
    os.mkdir("./Restart")
    os.mkdir("./Input")

    pg.writers.writeGrid(mb, "./Grid")
    pg.writers.writeRestart(mb, "./Restart", gridPath="../Grid", animate=True)
    pg.writers.writeConfigFile(config, "./Input")


if __name__ == "__main__":

    try:
        simulate()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
