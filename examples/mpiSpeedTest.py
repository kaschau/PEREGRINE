#!/usr/bin/env python
"""

A time trial test for testing and stuff.


"""

import peregrinepy as pg
import numpy as np
import os

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


np.seterr(all="raise")


def simulate():
    config = pg.files.configFile()
    config["simulation"]["physics"] = "navierStokes"
    config["simulation"]["mixture"] = air
    config["simulation"]["trans"] = "constantProps"

    config["simulation"]["niter"] = 100

    config["timeIntegration"]["integrator"] = "rk3"
    config["timeIntegration"]["controller"] = "cfl"
    config.validateConfig()

    ni = 30
    nbi = 10
    # a single species: the primitive variables are p, u, v, w, T
    mb = pg.multiBlock.restart(["p", "u", "v", "w", "T"])
    pg.mesher.CubeMesher(
        mbDims=[nbi, nbi, nbi],
        dimsPerBlock=[ni, ni, ni],
        lengths=[1, 1, 1],
        periodic=[True, True, True],
    ).fill(mb)

    for blk in mb.blocks:
        prims = blk.prims.get()
        prims[:, :, :, 0] = 101325.0
        prims[:, :, :, 4] = 300.0
        blk.prims.set(prims)

    # Create the case structure
    try:
        os.mkdir("./Grid")
    except FileExistsError:
        pass
    try:
        os.mkdir("./Restart")
    except FileExistsError:
        pass
    try:
        os.mkdir("./Input")
    except FileExistsError:
        pass

    pg.writers.GridWriter(mb, "./Grid").write(mb)
    pg.writers.RestartWriter(mb, "./Restart", gridPath="../Grid").write(mb)
    pg.writers.writeConfigFile(config, "./")


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
