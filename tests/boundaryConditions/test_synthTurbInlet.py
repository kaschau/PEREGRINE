import os
import shutil

import kokkos
import numpy as np
import peregrinepy as pg
import sempy
from scipy import interpolate as itrp


class TestInlets:
    def setup_method(self):
        kokkos.initialize()

    def teardown_method(self):
        kokkos.finalize()

    def test_synthTurbInlet(self):

        seminp = {}
        seminp["domainType"] = "channel"
        seminp["Uo"] = 4.519722
        seminp["totalTime"] = 1.0
        seminp["yHeight"] = 0.1
        seminp["zWidth"] = 0.3
        seminp["delta"] = 0.05
        seminp["utau"] = 0.21256278
        seminp["viscosity"] = 1.81e-5
        seminp["sigmasFrom"] = "jarrin"
        seminp["statsFrom"] = "moser"
        seminp["profileFrom"] = "channel"
        seminp["scaleFactor"] = 1.0
        seminp["cEddy"] = 2.0
        seminp["nframes"] = 10
        seminp["normalization"] = "exact"
        seminp["populationMethod"] = "PDF"
        seminp["interpolate"] = True
        seminp["convect"] = "uniform"
        seminp["shape"] = "tent"

        config = pg.files.configFile()

        mb = pg.multiBlock.generateMultiBlockSolver(1, config)

        pg.grid.create.multiBlockCube(
            mb,
            mbDims=[1, 1, 1],
            dimsPerBlock=[8, np.random.randint(3, 15), np.random.randint(3, 15)],
            lengths=[1, seminp["yHeight"], seminp["zWidth"]],
        )

        mb.initSolverArrays(config)

        mb.generateHalo()
        mb.computeMetrics(config["RHS"]["diffOrder"])

        blk = mb[0]
        face = blk.getFace(1)
        face.bcType = "cubicSplineSubsonicInlet"
        face.bcFam = "inlet"

        ###############################################################################
        # Create the domain based on above inputs
        ###############################################################################
        # Initialize domain
        domain = sempy.geometries.box(
            seminp["domainType"],
            seminp["Uo"],
            seminp["totalTime"],
            seminp["yHeight"],
            seminp["zWidth"],
            seminp["delta"],
            seminp["utau"],
            seminp["viscosity"],
        )

        # Set flow properties from existing data
        domain.setSemData(
            sigmasFrom=seminp["sigmasFrom"],
            statsFrom=seminp["statsFrom"],
            profileFrom=seminp["profileFrom"],
            scaleFactor=seminp["scaleFactor"],
        )

        # Populate the domain
        domain.populate(seminp["cEddy"], seminp["populationMethod"])
        # Create the eps
        domain.generateEps()
        # Compute sigmas
        domain.computeSigmas()
        # Make it periodic
        domain.makePeriodic(
            periodicX=False,
            periodicY=False,
            periodicZ=False,
        )

        ng = blk.ng
        yc = blk.array["iyc"][0, ng:-ng, ng:-ng]
        zc = blk.array["izc"][0, ng:-ng, ng:-ng]

        up, vp, wp = sempy.generatePrimes(
            yc,
            zc,
            domain,
            seminp["nframes"],
            normalization=seminp["normalization"],
            interpolate=seminp["interpolate"],
            convect=seminp["convect"],
            shape=seminp["shape"],
            progress=False,
        )
        bc = "not-a-knot"
        t = np.linspace(0, seminp["totalTime"], seminp["nframes"])
        # Add the mean profile here
        Upu = domain.ubarInterp(yc) + up
        fu = itrp.CubicSpline(t, Upu, bc_type=bc, axis=0)
        fv = itrp.CubicSpline(t, vp, bc_type=bc, axis=0)
        fw = itrp.CubicSpline(t, wp, bc_type=bc, axis=0)

        os.makedirs("Input")
        os.makedirs("Input/inletAlphas")
        with open("./Input/inletAlphas/alphas_0_1.npy", "wb") as f:
            #  shape =      4  nframes  ny nz
            np.save(f, fu.c[:, :, :, :])
            np.save(f, fv.c[:, :, :, :])
            np.save(f, fw.c[:, :, :, :])

        inputBcValues = {}
        inputBcValues["T"] = 300.0
        inputBcValues["intervalDt"] = seminp["totalTime"] / (seminp["nframes"] - 1)
        face.array["qBcVals"] = np.zeros((blk.array["q"][face.s1_].shape))
        for bcmodule in [pg.bcs.prepInlets, pg.bcs.prepExits, pg.bcs.prepWalls]:
            try:
                func = getattr(bcmodule, "prep_" + face.bcType)
                func(blk, face, inputBcValues)
                break
            except AttributeError:
                pass

        shutil.rmtree("./Input")
        pg.misc.createViewMirrorArray(face, ["qBcVals"], face.array["qBcVals"].shape)

        q = blk.array["q"][0, ng:-ng, ng:-ng, :]
        for tme in np.random.random(10):
            mb.tme = tme
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)

            targetU = fu(tme)
            targetV = fv(tme)
            targetW = fw(tme)

            pgU = q[:, :, 1]
            pgV = q[:, :, 2]
            pgW = q[:, :, 3]

            assert np.allclose(targetU, pgU)
            assert np.allclose(targetV, pgV)
            assert np.allclose(targetW, pgW)

        # now test that the signal looping logic works
        for tme in np.random.random(10):
            mb.tme = tme + seminp["totalTime"]
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)

            targetU = fu(tme)
            targetV = fv(tme)
            targetW = fw(tme)

            pgU = q[:, :, 1]
            pgV = q[:, :, 2]
            pgW = q[:, :, 3]

            assert np.allclose(targetU, pgU)
            assert np.allclose(targetV, pgV)
            assert np.allclose(targetW, pgW)
