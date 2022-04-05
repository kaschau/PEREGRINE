from .. import mpiComm


def unifySolverGrid(mb):

    for blk in mb:
        assert blk.blockType == "solver", "Only solverBlocks can be unified"
        blk.generateHalo()

    # Lets just be clean and create the edges and corners
    for _ in range(3):
        mpiComm.communicate(mb, ["x", "y", "z"])

    for blk in mb:
        for face in blk.faces:
            bc = face.bcType
            if not bc.startswith("periodic"):
                continue
            for i, sR in enumerate(face.sliceR3):
                x = blk.array["x"][sR]
                y = blk.array["y"][sR]
                z = blk.array["z"][sR]

                # Translate periodics
                if face.bcType == "periodicTransLow":
                    x[:] -= face.periodicAxis[0] * face.periodicSpan
                    y[:] -= face.periodicAxis[1] * face.periodicSpan
                    z[:] -= face.periodicAxis[2] * face.periodicSpan
                elif face.bcType == "periodicTransHigh":
                    x[:] += face.periodicAxis[0] * face.periodicSpan
                    y[:] += face.periodicAxis[1] * face.periodicSpan
                    z[:] += face.periodicAxis[2] * face.periodicSpan
                elif face.bcType == "periodicRotLow":
                    print(face.nface, face.periodicRotMatrixDown)
                    tempx = (
                        face.periodicRotMatrixDown[0, 0] * x[:]
                        + face.periodicRotMatrixDown[0, 1] * y[:]
                        + face.periodicRotMatrixDown[0, 2] * z[:]
                    )
                    tempy = (
                        face.periodicRotMatrixDown[1, 0] * x[:]
                        + face.periodicRotMatrixDown[1, 1] * y[:]
                        + face.periodicRotMatrixDown[1, 2] * z[:]
                    )
                    tempz = (
                        face.periodicRotMatrixDown[2, 0] * x[:]
                        + face.periodicRotMatrixDown[2, 1] * y[:]
                        + face.periodicRotMatrixDown[2, 2] * z[:]
                    )
                    x[:] = tempx[:]
                    y[:] = tempy[:]
                    z[:] = tempz[:]
                elif face.bcType == "periodicRotHigh":
                    tempx = (
                        face.periodicRotMatrixUp[0, 0] * x[:]
                        + face.periodicRotMatrixUp[0, 1] * y[:]
                        + face.periodicRotMatrixUp[0, 2] * z[:]
                    )
                    tempy = (
                        face.periodicRotMatrixUp[1, 0] * x[:]
                        + face.periodicRotMatrixUp[1, 1] * y[:]
                        + face.periodicRotMatrixUp[1, 2] * z[:]
                    )
                    tempz = (
                        face.periodicRotMatrixUp[2, 0] * x[:]
                        + face.periodicRotMatrixUp[2, 1] * y[:]
                        + face.periodicRotMatrixUp[2, 2] * z[:]
                    )
                    x[:] = tempx[:]
                    y[:] = tempy[:]
                    z[:] = tempz[:]

    for blk in mb:
        # Push back up the device
        blk.updateDeviceView(["x", "y", "z"])
