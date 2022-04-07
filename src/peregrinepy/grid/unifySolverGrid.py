from .. import mpiComm


def unifySolverGrid(mb):

    for blk in mb:
        assert blk.blockType == "solver", "Only solverBlocks can be unified"
        blk.generateHalo()

    # Lets just be clean and create the edges and corners
    for _ in range(3):
        mpiComm.communicate(mb, ["x", "y", "z"])

    # Device is up to date after communicate, so pull back down
    for blk in mb:
        blk.updateHostView(["x", "y", "z"])

    for blk in mb:
        for face in blk.faces:
            bc = face.bcType
            if not bc.startswith("periodic"):
                continue
            for i, s0 in enumerate(face.s0_):
                x = blk.array["x"][s0]
                y = blk.array["y"][s0]
                z = blk.array["z"][s0]

                # Translate periodics
                if face.bcType == "periodicTransLow":
                    x[:] -= face.periodicAxis[0] * face.periodicSpan
                    y[:] -= face.periodicAxis[1] * face.periodicSpan
                    z[:] -= face.periodicAxis[2] * face.periodicSpan
                elif face.bcType == "periodicTransHigh":
                    x[:] += face.periodicAxis[0] * face.periodicSpan
                    y[:] += face.periodicAxis[1] * face.periodicSpan
                    z[:] += face.periodicAxis[2] * face.periodicSpan
                elif face.bcType.startswith("periodicRot"):
                    if face.bcType == "periodicRotLow":
                        rotM = face.array["periodicRotMatrixDown"]
                    elif face.bcType == "periodicRotHigh":
                        rotM = face.array["periodicRotMatrixUp"]
                    tempx = rotM[0, 0] * x[:] + rotM[0, 1] * y[:] + rotM[0, 2] * z[:]
                    tempy = rotM[1, 0] * x[:] + rotM[1, 1] * y[:] + rotM[1, 2] * z[:]
                    tempz = rotM[2, 0] * x[:] + rotM[2, 1] * y[:] + rotM[2, 2] * z[:]
                    x[:] = tempx[:]
                    y[:] = tempy[:]
                    z[:] = tempz[:]

    # Push back up the device
    for blk in mb:
        blk.updateDeviceView(["x", "y", "z"])
