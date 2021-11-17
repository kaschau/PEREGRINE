import numpy as np


def prep_constantVelocitySubsonicInlet(blk, face):
    pass


def prep_supersonicInlet(blk, face):
    pass


def prep_constantMassFluxSubsonicInlet(blk, face):
    nface = face.nface

    s1_ = face.s1_
    # Estimate the inlet face normal
    if nface == 1:
        nx = np.mean(blk.array["inx"][s1_])
        ny = np.mean(blk.array["iny"][s1_])
        nz = np.mean(blk.array["inz"][s1_])
    elif nface == 2:
        nx = np.mean(-blk.array["inx"][s1_])
        ny = np.mean(-blk.array["iny"][s1_])
        nz = np.mean(-blk.array["inz"][s1_])
    elif nface == 3:
        nx = np.mean(blk.array["jnx"][s1_])
        ny = np.mean(blk.array["jny"][s1_])
        nz = np.mean(blk.array["jnz"][s1_])
    elif nface == 4:
        nx = np.mean(-blk.array["jnx"][s1_])
        ny = np.mean(-blk.array["jny"][s1_])
        nz = np.mean(-blk.array["jnz"][s1_])
    elif nface == 5:
        nx = np.mean(blk.array["knx"][s1_])
        ny = np.mean(blk.array["kny"][s1_])
        nz = np.mean(blk.array["knz"][s1_])
    elif nface == 6:
        nx = np.mean(-blk.array["knx"][s1_])
        ny = np.mean(-blk.array["kny"][s1_])
        nz = np.mean(-blk.array["knz"][s1_])

    # Thos value was set from readBcs
    mDotPerUnitArea = face.array["QBcVals"][0]

    # renormalize normals, just in case
    nx, ny, nz = np.array([nx, ny, nz]) / np.linalg.norm([nx, ny, nz])

    # Set target values of rhou, rhov, rhow
    face.array["QBcVals"][1] = nx * mDotPerUnitArea
    face.array["QBcVals"][2] = ny * mDotPerUnitArea
    face.array["QBcVals"][3] = nz * mDotPerUnitArea
