def adiabaticNoSlipWall(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        p = blk.array["q"][:, :, :, 0]
        u = blk.array["q"][:, :, :, 1:4]
        TN = blk.array["q"][:, :, :, 4::]
        for s0_ in face.s0_:
            p[s0_] = p[face.s1_]
            u[s0_] = -u[face.s1_]
            TN[s0_] = TN[face.s1_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        dvelodx = blk.array["dqdx"][:, :, :, 1:4]
        dvelody = blk.array["dqdy"][:, :, :, 1:4]
        dvelodz = blk.array["dqdz"][:, :, :, 1:4]

        dTNdx = blk.array["dqdx"][:, :, :, 4::]
        dTNdy = blk.array["dqdy"][:, :, :, 4::]
        dTNdz = blk.array["dqdz"][:, :, :, 4::]

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate velocity gradient
            dvelodx[s0_] = 2.0 * dvelodx[face.s1_] - dvelodx[s2_]
            dvelody[s0_] = 2.0 * dvelody[face.s1_] - dvelody[s2_]
            dvelodz[s0_] = 2.0 * dvelodz[face.s1_] - dvelodz[s2_]
            # negate temp and species gradient (so gradient evaluates to zero on wall)
            dTNdx[s0_] = -dTNdx[face.s1_]
            dTNdy[s0_] = -dTNdy[face.s1_]
            dTNdz[s0_] = -dTNdz[face.s1_]


def adiabaticSlipWall(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        p = blk.array["q"][:, :, :, 0]
        u = blk.array["q"][:, :, :, 1]
        v = blk.array["q"][:, :, :, 2]
        w = blk.array["q"][:, :, :, 3]
        if nface in [1, 2]:
            nx = blk.array["inx"]
            ny = blk.array["iny"]
            nz = blk.array["inz"]
        elif nface in [3, 4]:
            nx = blk.array["jnx"]
            ny = blk.array["jny"]
            nz = blk.array["jnz"]
        elif nface in [5, 6]:
            nx = blk.array["knx"]
            ny = blk.array["kny"]
            nz = blk.array["knz"]
        else:
            raise ValueError("Unknown nface")

        for s0_ in face.s0_:
            p[s0_] = p[face.s1_]

            u[s0_] = u[face.s1_] - 2.0 * u[face.s1_] * nx[face.s1_]
            v[s0_] = v[face.s1_] - 2.0 * v[face.s1_] * ny[face.s1_]
            w[s0_] = w[face.s1_] - 2.0 * w[face.s1_] * nz[face.s1_]

            TN = blk.array["q"][:, :, :, 4::]
            TN[s0_] = TN[face.s1_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":

        dvelodx = blk.array["dqdx"][:, :, :, 1:4]
        dvelody = blk.array["dqdy"][:, :, :, 1:4]
        dvelodz = blk.array["dqdz"][:, :, :, 1:4]
        dTNdx = blk.array["dqdx"][:, :, :, 4::]
        dTNdy = blk.array["dqdy"][:, :, :, 4::]
        dTNdz = blk.array["dqdz"][:, :, :, 4::]

        for s0_ in face.s0_:
            # negate velocity gradient (so gradient evaluates to zero on wall)
            dvelodx[s0_] = -dvelodx[face.s1_]
            dvelody[s0_] = -dvelody[face.s1_]
            dvelodz[s0_] = -dvelodz[face.s1_]
            # negate temp and species gradient (so gradient evaluates to zero on wall)
            dTNdx[s0_] = -dTNdx[face.s1_]
            dTNdy[s0_] = -dTNdy[face.s1_]
            dTNdz[s0_] = -dTNdz[face.s1_]


def adiabaticMovingWall(eos, blk, face, thtrdat, terms):

    nface = face.nface
    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]

    if terms == "euler":
        for s0_ in face.s0_:
            p[s0_] = p[face.s1_]

            u[s0_] = 2.0 * face.bcVals["u"] - u[face.s1_]
            v[s0_] = 2.0 * face.bcVals["v"] - v[face.s1_]
            w[s0_] = 2.0 * face.bcVals["w"] - w[face.s1_]

            TN[face.s0_] = TN[face.s1_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        # extrapolate velocity gradient
        dvelodx = blk.array["dqdx"][:, :, :, 1:4]
        dvelody = blk.array["dqdy"][:, :, :, 1:4]
        dvelodz = blk.array["dqdz"][:, :, :, 1:4]

        dTNdx = blk.array["dqdx"][:, :, :, 4::]
        dTNdy = blk.array["dqdy"][:, :, :, 4::]
        dTNdz = blk.array["dqdz"][:, :, :, 4::]

        for s0_, s2_ in zip(face.s0_, face.s2_):
            dvelodx[s0_] = 2.0 * dvelodx[face.s1_] - dvelodx[s2_]
            dvelody[s0_] = 2.0 * dvelody[face.s1_] - dvelody[s2_]
            dvelodz[s0_] = 2.0 * dvelodz[face.s1_] - dvelodz[s2_]

            # negate temp and species gradient (so gradient evaluates to zero on wall)
            dTNdx[s0_] = -dTNdx[face.s1_]
            dTNdy[s0_] = -dTNdy[face.s1_]
            dTNdz[s0_] = -dTNdz[face.s1_]


def isoTMovingWall(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        p = blk.array["q"][:, :, :, 0]
        u = blk.array["q"][:, :, :, 1]
        v = blk.array["q"][:, :, :, 2]
        w = blk.array["q"][:, :, :, 3]
        N = blk.array["q"][:, :, :, 4::]
        T = blk.array["q"][:, :, :, 4]

        for s0_ in face.s0_:
            p[s0_] = p[face.s1_]

            u[s0_] = 2.0 * face.bcVals["u"] - u[face.s1_]
            v[s0_] = 2.0 * face.bcVals["v"] - v[face.s1_]
            w[s0_] = 2.0 * face.bcVals["w"] - w[face.s1_]

            # Match species
            N[s0_] = N[face.s1_]

            # Set tempeerature
            T[s0_] = 2.0 * face.bcVals["T"] - T[face.s1_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        dvelodx = blk.array["dqdx"][:, :, :, 1:4]
        dvelody = blk.array["dqdy"][:, :, :, 1:4]
        dvelodz = blk.array["dqdz"][:, :, :, 1:4]
        dTNdx = blk.array["dqdx"][:, :, :, 4::]
        dTNdy = blk.array["dqdy"][:, :, :, 4::]
        dTNdz = blk.array["dqdz"][:, :, :, 4::]

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate velocity gradient
            dvelodx[s0_] = 2.0 * dvelodx[face.s1_] - dvelodx[s2_]
            dvelody[s0_] = 2.0 * dvelody[face.s1_] - dvelody[s2_]
            dvelodz[s0_] = 2.0 * dvelodz[face.s1_] - dvelodz[s2_]
            # extrapolate temp and species gradient
            dTNdx[s0_] = 2.0 * dTNdx[face.s1_] - dTNdx[s2_]
            dTNdy[s0_] = 2.0 * dTNdy[face.s1_] - dTNdy[s2_]
            dTNdz[s0_] = 2.0 * dTNdz[face.s1_] - dTNdz[s2_]
