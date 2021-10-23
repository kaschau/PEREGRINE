def constantPressureSubsonicExit(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        q = blk.array["q"][:, :, :, 1::]
        p = blk.array["q"][:, :, :, 0]
        for s0_, s2_ in zip(face.s0_, face.s2_):
            # Extrapolate everything
            q[s0_] = 2.0 * q[face.s1_] - q[s2_]

            # Set pressure at face
            p[s0_] = 2.0 * face.bcVals["p"] - p[face.s1_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        for s0_ in face.s0_:
            # neumann all gradients
            blk.array["dqdx"][s0_] = blk.array["dqdx"][face.s1_]
            blk.array["dqdy"][s0_] = blk.array["dqdy"][face.s1_]
            blk.array["dqdz"][s0_] = blk.array["dqdz"][face.s1_]


def supersonicExit(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        q = blk.array["q"][:, :, :, :]
        for s0_, s2_ in zip(face.s0_, face.s2_):
            # Extrapolate everything
            q[s0_] = 2.0 * q[face.s1_] - q[s2_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        for s0_ in face.s0_:
            # neumann all gradients
            blk.array["dqdx"][s0_] = blk.array["dqdx"][face.s1_]
            blk.array["dqdy"][s0_] = blk.array["dqdy"][face.s1_]
            blk.array["dqdz"][s0_] = blk.array["dqdz"][face.s1_]
