def constant_pressure_subsonic_exit(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        # Extrapolate everything
        q = blk.array["q"][:, :, :, 1::]
        q[face.s0_] = 2.0 * q[face.s1_] - q[face.s2_]

        # Set pressure at face
        p = blk.array["q"][:, :, :, 0]
        p[face.s0_] = 2.0 * face.bcVals["p"] - p[face.s1_]

        # Update conservatives
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        # neumann all gradients
        blk.array["dqdx"][face.s0_] = blk.array["dqdx"][face.s1_]
        blk.array["dqdy"][face.s0_] = blk.array["dqdy"][face.s1_]
        blk.array["dqdz"][face.s0_] = blk.array["dqdz"][face.s1_]
