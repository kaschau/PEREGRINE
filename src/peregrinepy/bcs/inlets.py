def constantVelocitySubsonicInlet(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == "euler":
        p = blk.array["q"][:, :, :, 0]
        u = blk.array["q"][:, :, :, 1]
        v = blk.array["q"][:, :, :, 2]
        w = blk.array["q"][:, :, :, 3]
        T = blk.array["q"][:, :, :, 4]

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            p[s0_] = 2.0 * p[face.s1_] - p[s2_]

            # apply velo on face
            u[s0_] = 2.0 * face.bcVals["u"] - u[face.s1_]
            v[s0_] = 2.0 * face.bcVals["v"] - v[face.s1_]
            w[s0_] = 2.0 * face.bcVals["w"] - w[face.s1_]

            T[s0_] = 2.0 * face.bcVals["T"] - T[face.s1_]

            for n, sn in enumerate(thtrdat.speciesNames[0:-1]):
                N = blk.array["q"][:, :, :, 5 + n]
                N[s0_] = 2.0 * face.bcVals[sn] - N[face.s1_]

        # Update conserved
        eos(blk, thtrdat, nface, "prims")

    elif terms == "viscous":
        for s0_ in face.s0_:
            # neumann all gradients
            blk.array["dqdx"][s0_] = blk.array["dqdx"][face.s1_]
            blk.array["dqdy"][s0_] = blk.array["dqdy"][face.s1_]
            blk.array["dqdz"][s0_] = blk.array["dqdz"][face.s1_]
