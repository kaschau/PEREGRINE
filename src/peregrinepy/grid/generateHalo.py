import numpy as np

s_ = np.s_


def generateHalo(blk):

    assert blk.blockType == "solver", "Only solver blocks can generate halos."

    ng = blk.ng

    fM = {}
    shapes = (
        (blk.nj, blk.nk),
        (blk.nj, blk.nk),
        (blk.ni, blk.nk),
        (blk.ni, blk.nk),
        (blk.ni, blk.nj),
        (blk.ni, blk.nj),
    )

    # Construct masks of the faces, edges, and corners.
    for i, shape in zip(range(6), shapes):
        face = i + 1
        fM[face] = {}
        fM[face]["faceMask"] = np.zeros((shape[0] + 2 * ng, shape[1] + 2 * ng))
        f = fM[face]["faceMask"]
        f[ng : shape[0] + ng, ng : shape[1] + ng] = 1.0
        fM[face]["faceMask"] = np.ma.make_mask(f)

        fM[face]["edgeMask"] = np.zeros((shape[0] + 2 * ng, shape[1] + 2 * ng))
        e = fM[face]["edgeMask"]
        e[0:ng, ng : shape[1] + ng] = 1.0
        e[ng : shape[0] + ng, 0:ng] = 1.0
        e[-ng::, ng : shape[1] + ng] = 1.0
        e[ng : shape[0] + ng, -ng::] = 1.0
        fM[face]["edgeMask"] = np.ma.make_mask(e)

        fM[face]["cornerMask"] = np.zeros((shape[0] + 2 * ng, shape[1] + 2 * ng))
        c = fM[face]["cornerMask"]
        c[0:ng, 0:ng] = 1.0
        c[0:ng, -ng::] = 1.0
        c[-ng::, 0:ng] = 1.0
        c[-ng::, -ng::] = 1.0
        fM[face]["cornerMask"] = np.ma.make_mask(c)

    varis = ["x", "y", "z"]

    # First make sure that the halo coordinates are zero
    for i, var in enumerate(varis):
        x = blk.array[var]
        x[0:ng, :, :] = 0.0
        x[-ng::, :, :] = 0.0
        x[:, 0:ng, :] = 0.0
        x[:, -ng::, :] = 0.0
        x[:, :, 0:ng] = 0.0
        x[:, :, -ng::] = 0.0

    # All faces
    for i, var in enumerate(varis):
        x = blk.array[var]
        for n in range(ng):
            s0 = ng - n - 1
            s1 = ng
            s2 = ng + n + 1
            # face 1
            mask = fM[1]["faceMask"]
            x[s0, :, :][mask] = 2.0 * x[s1, :, :][mask] - x[s2, :, :][mask]
            # face 3
            mask = fM[3]["faceMask"]
            x[:, s0, :][mask] = 2.0 * x[:, s1, :][mask] - x[:, s2, :][mask]
            # face 5
            mask = fM[5]["faceMask"]
            x[:, :, s0][mask] = 2.0 * x[:, :, s1][mask] - x[:, :, s2][mask]

            s0 = -ng + n
            s1 = -ng - 1
            s2 = -ng - n - 2
            # face 2
            mask = fM[2]["faceMask"]
            x[s0, :, :][mask] = 2.0 * x[s1, :, :][mask] - x[s2, :, :][mask]
            # face 4
            mask = fM[4]["faceMask"]
            x[:, s0, :][mask] = 2.0 * x[:, s1, :][mask] - x[:, s2, :][mask]
            # face 6
            mask = fM[6]["faceMask"]
            x[:, :, s0][mask] = 2.0 * x[:, :, s1][mask] - x[:, :, s2][mask]

    # All edges
    for i, var in enumerate(varis):
        x = blk.array[var]
        for n in range(ng):
            s0 = ng - n - 1
            s1 = ng
            s2 = ng + n + 1
            # face 1
            mask = fM[1]["edgeMask"]
            x[s0, :, :][mask] += 0.5 * (2.0 * x[s1, :, :][mask] - x[s2, :, :][mask])
            # face 3
            mask = fM[3]["edgeMask"]
            x[:, s0, :][mask] += 0.5 * (2.0 * x[:, s1, :][mask] - x[:, s2, :][mask])
            # face 5
            mask = fM[5]["edgeMask"]
            x[:, :, s0][mask] += 0.5 * (2.0 * x[:, :, s1][mask] - x[:, :, s2][mask])

            s0 = -ng + n
            s1 = -ng - 1
            s2 = -ng - n - 2
            # face 2
            mask = fM[2]["edgeMask"]
            x[s0, :, :][mask] += 0.5 * (2.0 * x[s1, :, :][mask] - x[s2, :, :][mask])
            # face 4
            mask = fM[4]["edgeMask"]
            x[:, s0, :][mask] += 0.5 * (2.0 * x[:, s1, :][mask] - x[:, s2, :][mask])
            # face 6
            mask = fM[6]["edgeMask"]
            x[:, :, s0][mask] += 0.5 * (2.0 * x[:, :, s1][mask] - x[:, :, s2][mask])

    # All corners
    for i, var in enumerate(varis):
        x = blk.array[var]
        for n in range(ng):
            s0 = ng - n - 1
            s1 = ng
            s2 = ng + n + 1
            # face 1
            mask = fM[1]["cornerMask"]
            x[s0, :, :][mask] += (
                1.0 / 3.0 * (2.0 * x[s1, :, :][mask] - x[s2, :, :][mask])
            )
            # face 3
            mask = fM[3]["cornerMask"]
            x[:, s0, :][mask] += (
                1.0 / 3.0 * (2.0 * x[:, s1, :][mask] - x[:, s2, :][mask])
            )
            # face 5
            mask = fM[5]["cornerMask"]
            x[:, :, s0][mask] += (
                1.0 / 3.0 * (2.0 * x[:, :, s1][mask] - x[:, :, s2][mask])
            )

            s0 = -ng + n
            s1 = -ng - 1
            s2 = -ng - n - 2
            # face 2
            mask = fM[2]["cornerMask"]
            x[s0, :, :][mask] += (
                1.0 / 3.0 * (2.0 * x[s1, :, :][mask] - x[s2, :, :][mask])
            )
            # face 4
            mask = fM[4]["cornerMask"]
            x[:, s0, :][mask] += (
                1.0 / 3.0 * (2.0 * x[:, s1, :][mask] - x[:, s2, :][mask])
            )
            # face 6
            mask = fM[6]["cornerMask"]
            x[:, :, s0][mask] += (
                1.0 / 3.0 * (2.0 * x[:, :, s1][mask] - x[:, :, s2][mask])
            )
