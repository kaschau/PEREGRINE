import numpy as np


def metrics(blk, fdOrder):

    x = blk.array["x"]
    y = blk.array["y"]
    z = blk.array["z"]

    if x is None:
        raise ValueError("You must initialize the grid arrays before computing metrics")

    # ----------------------------------------------------------------------------
    # Cell Centers
    # ----------------------------------------------------------------------------

    blk.array["xc"][:] = 0.125 * (
        x[0:-1, 0:-1, 0:-1]
        + x[0:-1, 0:-1, 1::]
        + x[0:-1, 1::, 0:-1]
        + x[0:-1, 1::, 1::]
        + x[1::, 0:-1, 0:-1]
        + x[1::, 0:-1, 1::]
        + x[1::, 1::, 0:-1]
        + x[1::, 1::, 1::]
    )

    blk.array["yc"][:] = 0.125 * (
        y[0:-1, 0:-1, 0:-1]
        + y[0:-1, 0:-1, 1::]
        + y[0:-1, 1::, 0:-1]
        + y[0:-1, 1::, 1::]
        + y[1::, 0:-1, 0:-1]
        + y[1::, 0:-1, 1::]
        + y[1::, 1::, 0:-1]
        + y[1::, 1::, 1::]
    )

    blk.array["zc"][:] = 0.125 * (
        z[0:-1, 0:-1, 0:-1]
        + z[0:-1, 0:-1, 1::]
        + z[0:-1, 1::, 0:-1]
        + z[0:-1, 1::, 1::]
        + z[1::, 0:-1, 0:-1]
        + z[1::, 0:-1, 1::]
        + z[1::, 1::, 0:-1]
        + z[1::, 1::, 1::]
    )
    if blk.blockType == "solver" and blk._isInitialized:
        for var in ["xc", "yc", "zc"]:
            blk.updateDeviceView(var)

    # ----------------------------------------------------------------------------
    # i face centers, area, normal vectors
    # ----------------------------------------------------------------------------
    blk.array["ixc"][:] = 0.25 * (
        x[:, 0:-1, 0:-1] + x[:, 0:-1, 1::] + x[:, 1::, 0:-1] + x[:, 1::, 1::]
    )
    blk.array["iyc"][:] = 0.25 * (
        y[:, 0:-1, 0:-1] + y[:, 0:-1, 1::] + y[:, 1::, 0:-1] + y[:, 1::, 1::]
    )
    blk.array["izc"][:] = 0.25 * (
        z[:, 0:-1, 0:-1] + z[:, 0:-1, 1::] + z[:, 1::, 0:-1] + z[:, 1::, 1::]
    )

    blk.array["isx"][:] = 0.5 * (
        (y[:, 1::, 1::] - y[:, 0:-1, 0:-1]) * (z[:, 0:-1, 1::] - z[:, 1::, 0:-1])
        - (z[:, 1::, 1::] - z[:, 0:-1, 0:-1]) * (y[:, 0:-1, 1::] - y[:, 1::, 0:-1])
    )

    blk.array["isy"][:] = 0.5 * (
        (z[:, 1::, 1::] - z[:, 0:-1, 0:-1]) * (x[:, 0:-1, 1::] - x[:, 1::, 0:-1])
        - (x[:, 1::, 1::] - x[:, 0:-1, 0:-1]) * (z[:, 0:-1, 1::] - z[:, 1::, 0:-1])
    )

    blk.array["isz"][:] = 0.5 * (
        (x[:, 1::, 1::] - x[:, 0:-1, 0:-1]) * (y[:, 0:-1, 1::] - y[:, 1::, 0:-1])
        - (y[:, 1::, 1::] - y[:, 0:-1, 0:-1]) * (x[:, 0:-1, 1::] - x[:, 1::, 0:-1])
    )

    blk.array["iS"][:] = np.sqrt(
        blk.array["isx"] ** 2 + blk.array["isy"] ** 2 + blk.array["isz"] ** 2
    )

    blk.array["inx"][:] = blk.array["isx"] / blk.array["iS"]
    blk.array["iny"][:] = blk.array["isy"] / blk.array["iS"]
    blk.array["inz"][:] = blk.array["isz"] / blk.array["iS"]

    if blk.blockType == "solver" and blk._isInitialized:
        for var in [
            "ixc",
            "iyc",
            "izc",
            "isx",
            "isy",
            "isz",
            "iS",
            "inx",
            "iny",
            "inz",
        ]:
            blk.updateDeviceView(var)

    # ----------------------------------------------------------------------------
    # j face center, area, normal vectors
    # ----------------------------------------------------------------------------
    blk.array["jxc"][:] = 0.25 * (
        x[0:-1, :, 0:-1] + x[0:-1, :, 1::] + x[1::, :, 0:-1] + x[1::, :, 1::]
    )
    blk.array["jyc"][:] = 0.25 * (
        y[0:-1, :, 0:-1] + y[0:-1, :, 1::] + y[1::, :, 0:-1] + y[1::, :, 1::]
    )
    blk.array["jzc"][:] = 0.25 * (
        z[0:-1, :, 0:-1] + z[0:-1, :, 1::] + z[1::, :, 0:-1] + z[1::, :, 1::]
    )

    blk.array["jsx"][:] = 0.5 * (
        (y[1::, :, 1::] - y[0:-1, :, 0:-1]) * (z[1::, :, 0:-1] - z[0:-1, :, 1::])
        - (z[1::, :, 1::] - z[0:-1, :, 0:-1]) * (y[1::, :, 0:-1] - y[0:-1, :, 1::])
    )

    blk.array["jsy"][:] = 0.5 * (
        (z[1::, :, 1::] - z[0:-1, :, 0:-1]) * (x[1::, :, 0:-1] - x[0:-1, :, 1::])
        - (x[1::, :, 1::] - x[0:-1, :, 0:-1]) * (z[1::, :, 0:-1] - z[0:-1, :, 1::])
    )

    blk.array["jsz"][:] = 0.5 * (
        (x[1::, :, 1::] - x[0:-1, :, 0:-1]) * (y[1::, :, 0:-1] - y[0:-1, :, 1::])
        - (y[1::, :, 1::] - y[0:-1, :, 0:-1]) * (x[1::, :, 0:-1] - x[0:-1, :, 1::])
    )

    blk.array["jS"][:] = np.sqrt(
        blk.array["jsx"] ** 2 + blk.array["jsy"] ** 2 + blk.array["jsz"] ** 2
    )

    blk.array["jnx"][:] = blk.array["jsx"] / blk.array["jS"]
    blk.array["jny"][:] = blk.array["jsy"] / blk.array["jS"]
    blk.array["jnz"][:] = blk.array["jsz"] / blk.array["jS"]

    if blk.blockType == "solver" and blk._isInitialized:
        for var in [
            "jxc",
            "jyc",
            "jzc",
            "jsx",
            "jsy",
            "jsz",
            "jS",
            "jnx",
            "jny",
            "jnz",
        ]:
            blk.updateDeviceView(var)

    # ----------------------------------------------------------------------------
    # k face center, area, normal vectors
    # ----------------------------------------------------------------------------
    blk.array["kxc"][:] = 0.25 * (
        x[0:-1, 0:-1, :] + x[0:-1, 1::, :] + x[1::, 0:-1, :] + x[1::, 1::, :]
    )
    blk.array["kyc"][:] = 0.25 * (
        y[0:-1, 0:-1, :] + y[0:-1, 1::, :] + y[1::, 0:-1, :] + y[1::, 1::, :]
    )
    blk.array["kzc"][:] = 0.25 * (
        z[0:-1, 0:-1, :] + z[0:-1, 1::, :] + z[1::, 0:-1, :] + z[1::, 1::, :]
    )

    blk.array["ksx"][:] = 0.5 * (
        (y[1::, 1::, :] - y[0:-1, 0:-1, :]) * (z[0:-1, 1::, :] - z[1::, 0:-1, :])
        - (z[1::, 1::, :] - z[0:-1, 0:-1, :]) * (y[0:-1, 1::, :] - y[1::, 0:-1, :])
    )

    blk.array["ksy"][:] = 0.5 * (
        (z[1::, 1::, :] - z[0:-1, 0:-1, :]) * (x[0:-1, 1::, :] - x[1::, 0:-1, :])
        - (x[1::, 1::, :] - x[0:-1, 0:-1, :]) * (z[0:-1, 1::, :] - z[1::, 0:-1, :])
    )

    blk.array["ksz"][:] = 0.5 * (
        (x[1::, 1::, :] - x[0:-1, 0:-1, :]) * (y[0:-1, 1::, :] - y[1::, 0:-1, :])
        - (y[1::, 1::, :] - y[0:-1, 0:-1, :]) * (x[0:-1, 1::, :] - x[1::, 0:-1, :])
    )

    blk.array["kS"][:] = np.sqrt(
        blk.array["ksx"] ** 2 + blk.array["ksy"] ** 2 + blk.array["ksz"] ** 2
    )

    blk.array["knx"][:] = blk.array["ksx"] / blk.array["kS"]
    blk.array["kny"][:] = blk.array["ksy"] / blk.array["kS"]
    blk.array["knz"][:] = blk.array["ksz"] / blk.array["kS"]

    if blk.blockType == "solver" and blk._isInitialized:
        for var in [
            "kxc",
            "kyc",
            "kzc",
            "ksx",
            "ksy",
            "ksz",
            "kS",
            "knx",
            "kny",
            "knz",
        ]:
            blk.updateDeviceView(var)

    # ----------------------------------------------------------------------------
    # Cell center volumes
    # ----------------------------------------------------------------------------
    #
    blk.array["J"][:] = (
        (x[1::, 1::, 1::] - x[0:-1, 0:-1, 0:-1])
        * (
            blk.array["isx"][1::, :, :]
            + blk.array["jsx"][:, 1::, :]
            + blk.array["ksx"][:, :, 1::]
        )
        + (y[1::, 1::, 1::] - y[0:-1, 0:-1, 0:-1])
        * (
            blk.array["isy"][1::, :, :]
            + blk.array["jsy"][:, 1::, :]
            + blk.array["ksy"][:, :, 1::]
        )
        + (z[1::, 1::, 1::] - z[0:-1, 0:-1, 0:-1])
        * (
            blk.array["isz"][1::, :, :]
            + blk.array["jsz"][:, 1::, :]
            + blk.array["ksz"][:, :, 1::]
        )
    ) / 3.0e0

    if blk.blockType == "solver" and blk._isInitialized:
        for var in ["J"]:
            blk.updateDeviceView(var)

    # ----------------------------------------------------------------------------
    # Cell center transformation metrics (ferda FD diffusion operator)
    # ----------------------------------------------------------------------------

    if fdOrder == 2:
        # cell corner
        x1 = x[0:-1, 0:-1, 0:-1]
        x2 = x[0:-1, 1::, 0:-1]
        x3 = x[1::, 1::, 0:-1]
        x4 = x[1::, 0:-1, 0:-1]
        x5 = x[0:-1, 0:-1, 1::]
        x6 = x[0:-1, 1::, 1::]
        x7 = x[1::, 1::, 1::]
        x8 = x[1::, 0:-1, 1::]

        y1 = y[0:-1, 0:-1, 0:-1]
        y2 = y[0:-1, 1::, 0:-1]
        y3 = y[1::, 1::, 0:-1]
        y4 = y[1::, 0:-1, 0:-1]
        y5 = y[0:-1, 0:-1, 1::]
        y6 = y[0:-1, 1::, 1::]
        y7 = y[1::, 1::, 1::]
        y8 = y[1::, 0:-1, 1::]

        z1 = z[0:-1, 0:-1, 0:-1]
        z2 = z[0:-1, 1::, 0:-1]
        z3 = z[1::, 1::, 0:-1]
        z4 = z[1::, 0:-1, 0:-1]
        z5 = z[0:-1, 0:-1, 1::]
        z6 = z[0:-1, 1::, 1::]
        z7 = z[1::, 1::, 1::]
        z8 = z[1::, 0:-1, 1::]

        # Derivative of (x,y,z) w.r.t. (E,N,X)
        dxdE = 0.25 * ((x4 - x1) + (x8 - x5) + (x3 - x2) + (x7 - x6))
        dydE = 0.25 * ((y4 - y1) + (y8 - y5) + (y3 - y2) + (y7 - y6))
        dzdE = 0.25 * ((z4 - z1) + (z8 - z5) + (z3 - z2) + (z7 - z6))

        dxdN = 0.25 * ((x2 - x1) + (x3 - x4) + (x7 - x8) + (x6 - x5))
        dydN = 0.25 * ((y2 - y1) + (y3 - y4) + (y7 - y8) + (y6 - y5))
        dzdN = 0.25 * ((z2 - z1) + (z3 - z4) + (z7 - z8) + (z6 - z5))

        dxdX = 0.25 * ((x5 - x1) + (x8 - x4) + (x6 - x2) + (x7 - x3))
        dydX = 0.25 * ((y5 - y1) + (y8 - y4) + (y6 - y2) + (y7 - y3))
        dzdX = 0.25 * ((z5 - z1) + (z8 - z4) + (z6 - z2) + (z7 - z3))

        blk.array["dEdx"][:] = (dydN * dzdX - dydX * dzdN) / blk.array["J"]
        blk.array["dEdy"][:] = (dxdN * dzdX - dxdX * dzdN) / -blk.array["J"]
        blk.array["dEdz"][:] = (dxdN * dydX - dxdX * dydN) / blk.array["J"]

        blk.array["dNdx"][:] = (dydE * dzdX - dydX * dzdE) / -blk.array["J"]
        blk.array["dNdy"][:] = (dxdE * dzdX - dxdX * dzdE) / blk.array["J"]
        blk.array["dNdz"][:] = (dxdE * dydX - dxdX * dzdE) / -blk.array["J"]

        blk.array["dXdx"][:] = (dydE * dzdN - dydN * dzdE) / blk.array["J"]
        blk.array["dXdy"][:] = (dxdE * dzdN - dxdN * dzdE) / -blk.array["J"]
        blk.array["dXdz"][:] = (dxdE * dydN - dxdN * dydE) / blk.array["J"]

    elif fdOrder == 4:
        xc = blk.array["xc"]
        yc = blk.array["yc"]
        zc = blk.array["zc"]

        dxdE = (
            -xc[4::, 2:-2, 2:-2]
            + 8.0 * xc[3:-1, 2:-2, 2:-2]
            - 8.0 * xc[1:-3, 2:-2, 2:-2]
            + xc[0:-4, 2:-2, 2:-2]
        ) / 12.0
        dxdN = (
            -xc[2:-2, 4::, 2:-2]
            + 8.0 * xc[2:-2, 3:-1, 2:-2]
            - 8.0 * xc[2:-2, 1:-3, 2:-2]
            + xc[2:-2, 0:-4, 2:-2]
        ) / 12.0
        dxdX = (
            -xc[2:-2, 2:-2, 4::]
            + 8.0 * xc[2:-2, 2:-2, 3:-1]
            - 8.0 * xc[2:-2, 2:-2, 1:-3]
            + xc[2:-2, 2:-2, 0:-4]
        ) / 12.0

        dydE = (
            -yc[4::, 2:-2, 2:-2]
            + 8.0 * yc[3:-1, 2:-2, 2:-2]
            - 8.0 * yc[1:-3, 2:-2, 2:-2]
            + yc[0:-4, 2:-2, 2:-2]
        ) / 12.0
        dydN = (
            -yc[2:-2, 4::, 2:-2]
            + 8.0 * yc[2:-2, 3:-1, 2:-2]
            - 8.0 * yc[2:-2, 1:-3, 2:-2]
            + yc[2:-2, 0:-4, 2:-2]
        ) / 12.0
        dydX = (
            -yc[2:-2, 2:-2, 4::]
            + 8.0 * yc[2:-2, 2:-2, 3:-1]
            - 8.0 * yc[2:-2, 2:-2, 1:-3]
            + yc[2:-2, 2:-2, 0:-4]
        ) / 12.0

        dzdE = (
            -zc[4::, 2:-2, 2:-2]
            + 8.0 * zc[3:-1, 2:-2, 2:-2]
            - 8.0 * zc[1:-3, 2:-2, 2:-2]
            + zc[0:-4, 2:-2, 2:-2]
        ) / 12.0
        dzdN = (
            -zc[2:-2, 4::, 2:-2]
            + 8.0 * zc[2:-2, 3:-1, 2:-2]
            - 8.0 * zc[2:-2, 1:-3, 2:-2]
            + zc[2:-2, 0:-4, 2:-2]
        ) / 12.0
        dzdX = (
            -zc[2:-2, 2:-2, 4::]
            + 8.0 * zc[2:-2, 2:-2, 3:-1]
            - 8.0 * zc[2:-2, 2:-2, 1:-3]
            + zc[2:-2, 2:-2, 0:-4]
        ) / 12.0

        blk.array["dEdx"][2:-2, 2:-2, 2:-2] = (dydN * dzdX - dydX * dzdN) / blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]
        blk.array["dEdy"][2:-2, 2:-2, 2:-2] = (dxdN * dzdX - dxdX * dzdN) / -blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]
        blk.array["dEdz"][2:-2, 2:-2, 2:-2] = (dxdN * dydX - dxdX * dydN) / blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]

        blk.array["dNdx"][2:-2, 2:-2, 2:-2] = (dydE * dzdX - dydX * dzdE) / -blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]
        blk.array["dNdy"][2:-2, 2:-2, 2:-2] = (dxdE * dzdX - dxdX * dzdE) / blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]
        blk.array["dNdz"][2:-2, 2:-2, 2:-2] = (dxdE * dydX - dxdX * dzdE) / -blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]

        blk.array["dXdx"][2:-2, 2:-2, 2:-2] = (dydE * dzdN - dydN * dzdE) / blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]
        blk.array["dXdy"][2:-2, 2:-2, 2:-2] = (dxdE * dzdN - dxdN * dzdE) / -blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]
        blk.array["dXdz"][2:-2, 2:-2, 2:-2] = (dxdE * dydN - dxdN * dydE) / blk.array[
            "J"
        ][2:-2, 2:-2, 2:-2]

    if blk.blockType == "solver" and blk._isInitialized:
        for var in [
            "dEdx",
            "dEdy",
            "dEdz",
            "dNdx",
            "dNdy",
            "dNdz",
            "dXdx",
            "dXdy",
            "dXdz",
        ]:
            blk.updateDeviceView(var)
