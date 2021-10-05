# -*- coding: utf-8 -*-

""" create.py

Authors:

Kyle Schau


This module holds functions for creating grid_block objects of various forms.

"""

import numpy as np


def cubicConnectivity(blk, mbDims, blkNum, i, j, k):

    # i faces
    if i == 0 and mbDims[0] != 1:
        conn = blk.getFaceConn(1)
        conn["bcType"] = "adiabaticNoSlipWall"
        conn["neighbor"] = None
        conn["orientation"] = None

        conn = blk.getFaceConn(2)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum + 1
        conn["orientation"] = "123"

    elif i == mbDims[0] - 1 and mbDims[0] != 1:
        conn = blk.getFaceConn(1)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum - 1
        conn["orientation"] = "123"

        conn = blk.getFaceConn(2)
        conn["bcType"] = "adiabaticNoSlipWall"
        conn["neighbor"] = None
        conn["orientation"] = None

    elif mbDims[0] != 1:
        conn = blk.getFaceConn(1)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum - 1
        conn["orientation"] = "123"

        conn = blk.getFaceConn(2)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum + 1
        conn["orientation"] = "123"

    # j faces
    if j == 0 and mbDims[1] != 1:
        conn = blk.getFaceConn(3)
        conn["bcType"] = "adiabaticNoSlipWall"
        conn["neighbor"] = None
        conn["orientation"] = None

        conn = blk.getFaceConn(4)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum + mbDims[0]
        conn["orientation"] = "123"

    elif j == mbDims[1] - 1 and mbDims[1] != 1:
        conn = blk.getFaceConn(3)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum - mbDims[0]
        conn["orientation"] = "123"

        conn = blk.getFaceConn(4)
        conn["bcType"] = "adiabaticNoSlipWall"
        conn["neighbor"] = None
        conn["orientation"] = None

    elif mbDims[1] != 1:
        conn = blk.getFaceConn(3)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum - mbDims[0]
        conn["orientation"] = "123"

        conn = blk.getFaceConn(4)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum + mbDims[0]
        conn["orientation"] = "123"

    # k faces
    if k == 0 and mbDims[2] != 1:
        conn = blk.getFaceConn(5)
        conn["bcType"] = "adiabaticNoSlipWall"
        conn["neighbor"] = None
        conn["orientation"] = None

        conn = blk.getFaceConn(6)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum + mbDims[0] * mbDims[1]
        conn["orientation"] = "123"

    elif k == mbDims[2] - 1 and mbDims[2] != 1:
        conn = blk.getFaceConn(5)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum - mbDims[0] * mbDims[1]
        conn["orientation"] = "123"

        conn = blk.getFaceConn(6)
        conn["bcType"] = "adiabaticNoSlipWall"
        conn["neighbor"] = None
        conn["orientation"] = None

    elif mbDims[2] != 1:
        conn = blk.getFaceConn(5)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum - mbDims[0] * mbDims[1]
        conn["orientation"] = "123"

        conn = blk.getFaceConn(6)
        conn["bcType"] = "b0"
        conn["neighbor"] = blkNum + mbDims[0] * mbDims[1]
        conn["orientation"] = "123"


def cube(blk, origin, lengths, dimensions):

    """Function to populate the coordinate arrays of a provided peregrinepy.block in the shape of a cube with prescribed location, extents, and discretization.
    If the input multiblock object is a restart block the shape and size of the flow data arrays are also updated.

    Parameters
    ----------

    blk : raptorpy.blocks.grid_block (or one of its descendants)

    origin : list, tuple
       List/tuple of length 3 containing the location of the origin of the cube to be created

    lengths : list, tuple
       List/tuple of length 3 containing the extents in x, y, and z of the cube relative to the origin

    dimensions : list, tuple
       List/tuple of length 3 containing discretization (nx,nj,nk) in each dimension of the cube.

    Returns
    -------
    None
        Updates attributes of parameter blk.

    """
    blk.ni = dimensions[0]
    blk.nj = dimensions[1]
    blk.nk = dimensions[2]

    blk.initGridArrays()

    x = np.linspace(origin[0], origin[0] + lengths[0], dimensions[0], dtype=np.float64)
    y = np.linspace(origin[1], origin[1] + lengths[1], dimensions[1], dtype=np.float64)
    z = np.linspace(origin[2], origin[2] + lengths[2], dimensions[2], dtype=np.float64)

    if blk.blockType == "solver":
        s_i = np.s_[1:-1, 1:-1, 1:-1]
    else:
        s_i = np.s_[:, :, :]

    blk.array["x"][s_i], blk.array["y"][s_i], blk.array["z"][s_i] = np.meshgrid(
        x, y, z, indexing="ij"
    )

    if blk.blockType in ["restart", "solver"]:
        blk.initRestartArrays()


def multiblockCube(
    mb,
    origin=[0, 0, 0],
    lengths=[1, 1, 1],
    mbDims=[1, 1, 1],
    dimsPerBlock=[10, 10, 10],
):

    """Function to populate the coordinate arrays of a raptorpy.multiblock.grid (or one of its descendants) in the shape of a cube
       with prescribed location, extents, and discretization split into as manj  blocks as mb.nblks. Will also update
       connectivity of interblock faces. If the input multiblock object is a restart block the shape and size of the flow
       data arrays are also updated.

    Parameters
    ----------

    mb : raptorpy.multiblock.grid (or one of its descendants)

    origin : list, tuple
       List/tuple of length 3 containing the location of the origin of the ENTIRE cube to be created

    lengths : list, tuple
       List/tuple of length 3 containing the extents in x, y, and z of the ENTIRE cube relative to the origin

    mbDims : list, tuple
       List/tuple of length 3 containing number of blocks in x, y, and z. NOTE: product of mbDims must equal mb.nblks!

    dimsPerBlock : list, tuple
       List/tuple of length 3 containing discretization (nx,nj,nk) in each dimension of each block to be created.

    Returns
    -------
    None
        Updates elements in mb

    """

    if np.product(mbDims) != mb.nblks:
        raise ValueError(
            "Warning, multiblock dimensions does not equal number of blocks!"
        )

    blk_origins_x = np.linspace(origin[0], origin[0] + lengths[0], mbDims[0] + 1)
    blk_origins_y = np.linspace(origin[1], origin[1] + lengths[1], mbDims[1] + 1)
    blk_origins_z = np.linspace(origin[2], origin[2] + lengths[2], mbDims[2] + 1)

    for k in range(int(mbDims[2])):
        for j in range(int(mbDims[1])):
            for i in range(int(mbDims[0])):
                blkNum = k * mbDims[1] * mbDims[0] + j * mbDims[0] + i

                blk = mb[blkNum]
                blk.nblki = blkNum

                origin = [blk_origins_x[i], blk_origins_y[j], blk_origins_z[k]]
                lengths = [
                    blk_origins_x[i + 1] - blk_origins_x[i],
                    blk_origins_y[j + 1] - blk_origins_y[j],
                    blk_origins_z[k + 1] - blk_origins_z[k],
                ]
                dimensions = [
                    dimsPerBlock[0],
                    dimsPerBlock[1],
                    dimsPerBlock[2],
                ]

                cube(blk, origin, lengths, dimensions)

                # Update connectivity
                cubicConnectivity(blk, mbDims, blkNum, i, j, k)


def annulus(blk, p1, p2, p3, sweep, thickness, dimensions):

    raise ValueError("The annulus is jacked, needs to be updated.")

    """Function to populate the coordinate arrays of a provided raptorpy.grid.grid_block in the shape of an annulus with prescribed location, extents, and discretization.
    If the input multiblock object is a restart block the shape and size of the flow data arrays are also updated.

    Parameters
    ----------

    blk : raptorpy.blocks.grid_block (or one of its descendants)

    p1 : list, tuple
       List/tuple of length 3 containing the location of the origin of the annulus to be created, i.e.
       the center of the beginning of the cylindrical segment

    p2 : list, tuple
       List/tuple of length 3 containing the location of the end of the annulus to be created, i.e.
       the center of the end of the cylindrical segment

    p3 : list, tuple
       List/tuple of length 3 containing the location of a point orthogonal to the line (p1,p2) marking
       the inner most corner point of the cylindrical segment. This point also serves as the inner radius
       of the cylindrical segment. The outer radius is measured  from :p3: outward along the line (p1,p3)
       a distance of :thickness:. This point also serves as the starting angular point for :sweep: to be measured
       according to the right hand rule about the line (p1,p2). I.e. the variable :sweep: measures the angle
       about which the cylindrical segment "sweeps" starting from the line (p1,p3).

    sweep : float
       Float denoting the angle (in degrees) of sweep of the annular segment in the direction according to the right
       hand rule about the line (p1,p2) starting using the line (p1,p3) as the starting point for the sweep.

    thickness : float
       Float denoting (outer radius - inner radius) of the annulus, where the inner radius is determined by
       the length of the line (p1,p3).

    dimensions : list, tuple
       List/tuple of length 3 containing discretization (ni,nj,nk) in each dimension of the cube. Where the "x"
       direction is along the annulus axis, the "y" direction is along the radial direction, and the "z" direction
       is along the theta direction.

    Returns
    -------
    None
        Updates attributes of parameter blk.

    """

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    if np.dot(p2 - p1, p3 - p1) != 0.0:
        raise ValueError("Error: The line (p1,p2) is not orthogonal to (p1,p3)")

    if abs(sweep) < -360 or abs(sweep) > 360.0:
        raise ValueError("Error: sweep parameter must be >-360 and <360")

    n12 = (p2 - p1) / np.linalg.norm(p2 - p1)
    n13 = (p3 - p1) / np.linalg.norm(p3 - p1)

    shape = (blk.ni + 2, blk.nj + 2, blk.nk + 2)
    blk.array["x"] = np.zeros(shape)
    blk.array["y"] = np.zeros(shape)
    blk.array["z"] = np.zeros(shape)

    dx = np.linalg.norm(p2 - p1) / (blk.ni - 1)
    dr = thickness / (blk.nj - 1)
    dtheta = sweep / (blk.nk - 1)

    for j in range(blk.nj):
        for i in range(blk.ni):
            p_ij = np.append(p3 + dx * i * n12 + dr * j * n13, 1)

            blk.array["x"][i, j, 0] = p_ij[0]
            blk.array["y"][i, j, 0] = p_ij[1]
            blk.array["z"][i, j, 0] = p_ij[2]

    xflat = np.reshape(blk.array["x"][:, :, 0], (blk.ni * blk.nj, 1))
    yflat = np.reshape(blk.array["y"][:, :, 0], (blk.ni * blk.nj, 1))
    zflat = np.reshape(blk.array["z"][:, :, 0], (blk.ni * blk.nj, 1))

    pts = np.hstack((xflat, yflat, zflat))
    p = pts - p1
    shape = blk.array["x"][:, :, 0].shape
    for k in range(1, blk.nk):

        # See http://paulbourke.net/geometry/rotate/
        theta = k * dtheta * np.pi / 180.0
        ct = np.cos(theta)
        st = np.sin(theta)

        q = np.zeros(pts.shape)
        q[:, 0] += (ct + (1 - ct) * n12[0] * n12[0]) * p[:, 0]
        q[:, 0] += ((1 - ct) * n12[0] * n12[1] - n12[2] * st) * p[:, 1]
        q[:, 0] += ((1 - ct) * n12[0] * n12[2] + n12[1] * st) * p[:, 2]

        q[:, 1] += ((1 - ct) * n12[0] * n12[1] + n12[2] * st) * p[:, 0]
        q[:, 1] += (ct + (1 - ct) * n12[1] * n12[1]) * p[:, 1]
        q[:, 1] += ((1 - ct) * n12[1] * n12[2] - n12[0] * st) * p[:, 2]

        q[:, 2] += ((1 - ct) * n12[0] * n12[2] - n12[1] * st) * p[:, 0]
        q[:, 2] += ((1 - ct) * n12[1] * n12[2] + n12[0] * st) * p[:, 1]
        q[:, 2] += (ct + (1 - ct) * n12[2] * n12[2]) * p[:, 2]

        q[:, 0] += p1[0]
        q[:, 1] += p1[1]
        q[:, 2] += p1[2]

        blk.array["x"][:, :, k] = np.reshape(q[:, 0], shape)
        blk.array["y"][:, :, k] = np.reshape(q[:, 1], shape)
        blk.array["z"][:, :, k] = np.reshape(q[:, 2], shape)


def multiblockAnnulus(
    mb, p1, p2, p3, sweep, thickness, mbDims, dimsPerBlock, periodic=False
):

    """Function to populate the coordinate arrays of a raptorpy.multiblock.grid (or one of its descendants) in the shape
       of an annulus with prescribed location, extents, and discretization split into as manj blocks as mb.nblks.
       Will also update connectivity of interblock faces. If the input multiblock object is a restart block the shape
       and size of the flow data arrays are also updated.

    Parameters
    ----------

    mb : raptorpy.multiblock.grid (or one of its descendants)

    p1 : list, tuple
       List/tuple of length 3 containing the location of the origin of the annulus to be created, i.e.
       the center of the beginning of the whole annulus.

    p2 : list, tuple
       List/tuple of length 3 containing the location of the end of the annulus to be created, i.e.
       the center of the end of the the whole.

    p3 : list, tuple
       List/tuple of length 3 containing the location of a point orthogonal to the line (p1,p2) marking
       the inner most corner point of the cylindrical segment. This point also serves as the inner radius
       of the cylindrical segment. The outer radius is measured  from :p3: outward along the line (p1,p3)
       a distance of :thickness:. This point also serves as the starting angular point for :sweep: to be measured
       according to the right hand rule about the line (p1,p2). I.e. the variable :sweep: measures the angle
       about which the cylindrical segment "sweeps" starting from the line (p1,p3).

    sweep : float
       Float denoting the angle (in degrees) of sweep of the annulus in the direction according to the right
       hand rule about the line (p1,p2) starting using the line (p1,p3) as the starting point for the sweep.

    thickness : float
       Float denoting (outer radius - inner radius) of the annulus, where the inner radius is determined by
       the length of the line (p1,p3).

    mbDims : list, tuple
       List/tuple of length 3 containing number of blocks in axial direction, radial direction, and theta direction.
       NOTE: product of mbDims must equal mb.nblks!

    dimensions : list, tuple
       List/tuple of length 3 containing discretization (ni,nj,nk) in each dimension of every block (all will be uniform). Where the "x,i,xi"
       direction is along the annulus axis, the "y,j,eta" direction is along the radial direction, and the "z,k,zeta" direction
       is along the theta direction.

    Returns
    -------
    None
        Updates elements in mb

    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    if np.product(mbDims) != mb.nblks:
        raise ValueError(
            "Error: multiblock dimensions does not equal number of blocks!"
        )

    if np.dot(p2 - p1, p3 - p1) != 0.0:
        raise ValueError("Error: The line (p1,p2) is not orthogonal to (p1,p3)")

    if sweep < 0.0 or sweep > 360.0:
        raise ValueError("Error: sweep parameter must be >0 and <360")

    if periodic or float(sweep) == 360.0:
        connect = True
    else:
        connect = False

    n12 = (p2 - p1) / np.linalg.norm(p2 - p1)
    p = p3 - p1

    dx = np.linalg.norm(p2 - p1) / mbDims[0]
    dr = thickness / mbDims[1]
    dtheta = sweep / mbDims[2]

    for k in range(int(mbDims[2])):

        # See http://paulbourke.net/geometry/rotate/
        theta = k * dtheta * np.pi / 180.0
        ct = np.cos(theta)
        st = np.sin(theta)

        q = np.array([0.0, 0.0, 0.0])
        q[0] += (ct + (1 - ct) * n12[0] * n12[0]) * p[0]
        q[0] += ((1 - ct) * n12[0] * n12[1] - n12[2] * st) * p[1]
        q[0] += ((1 - ct) * n12[0] * n12[2] + n12[1] * st) * p[2]

        q[1] += ((1 - ct) * n12[0] * n12[1] + n12[2] * st) * p[0]
        q[1] += (ct + (1 - ct) * n12[1] * n12[1]) * p[1]
        q[1] += ((1 - ct) * n12[1] * n12[2] - n12[0] * st) * p[2]

        q[2] += ((1 - ct) * n12[0] * n12[2] - n12[1] * st) * p[0]
        q[2] += ((1 - ct) * n12[1] * n12[2] + n12[0] * st) * p[1]
        q[2] += (ct + (1 - ct) * n12[2] * n12[2]) * p[2]

        q[0] += p1[0]
        q[1] += p1[1]
        q[2] += p1[2]

        n13 = (q - p1) / np.linalg.norm(q - p1)

        for j in range(int(mbDims[1])):
            for i in range(int(mbDims[0])):

                blkNum = k * mbDims[1] * mbDims[0] + j * mbDims[0] + i
                blk = mb[blkNum]
                blk.nblki = blkNum
                blk.ni = dimsPerBlock[0]
                blk.nj = dimsPerBlock[1]
                blk.nk = dimsPerBlock[2]

                newp1 = p1 + dx * i * n12
                newp2 = p1 + dx * (i + 1) * n12
                newp3 = q + dx * i * n12 + dr * j * n13

                annulus(blk, newp1, newp2, newp3, dtheta, dr, dimsPerBlock)

                # Update connectivity
                conn = blk.connectivity
                cubicConnectivity(conn, mbDims, blkNum, i, j, k)

                # k faces
                if connect:
                    if k == 0:
                        if float(sweep) == 360.0:
                            conn["5"]["bc"] = "b0"
                        else:
                            conn["5"]["bc"] = "b1"
                        conn["5"]["neighbor"] = blkNum + mbDims[0] * mbDims[1] * (
                            mbDims[2] - 1
                        )
                        conn["5"]["orientation"] = "123"

                    elif k == mbDims[2] - 1:
                        if float(sweep) == 360.0:
                            conn["6"]["bc"] = "b0"
                        else:
                            conn["6"]["bc"] = "b1"
                        conn["6"]["neighbor"] = blkNum - mbDims[0] * mbDims[1] * (
                            mbDims[2] - 1
                        )
                        conn["6"]["orientation"] = "123"
