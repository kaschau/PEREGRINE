#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility executes a copy/rotate operation on a sector grid.

Requires a path to a "from" folder that contains the g.* files.

The utility will output the copy/rotated grid in the "to" folder.

"""

import argparse
import peregrinepy as pg
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate from one grid to another")
    parser.add_argument(
        "-from",
        "--fromDir",
        action="store",
        metavar="<fromDir>",
        dest="fromDir",
        default="./from",
        help="Directory containing the g.*.h5 and q.*.h5 files to interpolate from. Default is ./from",
        type=str,
    )
    parser.add_argument(
        "-to",
        "--toDir",
        action="store",
        metavar="<toDir>",
        dest="toDir",
        default="./to",
        help="Directory containing the g.*.h5 files to interpolate to. Default is ./to",
        type=str,
    )
    parser.add_argument(
        "-nseg",
        "--numSegments",
        action="store",
        metavar="<nseg>",
        dest="nseg",
        default="2",
        help="Number of output segments (must be > 1)",
        type=int,
    )
    parser.add_argument(
        "-sectorAngle",
        "--sectorAngle",
        action="store",
        metavar="<sectorAngle>",
        dest="sectorAngle",
        help="Angle of sector in degrees.",
        type=float,
    )
    parser.add_argument(
        "-axis",
        "--rotationAxis",
        action="store",
        metavar="<axis>",
        dest="axis",
        help="List of sector axis components, delimited with comma",
        type=str,
    )

    args = parser.parse_args()
    fromDir = args.fromDir
    toDir = args.toDir
    nseg = args.nseg
    sectorAngle = args.sectorAngle
    axis = [float(item) for item in args.axis.split(",")]

    if nseg <= 1:
        raise ValueError(
            "nseg must be > 1 (it corresponds to the total number of output segments)"
        )

    fromGrid = pg.multiBlock.grid.fromGrid(fromDir)
    nblks = len(fromGrid)

    toGrid = pg.multiBlock.grid(nblks * nseg)

    # Copy the original grid to new grid's first sector
    # Also collect "high" and "low" faces.
    lowside = []
    highside = []
    for i, fromBlk in enumerate(fromGrid):
        toBlk = toGrid[i]
        # Copy coordinates
        toBlk.array = fromBlk.array
        toBlk.ni = fromBlk.ni
        toBlk.nj = fromBlk.nj
        toBlk.nk = fromBlk.nk

        # Copy connectivity (must be explicitely copied, not just pointed to
        # like coordinates can)
        for toFace, fromFace in zip(toBlk.faces, fromBlk.faces):
            toFace.bcName = fromFace.bcName
            toFace.bcType = fromFace.bcType
            toFace.orientation = fromFace.orientation
            toFace.neighbor = fromFace.neighbor

            if fromFace.amILow:
                lowside.append(fromBlk.nblki)
            elif not fromFace.amILow:
                highside.append(fromBlk.nblki)

    # Now copy/rotate sector by sector
    # We will update the high and low side
    # faces and the new sector faces on this
    # pass
    for i in range(nseg - 1):
        angle = sectorAngle * (i + 1) * np.pi / 180.0
        # Compute rotation matrix for positive and negative rotatoin
        rotM = np.zeros((3, 3))
        ct = np.cos(angle)
        st = np.sin(angle)
        ux, uy, uz = tuple(axis)
        rotM[0, 0] = ct + ux**2 * (1 - ct)
        rotM[0, 1] = ux * uy * (1 - ct) - uz * st
        rotM[0, 2] = ux * uz * (1 - ct) + uy * st

        rotM[1, 0] = uy * ux * (1 - ct) + uz * st
        rotM[1, 1] = ct + uy**2 * (1 - ct)
        rotM[1, 2] = uy * uz * (1 - ct) - ux * st

        rotM[2, 0] = uz * ux * (1 - ct) - uy * st
        rotM[2, 1] = uz * uy * (1 - ct) + ux * st
        rotM[2, 2] = ct + uz**2 * (1 - ct)

        for j in range(nblks):
            fromBlk = fromGrid[j]

            rotNblki = (i + 1) * nblks + j
            rotBlk = toGrid[rotNblki]

            # copy/rotate block coordinates
            rotBlk.setExtents(fromBlk.ni, fromBlk.nj, fromBlk.nk)

            shape = fromBlk.array["nodes"].shape
            points = fromBlk.array["nodes"].reshape(-1, 3)
            points = np.matmul(rotM, points.T).T
            rotBlk.array["nodes"][:] = points.reshape(shape)

            # transfer connectivity
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                # treat boundary faces
                if fromFace.neighbor is None:
                    toFace.bcName = fromFace.bcName
                    toFace.bcType = fromFace.bcType
                    toFace.orientation = None
                    toFace.neighbor = None
                    continue

                # treat the rotated faces
                elif fromFace.bcType.startswith("periodic"):
                    toFace.bcName = fromFace.bcName
                    toFace.orientation = fromFace.orientation
                    # low side faces
                    if fromBlk.nblki in lowside:
                        toFace.bcType = "interior"
                        toFace.neighbor = fromFace.neighbor + nblks * i
                        toFace.bcName = None
                    # high side faces
                    elif fromBlk.nblki in highside:
                        if i == nseg - 2:
                            toFace.bcType = "periodicRot"
                            toFace.neighbor = fromFace.neighbor
                            toFace.bcName = fromFace.bcName
                        else:
                            toFace.bcType = "interior"
                            toFace.neighbor = fromFace.neighbor + nblks * (i + 2)
                            toFace.bcName = None
                # treat the internal faces
                elif fromFace.bcType == "interior":
                    toFace.bcType = "interior"
                    toFace.bcName = None
                    toFace.orientation = fromFace.orientation
                    toFace.neighbor = fromFace.neighbor + nblks * (i + 1)
                else:
                    raise ValueError("What is this face?")

    # At this point, the original sector high and low side faces are out of
    # date, as is the last sector's high side
    if abs(nseg * sectorAngle - 360.0) < 1e-10:
        is360 = True
    else:
        is360 = False

    for i, fromBlk in enumerate(fromGrid):
        # lowside
        if fromBlk.nblki in lowside:
            rotBlk = toGrid[i]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                if fromFace.amILow:
                    # set neighbor to new high side
                    toFace.neighbor = fromFace.neighbor + nblks * (nseg - 1)
                    if is360:
                        toFace.bcType = "interior"
                        toFace.bcName = None

        # original and far highside
        elif fromBlk.nblki in highside:
            # original
            rotBlk = toGrid[i]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                # set to internal with new neighbor
                if not fromFace.amILow:
                    toFace.bcType = "interior"
                    toFace.neighbor = fromFace.neighbor + nblks
                    toFace.bcName = None
            # new far high side
            rotBlk = toGrid[i + nblks * (nseg - 1)]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                # if360, the new high side is an internal face
                if not fromFace.amILow:
                    if is360:
                        toFace.bcType = "interior"
                        toFace.bcName = None

    pg.writers.GridWriter(toGrid, toDir).write(toGrid)
