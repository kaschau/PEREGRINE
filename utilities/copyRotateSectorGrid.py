#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility executes a copy/rotate operation on a sector grid.

Requires a path to a "from" folder that contains g.* and conn.yaml files.

The utility will output the copy/rotated grid in the "to" folder.

"""

import argparse
import peregrinepy as pg
import numpy as np
import os

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

    nblks = len(
        [f for f in os.listdir(fromDir) if f.startswith("g.") and f.endswith(".h5")]
    )

    fromGrid = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(fromGrid, fromDir)
    pg.readers.readConnectivity(fromGrid, fromDir)

    toGrid = pg.multiBlock.grid(nblks * nseg)

    # Copy the original grid to new grid's first sector
    # Also collect "high" and "low" faces.
    lowside = []
    highside = []
    for i, fromBlk in enumerate(fromGrid):
        toGrid[i] = fromBlk

        for face in fromBlk.faces:
            if face.bcType == "periodicRotLow":
                lowside.append(fromBlk.nblki)
            elif face.bcType == "periodicRotHigh":
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
        rotM[0, 0] = ct + ux ** 2 * (1 - ct)
        rotM[0, 1] = ux * uy * (1 - ct) - uz * st
        rotM[0, 2] = ux * uz * (1 - ct) + uy * st

        rotM[1, 0] = uy * ux * (1 - ct) + uz * st
        rotM[1, 1] = ct + uy ** 2 * (1 - ct)
        rotM[1, 2] = uy * uz * (1 - ct) - ux * st

        rotM[2, 0] = uz * ux * (1 - ct) - uy * st
        rotM[2, 1] = uz * uy * (1 - ct) + ux * st
        rotM[2, 2] = ct + uz ** 2 * (1 - ct)

        for j in range(nblks):
            fromBlk = fromGrid[j]

            rotNblki = (i + 1) * nblks + j
            rotBlk = toGrid[rotNblki]

            # copy/rotate block coordinates
            rotBlk.ni = fromBlk.ni
            rotBlk.nj = fromBlk.nj
            rotBlk.nk = fromBlk.nk

            shape = fromBlk.array["x"].shape
            points = np.column_stack(
                (
                    fromBlk.array["x"].ravel(),
                    fromBlk.array["y"].ravel(),
                    fromBlk.array["z"].ravel(),
                )
            )
            points = np.matmul(rotM, points.T).T
            rotBlk.array["x"] = points[:, 0].reshape(shape)
            rotBlk.array["y"] = points[:, 1].reshape(shape)
            rotBlk.array["z"] = points[:, 2].reshape(shape)

            # transfer connectivity
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):

                # treat boundary faces
                if fromFace.neighbor is None:
                    toFace.bcFam = fromFace.bcFam
                    toFace.bcType = fromFace.bcType
                    toFace.orientation = None
                    toFace.neighbor = None
                    continue

                # treat the rotated faces
                elif fromFace.bcType.startswith("periodic"):
                    toFace.bcFam = fromFace.bcFam
                    toFace.orientation = fromFace.orientation
                    # low side faces
                    if fromBlk.nblki in lowside:
                        toFace.bcType = "b0"
                        toFace.neighbor = fromFace.neighbor + nblks * i
                        toFace.bcFam = None
                    # high side faces
                    elif fromBlk.nblki in highside:
                        if i == nseg - 2:
                            toFace.bcType = "periodicRotHigh"
                            toFace.neighbor = fromFace.neighbor
                            toFace.bcFam = fromFace.bcFam
                        else:
                            toFace.bcType = "b0"
                            toFace.neighbor = fromFace.neighbor + nblks * (i + 2)
                            toFace.bcFam = None
                # treat the internal faces
                elif fromFace.bcType == "b0":
                    toFace.bcType = "b0"
                    toFace.bcFam = None
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
                if fromFace.bcType == "periodicRotLow":
                    toFace.neighbor = fromFace.neighbor + nblks * (nseg - 1)
                    if is360:
                        toFace.bcType = "b0"
                        toFace.bcFam = None

        # original and far highside
        elif fromBlk.nblki in highside:
            # original
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                if fromFace.bcType == "periodicRotHigh":
                    toFace.bcType = "b0"
                    toFace.neighbor = fromFace.neighbor + nblks
                    toFace.bcFam = None
            # new far high side
            rotBlk = toGrid[i + nblks * (nseg - 1)]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                if fromFace.bcType == "periodicRotHigh":
                    if is360:
                        toFace.bcType = "b0"
                        toFace.bcFam = None

    pg.writers.writeGrid(toGrid, toDir)
    pg.writers.writeConnectivity(toGrid, toDir)
