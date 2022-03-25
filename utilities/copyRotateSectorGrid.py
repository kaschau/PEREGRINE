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

    args = parser.parse_args()
    fromDir = args.fromDir
    toDir = args.toDir
    nseg = args.nseg
    sectorAngle = args.sectorAngle

    if nseg <= 1:
        raise ValueError(
            "nseg must be >= 1 (it corresponds to the total number of output segments)"
        )

    nblks = len(
        [f for f in os.listdir(fromDir) if f.startswith("g.") and f.endswith(".h5")]
    )

    fromGrid = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(fromGrid, fromDir)
    pg.readers.readConnectivity(fromGrid, fromDir)

    toGrid = pg.multiBlock.grid(nblks * nseg)

    # Copy the original grid to new grid's first sector
    # Also collect "front" and "back" faces.
    backside = []
    frontside = []
    for i, fromBlk in enumerate(fromGrid):
        toGrid[i] = fromBlk

        for face in fromBlk.faces:
            if face.bcType == "b1":
                if np.mean(fromBlk.array["z"]) > 0.0:
                    backside.append(fromBlk.nblki)
                elif np.mean(fromBlk.array["z"] < 0.0):
                    frontside.append(fromBlk.nblki)
                else:
                    print("HELP")

    # Now copy/rotate sector by sector
    # We will update the front and back side
    # faces and the new sector faces on this
    # pass
    for i in range(nseg - 1):
        angle = sectorAngle * (i + 1) * np.pi / 180.0
        for j in range(nblks):
            fromBlk = fromGrid[j]

            rotNblki = (i + 1) * nblks + j
            rotBlk = toGrid[rotNblki]

            # copy/rotate block coordinates
            rotBlk.ni = fromBlk.ni
            rotBlk.nj = fromBlk.nj
            rotBlk.nk = fromBlk.nk

            rotBlk.array["x"] = fromBlk.array["z"] * np.sin(angle) + fromBlk.array[
                "x"
            ] * np.cos(angle)
            rotBlk.array["y"] = fromBlk.array["y"]
            rotBlk.array["z"] = fromBlk.array["z"] * np.cos(angle) - fromBlk.array[
                "x"
            ] * np.sin(angle)

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
                if fromFace.bcType == "b1":
                    toFace.bcFam = None
                    toFace.orientation = fromFace.orientation
                    # back side faces
                    if fromBlk.nblki in backside:
                        toFace.bcType = "b0"
                        toFace.neighbor = fromFace.neighbor + nblks * i
                    # front side faces
                    elif fromBlk.nblki in frontside:
                        if i == nseg - 2:
                            toFace.bcType = "b1"
                            toFace.neighbor = fromFace.neighbor
                        else:
                            toFace.bcType = "b0"
                            toFace.neighbor = fromFace.neighbor + nblks * (i + 2)
                # treat the internal faces
                elif fromFace.bcType == "b0":
                    toFace.bcType = "b0"
                    toFace.bcFam = None
                    toFace.orientation = fromFace.orientation
                    toFace.neighbor = fromFace.neighbor + nblks * (i + 1)

    # At this point, the original sector front and back side faces are out of
    # date, as is the last sector's front side
    if abs(nseg * sectorAngle - 360.0) < 1e-10:
        is360 = True
    else:
        is360 = False

    for i, fromBlk in enumerate(fromGrid):
        # backside
        if fromBlk.nblki in backside:
            rotBlk = toGrid[i]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                if fromFace.bcType == "b1":
                    toFace.neighbor = fromFace.neighbor + nblks * (nseg - 1)
                if is360:
                    toFace.bcType = "b0"

        # original and far frontside
        elif fromBlk.nblki in frontside:
            # original
            rotBlk = toGrid[i]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                if fromFace.bcType == "b1":
                    toFace.bcType = "b0"
                    toFace.neighbor = fromFace.neighbor + nblks
            # far frontside
            rotBlk = toGrid[i + nblks * (nseg - 1)]
            for toFace, fromFace in zip(rotBlk.faces, fromBlk.faces):
                if is360:
                    toFace.bcType = "b0"

    pg.writers.writeGrid(toGrid, toDir)
    pg.writers.writeConnectivity(toGrid, toDir)
    # match_faces(toGrid)
