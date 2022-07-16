#!/usr/bin/env python3

"""
This utility goes through a grid face by face,
verifying that all the block's connectivities agree,
and that the coordinates of matching faces are identical.

Inputs are the path to the grid files, and path to the conn.yaml file.

If you have periodicity in the grid, you must have the periodic
information populated in the grid, i.e. periodicSpan and periodicAxis

Output will print any discrepencies to the screen

"""

import argparse
import os

import numpy as np
import peregrinepy as pg


faceToOrientIndexMapping = {
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
}

largeIndexMapping = {0: "k", 1: "k", 2: "j"}
needToTranspose = {
    "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
    "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
}


def extractFace(blk, nface):

    faceSliceMapping = {
        1: {"i": 0, "j": slice(None), "k": slice(None)},
        2: {"i": -1, "j": slice(None), "k": slice(None)},
        3: {"i": slice(None), "j": 0, "k": slice(None)},
        4: {"i": slice(None), "j": -1, "k": slice(None)},
        5: {"i": slice(None), "j": slice(None), "k": 0},
        6: {"i": slice(None), "j": slice(None), "k": -1},
    }

    face_i = faceSliceMapping[nface]

    x = np.copy(blk.array["x"][face_i["i"], face_i["j"], face_i["k"]])
    y = np.copy(blk.array["y"][face_i["i"], face_i["j"], face_i["k"]])
    z = np.copy(blk.array["z"][face_i["i"], face_i["j"], face_i["k"]])

    return x, y, z


def verify(mb):

    warn = False
    for blk in mb:
        for face in blk.faces:

            nface = face.nface
            neighbor = face.neighbor
            bc = face.bcType
            orientation = face.orientation
            bcFam = face.bcFam

            if neighbor is None:
                assert bc in (
                    # Inlets
                    "constantVelocitySubsonicInlet",
                    "supersonicInlet",
                    "constantMassFluxSubsonicInlet",
                    "cubicSplineSubsonicInlet",
                    # Exits
                    "constantPressureSubsonicExit",
                    "supersonicExit",
                    # Walls
                    "adiabaticNoSlipWall",
                    "adiabaticSlipWall",
                    "adiabaticMovingWall",
                    "isoTNoSlipWall",
                    "isoTSlipWall",
                    "isoTMovingWall",
                ), f"Block #{blk.nblki} face {nface} has no neighbor, but has bcType {bc}"

                assert bc not in (
                    # Interior, periodic
                    "b0",
                    "periodicTransLow",
                    "periodicTransHigh",
                    "periodicRotLow",
                    "periodicRotHigh",
                ), f"Block #{blk.nblki} face {nface} has no neighbor, but has bcType {bc}"

                assert (
                    orientation is None
                ), f"Block #{blk.nblki} face {nface} has no neighbor, but has orientation {orientation}"

                if bc not in (
                    "supersonicExit",
                    "adiabaticNoSlipWall",
                    "adiabaticSlipWall",
                ):
                    assert (
                        bcFam is not None
                    ), f"Block #{blk.nblki} face {nface} is {bc}, but has no bcFam"

                continue

            (face_x, face_y, face_z) = extractFace(blk, face.nface)

            blk2 = mb.getBlock(neighbor)
            nface2 = face.neighborNface
            nOrientation = face.neighborOrientation

            if int(blk2.getFace(nface2).neighbor) != blk.nblki:
                raise ValueError(
                    f"Block {blk.nblki}'s' face {nface} says it is connected to\nblock {blk2.nblki}'s' face {nface2}, however block {blk2.nblki}'s\nface {nface2} says it is connected to a different block."
                )

            (face2_x, face2_y, face2_z) = extractFace(blk2, nface2)

            faceOrientations = [
                int(i)
                for j, i in enumerate(nOrientation)
                if j != faceToOrientIndexMapping[nface2]
            ]
            normalIndex = [
                j for j in range(3) if j == faceToOrientIndexMapping[nface2]
            ][0]
            normalIndex2 = [
                j for j in range(3) if j == faceToOrientIndexMapping[nface]
            ][0]

            bigIndex = largeIndexMapping[normalIndex]
            bigIndex2 = largeIndexMapping[normalIndex2]

            if faceOrientations[1] in needToTranspose[bigIndex][bigIndex2]:
                face_x = face_x.T
                face_y = face_y.T
                face_z = face_z.T

            if faceOrientations[0] in [4, 5, 6]:
                face_x = np.flip(face_x, 0)
                face_y = np.flip(face_y, 0)
                face_z = np.flip(face_z, 0)

            if faceOrientations[1] in [4, 5, 6]:
                face_x = np.flip(face_x, 1)
                face_y = np.flip(face_y, 1)
                face_z = np.flip(face_z, 1)

            # Here we translate the coordinates to periodic face
            if bc == "periodicTransLow":
                face_x += face.periodicAxis[0] * face.periodicSpan
                face_y += face.periodicAxis[1] * face.periodicSpan
                face_z += face.periodicAxis[2] * face.periodicSpan
            elif bc == "periodicTransHigh":
                face_x -= face.periodicAxis[0] * face.periodicSpan
                face_y -= face.periodicAxis[1] * face.periodicSpan
                face_z -= face.periodicAxis[2] * face.periodicSpan
            elif bc == "periodicRotLow":
                shape = face_x.shape
                points = np.column_stack(
                    (face_x.ravel(), face_y.ravel(), face_z.ravel())
                )
                points = np.matmul(face.periodicRotMatrixUp, points.T).T
                face_x = points[:, 0].reshape(shape)
                face_y = points[:, 1].reshape(shape)
                face_z = points[:, 2].reshape(shape)
            elif bc == "periodicRotHigh":
                shape = face_x.shape
                points = np.column_stack(
                    (face_x.ravel(), face_y.ravel(), face_z.ravel())
                )
                points = np.matmul(face.periodicRotMatrixDown, points.T).T
                face_x = points[:, 0].reshape(shape)
                face_y = points[:, 1].reshape(shape)
                face_z = points[:, 2].reshape(shape)

            # Compare the face points to the neighbor face
            try:
                off_x = np.max(np.abs(face_x - face2_x))
                off_y = np.max(np.abs(face_y - face2_y))
                off_z = np.max(np.abs(face_z - face2_z))
            except ValueError:
                raise ValueError(
                    f"Error when comparing block {blk.nblki} and block {blk2.nblki} connection"
                )

            if off_x > 1e-10:
                print(
                    f"Warning, the x coordinates of face {nface} on block {blk.nblki} are not matching the x coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {off_x}")
                warn = True

            if off_y > 1e-10:
                print(
                    f"Warning, the y coordinates of face {nface} on block {blk.nblki} are not matching the y coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {off_y}")
                warn = True

            if off_z > 1e-10:
                print(
                    f"Warning, the z coordinates of face {nface} on block {blk.nblki} are not matching the z coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {off_z}")
                warn = True

        # Now we check that all blocks are right handed
        pO = np.array(
            [blk.array["x"][0, 0, 0], blk.array["y"][0, 0, 0], blk.array["z"][0, 0, 0]]
        )
        pI = np.array(
            [blk.array["x"][1, 0, 0], blk.array["y"][1, 0, 0], blk.array["z"][1, 0, 0]]
        )
        pJ = np.array(
            [blk.array["x"][0, 1, 0], blk.array["y"][0, 1, 0], blk.array["z"][0, 1, 0]]
        )
        pK = np.array(
            [blk.array["x"][0, 0, 1], blk.array["y"][0, 0, 1], blk.array["z"][0, 0, 1]]
        )
        vI = pI - pO
        vJ = pJ - pO
        vK = pK - pO

        cross = np.cross(vI, vJ)
        if np.dot(vK, cross) < 0.0:
            print(f"Warning, block {blk.nblki} is left handed. This must be fixed.")
            warn = True

    if warn:
        return False
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify grid is consistent in terms of connectivity and matching face coordinates."
    )
    parser.add_argument(
        "-gpath",
        action="store",
        metavar="<gridPath>",
        dest="gridPath",
        default="./",
        help="Path to grid files",
        type=str,
    )
    parser.add_argument(
        "-connPath",
        action="store",
        metavar="<connPath>",
        dest="connPath",
        default="./",
        help="Path to conn.yaml",
        type=str,
    )
    parser.add_argument(
        "-bcFamPath",
        action="store",
        metavar="<bcFamPath>",
        dest="bcFamPath",
        default="./",
        help="""If your grid has periodics, we need the periodic data from bcFams.""",
        type=str,
    )

    args = parser.parse_args()

    gp = args.gridPath
    cp = args.connPath
    bcFamPath = args.bcFamPath
    nblks = len([i for i in os.listdir(gp) if i.startswith("g.") and i.endswith(".h5")])
    assert nblks > 0
    mb = pg.multiBlock.grid(nblks)

    pg.readers.readGrid(mb, gp)
    pg.readers.readConnectivity(mb, cp)
    try:
        pg.readers.readBcs(mb, bcFamPath)
    except FileNotFoundError:
        print("No bcFam.yaml file provided, assuming no periodics.")

    if verify(mb):
        print("Grid is valid!")
