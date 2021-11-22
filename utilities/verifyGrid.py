#!/usr/bin/env python3

"""This utility goes through a grid face by face, verifying that all the block's connectivities agree,
and that the coordinates of matching faces are identical.

Inputs are the path to the grid files, and path to the conn.inp file.

It can handle b 1 periodicity in the conn.inp, however it will not compare the x,y,z coordinate locations
of the faces.

Output will print any discrepencies to the screen

"""

import os
import argparse
import peregrinepy as pg
import numpy as np


def verify(mb):

    faceSliceMapping = {
        1: {"i": 0, "j": slice(None), "k": slice(None)},
        2: {"i": -1, "j": slice(None), "k": slice(None)},
        3: {"i": slice(None), "j": 0, "k": slice(None)},
        4: {"i": slice(None), "j": -1, "k": slice(None)},
        5: {"i": slice(None), "j": slice(None), "k": 0},
        6: {"i": slice(None), "j": slice(None), "k": -1},
    }

    faceToOrientIndexMapping = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
    }
    orientToSmallFaceMapping = {
        1: 2,
        2: 4,
        3: 6,
        4: 1,
        5: 3,
        6: 5,
    }
    orientToLargeFaceMapping = {
        1: 1,
        2: 3,
        3: 5,
        4: 2,
        5: 4,
        6: 6,
    }

    largeIndexMapping = {0: "k", 1: "k", 2: "j"}
    needToTranspose = {
        "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
        "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
    }

    def extractFace(blk, nface):

        face_i = faceSliceMapping[nface]

        x = blk.array["x"][face_i["i"], face_i["j"], face_i["k"]]
        y = blk.array["y"][face_i["i"], face_i["j"], face_i["k"]]
        z = blk.array["z"][face_i["i"], face_i["j"], face_i["k"]]

        return x, y, z

    warn = False
    for blk in mb:
        for face in blk.faces:

            nface = face.nface
            neighbor = face.neighbor
            bc = face.bcType

            if neighbor is None:
                continue
            if bc == "b1":
                periodic = True
            else:
                periodic = False

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

            try:
                diff_x = np.mean(np.abs(face_x - face2_x))
                diff_y = np.mean(np.abs(face_y - face2_y))
                diff_z = np.mean(np.abs(face_z - face2_z))
            except ValueError:
                raise ValueError(
                    f"Error when comparing block {blk.nblki} and block {blk2.nblki} connection"
                )

            if diff_x > 1e-08 and not periodic:
                print(
                    f"Warning, the x coordinates of face {nface} on block {blk.nblki} are not matching the x coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {diff_x}")
                warn = True

            if diff_y > 1e-08 and not periodic:
                print(
                    f"Warning, the y coordinates of face {nface} on block {blk.nblki} are not matching the y coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {diff_y}")
                warn = True

            if diff_z > 1e-08 and not periodic:
                print(
                    f"Warning, the z coordinates of face {nface} on block {blk.nblki} are not matching the z coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {diff_z}")
                warn = True

    if not warn:
        return True
    else:
        return False


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

    args = parser.parse_args()

    gp = args.gridPath
    cp = args.connPath
    nblks = len(
        [i for i in os.listdir(gp) if i.startswith("gv.") and i.endswith(".h5")]
    )
    assert nblks > 0
    mb = pg.multiBlock.grid(nblks)

    pg.readers.readGrid(mb, gp)
    pg.readers.readConnectivity(mb, cp)

    if verify(mb):
        print("Grid is valid!")
