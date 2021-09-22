#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This utility goes through a grid face by face, verifying that all the block's connectivities agree,
and that the coordinates of matching faces are identical.

Inputs are the path to the grid files, and path to the conn.inp file.

It can handle b 1 periodicity in the conn.inp, however it will now compare the x,y,z coordinate locations
of the faces.

Example
-------
/path/to/raptorpy/utilities/verify_grid.py -gpath './' -cpath './'

Output will print any errors to the screen

"""

import os
import argparse
from peregrinepy.readers import read_grid, read_connectivity
from peregrinepy.multiblock import grid as mbg
import numpy as np


def verify(mb_data):

    face_slice_mapping = {
        "1": {"i": 0, "j": slice(None), "k": slice(None)},
        "2": {"i": -1, "j": slice(None), "k": slice(None)},
        "3": {"i": slice(None), "j": 0, "k": slice(None)},
        "4": {"i": slice(None), "j": -1, "k": slice(None)},
        "5": {"i": slice(None), "j": slice(None), "k": 0},
        "6": {"i": slice(None), "j": slice(None), "k": -1},
    }

    face_to_orient_place_mapping = {
        "1": "0",
        "2": "0",
        "3": "1",
        "4": "1",
        "5": "2",
        "6": "2",
    }
    orient_to_small_face_mapping = {
        "1": "2",
        "2": "4",
        "3": "6",
        "4": "1",
        "5": "3",
        "6": "5",
    }
    orient_to_large_face_mapping = {
        "1": "1",
        "2": "3",
        "3": "5",
        "4": "2",
        "5": "4",
        "6": "6",
    }

    large_index_mapping = {0: "k", 1: "k", 2: "j"}
    need_to_transpose = {
        "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
        "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
    }

    def extract_face(blk, nface):

        face_i = face_slice_mapping[nface]

        x = blk.array["x"][face_i["i"], face_i["j"], face_i["k"]]
        y = blk.array["y"][face_i["i"], face_i["j"], face_i["k"]]
        z = blk.array["z"][face_i["i"], face_i["j"], face_i["k"]]

        return x, y, z

    def get_neighbor_face(nface, orientation, blk2):

        direction = orientation[int(face_to_orient_place_mapping[nface])]

        if nface in ["2", "4", "6"]:
            nface2 = orient_to_large_face_mapping[direction]
        elif nface in ["1", "3", "5"]:
            nface2 = orient_to_small_face_mapping[direction]

        return nface2

    warn = False
    for blk in mb_data:
        for face in blk.faces:

            nface = face.nface
            nneighbor = face.connectivity["neighbor"]
            orientation = face.connectivity["orientation"]
            bc = face.connectivity["bctype"]

            if nneighbor == 0:
                continue
            if bc == "b 1":
                periodic = True
            else:
                periodic = False

            (face_x, face_y, face_z) = extract_face(blk, face.nface)

            blk2 = mb_data[nneighbor - 1]
            nface2 = get_neighbor_face(nface, orientation, blk2)

            if int(blk2.connectivity[nface2]["connection"]) != blk.nblki:
                raise ValueError(
                    f"Block {blk.nblki}'s' face {nface} says it is connected to\nblock {blk2.nblki}'s' face {nface2}, however block {blk2.nblki}'s\nface {nface2} says it is connected to a different block."
                )

            (face2_x, face2_y, face2_z) = extract_face(blk2, nface2)
            orientation2 = blk2.connectivity[nface2]["orientation"]

            face_orientations = [
                i
                for j, i in enumerate(orientation)
                if j != int(face_to_orient_place_mapping[nface])
            ]
            normal_index = [
                j for j in range(3) if j == int(face_to_orient_place_mapping[nface])
            ][0]
            face_orientations2 = [
                i
                for j, i in enumerate(orientation2)
                if j != int(face_to_orient_place_mapping[nface2])
            ]
            normal_index2 = [
                j for j in range(3) if j == int(face_to_orient_place_mapping[nface2])
            ][0]

            big_index = large_index_mapping[normal_index]
            big_index2 = large_index_mapping[normal_index2]

            if int(face_orientations[1]) in need_to_transpose[big_index][big_index2]:
                face2_x = face2_x.T
                face2_y = face2_y.T
                face2_z = face2_z.T

            if face_orientations[1] in ["4", "5", "6"]:
                face2_x = np.flip(face2_x, 0)
                face2_y = np.flip(face2_y, 0)
                face2_z = np.flip(face2_z, 0)

            if face_orientations[0] in ["4", "5", "6"]:
                face2_x = np.flip(face2_x, 1)
                face2_y = np.flip(face2_y, 1)
                face2_z = np.flip(face2_z, 1)

            try:
                diff_x = np.mean(np.abs(face_x - face2_x))
                diff_y = np.mean(np.abs(face_y - face2_y))
                diff_z = np.mean(np.abs(face_z - face2_z))
            except:
                raise ValueError(
                    f"Error when comparing block {blk.nblki} and block {blk2.nblki} connection"
                )

            if diff_x > 1e-14 and not periodic:
                print(
                    f"Warning, the x coordinates of face {nface} on block {blk.nblki} are not matching the x coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {diff_x}")
                warn = True

            if diff_y > 1e-14 and not periodic:
                print(
                    f"Warning, the y coordinates of face {nface} on block {blk.nblki} are not matching the y coordinates of face {nface2} of block {blk2.nblki}"
                )
                print(f"Off by average of {diff_y}")
                warn = True

            if diff_z > 1e-14 and not periodic:
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
        metavar="<grid_path>",
        dest="grid_path",
        default="./",
        help="Path to grid files",
        type=str,
    )
    parser.add_argument(
        "-cpath",
        action="store",
        metavar="<conn_path>",
        dest="conn_path",
        default="./",
        help="Path to conn.inp file",
        type=str,
    )

    args = parser.parse_args()

    gp = args.grid_path
    cp = args.conn_path
    nblks = len([i for i in os.listdir(gp) if i.startswith("g.")])
    mb = mbg(nblks)

    read_grid(mb, gp)
    read_connectivity(mb, cp + "/conn.inp")

    if verify(mb):
        print("Grid is valid!")
