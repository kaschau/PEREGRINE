#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This utility converts the output of ICEM Multiblock-Info into PEREGRINE grid and connectivity files.

Inputs are the ICEM info connectivity file (info.topo), and ICEM property file (blk.tmp.pty).

The ICEM **ascii** info.* grid file (i.e. info.dom*) must be present in same directory

Example
-------
icem2raptor.py -topo <info.topo>

Output will be a PEREGRINE compatible connectivity file 'conn.yaml' as well as hdf5 binary grid files.

"""

# For info on the info.topo file format see
# https://support.ansys.com/staticassets/ANSYS/Initial%20Content%20Entry/General%20Articles%20-%20Products/ICEM%20CFD%20Interfaces/multiblock.htm


import argparse
import os
import peregrinepy as pg

# from verify_grid import verify
import numpy as np
from scipy.io import FortranFile

parser = argparse.ArgumentParser(
    description="Convert ICEM Multioblock-Info files into grid and connectivity files used by PEREGRINE"
)
parser.add_argument(
    "-topo",
    action="store",
    metavar="<topo_FILE>",
    dest="topo_file_name",
    default="info.topo",
    help="topology file from Multiblock-Info export from ICEM. Default is info.topo",
    type=str,
)
parser.add_argument(
    "-fmt",
    action="store",
    metavar="<file_format>",
    dest="fmt",
    default="tns3dmb",
    help="""Format used to export from ICEM **PAY ATTENTION TO THIS**. You must always export your ICEM grid in Multi-Block Info format to get the topo file. \nBut the
                    Multi-Block Info node files are limited to single precision. This may not be desirable. So the best practice is to use TLNS3D-MB format in double precision (see web docs), this is the default assumption.\n
                    Options:\n
                            tns3dmb  <Default>\n
                            mbi      Multi-Block Info""",
    type=str,
)

args = parser.parse_args()

# ----------------------------------------------------------------- #
# ------------------------- Coordinates --------------------------- #
# ----------------------------------------------------------------- #

##################
# TNS3DMB FORMAT #
##################

if args.fmt == "tns3dmb":
    file_name = "tns3dmb.dat"
    with FortranFile(file_name, "r") as f90:

        nblks = f90.read_ints(dtype=np.int32)[0]
        mb = pg.multiblock.grid(nblks)

        nxyzs = np.array_split(f90.read_ints(dtype=np.int32), nblks)

        for nxyz, blk in zip(nxyzs, mb):
            nx, ny, nz = nxyz
            temp = f90.read_reals(dtype=np.float64).reshape(3, nz, ny, nx)
            blk.x = temp[0]
            blk.y = temp[1]
            blk.z = temp[2]

##########################
# MULTIBLOCK-INFO FORMAT #
##########################
elif args.fmt == "mbi":
    nblks = len([f for f in os.listdir() if f.startswith("info.dom")])
    print(f"Reading in {nblks} ICEM domain files")
    print("    {}".format(args.topo_file_name))
    mb = pg.multiblock.grid(nblks)

    for blk in mb:
        file_name = "info.dom{}".format(blk.nblki)
        with open(file_name, "r") as f:
            line = f.readline().strip().split()
            nx = int(line[1])
            ny = int(line[2])
            nz = int(line[3])
        points = np.genfromtxt(file_name, comments="domain.")

        blk.x = np.reshape(points[:, 0], (nx, ny, nz))
        blk.y = np.reshape(points[:, 1], (nx, ny, nz))
        blk.z = np.reshape(points[:, 2], (nx, ny, nz))

else:
    raise ValueError("Unknown file format given, see help menu")

for blk in mb:
    # Set all bc types to internal... we will set the external bc's later
    for i in range(6):
        blk.connectivity["{}".format(i + 1)]["bc"] = "b0"

face_mapping = {
    "small_i": 1,
    "large_i": 2,
    "small_j": 3,
    "large_j": 4,
    "small_k": 5,
    "large_k": 6,
}
orientation_mapping = {"i": 1, "j": 2, "k": 3, "-i": 4, "-j": 5, "-k": 6}

# ----------------------------------------------------------------- #
# ------------- External Face Boundary Conditions ----------------- #
# ----------------------------------------------------------------- #

# Set boundary conditions
reading_block = False
with open(args.topo_file_name, "r") as f:
    iter_lines = iter(f.readlines())
    for raw_line in iter_lines:
        # Are we starting a domain boundary condition section?
        if raw_line.startswith("# Boundary conditions and/or"):
            current_block = int(raw_line.strip().split(".")[-1])
            blk = mb[current_block - 1]
            reading_block = True
            continue
        elif raw_line == "\n":
            blk = None
            reading_block = False
            pass

        if reading_block:
            tag = raw_line[
                0:2
            ]  # Only 'I ', 'E ', and 'S ' will be picked up as RAPTOR tags
            line = raw_line.strip().split()
            bc_type = line[2]  # f= face, b=block, e=edge, v=vertex
            tag_num = line[1]
            # Is this line an external face (not edge, vertex, etc.)
            # Is this face tagged with a RAPTOR BC tag?
            if bc_type == "f" and tag in ["I ", "E ", "S "]:
                mins = [line[3], line[4], line[5]]
                maxs = [line[6], line[7], line[8]]
                directions = ["i", "j", "k"]
                for mini, maxi, direc in zip(mins, maxs, directions):
                    if mini == maxi and mini == "1":
                        face = face_mapping[f"small_{direc}"]
                        rp_bc = f"{tag.lower()}{tag_num}".replace(" ", "")
                        blk.connectivity[str(face)]["bc"] = rp_bc
                    elif mini == maxi and mini != "1":
                        face = face_mapping[f"large_{direc}"]
                        rp_bc = f"{tag.lower()}{tag_num}".replace(" ", "")
                        blk.connectivity[str(face)]["bc"] = rp_bc

# ----------------------------------------------------------------- #
# ------------------ Periodic Face Connectivity ------------------- #
# ----------------------------------------------------------------- #
with open(args.topo_file_name, "r") as f:
    iter_lines = iter(f.readlines())
    for raw_line in iter_lines:
        # Are we starting a periodic info section?
        if raw_line.startswith("# Periodic info for domain"):
            current_block = int(raw_line.strip().split(".")[-1])
            blk = mb[current_block - 1]

            # Periodic info always a single pair format
            this_block_line = next(iter_lines).replace("-", " -").strip().split()
            if this_block_line == []:
                continue
            adjacent_block_line = next(iter_lines).replace("-", " -").strip().split()
            adjacent_block_number = int(adjacent_block_line[1].split(".")[-1])
            # Seems like periodics do not get set to Part Name in info.topo, so we must rely
            # on only what is in the Periodic info section for block #'s, face #'s and orientation
            mins = [this_block_line[6], this_block_line[7], this_block_line[8]]
            maxs = [this_block_line[9], this_block_line[10], this_block_line[11]]
            directions = [i.replace("-", "") for i in this_block_line[2:5]]
            for mini, maxi, direc in zip(mins, maxs, directions):
                if mini == maxi and mini == "1":
                    face = str(face_mapping[f"small_{direc}"])
                    opp_face = str(face_mapping[f"large_{direc}"])
                    break
                elif mini == maxi and mini != "1":
                    face = str(face_mapping[f"large_{direc}"])
                    opp_face = str(face_mapping[f"small_{direc}"])
                    break

            # We will also double check that this is in fact a face connectivity, not a edge or vertex connectivity
            # b/c again, it seems like these files always list the faces first, then edges, then verticies.
            bc_type = this_block_line[5]  # f= face, b=block, e=edge, v=vertex
            if bc_type != "f":
                raise ValueError(
                    f"ERROR: we are expecting a face to be here to set face {face} of block{current_block}, but instead we are seeing a {bc_type}."
                )

            # It's clear something is different about the periodic orientations... this is my best guess as to how to treat them...
            if this_block_line[2:5] == adjacent_block_line[2:5]:
                orientation = {"i": "i", "j": "j", "k": "k"}
            else:
                orientation = dict()
                for i in range(3):
                    curr = this_block_line[2 + i]
                    adjc = adjacent_block_line[2 + i]

                    # Check if the actual +/- i,j,k of the current block is the joining face between the adjacent block.
                    # If it is, then we honor the +/- of the adjacent block orientation.
                    # If not, we do not honor the +/- of the adjacent orientation and use the other.
                    face_num = (
                        f"small_{curr[-1]}"
                        if curr.startswith("-")
                        else f"large_{curr[-1]}"
                    )
                    if face_mapping[face_num] == int(face):
                        if curr.startswith("-"):
                            pass
                        else:
                            if adjc.startswith("-"):
                                adjc = adjc.replace("-", "")
                            else:
                                adjc = f"-{adjc}"

                    elif curr.startswith("-"):
                        if adjc.startswith("-"):
                            adjc = adjc.replace("-", "")
                        else:
                            adjc = f"-{adjc}"

                    if curr.startswith("-"):
                        curr = curr.replace("-", "")

                    orientation[curr] = adjc

            blk.faces[face].connectivity["neighbor"] = str(adjacent_block_number)
            blk.faces[face].connectivity["bc"] = "b1"
            i_orient_num = orientation_mapping[orientation["i"]]
            j_orient_num = orientation_mapping[orientation["j"]]
            k_orient_num = orientation_mapping[orientation["k"]]

            pg_orientation = f"{i_orient_num}{j_orient_num}{k_orient_num}"
            blk.face[face].connectivity["orientation"] = pg_orientation
            # If a block is periodic with itself, we must also set the opposite face as it will not be referenced again later
            if current_block == adjacent_block_number:
                blk.face[opp_face].connectivity["neighbor"] = str(current_block)
                blk.face[opp_face].connectivity["bc"] = "b1"
                blk.face[opp_face].connectivity["orientation"] = pg_orientation

# ----------------------------------------------------------------- #
# ---------------- Internal Face Connectivity --------------------- #
# ----------------------------------------------------------------- #
# Update connectivity
with open(args.topo_file_name, "r") as f:
    iter_lines = iter(f.readlines())
    for raw_line in iter_lines:
        # Are we starting a domain connectivity section?
        if raw_line.startswith("# Connectivity for domain"):
            current_block = int(raw_line.strip().split(".")[-1])
            blk = mb[current_block - 1]
            # Collect the faces that were not set as external BCs and need connectivity info
            internal_faces = []
            for i in range(6):
                if blk.connectivity[str(i + 1)]["bc"].replace(" ", "") == "b0":
                    internal_faces.append(str(i + 1))
            # March through the required faces
            for internal_face in internal_faces:
                this_block_line = next(iter_lines).replace("-", " -").strip().split()
                adjacent_block_line = (
                    next(iter_lines).replace("-", " -").strip().split()
                )

                # As a check, lets make sure that the face we are setting is the same as this section ICEM dictates
                # Looking at the connectivity section in the info.topo file, it seems like these face connectivities
                # go in order of internal faces that need to be set, i.e. [1,2,3,4,5,6]
                # Thats what we can are just cycling through the internal faces left in this block and trusting that
                # This order will hold up.
                mins = [this_block_line[6], this_block_line[7], this_block_line[8]]
                maxs = [this_block_line[9], this_block_line[10], this_block_line[11]]
                directions = [i.replace("-", "") for i in this_block_line[2:5]]
                for mini, maxi, direc in zip(mins, maxs, directions):
                    if mini == maxi and mini == "1":
                        face = face_mapping[f"small_{direc}"]
                        if str(face) != internal_face:
                            raise ValueError(
                                f"Error, the order of the info.topo connectivity of block {current_block} is giving face {face} when we are expecting face {internal_face}"
                            )
                    elif mini == maxi and mini != "1":
                        face = face_mapping[f"large_{direc}"]
                        if str(face) != internal_face:
                            raise ValueError(
                                f"Error, the order of the info.topo connectivity of block {current_block} is giving face {face} when we are expecting face {internal_face}"
                            )

                # We will also double check that this is in fact a face connectivity, not a edge or vertex connectivity
                # b/c again, it seems like these files always list the faces first, then edges, then verticies.
                bc_type = this_block_line[5]  # f= face, b=block, e=edge, v=vertex
                if bc_type != "f":
                    raise ValueError(
                        f"ERROR: we are expecting a face to be here to set face {face} of block{current_block}, but instead we are seeing a {bc_type}."
                    )

                # With those checks done, lets actually set the connectivity, orientation, etc.
                adjacent_block_number = int(
                    adjacent_block_line[1].strip().split(".")[-1]
                )

                # I don't know why, but the ICEM orientation is really, weird, so lets create a dict whose keys are
                # the i,j,k of the current block, and the values are the +/- i,j,k of the adjacent block
                orientation = dict()
                for i in range(3):
                    curr = this_block_line[2 + i]
                    adjc = adjacent_block_line[2 + i]

                    # Check if the actual +/- i,j,k of the current block is the joining face between the adjacent block.
                    # If it is, then we honor the +/- of the adjacent block orientation.
                    # If not, we do not honor the +/- of the adjacent orientation and use the other.
                    face_num = (
                        f"small_{curr[-1]}"
                        if curr.startswith("-")
                        else f"large_{curr[-1]}"
                    )
                    if face_mapping[face_num] == int(internal_face):
                        if curr.startswith("-"):
                            pass
                        else:
                            if adjc.startswith("-"):
                                adjc = adjc.replace("-", "")
                            else:
                                adjc = f"-{adjc}"
                    elif curr.startswith("-"):
                        if adjc.startswith("-"):
                            adjc = adjc.replace("-", "")
                        else:
                            adjc = f"-{adjc}"

                    if curr.startswith("-"):
                        curr = curr.replace("-", "")

                    orientation[curr] = adjc

                # Set the values
                blk.faces[internal_face].connectivity["neighbor"] = str(
                    adjacent_block_number
                )

                i_orient_num = orientation_mapping[orientation["i"]]
                j_orient_num = orientation_mapping[orientation["j"]]
                k_orient_num = orientation_mapping[orientation["k"]]

                pg_orientation = f"{i_orient_num}{j_orient_num}{k_orient_num}"
                blk.faces[internal_face].connectivity["orientation"] = pg_orientation

# if verify(mb):
#     pass

print("Writing out PEREGRINE connectivity file: conn.inp...")
pg.writers.write_connectivity(mb, "./")

print("Writing out {} block RAPTOR grid files".format(mb.nblks))
pg.writers.write_grid(mb, "./")


print("ICEM to PEREGRINE translation done.")
