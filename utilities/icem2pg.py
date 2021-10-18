#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This utility converts the output of ICEM MultiBlock-Info into PEREGRINE grid and connectivity files.

Inputs are the ICEM info connectivity file (info.topo), and ICEM property file (blk.tmp.pty).

The ICEM **ascii** info.* grid file (i.e. info.dom*) must be present in same directory

Example
-------
icem2raptor.py -topo <info.topo>

Output will be a PEREGRINE compatible connectivity file 'conn.yaml' as well as hdf5 binary grid files.

"""

# For info on the info.topo file format see
# https://support.ansys.com/staticassets/ANSYS/Initial%20Content%20Entry/General%20Articles%20-%20Products/ICEM%20CFD%20Interfaces/multiBlock.htm


import argparse
import os
import peregrinepy as pg
import yaml

from verifyGrid import verify
import numpy as np
from scipy.io import FortranFile

parser = argparse.ArgumentParser(
    description="Convert ICEM Multioblock-Info files into grid and connectivity files used by PEREGRINE"
)
parser.add_argument(
    "-topo",
    action="store",
    metavar="<topoFile>",
    dest="topoFileName",
    default="info.topo",
    help="topology file from MultiBlock-Info export from ICEM. Default is info.topo",
    type=str,
)
parser.add_argument(
    "-fmt",
    action="store",
    metavar="<fileFormat>",
    dest="fmt",
    default="tns3dmb",
    help="""Format used to export from ICEM **PAY ATTENTION TO THIS**. You must always export your ICEM grid in Multi-Block Info format to get the topo file. \nBut the
                    Multi-Block Info node files are limited to single precision. This may not be desirable. So the best practice is to use TLNS3D-MB format in double precision (see web docs), this is the default assumption.\n
                    Options:\n
                            tns3dmb  <Default>\n
                            mbi      Multi-Block Info""",
    type=str,
)
parser.add_argument(
    "-bcFam",
    action="store",
    metavar="<bcFam>",
    dest="bcFam",
    default="./bcFam.yaml",
    help="""File to translate the labels given to boundary conditions in ICEM
            to PEREGRINE bcType (i.e. constantVelocitySubsonicInlet, adiabaticNoSlipWall, etc.)\n
            \n
            NOTE: ALL labels from ICEM need an entry in the yaml file. Even walls.""",
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
    fileName = "tns3dmb.dat"
    with FortranFile(fileName, "r") as f90:

        nblks = f90.read_ints(dtype=np.int32)[0]
        mb = pg.multiBlock.grid(nblks)

        nxyzs = np.array_split(f90.read_ints(dtype=np.int32), nblks)

        for nxyz, blk in zip(nxyzs, mb):
            nx, ny, nz = nxyz
            temp = f90.read_reals(dtype=np.float64).reshape((nx, ny, nz, 3), order='F')
            blk.array["x"] = temp[:, :, :, 0]
            blk.array["y"] = temp[:, :, :, 1]
            blk.array["z"] = temp[:, :, :, 2]

##########################
# MULTIBLOCK-INFO FORMAT #
##########################
elif args.fmt == "mbi":
    nblks = len([f for f in os.listdir() if f.startswith("info.dom")])
    print(f"Reading in {nblks} ICEM domain files")
    print("    {}".format(args.topoFileName))
    mb = pg.multiBlock.grid(nblks)

    for blk in mb:
        fileName = "info.dom{}".format(blk.nblki)
        with open(fileName, "r") as f:
            line = f.readline().strip().split()
            nx = int(line[1])
            ny = int(line[2])
            nz = int(line[3])
        points = np.genfromtxt(fileName, comments="domain.")

        blk.array["x"] = np.reshape(points[:, 0], (nx, ny, nz))
        blk.array["y"] = np.reshape(points[:, 1], (nx, ny, nz))
        blk.array["z"] = np.reshape(points[:, 2], (nx, ny, nz))

else:
    raise ValueError("Unknown file format given, see help menu")

# Set all bc types to internal... we will set the external bc's later
for blk in mb:
    for face in blk.faces:
        blk.bcType = "b0"

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
validBcTypes = (
    # Inlets
    "constantVelocitySubsonicInlet",
    # Exits
    "constantPressureSubsonicExit",
    # Walls
    "adiabaticNoSlipWall",
    "adiabaticSlipWall",
    "adiabaticMovingWall",
    "isoTMovingWall",
)
# Read in bcFam.yaml file so we know what the bcType is for each label.
with open(args.bcFam, "r") as f:
    bcFam2Type = yaml.load(f, Loader=yaml.FullLoader)


# Set boundary conditions
readingBlock = False
with open(args.topoFileName, "r") as f:
    iterLines = iter(f.readlines())
    for rawLine in iterLines:
        # Are we starting a domain boundary condition section?
        if rawLine.startswith("# Boundary conditions and/or"):
            currentBlock = int(rawLine.strip().split(".")[-1]) - 1
            blk = mb.getBlock(currentBlock)
            readingBlock = True
            continue
        elif rawLine == "\n":
            blk = None
            readingBlock = False
            pass

        if readingBlock:
            tag = rawLine[
                0:2
            ]
            line = rawLine.strip().split()
            faceBlockEdgeVertex = line[2]  # f= face, b=block, e=edge, v=vertex
            tagNum = line[1]
            # Is this line an external face (not edge, vertex, etc.)
            # Is this face tagged with a PEREGRINE BC tag?
            if faceBlockEdgeVertex == "f":
                mins = [line[3], line[4], line[5]]
                maxs = [line[6], line[7], line[8]]
                directions = ["i", "j", "k"]
                for mini, maxi, direc in zip(mins, maxs, directions):
                    if mini == maxi and mini == "1":
                        face = face_mapping[f"small_{direc}"]
                    elif mini == maxi and mini != "1":
                        face = face_mapping[f"large_{direc}"]
                    bcFam = f"{tag.lower()}{tagNum}".replace(" ", "")
                    blk.getFace(face).bcFam = bcFam
                    try:
                        bcType = bcFam2Type[bcFam]
                        if bcType not in validBcTypes:
                            raise ValueError(f"{bcType} is not a valid PEREGRINE bcType")
                        blk.getFace(face).bcType = bcType
                    except KeyError:
                        raise KeyError(f"No entry for bcFam {bcFam} in {args.bcFam} file.")

# ----------------------------------------------------------------- #
# ------------------ Periodic Face Connectivity ------------------- #
# ----------------------------------------------------------------- #
with open(args.topoFileName, "r") as f:
    iterLines = iter(f.readlines())
    for rawLine in iterLines:
        # Are we starting a periodic info section?
        if rawLine.startswith("# Periodic info for domain"):
            currentBlock = int(rawLine.strip().split(".")[-1]) - 1
            blk = mb.getBlock(currentBlock)

            # Periodic info always a single pair format
            thisBlockLine = next(iterLines).replace("-", " -").strip().split()
            if thisBlockLine == []:
                continue
            adjacentBlockLine = next(iterLines).replace("-", " -").strip().split()
            adjacentBlockNumber = int(adjacentBlockLine[1].split(".")[-1]) - 1
            # Seems like periodics do not get set to Part Name in info.topo, so we must rely
            # on only what is in the Periodic info section for block #'s, face #'s and orientation
            mins = [thisBlockLine[6], thisBlockLine[7], thisBlockLine[8]]
            maxs = [thisBlockLine[9], thisBlockLine[10], thisBlockLine[11]]
            directions = [i.replace("-", "") for i in thisBlockLine[2:5]]
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
            faceBlockEdgeVertex = thisBlockLine[5]  # f= face, b=block, e=edge, v=vertex
            if faceBlockEdgeVertex != "f":
                raise ValueError(
                    f"ERROR: we are expecting a face to be here to set face {face} of block{currentBlock}, but instead we are seeing a {faceBlockEdgeVertex}."
                )

            # It's clear something is different about the periodic orientations... this is my best guess as to how to treat them...
            if thisBlockLine[2:5] == adjacentBlockLine[2:5]:
                orientation = {"i": "i", "j": "j", "k": "k"}
            else:
                orientation = dict()
                for i in range(3):
                    curr = thisBlockLine[2 + i]
                    adjc = adjacentBlockLine[2 + i]

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

            blk.getFace(face).neighbor = str(adjacentBlockNumber)
            blk.getFace(face).bcType = "b1"
            i_orient_num = orientation_mapping[orientation["i"]]
            j_orient_num = orientation_mapping[orientation["j"]]
            k_orient_num = orientation_mapping[orientation["k"]]

            pg_orientation = f"{i_orient_num}{j_orient_num}{k_orient_num}"
            blk.getFace(face).orientation = pg_orientation
            # If a block is periodic with itself, we must also set the opposite face as it will not be referenced again later
            if currentBlock == adjacentBlockNumber:
                blk.getFace(opp_face).neighbor = str(currentBlock)
                blk.getFace(opp_face).bc = "b1"
                blk.getFace(opp_face).orientation = pg_orientation

# ----------------------------------------------------------------- #
# ---------------- Internal Face Connectivity --------------------- #
# ----------------------------------------------------------------- #
# Update connectivity
with open(args.topoFileName, "r") as f:
    iterLines = iter(f.readlines())
    for rawLine in iterLines:
        # Are we starting a domain connectivity section?
        if rawLine.startswith("# Connectivity for domain"):
            currentBlock = int(rawLine.strip().split(".")[-1]) - 1
            blk = mb.getBlock(currentBlock)
            # Collect the faces that were not set as external BCs and need connectivity info
            internalFaces = []
            for face in blk.faces:
                if face.bcType == "b0":
                    internalFaces.append(face.nface)
            # March through the required faces
            for internalFace in internalFaces:
                thisBlockLine = next(iterLines).replace("-", " -").strip().split()
                adjacentBlockLine = (
                    next(iterLines).replace("-", " -").strip().split()
                )

                # As a check, lets make sure that the face we are setting is the same as this section ICEM dictates
                # Looking at the connectivity section in the info.topo file, it seems like these face connectivities
                # go in order of internal faces that need to be set, i.e. [1,2,3,4,5,6]
                # Thats what we can are just cycling through the internal faces left in this block and trusting that
                # This order will hold up.
                mins = [thisBlockLine[6], thisBlockLine[7], thisBlockLine[8]]
                maxs = [thisBlockLine[9], thisBlockLine[10], thisBlockLine[11]]
                directions = [i.replace("-", "") for i in thisBlockLine[2:5]]
                for mini, maxi, direc in zip(mins, maxs, directions):
                    if mini == maxi and mini == "1":
                        face = face_mapping[f"small_{direc}"]
                        if str(face) != internalFace:
                            raise ValueError(
                                f"Error, the order of the info.topo connectivity of block {currentBlock} is giving face {face} when we are expecting face {internalFace}"
                            )
                    elif mini == maxi and mini != "1":
                        face = face_mapping[f"large_{direc}"]
                        if str(face) != internalFace:
                            raise ValueError(
                                f"Error, the order of the info.topo connectivity of block {currentBlock} is giving face {face} when we are expecting face {internalFace}"
                            )

                # We will also double check that this is in fact a face connectivity, not a edge or vertex connectivity
                # b/c again, it seems like these files always list the faces first, then edges, then verticies.
                faceBlockEdgeVertex = thisBlockLine[5]  # f= face, b=block, e=edge, v=vertex
                if faceBlockEdgeVertex != "f":
                    raise ValueError(
                        f"ERROR: we are expecting a face to be here to set face {face} of block{currentBlock}, but instead we are seeing a {faceBlockEdgeVertex}."
                    )

                # With those checks done, lets actually set the connectivity, orientation, etc.
                adjacentBlockNumber = int(
                    adjacentBlockLine[1].strip().split(".")[-1]
                ) - 1

                # I don't know why, but the ICEM orientation is really, weird, so lets create a dict whose keys are
                # the i,j,k of the current block, and the values are the +/- i,j,k of the adjacent block
                orientation = dict()
                for i in range(3):
                    curr = thisBlockLine[2 + i]
                    adjc = adjacentBlockLine[2 + i]

                    # Check if the actual +/- i,j,k of the current block is the joining face between the adjacent block.
                    # If it is, then we honor the +/- of the adjacent block orientation.
                    # If not, we do not honor the +/- of the adjacent orientation and use the other.
                    face_num = (
                        f"small_{curr[-1]}"
                        if curr.startswith("-")
                        else f"large_{curr[-1]}"
                    )
                    if face_mapping[face_num] == int(internalFace):
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
                blk.getFaces(internalFace).neighbor = adjacentBlockNumber

                i_orient_num = orientation_mapping[orientation["i"]]
                j_orient_num = orientation_mapping[orientation["j"]]
                k_orient_num = orientation_mapping[orientation["k"]]

                pg_orientation = f"{i_orient_num}{j_orient_num}{k_orient_num}"
                blk.getFaces(internalFace).orientation = pg_orientation

if verify(mb):
    pass

print("Writing out PEREGRINE connectivity file: conn.inp...")
pg.writers.writeConnectivity(mb)

print("Writing out {} block PEREGRINE grid files".format(mb.nblks))
pg.writers.writeGrid(mb)


print("ICEM to PEREGRINE translation done.")
