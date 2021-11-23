#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility converts the output of gridpro into PEREGRINE readable grid and connectivity files.

Inputs are the GridPro **binary** grid file (blk.tmp), GridPro connectivity file (blk.tmp.conn), and
GridPro property file (blk.tmp.pty).

Example
-------
/path/to/pergrinepy/utilities/gp2pg.py <gp_binary_grid_file> <gp_connnectivity_file> <gp_property_file>

Default vaules to read in are the GridPro generic names of blk.tmp, blk.tmp.conn, and blk.tmp.pty

Output will be a PEREGRINE compatible connectivity file 'conn.inp' as well as fortran binary grid files.

"""

import argparse
from peregrinepy.writers import writeGrid, writeConnectivity
import numpy as np
from peregrinepy.multiBlock import grid as mbg
from verifyGrid import verify
import yaml

parser = argparse.ArgumentParser(
    description="Convert binary GridPro files into grid and connectivity files used by PEREGRINE"
)
parser.add_argument(
    "-blk",
    action="store",
    metavar="<GP_blk_FILE>",
    dest="gpBlkFileName",
    default="blk.tmp",
    help="Binary GridPro block node file. Default is blk.tmp",
    type=str,
)
parser.add_argument(
    "-conn",
    action="store",
    metavar="<GP_conn_FILE>",
    dest="gpConnFileName",
    default="blk.tmp.conn",
    help="GridPro block connectivity file. Default is blk.tmp.conn",
    type=str,
)
parser.add_argument(
    "-pty",
    action="store",
    metavar="<GP_pty_FILE>",
    dest="gpPtyFileName",
    default="blk.tmp.pty",
    help="GridPro property file. Default is blk.tmp.pty",
    type=str,
)
parser.add_argument(
    "-bcFam",
    action="store",
    metavar="<bcFam>",
    dest="bcFam",
    default="./bcFams.yaml",
    help="""File to translate the labels given to boundary conditions in ICEM
            to PEREGRINE bcType (i.e. constantVelocitySubsonicInlet, adiabaticNoSlipWall, etc.)\n
            \n
            NOTE: ALL labels from ICEM need an entry in the yaml file. Even walls.""",
    type=str,
)
parser.add_argument(
    "--binary",
    action="store_true",
    dest="isBinary",
    help="Block file in binary format (default is ascii).",
)
args = parser.parse_args()

print("Reading in {0} GridPro files\n".format("binary" if args.isBinary else "ascii"))
print("    {}".format(args.gpBlkFileName))
print("    {}".format(args.gpConnFileName))
print("    {}\n".format(args.gpPtyFileName))


gpConnFile = open(args.gpConnFileName, "r")
gpPtyFile = open(args.gpPtyFileName, "r")
if args.isBinary:
    gpBlkFile = open(args.gpBlkFileName, "rb")
else:
    gpBlkFile = open(args.gpBlkFileName, "r")


# Read in bcFam.yaml file so we know what the bcType is for each label.
with open(args.bcFam, "r") as f:
    bcFamDict = yaml.load(f, Loader=yaml.FullLoader)
gpSurfaceToPgBcType = {
    "pdc:INTERBLK": "b0",
    "pdc:PERIODIC": "b1",
    "pdc:WALL": "adiabaticNoSlipWall",
    "pdc:user8": "adiabaticSlipWall",
}

go = True
while go:
    line = gpPtyFile.readline().split()
    if line[0].startswith("#"):
        pass
    else:
        try:
            nblks = int(line[0])
            print("Found {} blocks".format(nblks))
            go = False
        except ValueError:
            raise TypeError("Unrecognized format for property file.")

presentSurfaces = []
blkPtys = []
for i in range(nblks):
    go = True
    while go:
        line = gpPtyFile.readline().split()
        if line[0].startswith("#"):
            pass
        else:
            go = False

    blkPtys.append([line[4], line[6], line[8], line[10], line[12], line[14]])

    for surf in blkPtys[i]:
        if int(surf) > 0 and surf not in presentSurfaces:
            presentSurfaces.append(surf)

go = True
while go:
    line = gpPtyFile.readline().split()
    if "2D" in line and "properties" in line:
        nprops = int(line[0])
        go = False

surfaceToBc = dict()
for i in range(nprops):
    line = gpPtyFile.readline().replace("(", "").replace(")", "").split()
    if line[0] in presentSurfaces:
        try:
            surfaceToBc[line[0]] = gpSurfaceToPgBcType[line[2]]
        except KeyError:
            surfaceToBc[line[0]] = line[2].strip("pdc:")

pgBlkPtys = []
for i in range(nblks):
    pgBlkPtys.append([0, 0, 0, 0, 0, 0])
    for j in range(6):
        pgBlkPtys[i][j] = surfaceToBc[blkPtys[i][j]]

go = True
while go:
    line = gpConnFile.readline().split()
    if line[0].startswith("#"):
        pass
    else:
        try:
            nblks = int(line[0])
            go = False
        except ValueError:
            raise TypeError("Unrecognized format for connectivity file.")

pgConns = []
for i in range(nblks):
    line = gpConnFile.readline().split()
    for j in range(6):
        line[2 + j * 4] = pgBlkPtys[i][j].split()[0]

    pgConns.append(line[2:-1])

mb = mbg(nblks)
for temp, blk in zip(pgConns, mb):
    for face in blk.faces:
        faceData = temp[(int(face.nface) - 1) * 4 : (int(face.nface) - 1) * 4 + 4]
        if faceData[0] in bcFamDict:
            face.bcFam = f"{faceData[0]}"
            face.bcType = bcFamDict[faceData[0]]["bcType"]
        else:
            face.bcType = f"{faceData[0]}"
            face.bcFam = None
        face.neighbor = None if int(faceData[2]) == 0 else int(faceData[2]) - 1
        face.orientation = None if "0" in faceData[3] else faceData[3]

go = True
while go:
    blockStart = gpBlkFile.tell()
    line = gpBlkFile.readline()
    try:
        test = [int(b) for b in line.strip().split()]
        go = False
    except ValueError:
        go = True

go = True

for blk in mb:

    gpBlkFile.seek(blockStart)
    blk_shape = tuple([int(b) for b in gpBlkFile.readline().strip().split()])

    blk.ni = blk_shape[0]
    blk.nj = blk_shape[1]
    blk.nk = blk_shape[2]

    if args.isBinary:
        byte = gpBlkFile.read(8 * blk_shape[0] * blk_shape[1] * blk_shape[2] * 3)
        temp = np.frombuffer(byte, dtype=np.float64).reshape(
            (blk_shape[0] * blk_shape[1] * blk_shape[2], 3)
        )
        blk.array["x"] = temp[:, 0].reshape(blk_shape)
        blk.array["y"] = temp[:, 1].reshape(blk_shape)
        blk.array["z"] = temp[:, 2].reshape(blk_shape)

        gpBlkFile.read(1)
    else:
        blk.array["x"] = np.empty(blk_shape, dtype=np.float64)
        blk.array["y"] = np.empty(blk_shape, dtype=np.float64)
        blk.array["z"] = np.empty(blk_shape, dtype=np.float64)

        for i in range(blk_shape[0]):
            for j in range(blk_shape[1]):
                for k in range(blk_shape[2]):
                    line = gpBlkFile.readline().strip().split()
                    blk.array["x"][i, j, k] = float(line[0])
                    blk.array["y"][i, j, k] = float(line[1])
                    blk.array["z"][i, j, k] = float(line[2])

    blockStart = gpBlkFile.tell()

gpConnFile.close()
gpPtyFile.close()
gpBlkFile.close()

if verify(mb):
    pass

print("Writing out PEREGRINE connectivity file: conn.yaml...")
writeConnectivity(mb, "./")

print("Writing out {} block PEREGRINE grid files".format(len(mb)))
writeGrid(mb, "./")

print("GridPro to PEREGRINE translation done.")
