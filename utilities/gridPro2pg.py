#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility converts the output of gridpro into PEREGRINE readable grid and connectivity files.

Inputs are the GridPro **binary** grid file (blk.tmp), GridPro connectivity file (blk.tmp.conn), and
GridPro property file (blk.tmp.pty).

Example
-------
/path/to/pergrinepy/utilities/gp2pg.py <gp_binary_grid_file> <gp_connnectivity_file> <gp_property_file>

Default vaules to read in are the GridPro generic names of blk.tmp, blk.tmp.conn, and blk.tmp.pty

Output will be a PEREGRINE grid file 'g.h5', which carries the connectivity.

"""

import argparse
from peregrinepy.partition import Conditioner
import numpy as np
from peregrinepy.multiBlock import grid as mbg
from peregrinepy.writers import GridWriter
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
    help="""File to translate the labels given to boundary conditions in GridPro
            to PEREGRINE bcType (i.e. constantVelocitySubsonicInlet, adiabaticNoSlipWall, etc.)\n
            \n
            NOTE: ALL labels from GridPro need an entry in the yaml file. Even walls.""",
    type=str,
)
parser.add_argument(
    "-units",
    action="store",
    metavar="<units>",
    dest="units",
    default="m",
    help="""Grid units, [in,mm,cm], to be converted to meters. Default is meters.""",
    type=str,
)
parser.add_argument(
    "--binary",
    action="store_true",
    dest="isBinary",
    help="Block file in binary format (default is ascii).",
)
args = parser.parse_args()

conversion = {"m": 1.0, "in": 0.0254, "mm": 0.001, "cm": 0.01}
factor = conversion[args.units]

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
    "pdc:INTERBLK": "interior",
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
gridIsPeriodic = False
for temp, blk in zip(pgConns, mb):
    for face in blk.faces:
        faceData = temp[(int(face.nface) - 1) * 4 : (int(face.nface) - 1) * 4 + 4]
        # Set the named BCs in the bcFam.yaml
        if faceData[0] in bcFamDict:
            face.bcName = f"{faceData[0]}"
            face.bcType = bcFamDict[faceData[0]]["bcType"]
        # Need to treat periodics
        elif faceData[0] == "PERIODIC":
            gridIsPeriodic = True
            # get the periodic from bcValues
            key = [key for key in bcFamDict.keys() if key.startswith("periodic")][0]
            face.bcName = key
            # We will set High/Low bcType after we have all the point data
            # the sign is settled once every block has its points
            face.bcType = bcFamDict[key]["bcType"]
            periodicSpan = bcFamDict[key]["bcVals"]["periodicSpan"]
            periodicAxis = np.array(bcFamDict[key]["bcVals"]["periodicAxis"], float)
            periodicAxis /= np.linalg.norm(periodicAxis)
        else:
            face.bcType = f"{faceData[0]}"
            face.bcName = None
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
    blkShape = tuple([int(b) for b in gpBlkFile.readline().strip().split()])

    blk.setExtents(*blkShape)

    if args.isBinary:
        byte = gpBlkFile.read(8 * blkShape[0] * blkShape[1] * blkShape[2] * 3)
        temp = np.frombuffer(byte, dtype=np.float64).reshape(
            (blkShape[0] * blkShape[1] * blkShape[2], 3)
        )
        blk.array["x"][:] = temp[:, 0].reshape(blkShape) * factor
        blk.array["y"][:] = temp[:, 1].reshape(blkShape) * factor
        blk.array["z"][:] = temp[:, 2].reshape(blkShape) * factor

        gpBlkFile.read(1)
    else:
        for i in range(blkShape[0]):
            for j in range(blkShape[1]):
                for k in range(blkShape[2]):
                    line = gpBlkFile.readline().strip().split()
                    blk.array["x"][i, j, k] = float(line[0]) * factor
                    blk.array["y"][i, j, k] = float(line[1]) * factor
                    blk.array["z"][i, j, k] = float(line[2]) * factor

    blockStart = gpBlkFile.tell()

gpConnFile.close()
gpPtyFile.close()
gpBlkFile.close()

# Now we go through the grid to set periodic High/Low
if gridIsPeriodic:
    from verifyGrid import extractFace

    for blk in mb:
        for face in blk.faces:
            if face.bcType.startswith("periodic"):
                myX, myY, myZ = extractFace(blk, face.nface)
                myFaceCenter = np.array([np.mean(i) for i in (myX, myY, myZ)])
                nBlk = mb.getBlock(face.neighbor)
                nX, nY, nZ = extractFace(nBlk, face.neighborNface)
                nFaceCenter = np.array([np.mean(i) for i in (nX, nY, nZ)])
            else:
                continue

            if face.bcType == "periodicTrans":
                up = periodicAxis * periodicSpan
                faceUp, faceDown = myFaceCenter + up, myFaceCenter - up
            elif face.bcType == "periodicRot":
                R = face.rotationAbout(periodicAxis, periodicSpan)
                faceUp, faceDown = R @ myFaceCenter, R.T @ myFaceCenter
            else:
                raise ValueError("Didnt find either periodic Trans or Rot")

            # whichever way lands on the partner is the way this face moves
            if np.allclose(faceUp, nFaceCenter):
                sign = 1.0
            elif np.allclose(faceDown, nFaceCenter):
                sign = -1.0
            else:
                raise ValueError(
                    f"Not a periodic match between block {blk.nblki} face {face.nface} and block {nBlk.nblki} face {face.neighborNface}."
                )

            if face.bcType == "periodicTrans":
                face.setPeriodic(translation=sign * periodicAxis * periodicSpan)
            else:
                face.setPeriodic(
                    rotation=face.rotationAbout(periodicAxis, sign * periodicSpan)
                )

# a mesher's own precision is its business, so this is a warning, not a
# gate; what must not happen is conditioning making a good grid bad
verified = verify(mb)
if not verified:
    print("  NOTE: the translated grid does not verify, see above")

Conditioner().condition(mb)

if verified and not verify(mb):
    raise ValueError("Conditioning invalidated the grid.")

print("Writing out {} block PEREGRINE grid files".format(len(mb)))
GridWriter(mb, "./").write(mb)

print("GridPro to PEREGRINE translation done.")
