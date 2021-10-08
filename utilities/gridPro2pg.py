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
from verify_grid import verify

parser = argparse.ArgumentParser(
    description="Convert binary GridPro files into grid and connectivity files used by PEREGRINE"
)
parser.add_argument(
    "-blk",
    action="store",
    metavar="<GP_blk_FILE>",
    dest="gp_blk_file_name",
    default="blk.tmp",
    help="Binary GridPro block node file. Default is blk.tmp",
    type=str,
)
parser.add_argument(
    "-conn",
    action="store",
    metavar="<GP_conn_FILE>",
    dest="gp_conn_file_name",
    default="blk.tmp.conn",
    help="GridPro block connectivity file. Default is blk.tmp.conn",
    type=str,
)
parser.add_argument(
    "-pty",
    action="store",
    metavar="<GP_pty_FILE>",
    dest="gp_pty_file_name",
    default="blk.tmp.pty",
    help="GridPro property file. Default is blk.tmp.pty",
    type=str,
)
parser.add_argument(
    "--binary",
    action="store_true",
    dest="is_binary",
    help="Block file in binary format (default is ascii).",
)
args = parser.parse_args()

print("Reading in {0} GridPro files\n".format("binary" if args.is_binary else "ascii"))
print("    {}".format(args.gp_blk_file_name))
print("    {}".format(args.gp_conn_file_name))
print("    {}\n".format(args.gp_pty_file_name))


gp_conn_file = open(args.gp_conn_file_name, "r")
gp_pty_file = open(args.gp_pty_file_name, "r")
if args.is_binary:
    gp_blk_file = open(args.gp_blk_file_name, "rb")
else:
    gp_blk_file = open(args.gp_blk_file_name, "r")

gp_surface_to_pg_bc = {
    "pdc:INTERBLK": "b0",
    "pdc:PERIODIC": "b1",
    "pdc:WALL": "adiabaticNoSlipWall",
    "pdc:user5": "constantPressureSubsonicInlet",
    "pdc:user6": "constantPressureSubsonicInlet",
    "pdc:user7": "user7",
    "pdc:user8": "user8",
    "pdc:user9": "user9",
    "pdc:user10": "user10",
    "pdc:user11": "user11",
}

blk_ptys = []
go = True
while go:
    line = gp_pty_file.readline().split()
    if line[0].startswith("#"):
        pass
    else:
        try:
            nblks = int(line[0])
            print("Found {} blocks".format(nblks))
            go = False
        except ValueError:
            raise TypeError("Unrecognized format for property file.")

present_surfaces = []

for i in range(nblks):
    go = True
    while go:
        line = gp_pty_file.readline().split()
        if line[0].startswith("#"):
            pass
        else:
            go = False

    blk_ptys.append([line[4], line[6], line[8], line[10], line[12], line[14]])

    for surf in blk_ptys[i]:
        if int(surf) > 0 and surf not in present_surfaces:
            present_surfaces.append(surf)

go = True
while go:
    line = gp_pty_file.readline().split()
    if "2D" in line and "properties" in line:
        nprops = int(line[0])
        go = False

surface_to_bc = dict()
for i in range(nprops):
    line = gp_pty_file.readline().replace("(", "").replace(")", "").split()
    if line[0] in present_surfaces:
        surface_to_bc[line[0]] = gp_surface_to_pg_bc[line[2]]

pg_blk_ptys = []
for i in range(nblks):
    pg_blk_ptys.append([0, 0, 0, 0, 0, 0])
    for j in range(6):
        pg_blk_ptys[i][j] = surface_to_bc[blk_ptys[i][j]]

go = True
while go:
    line = gp_conn_file.readline().split()
    if line[0].startswith("#"):
        pass
    else:
        try:
            nblks = int(line[0])
            go = False
        except ValueError:
            raise TypeError("Unrecognized format for connectivity file.")

pg_conns = []
for i in range(nblks):
    line = gp_conn_file.readline().split()
    for j in range(6):
        line[2 + j * 4] = pg_blk_ptys[i][j].split()[0]
        line[3 + j * 4] = pg_blk_ptys[i][j].split()[1]
    pg_conns.append(line[2:-1])

mb = mbg(nblks)
for temp, blk in zip(pg_conns, mb):
    for face in blk.faces:
        face_data = temp[(int(face.nface) - 1) * 4 : (int(face.nface) - 1) * 4 + 4]
        face.bcType = "{}{}".format(face_data[0], face_data[1])
        face.connection = face_data[2]
        face.orientation = face_data[3]

go = True
while go:
    block_start = gp_blk_file.tell()
    line = gp_blk_file.readline()
    try:
        test = [int(b) for b in line.strip().split()]
        go = False
    except ValueError:
        go = True

blk_count = 1
go = True

for blk in mb:

    gp_blk_file.seek(block_start)
    blk_shape = tuple([int(b) for b in gp_blk_file.readline().strip().split()])

    blk.nblki = blk_count

    if args.is_binary:
        byte = gp_blk_file.read(8 * blk_shape[0] * blk_shape[1] * blk_shape[2] * 3)
        temp = np.fromstring(byte, dtype=np.float64).reshape(
            (blk_shape[0] * blk_shape[1] * blk_shape[2], 3)
        )

        blk.array["x"] = temp[:, 0].reshape(blk_shape, order="F")
        blk.array["y"] = temp[:, 1].reshape(blk_shape, order="F")
        blk.array["z"] = temp[:, 2].reshape(blk_shape, order="F")

        gp_blk_file.read(1)
    else:
        blk.array["x"] = np.empty(blk_shape, dtype=np.float64)
        blk.array["y"] = np.empty(blk_shape, dtype=np.float64)
        blk.array["z"] = np.empty(blk_shape, dtype=np.float64)

        for i in range(blk_shape[0]):
            for j in range(blk_shape[1]):
                for k in range(blk_shape[2]):
                    line = gp_blk_file.readline().strip().split()
                    blk.array["x"][i, j, k] = float(line[0])
                    blk.array["y"][i, j, k] = float(line[1])
                    blk.array["z"][i, j, k] = float(line[2])

    block_start = gp_blk_file.tell()
    blk_count += 1

gp_conn_file.close()
gp_pty_file.close()
gp_blk_file.close()

if verify(mb):
    pass

print("Writing out PEREGRINE connectivity file: conn.inp...")
writeConnectivity(mb, "./")

print("Writing out {} block PEREGRINE grid files".format(len(mb)))
writeGrid(mb, "./")

print("GridPro to PEREGRINE translation done.")
