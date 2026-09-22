"""An ICEM grid translated into a PEREGRINE grid file: the Multi-Block
Info topology file for the connectivity and the faces' names, and the
nodes from a TLNS3D-MB file, or from the info.dom files beside the topology
file. The grid is conditioned on the way out.

The topology format is documented at
https://support.ansys.com/staticassets/ANSYS/Initial%20Content%20Entry/General%20Articles%20-%20Products/ICEM%20CFD%20Interfaces/multiBlock.htm
"""

from pathlib import Path

import numpy as np

import peregrinepy as pg
from ..partition import Conditioner
from .verify import verify

name = "icem2pg"
help = "translate an ICEM Multi-Block Info grid into a grid file"

faceMapping = {
    "small_i": 1,
    "large_i": 2,
    "small_j": 3,
    "large_j": 4,
    "small_k": 5,
    "large_k": 6,
}
orientationMapping = {"i": 1, "j": 2, "k": 3, "-i": 4, "-j": 5, "-k": 6}


def addArguments(parser):
    parser.add_argument("topo", help="the topology file, info.topo")
    parser.add_argument("out", help="the grid file to write")
    parser.add_argument(
        "-nodes",
        help="the TLNS3D-MB node file, double precision; without it the nodes "
        "are read from the single precision info.dom files beside the topology",
    )
    parser.add_argument(
        "-units", default="m", choices=("m", "in", "mm", "cm"), help="the grid's units"
    )


def readTlns3dmb(fileName, mb, factor):
    """Every block's nodes from a TLNS3D-MB file, a Fortran unformatted
    file: the block count, the extents, then each block's points."""
    from scipy.io import FortranFile

    with FortranFile(fileName, "r") as f90:
        nblks = f90.read_ints(dtype=np.int32)[0]
        for _ in range(nblks):
            mb.addBlock()
        nijks = np.array_split(f90.read_ints(dtype=np.int32), nblks)
        for nijk, blk in zip(nijks, mb.blocks):
            ni, nj, nk = nijk
            nodes = f90.read_reals(dtype=np.float64).reshape((ni, nj, nk, 3), order="F")
            blk.setExtents(ni, nj, nk)
            blk.nodes.set(nodes * factor)


def readDomains(here, mb, factor):
    """Every block's nodes from the info.dom<n> files in :here:."""
    nblks = len(list(here.glob("info.dom*")))
    print(f"Reading {nblks} ICEM domain files")
    for _ in range(nblks):
        mb.addBlock()
    for blk in mb.blocks:
        fileName = here / f"info.dom{blk.nblki}"
        with open(fileName) as f:
            _, ni, nj, nk = (int(v) for v in f.readline().strip().split()[:4])
        points = np.genfromtxt(fileName, comments="domain.")
        blk.setExtents(ni, nj, nk)
        blk.nodes.set(np.reshape(points[:, 0:3], (ni, nj, nk, 3)) * factor)


def axesOf(line):
    """The i, j, k a topology line's +/- axes name, in its order."""
    return [d.replace("-", "") for d in line[2:5]]


def faceOf(directions, mins, maxs):
    """Which face of a block a topology line describes: the one axis held
    at its low or high end."""
    for lo, hi, direction in zip(mins, maxs, directions):
        if lo == hi:
            side = "small" if lo == "1" else "large"
            return faceMapping[f"{side}_{direction}"]
    return None


def orientationOf(thisLine, adjacentLine, thisFace):
    """PEREGRINE's orientation string for a connection ICEM describes by
    the +/- i, j, k of each block."""
    if thisLine[2:5] == adjacentLine[2:5]:
        return "123"
    orientation = {}
    for n in range(3):
        curr, adjc = thisLine[2 + n], adjacentLine[2 + n]
        # the axis normal to the joined face keeps the neighbor's sign; the
        # others take the opposite
        side = "small" if curr.startswith("-") else "large"
        onFace = faceMapping[f"{side}_{curr[-1]}"] == thisFace
        if (onFace and not curr.startswith("-")) or (
            not onFace and curr.startswith("-")
        ):
            adjc = adjc.replace("-", "") if adjc.startswith("-") else f"-{adjc}"
        orientation[curr.replace("-", "")] = adjc
    return "".join(str(orientationMapping[orientation[a]]) for a in "ijk")


def readTopology(topo, mb):
    """The faces' names, the periodic connections and the interfaces, from
    the topology file's three sections."""
    lines = Path(topo).read_text().splitlines(keepends=True)

    # the tagged boundary faces: the tag is the face's name
    blk = None
    for rawLine in lines:
        if rawLine.startswith("# Boundary conditions and/or"):
            blk = mb.getBlock(int(rawLine.strip().split(".")[-1]) - 1)
            continue
        if rawLine == "\n":
            blk = None
        if blk is None:
            continue
        line = rawLine.strip().split()
        if line[1] != "f" or line[0] == "DEFAULT_SUBFACE":
            continue
        nface = faceOf("ijk", line[2:5], line[5:8])
        blk.getFace(nface).bcName = line[0]

    # the periodics, joined like interfaces; conditioning reads their transform
    it = iter(lines)
    for rawLine in it:
        if not rawLine.startswith("# Periodic info for domain"):
            continue
        current = int(rawLine.strip().split(".")[-1]) - 1
        blk = mb.getBlock(current)
        thisLine = next(it).replace("-", " -").strip().split()
        if not thisLine:
            continue
        adjacentLine = next(it).replace("-", " -").strip().split()
        adjacent = int(adjacentLine[1].split(".")[-1]) - 1
        thisFace = faceOf(axesOf(thisLine), thisLine[6:9], thisLine[9:12])
        if thisLine[5] != "f":
            raise ValueError(
                f"expected a face to set face {thisFace} of block {current},"
                f" not a {thisLine[5]}"
            )
        orientation = orientationOf(thisLine, adjacentLine, thisFace)
        blk.getFace(thisFace).setInterior(adjacent, orientation)
        # a block periodic with itself is not listed again from the other side
        if current == adjacent:
            oppFace = thisFace + 1 if thisFace % 2 else thisFace - 1
            blk.getFace(oppFace).setInterior(current, orientation)

    # the interfaces, in the order of the faces still unassigned
    it = iter(lines)
    for rawLine in it:
        if not rawLine.startswith("# Connectivity for domain"):
            continue
        current = int(rawLine.strip().split(".")[-1]) - 1
        blk = mb.getBlock(current)
        unassigned = [
            f.nface for f in blk.faces if f.bcName is None and f.neighbor is None
        ]
        for nface in unassigned:
            thisLine = next(it).replace("-", " -").strip().split()
            adjacentLine = next(it).replace("-", " -").strip().split()
            thisFace = faceOf(axesOf(thisLine), thisLine[6:9], thisLine[9:12])
            if thisFace != nface:
                raise ValueError(
                    f"the connectivity of block {current} gives face {thisFace}"
                    f" where face {nface} was expected"
                )
            if thisLine[5] != "f":
                raise ValueError(
                    f"expected a face to set face {thisFace} of block {current},"
                    f" not a {thisLine[5]}"
                )
            adjacent = int(adjacentLine[1].strip().split(".")[-1]) - 1
            orientation = orientationOf(thisLine, adjacentLine, nface)
            blk.getFace(nface).setInterior(adjacent, orientation)


def main(args):
    factor = {"m": 1.0, "in": 0.0254, "mm": 0.001, "cm": 0.01}[args.units]
    mb = pg.multiBlock.grid()
    if args.nodes:
        readTlns3dmb(args.nodes, mb, factor)
    else:
        readDomains(Path(args.topo).parent, mb, factor)
    readTopology(args.topo, mb)

    found = mb.detectPeriodics()
    for kind, n in sorted(found.items()):
        print(f"Found {n} {kind} face(s) among the interfaces")

    # a mesher's own precision is its business, so this is a warning, not a
    # gate; what must not happen is conditioning making a good grid bad
    verified = verify(mb)
    if not verified:
        print("  NOTE: the translated grid does not verify, see above")
    Conditioner().condition(mb)
    if verified and not verify(mb):
        raise ValueError("Conditioning invalidated the grid.")

    print(f"Writing the {len(mb.blocks)} block grid to {args.out}")
    pg.writers.GridWriter(mb, args.out, quiet=False).write(mb)
    print("ICEM to PEREGRINE translation done.")
