"""A GridPro grid translated into a PEREGRINE grid file: the block node
file, the connectivity file and the property file, whose surface labels
become the faces' names. The grid is conditioned on the way out."""

import numpy as np

import peregrinepy as pg
from ..partition import Conditioner
from .verify import verify

name = "gridpro2pg"
help = "translate a GridPro grid (blk.tmp, .conn, .pty) into a grid file"


def addArguments(parser):
    parser.add_argument("blk", help="the GridPro block node file, blk.tmp")
    parser.add_argument("conn", help="the GridPro connectivity file, blk.tmp.conn")
    parser.add_argument("pty", help="the GridPro property file, blk.tmp.pty")
    parser.add_argument("out", help="the grid file to write")
    parser.add_argument(
        "-units", default="m", choices=("m", "in", "mm", "cm"), help="the grid's units"
    )
    parser.add_argument(
        "--binary", action="store_true", help="the node file is binary, not ascii"
    )


def _skipComments(f):
    """The next line of :f: that is not a comment, split into words."""
    while True:
        line = f.readline().split()
        if line and not line[0].startswith("#"):
            return line


def readProperties(pty, nblks):
    """The name of each block face's surface, (nblks, 6), from the property
    file: a GridPro label with its pdc: stripped, and None for an interface."""
    with open(pty) as f:
        assert int(_skipComments(f)[0]) == nblks, "property and connectivity disagree"
        blkSurfaces = []
        for _ in range(nblks):
            line = _skipComments(f)
            blkSurfaces.append(
                [line[4], line[6], line[8], line[10], line[12], line[14]]
            )
        while True:
            line = f.readline().split()
            if "2D" in line and "properties" in line:
                nprops = int(line[0])
                break
        surfaceName = {}
        for _ in range(nprops):
            line = f.readline().replace("(", "").replace(")", "").split()
            label = line[2].removeprefix("pdc:")
            surfaceName[line[0]] = None if label == "INTERBLK" else label
    return [[surfaceName[s] for s in surfaces] for surfaces in blkSurfaces]


def readConnectivity(conn):
    """Every block's six faces from the connectivity file: [(neighbor,
    orientation)] * 6 per block, neighbor counted from zero or None."""
    with open(conn) as f:
        nblks = int(_skipComments(f)[0])
        faces = []
        for _ in range(nblks):
            line = f.readline().split()[2:-1]
            faces.append(
                [
                    (
                        None if int(line[4 * n + 2]) == 0 else int(line[4 * n + 2]) - 1,
                        None if "0" in line[4 * n + 3] else line[4 * n + 3],
                    )
                    for n in range(6)
                ]
            )
    return faces


def readNodes(blk, mb, binary, factor):
    """Every block's nodes from the node file, ascii or binary."""
    with open(blk, "rb" if binary else "r") as f:
        # past the header to the first block's extents
        while True:
            start = f.tell()
            line = f.readline()
            try:
                [int(b) for b in line.strip().split()]
                break
            except ValueError:
                pass
        for block in mb.blocks:
            f.seek(start)
            shape = tuple(int(b) for b in f.readline().strip().split())
            block.setExtents(*shape)
            count = shape[0] * shape[1] * shape[2]
            if binary:
                nodes = np.frombuffer(f.read(8 * count * 3), dtype=np.float64)
                nodes = nodes.reshape(shape + (3,))
                f.read(1)
            else:
                nodes = np.empty(shape + (3,))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            line = f.readline().strip().split()
                            nodes[i, j, k] = [float(v) for v in line[0:3]]
            block.nodes.set(nodes * factor)
            start = f.tell()


def main(args):
    factor = {"m": 1.0, "in": 0.0254, "mm": 0.001, "cm": 0.01}[args.units]
    print(f"Reading the {'binary' if args.binary else 'ascii'} GridPro files")
    print(f"    {args.blk}\n    {args.conn}\n    {args.pty}\n")

    faces = readConnectivity(args.conn)
    nblks = len(faces)
    print(f"Found {nblks} blocks")
    names = readProperties(args.pty, nblks)

    mb = pg.multiBlock.grid()
    for _ in range(nblks):
        mb.addBlock()
    for blk, blkFaces, blkNames in zip(mb.blocks, faces, names):
        for face, (neighbor, orientation), faceName in zip(
            blk.faces, blkFaces, blkNames
        ):
            if neighbor is None:
                # GridPro's label is the name the case binds values to
                face.bcName = faceName
            else:
                # an interface, or a periodic: conditioning tells them apart
                face.setInterior(neighbor, orientation)
    readNodes(args.blk, mb, args.binary, factor)

    # a periodic arrives looking like any other interface; which ones really
    # are, and how they move, is read off the points, and verify needs it
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
    print("GridPro to PEREGRINE translation done.")
