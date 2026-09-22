"""A sector grid copied and turned about its axis into more sectors: the
sector's two periodic faces join sector to sector, and the two at the ends
stay periodic across the whole sweep, or become interfaces when the
sectors close a full turn."""

import numpy as np

import peregrinepy as pg
from ..multiBlock.topologyFace import topologyFace

name = "rotate"
help = "copy a sector grid about its axis into so many sectors"


def addArguments(parser):
    parser.add_argument("grid", help="the sector's grid file")
    parser.add_argument("out", help="the grid file to write")
    parser.add_argument(
        "-segments", type=int, required=True, help="how many sectors to end with"
    )
    parser.add_argument(
        "-angle", type=float, required=True, help="the sector's angle, degrees"
    )
    parser.add_argument(
        "-axis", required=True, help="the axis the sector turns about, x,y,z"
    )


def turnOf(face, axis):
    """How far a halo through this periodic face is turned about the axis,
    in degrees, signed."""
    R = face.periodicRotation
    sine = 0.5 * (
        (R[2, 1] - R[1, 2]) * axis[0]
        + (R[0, 2] - R[2, 0]) * axis[1]
        + (R[1, 0] - R[0, 1]) * axis[2]
    )
    cosine = 0.5 * (np.trace(R) - 1.0)
    return np.degrees(np.arctan2(sine, cosine))


def main(args):
    nseg, angle = args.segments, args.angle
    if nseg <= 1:
        raise SystemExit(
            "peregrine rotate: -segments is the sectors to end with, above one"
        )
    axis = np.array([float(a) for a in args.axis.split(",")])
    axis /= np.linalg.norm(axis)
    closed = abs(nseg * angle - 360.0) < 1e-10

    fromGrid = pg.multiBlock.grid.fromGrid(args.grid, quiet=False)
    nblks = len(fromGrid.blocks)
    # a halo through the sector's high face is turned by +angle, through
    # its low face by -angle
    high = {}
    for blk, face in fromGrid.faces():
        if face.periodicRotation is None:
            continue
        turn = turnOf(face, axis)
        if abs(abs(turn) - angle) > 1e-6:
            raise ValueError(
                f"block {blk.nblki} face {face.nface} is periodic by {turn:.6f}"
                f" degrees, not the sector's {angle}"
            )
        high[(blk.nblki, face.nface)] = turn > 0.0
    if not high:
        raise ValueError(f"{args.grid} has no periodic faces to turn about")

    toGrid = pg.multiBlock.grid()
    for _ in range(nblks * nseg):
        toGrid.addBlock()
    whole = topologyFace.rotationAbout(axis, nseg * angle)
    for k in range(nseg):
        R = topologyFace.rotationAbout(axis, k * angle)
        for fromBlk in fromGrid.blocks:
            toBlk = toGrid.blocks[k * nblks + fromBlk.nblki]
            toBlk.setExtents(fromBlk.ni, fromBlk.nj, fromBlk.nk)
            nodes = fromBlk.nodes.get()
            toBlk.nodes.set((nodes.reshape(-1, 3) @ R.T).reshape(nodes.shape))
            for toFace, fromFace in zip(toBlk.faces, fromBlk.faces):
                if fromFace.neighbor is None:
                    toFace.copyFrom(fromFace)
                    continue
                if fromFace.periodicRotation is None:
                    toFace.setInterior(
                        fromFace.neighbor + k * nblks, fromFace.orientation
                    )
                    continue
                # a periodic: joined to the next sector round, or across the
                # whole sweep at the ends
                step = 1 if high[(fromBlk.nblki, fromFace.nface)] else -1
                partner = k + step
                if 0 <= partner < nseg:
                    toFace.setInterior(
                        fromFace.neighbor + partner * nblks, fromFace.orientation
                    )
                elif closed:
                    toFace.setInterior(
                        fromFace.neighbor + (partner % nseg) * nblks,
                        fromFace.orientation,
                    )
                else:
                    toFace.copyFrom(fromFace)
                    toFace.neighbor = fromFace.neighbor + (partner % nseg) * nblks
                    toFace.setPeriodic(rotation=whole if step > 0 else whole.T)

    print(f"Writing the {len(toGrid.blocks)} block grid to {args.out}")
    pg.writers.GridWriter(toGrid, args.out, quiet=False).write(toGrid)
