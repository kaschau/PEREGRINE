"""A grid checked face by face: every connection agreed to from both
sides, the coordinates of joined faces on top of each other, a periodic's
partner where its transform puts it, and every block right handed."""

import numpy as np

import peregrinepy as pg

name = "verify"
help = "check a grid's connectivity and that joined faces' coordinates match"

faceToOrientIndexMapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
largeIndexMapping = {0: "k", 1: "k", 2: "j"}
needToTranspose = {
    "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
    "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
}


def addArguments(parser):
    parser.add_argument("grid", help="the grid file")


def extractFace(blk, nface):
    faceSliceMapping = {
        1: {"i": 0, "j": slice(None), "k": slice(None)},
        2: {"i": -1, "j": slice(None), "k": slice(None)},
        3: {"i": slice(None), "j": 0, "k": slice(None)},
        4: {"i": slice(None), "j": -1, "k": slice(None)},
        5: {"i": slice(None), "j": slice(None), "k": 0},
        6: {"i": slice(None), "j": slice(None), "k": -1},
    }
    face_i = faceSliceMapping[nface]
    plane = blk.nodes.get()[face_i["i"], face_i["j"], face_i["k"]]
    return (np.copy(plane[..., n]) for n in range(3))


def verify(mb):
    """Says whether the grid is consistent, printing every discrepancy."""
    warn = False
    tol = 1e-7
    for blk in mb.blocks:
        for face in blk.faces:
            nface = face.nface
            neighbor = face.neighbor
            bc = face.bcType
            orientation = face.orientation
            bcName = face.bcName

            if neighbor is None:
                assert bc != "interior" and face.periodicRotation is None, (
                    f"Block #{blk.nblki} face {nface} has no neighbor, "
                    f"but has bcType {bc}"
                )
                assert (
                    orientation is None
                ), f"Block #{blk.nblki} face {nface} has no neighbor, but has orientation {orientation}"
                if pg.simulator.BaseBC.named(bc).values:
                    assert (
                        bcName is not None
                    ), f"Block #{blk.nblki} face {nface} is {bc}, but has no bcName"
                continue

            face_x, face_y, face_z = extractFace(blk, face.nface)

            blk2 = mb.getBlock(neighbor)
            nface2 = face.neighborNface
            nOrientation = face.neighborOrientation

            if int(blk2.getFace(nface2).neighbor) != blk.nblki:
                raise ValueError(
                    f"Block {blk.nblki}'s' face {nface} says it is connected to\nblock {blk2.nblki}'s' face {nface2}, however block {blk2.nblki}'s\nface {nface2} says it is connected to a different block."
                )

            face2_x, face2_y, face2_z = extractFace(blk2, nface2)

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

            # move the face onto its partner, the way a halo through it goes
            if face.periodicRotation is not None:
                shape = face_x.shape
                points = np.column_stack(
                    (face_x.ravel(), face_y.ravel(), face_z.ravel())
                )
                # the partner is the other way round from the halo
                points = (points - face.periodicTranslation) @ face.periodicRotation
                face_x = points[:, 0].reshape(shape)
                face_y = points[:, 1].reshape(shape)
                face_z = points[:, 2].reshape(shape)

            try:
                off_x = np.max(np.abs(face_x - face2_x))
                off_y = np.max(np.abs(face_y - face2_y))
                off_z = np.max(np.abs(face_z - face2_z))
            except ValueError:
                raise ValueError(
                    f"Error when comparing block {blk.nblki} and block {blk2.nblki} connection"
                )

            for axis, off in zip("xyz", (off_x, off_y, off_z)):
                if off > tol:
                    print(
                        f"Warning, the {axis} coordinates of face {nface} on block {blk.nblki} are not matching the {axis} coordinates of face {nface2} of block {blk2.nblki}"
                    )
                    print(f"Off by average of {off}")
                    warn = True

        # every block right handed
        nodes = blk.nodes.get()
        pO, pI, pJ, pK = nodes[0, 0, 0], nodes[1, 0, 0], nodes[0, 1, 0], nodes[0, 0, 1]
        if np.dot(pK - pO, np.cross(pI - pO, pJ - pO)) < 0.0:
            print(f"Warning, block {blk.nblki} is left handed. This must be fixed.")
            warn = True

    return not warn


def main(args):
    mb = pg.multiBlock.grid.fromGrid(args.grid, quiet=False)
    if not verify(mb):
        raise SystemExit(1)
    print("Grid is valid!")
