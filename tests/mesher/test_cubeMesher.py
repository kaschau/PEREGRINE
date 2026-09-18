"""A cube mesher wires its blocks: interior faces meet their neighbors,
periodic sides carry the translation that brings the far side over,
outer sides are walls unless named, and the halo of a periodic grid is
the far side's nodes moved by that translation."""

import numpy as np

import peregrinepy as pg

from ..gases import configure


def test_facesAreWiredAndNamed():
    mb = pg.multiBlock.grid()
    pg.mesher.CubeMesher(
        mbDims=[2, 1, 1],
        dimsPerBlock=[4, 3, 3],
        lengths=[2, 1, 1],
        periodic=[False, True, False],
        boundaryNames={1: "inlet", 6: "top"},
    ).fill(mb)
    a, b = mb.blocks
    assert a.getFace(2).neighbor == 1 and b.getFace(1).neighbor == 0
    assert a.getFace(2).bcType == "interior" and a.getFace(2).bcName is None
    # periodic in j: each block's j sides meet themselves, moved by the length
    for blk in mb.blocks:
        for nface in (3, 4):
            face = blk.getFace(nface)
            assert face.neighbor == blk.nblki and face.bcType == "periodicTrans"
            assert np.allclose(np.abs(face.periodicTranslation), [0, 1, 0])
            assert np.allclose(face.periodicRotation, np.eye(3))
    assert (
        a.getFace(1).bcName == "inlet" and a.getFace(1).bcType == "adiabaticNoSlipWall"
    )
    assert a.getFace(6).bcName == "top" and b.getFace(6).bcName == "top"
    assert a.getFace(5).bcName is None and a.getFace(5).bcType == "adiabaticNoSlipWall"
    # the second block sits beside the first
    assert np.isclose(b.nodes.get()[..., 0].min(), a.nodes.get()[..., 0].max())


def test_periodicHalosAreTheFarSideMoved(my_setup):
    config = pg.files.configFile()
    configure(config, "air", "euler")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[5, 4, 3],
        lengths=[2, 1, 1],
        periodic=[True, True, True],
    )
    mb = pg.multiBlock.solver(config, mesh)
    blk = mb.blocks[0]
    ng = blk.ng
    nodes = blk.nodes.get()
    # the node halo past the high i side continues the grid with the same spacing
    dx = nodes[ng + 1, ng, ng, 0] - nodes[ng, ng, ng, 0]
    assert np.allclose(np.diff(nodes[:, ng, ng, 0]), dx)
    # and the cell centers likewise, both ends
    cells = blk.cells.get()
    assert np.allclose(np.diff(cells[:, ng, ng, 0]), dx)
    assert np.allclose(cells[ng:-ng, ng, ng, 0].min(), dx / 2) and np.allclose(
        cells[0, ng, ng, 0], -dx / 2
    )
