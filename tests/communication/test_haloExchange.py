"""One rank, two blocks: after an exchange a block's halo holds its
neighbor's cells, plane for plane, as deep as the array trades; the pair
met on this rank is filled directly, with no buffer; a wall's halo
is untouched."""

import numpy as np

import peregrinepy as pg
from peregrinepy.graph import ExchangeGraphs

from ..gases import configure


def pair(periodic=(False, False, False)):
    config = pg.files.configFile()
    configure(config, "air", "navierStokes")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    mesh = pg.mesher.CubeMesher(
        mbDims=[2, 1, 1],
        dimsPerBlock=[5, 4, 3],
        lengths=[2, 1, 1],
        periodic=list(periodic),
    )
    return pg.multiBlock.solver(config, mesh)


def stamp(mb, name):
    """Fills an array with a value that says which block and cell it is."""
    for blk in mb.blocks:
        a = getattr(blk, name).get()
        i, j, k = np.indices(a.shape[:3])
        cell = 1000 * blk.nblki + 100 * i + 10 * j + k
        a[...] = cell.reshape(cell.shape + (1,) * (a.ndim - 3))
        getattr(blk, name).set(a)


def exchange(mb, name):
    for g in ExchangeGraphs("test", name).bind(*mb.means):
        g.run()


def test_theHaloHoldsTheNeighborsCells(my_setup):
    mb = pair()
    stamp(mb, "Q")
    exchange(mb, "Q")
    a, b = mb.blocks
    ng = mb.ng
    Qa, Qb = a.Q.get(), b.Q.get()
    # block 0's high i halo is block 1's first interior planes, in order
    for g in range(ng):
        assert np.array_equal(Qa[-ng + g, ng:-ng, ng:-ng], Qb[ng + g, ng:-ng, ng:-ng])
        assert np.array_equal(Qb[g, ng:-ng, ng:-ng], Qa[-2 * ng + g, ng:-ng, ng:-ng])
    # a wall's halo is not the exchange's to touch: still its own stamp
    j, k = np.indices(Qa.shape[1:3])
    assert np.array_equal(Qa[0, ..., 0], 10 * j + k)


def test_aGradientTradesOnePlane(my_setup):
    mb = pair()
    assert mb.exchanges["grads"].depth == 1 and mb.exchanges["Q"].depth == mb.ng
    stamp(mb, "grads")
    before = mb.blocks[0].grads.get().copy()
    exchange(mb, "grads")
    after = mb.blocks[0].grads.get()
    ng = mb.ng
    # the plane next to the face came across; deeper planes did not
    assert np.array_equal(
        after[-1, ng:-ng, ng:-ng], mb.blocks[1].grads.get()[ng, ng:-ng, ng:-ng]
    )
    if ng > 1:
        assert np.array_equal(after[-ng, ng:-ng, ng:-ng], before[-ng, ng:-ng, ng:-ng])


def test_theLocalPairHasNoBuffer(my_setup):
    mb = pair()
    a, b = mb.blocks
    ex = mb.exchanges["Q"]
    left, right = a.getFace(2), b.getFace(1)
    assert mb.neighborFace(left) is right and mb.neighborFace(right) is left
    for face in (left, right):
        assert getattr(face, ex.sendBuffer) is None
        assert getattr(face, ex.recvBuffer) is None
    assert not ex.messages and ex.ranks == []
    assert set(mb.connOnRankFaces) == {left, right}
    assert mb.connOffRankFaces == []


def test_periodicHalosWrapAround(my_setup):
    mb = pair(periodic=(True, False, False))
    stamp(mb, "Q")
    exchange(mb, "Q")
    a, b = mb.blocks
    ng = mb.ng
    # block 0's low i halo is block 1's last interior planes
    for g in range(ng):
        assert np.array_equal(
            a.Q.get()[g, ng:-ng, ng:-ng], b.Q.get()[-2 * ng + g, ng:-ng, ng:-ng]
        )
