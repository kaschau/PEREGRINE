import itertools

import numpy as np
import peregrinepy as pg
import pytest
from peregrinepy.graph import ExchangeGraphs
from peregrinepy.partition import getPartitioner

from ..gases import configure

partitioner = getPartitioner()

##############################################
# Every valid (right handed) block-to-block orientation, checked against the
# trivial "123" case rather than against a hand written index mapping.
#
# An orientation is three digits, one per index direction of ours, saying
# which of the neighbor's directions it runs along and whether it runs
# backwards: 1,2,3 for +i,+j,+k and 4,5,6 for -i,-j,-k. So "123" is the
# identity, and "162" means our i runs along their i, our j along their -k,
# and our k along their j.
#
# Block 1's storage is reoriented while its physical data is unchanged, so
# after communication block 0 must be bit-identical to the identity run --
# it cannot care how its neighbor stores data -- and block 1 must be exactly
# the identity result put through the same reorientation.
##############################################


def signedPermutation(S):
    """The axis permutation and per-axis flip an orientation string encodes."""
    axes = tuple((int(d) - 1) % 3 for d in S)
    flips = tuple(int(d) > 3 for d in S)
    return axes, flips


def allValidOrientations():
    """All 24 proper (det=+1) signed axis permutations as orientation strings.

    The other 24 signed permutations are reflections, which no right handed
    grid can produce.
    """
    valid = []
    for axes in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for m, (a, s) in enumerate(zip(axes, signs)):
                P[m, a] = s
            if np.linalg.det(P) > 0:
                valid.append(
                    "".join(
                        str(a + 1 + (3 if s < 0 else 0)) for a, s in zip(axes, signs)
                    )
                )
    assert len(valid) == 24
    return valid


def reorient(a, S):
    """Transform a reference-layout array into the S-oriented storage layout."""
    axes, flips = signedPermutation(S)
    out = a
    for m, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=m)
    return np.moveaxis(out, (0, 1, 2), axes)


class ReorientedPair(pg.mesher.CubeMesher):
    """Two blocks along i, the second's storage laid out per S while its
    physical data is unchanged: the partitioner relabels it, which is what
    makes the orientation strings."""

    def __init__(self, S, **kwargs):
        super().__init__(**kwargs)
        self.S = S

    def fill(self, mb):
        super().fill(mb)
        if self.S == "123":
            return
        axes, flips = signedPermutation(self.S)
        # storage axis axes[m] holds reference axis m: the partitioner takes
        # which old axis each new one holds, and which new ones run backwards
        perm = [axes.index(m) for m in range(3)]
        newFlips = [flips[perm[m]] for m in range(3)]
        partitioner.reorientBlock(mb, mb.getBlock(1), perm, newFlips)


VARLIST = ["nodes", "Q", "grads"]

pytestmark = pytest.mark.parametrize(
    "adv,gas",
    list(
        itertools.product(
            ("KEPaEC",),
            ("air", "CH4_O2"),
        )
    ),
)


def buildAndCommunicate(S, adv, gas, seed):
    """A case on the reoriented pair, random data in every exchanged array
    -- the same physical data whatever S -- traded once."""
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = adv
    configure(config, gas, "navierStokes")

    mesh = ReorientedPair(
        S, mbDims=[2, 1, 1], dimsPerBlock=[6, 3, 2], lengths=[2, 1, 1]
    )
    mb = pg.multiBlock.solver(config, mesh)

    # the data drawn in the reference layout, so block 1 gets the same
    # physical field however it is stored
    np.random.seed(seed)
    # both blocks have the reference shape in the reference layout
    for blk in mb.blocks:
        for var in ("Q", "grads"):
            values = np.random.random(getattr(mb.blocks[0], var).shape)
            getattr(blk, var).set(values if blk.nblki == 0 else reorient(values, S))

    for var in VARLIST:
        for g in ExchangeGraphs("test", var).bind(*mb.means):
            g.run()
    return mb


# the identity run is the ground truth, built once per configuration
_refCache = {}


def _reference(adv, gas, seed):
    key = (adv, str(gas), seed)
    if key not in _refCache:
        mb = buildAndCommunicate("123", adv, gas, seed)
        _refCache[key] = (
            {v: getattr(mb.blocks[0], v).get() for v in VARLIST},
            {v: getattr(mb.blocks[1], v).get() for v in VARLIST},
        )
    return _refCache[key]


@pytest.mark.parametrize("S", allValidOrientations())
def test_orientation(my_setup, adv, gas, S):
    seed = 20260901
    ref0, ref1 = _reference(adv, gas, seed)

    mb = buildAndCommunicate(S, adv, gas, seed)
    for var in VARLIST:
        assert np.array_equal(getattr(mb.blocks[0], var).get(), ref0[var]), (
            S,
            var,
            "blk0",
        )
        assert np.array_equal(
            getattr(mb.blocks[1], var).get(), reorient(ref1[var], S)
        ), (
            S,
            var,
            "blk1",
        )
