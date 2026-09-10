import itertools

import numpy as np
import peregrinepy as pg
import pytest

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


def reorientBlock1(mb, S, varList):
    """Restore the two block "blk0 face 2 <-> blk1" interface with block 1's
    storage laid out per S, holding the same physical data."""
    if S == "123":
        return
    blk0, blk1 = mb[0], mb[1]
    for var in varList:
        blk1.array[var] = reorient(blk1.array[var], S)

    # storage axis axes[m] now holds reference axis m
    axes, _ = signedPermutation(S)
    refDims = (blk1.ni, blk1.nj, blk1.nk)
    newDims = [0, 0, 0]
    for m in range(3):
        newDims[axes[m]] = refDims[m]
    blk1.setExtents(*newDims)

    blk0.getFace(2).orientation = S
    nn = blk0.getFace(2).neighborNface
    inverseS = blk0.getFace(2).neighborOrientation
    # the interface may no longer be block 1's face 1
    if nn != 1:
        old = blk1.getFace(1)
        old.neighbor = None
        old.bcType = "adiabaticSlipWall"
        old.orientation = None
        old.commRank = None
        new = blk1.getFace(nn)
        new.neighbor = 0
        new.bcType = "interior"
        new.commRank = 0
    blk1.getFace(nn).orientation = inverseS


VARLIST = ["x", "y", "z", "q", "Q", "dqdx", "dqdy", "dqdz", "phi"]

pytestmark = pytest.mark.parametrize(
    "adv,spdata",
    list(
        itertools.product(
            ("KEEPpe", "fourthOrderKEEP"),
            (["Air"], "thtr_CH4_O2_FFCMY.yaml"),
        )
    ),
)


def buildAndCommunicate(S, adv, spdata, seed):
    np.random.seed(seed)

    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = adv
    config["RHS"]["shockHandling"] = "hybrid"
    config["RHS"]["secondaryAdvFlux"] = "rusanov"
    config["RHS"]["diffusion"] = True
    config["thermochem"]["spdata"] = spdata

    mb = pg.multiBlock.buildSolver(config, 2)
    pg.mesher.CubeMesher(
        mbDims=[2, 1, 1], dimsPerBlock=[6, 3, 2], lengths=[2, 1, 1]
    ).mesh(mb)
    mb.generateHalo()
    mb.computeMetrics()

    mb[0].getFace(2).commRank = 0
    mb[1].getFace(1).commRank = 0

    for blk in mb:
        for var in VARLIST:
            blk.array[var][:] = np.random.random(blk.array[var].shape)

    if S != "123":
        # allocation copies existing numpy arrays into the new views and
        # asserts the shapes agree; block 1's other arrays still carry the
        # pre-reorientation shape, so drop them and let them be rebuilt
        for blk in mb:
            for v in blk.array.keys():
                if v not in VARLIST:
                    blk.array[v] = None
                    blk.mirror[v] = None
        reorientBlock1(mb, S, VARLIST)
        # reorienting re-sized block 1, which rebuilt its views around what
        # it left; block 0 keeps its size but lost the arrays dropped above
        mb[0].allocate()

    mb.setBlockCommunication()

    for blk in mb:
        blk.updateDeviceView(VARLIST)
    pg.mpiComm.communicate(mb, VARLIST)
    for blk in mb:
        blk.updateHostView(VARLIST)

    return mb


# the identity run is the ground truth, built once per configuration
_refCache = {}


def _reference(adv, spdata, seed):
    key = (adv, str(spdata), seed)
    if key not in _refCache:
        mb = buildAndCommunicate("123", adv, spdata, seed)
        _refCache[key] = (
            {v: mb[0].array[v].copy() for v in VARLIST},
            {v: mb[1].array[v].copy() for v in VARLIST},
        )
    return _refCache[key]


@pytest.mark.parametrize("S", allValidOrientations())
def test_orientation(my_setup, adv, spdata, S):
    seed = 20260901
    ref0, ref1 = _reference(adv, spdata, seed)

    mb = buildAndCommunicate(S, adv, spdata, seed)
    for var in VARLIST:
        assert np.array_equal(mb[0].array[var], ref0[var]), (S, var, "blk0")
        assert np.array_equal(mb[1].array[var], reorient(ref1[var], S)), (
            S,
            var,
            "blk1",
        )
