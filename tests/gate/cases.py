"""The gate's cases: small, fixed, perturbed so every step moves every
cell, and long enough that a one-ulp change anywhere reaches the hash.
Each names its physics, gas, integrator and step count; the hash is of the
interior conserved state after the steps, over the blocks in order."""

import hashlib

import numpy as np
import peregrinepy as pg

from ..gases import configure, primitives

cases = {
    "navierStokes rk3": dict(physics="navierStokes", gas="air", integrator="rk3"),
    "navierStokes dualTime": dict(
        physics="navierStokes", gas="air", integrator="dualTime"
    ),
    "euler rk3": dict(physics="euler", gas="air", integrator="rk3"),
    "navierStokes rk3 CH4_O2": dict(
        physics="navierStokes", gas="CH4_O2", integrator="rk3"
    ),
}


def mesh():
    """The gate's box: four blocks, periodic in i and k, walls on j."""
    return pg.mesher.CubeMesher(
        mbDims=[2, 2, 1],
        dimsPerBlock=[8, 7, 6],
        lengths=[1, 1, 1],
        periodic=[True, False, True],
        boundaryNames={3: "walls", 4: "walls"},
    )


def build(physics, gas, integrator, ranks=(1, 1), grid=None):
    """Makes a case on a 2x2x1 box periodic in i and k, or on a grid file
    partitioned for the ranks, its state perturbed by block."""
    config = pg.files.configFile()
    configure(config, gas, physics)
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["integrator"] = integrator
    config["timeIntegration"]["dt"] = 1e-9
    # the j sides are walls: no-slip where the flow sticks, slip where it cannot
    wall = "adiabaticNoSlipWall" if physics == "navierStokes" else "adiabaticSlipWall"
    config["bcValues"]["walls"] = {"bcType": wall}
    mb = pg.multiBlock.solver(
        config, pg.readers.GridReader(grid, ranks=ranks) if grid else mesh()
    )
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        if mb.ne > 5:
            base = rng.random(mb.ne - 5) / (mb.ne - 5)
            q[..., 5:] = base * (1 + 0.01 * rng.random(q.shape[:3] + (mb.ne - 5,)))
        prims.append(q)
    mb.setPrimitives(prims)
    mb.integrator.initialize()
    # what each block started from, for the digest to refuse a state that stood still
    for blk in mb.blocks:
        blk.started = blk.Q.get()[mb.ng : -mb.ng, mb.ng : -mb.ng, mb.ng : -mb.ng].copy()
    return mb


def step(mb, n=12):
    dt = mb.config["timeIntegration"]["dt"]
    for _ in range(n):
        mb.integrator.step(dt)


def blockDigests(mb):
    """Gives each block's interior state digest, by block number. A NaN
    hashes as steadily as a number, so the state is refused first if any
    of it is not finite, or if it never moved from where it started."""
    ng = mb.ng
    digests = {}
    for blk in mb.blocks:
        Q = np.ascontiguousarray(blk.Q.get()[ng:-ng, ng:-ng, ng:-ng])
        assert np.isfinite(Q).all(), f"block {blk.nblki}: the state is not finite"
        assert not np.array_equal(
            Q, blk.started
        ), f"block {blk.nblki}: the state never moved"
        digests[blk.nblki] = hashlib.sha256(Q.tobytes()).hexdigest()
    return digests


def digest(blockDigests):
    """Folds the blocks' digests, in block order, into one: the same
    whatever rank holds which block."""
    h = hashlib.sha256()
    for nblki in sorted(blockDigests):
        h.update(blockDigests[nblki].encode())
    return h.hexdigest()[:16]


def platform():
    """Names what the kernels ran on, which is what a reference is for: the
    runtime's backend, and the device architecture the toolchain targets."""
    from peregrinepy.backend.abi import lib
    from peregrinepy.backend.jit import Jit
    from peregrinepy.backend.toolchain import Toolchain

    name = lib.pgBackend().decode().lower()
    flags = Toolchain.read(Jit.package / "toolchain.json").flags
    arch = [f for f in flags if "offload-arch=" in f or f.startswith("-arch=")]
    return name + ("-" + arch[0].split("=")[-1] if arch else "")
