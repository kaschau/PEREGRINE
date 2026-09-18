import peregrinepy as pg
import numpy as np
from ..gases import configure, primitives


class PerturbedCube(pg.mesher.CubeMesher):
    """A cube whose interior nodes are jostled a little, so no face normal
    is exactly an axis."""

    def shapeBlock(self, blk, i, j, k):
        super().shapeBlock(blk, i, j, k)
        nodes = blk.nodes.get()
        inner = np.s_[1:-1, 1:-1, 1:-1]
        nodes[inner] += np.random.uniform(-1, 1, nodes[inner].shape) * 0.02
        blk.nodes.set(nodes)


def create(bc, adv, gas):
    """A one-block Navier-Stokes case with every face carrying :bc: at
    random values, a random state on a perturbed grid, and its gradients
    taken."""
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = adv
    configure(config, gas, "navierStokes")
    # every face the same boundary, at random values the config's entry sets
    values = {
        "bcType": bc,
        "p": np.random.uniform(low=101325 * 0.9, high=101325 * 1.1),
        "u": np.random.uniform(low=1, high=1000),
        "v": np.random.uniform(low=1, high=1000),
        "w": np.random.uniform(low=1, high=1000),
        "T": np.random.uniform(low=300 * 0.9, high=300 * 1.1),
        "mDotPerUnitArea": np.random.uniform(low=1, high=1000),
    }
    values["pt"], values["Tt"] = values["p"], values["T"]
    speciesNames = pg.mixture.Mixture(config["simulation"]).speciesNames
    if len(speciesNames) > 1:
        Ybc = np.random.uniform(low=0.0, high=1.0, size=len(speciesNames))
        Ybc = Ybc / np.sum(Ybc)
        values.update(zip(speciesNames[:-1], Ybc))
    config["bcValues"]["outer"] = values

    mesh = PerturbedCube(
        mbDims=[1, 1, 1],
        dimsPerBlock=[8, 6, 4],
        lengths=[1, 1, 1],
        boundaryNames=dict.fromkeys(range(1, 7), "outer"),
    )
    mb = pg.multiBlock.solver(config, mesh)
    mixture = mb.simulator.mixture
    blk = mb.blocks[0]
    for face in blk.faces:
        if face.QBcVals is None:
            continue
        # the target mdot goes in the zeroth (unused) index of QBcVals, for the check
        QBcVals = face.QBcVals.get()
        QBcVals[:, :, 0] = values["mDotPerUnitArea"]
        face.QBcVals.set(QBcVals)

    q = primitives(mb, blk)
    qshape = q.shape[:3]
    p = np.random.uniform(low=101325.0 * 0.1, high=101325 * 10, size=qshape)
    u = np.random.uniform(low=-200, high=200, size=qshape)
    v = np.random.uniform(low=-200, high=200, size=qshape)
    w = np.random.uniform(low=-200, high=200, size=qshape)
    T = np.random.uniform(low=200, high=3000, size=qshape)
    if mixture.ns > 1:
        Y = np.random.uniform(low=0.0, high=1.0, size=qshape + (mixture.ns - 1,))
        Y = Y / np.sum(Y, axis=-1)[:, :, :, np.newaxis]

    q[:, :, :, 0] = p
    q[:, :, :, 1] = u
    q[:, :, :, 2] = v
    q[:, :, :, 3] = w
    q[:, :, :, 4] = T
    if mixture.ns > 1:
        q[:, :, :, 5::] = Y
    mb.setPrimitives([q])

    mb.launch("dqdxyz", "interior")

    return mb
