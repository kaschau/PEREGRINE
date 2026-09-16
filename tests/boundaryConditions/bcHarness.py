import numpy as np

from .bcBlock import create


class BaseBC:
    """One boundary condition, applied to every face of a single block."""

    bcType = None
    # what each variable's gradient does in the first halo
    gradRules = {"all": "neumann"}

    # what a stage writes, and so what to snapshot off the device after it:
    # the primitive vector, derived from the state, or the gradients
    _pull = {
        "euler": ["q"],
        "preDqDxyz": ["q"],
        "postDqDxyz": ["grads"],
    }
    # the slices of the primitive vector that share a rule
    _slice = {
        "p": np.s_[0],
        "u": np.s_[1],
        "v": np.s_[2],
        "w": np.s_[3],
        "velo": np.s_[1:4],
        "T": np.s_[4],
        "Y": np.s_[5:],
        "TY": np.s_[4:],
        "all": np.s_[:],
    }
    # the slices of grads, which holds no pressure
    _gradSlice = {
        "velo": np.s_[0:3],
        "T": np.s_[3],
        "Y": np.s_[4:],
        "TY": np.s_[3:],
        "all": np.s_[:],
    }
    # "negate" makes the gradient average to zero on the face, "neumann"
    # carries it over
    _sign = {"negate": -1.0, "neumann": 1.0}
    # the index into a face's qBcVals holding each variable
    _bcIndex = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}

    def __init__(self, adv, spdata):
        self.mb = create(self.bcType, adv, spdata)
        self.blk = self.mb.blocks[0]
        # the latest snapshot of each block array a stage wrote
        self.host = {}

    def check(self):
        for face in self.blk.faces:
            self.euler(face)
            self.viscous(face)
            self.run(face, "postDqDxyz")
            rules = dict(self.gradRules)
            if self.blk.ns == 1:
                rules.pop("Y", None)
            self._gradients(face, rules)

    def euler(self, face):
        raise NotImplementedError

    def viscous(self, face):
        """most bcs have nothing to say before the gradients are taken"""
        pass

    def run(self, face, stage):
        self.mb.applyBcs(stage, faces=[face])
        for name in self._pull[stage]:
            self.host[name] = (
                self.blk.primitives() if name == "q" else getattr(self.blk, name).get()
            )

    def q(self, name):
        """the q slice a variable name refers to; a species is "Y3" and so on"""
        if name.startswith("Y") and name[1:].isdigit():
            return self.host["q"][:, :, :, 5 + int(name[1:])]
        return self.host["q"][:, :, :, self._slice[name]]

    def normals(self, face):
        """the unit normal of the face plane, and the sign that makes it point
        out of the block (faces 1, 3, 5 store the inward normal)"""
        _, normals = self.blk.faceNormals(face.direction)
        n = tuple(face.boundary(c) for c in normals)
        return n, (-1.0 if face.amILow else 1.0)

    def _bcVals(self, face, name):
        if name.startswith("Y") and name[1:].isdigit():
            return face.qBcVals.get()[:, :, 5 + int(name[1:])]
        return face.qBcVals.get()[:, :, self._bcIndex[name]]

    def species(self, face, rule):
        """apply a rule to every species the case actually carries"""
        for n in range(self.blk.ns - 1):
            rule(face, f"Y{n}")

    # every rule checks all halo layers; `where` restricts the check to a
    # mask over the face plane

    def mirror(self, face, name, where=None, sign=1.0):
        """halo = +- the first interior cell"""
        a = self.q(name)
        for h in face.halo(a):
            self._close(h, sign * face.interior(a)[0], where)

    def negate(self, face, name, where=None):
        self.mirror(face, name, where, sign=-1.0)

    def extrapolate(self, face, name, where=None, lo=None, hi=None):
        """halo = linear extrapolation through the first two interior cells"""
        a = self.q(name)
        first = face.interior(a)[0]
        for h, deeper in zip(face.halo(a), face.interior(a)[1:]):
            want = 2.0 * first - deeper
            if lo is not None or hi is not None:
                want = np.clip(
                    want,
                    lo if lo is not None else -np.inf,
                    hi if hi is not None else np.inf,
                )
            self._close(h, want, where)

    def imposed(self, face, name, where=None):
        """halo = the value set on the face"""
        a = self.q(name)
        for h in face.halo(a):
            self._close(h, self._bcVals(face, name), where)

    def straddles(self, face, name, where=None):
        """halo = 2 * the face value - interior, so the face sits at the value"""
        a = self.q(name)
        for h in face.halo(a):
            self._close(h, 2.0 * self._bcVals(face, name) - face.interior(a)[0], where)

    def reflect(self, face, where=None):
        """velocity mirrored about the face plane, so it carries no normal flux"""
        n, _ = self.normals(face)
        velo = [self.q(c) for c in "uvw"]
        first = [face.interior(c)[0] for c in velo]
        uDotn = sum(f * ni for f, ni in zip(first, n))
        for g in range(self.blk.ng):
            for c, f, ni in zip(velo, first, n):
                self._close(face.halo(c)[g], f - 2.0 * uDotn * ni, where)

    def alongNormal(self, face):
        """the halo velocity lies entirely along the face normal"""
        n, _ = self.normals(face)
        velo = [self.q(c) for c in "uvw"]
        for g in range(self.blk.ng):
            Vb = sum(face.halo(c)[g] * ni for c, ni in zip(velo, n))
            for c, ni in zip(velo, n):
                self._close(face.halo(c)[g], Vb * ni)

    def _gradients(self, face, rules):
        """the first halo layer of every gradient, one rule per variable"""
        d = self.host["grads"]
        h, first = face.halo(d)[0], face.interior(d)[0]
        for name, rule in rules.items():
            sl = self._gradSlice[name]
            self._close(h[..., sl, :], self._sign[rule] * first[..., sl, :])

    @staticmethod
    def _close(got, want, where=None):
        if where is not None:
            got, want = got[where], want[where]
        assert np.allclose(got, want)
