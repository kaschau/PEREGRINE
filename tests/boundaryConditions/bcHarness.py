import numpy as np

from .bcBlock import create


class BcCase:
    """One boundary condition, applied to every face of a single block."""

    bcType = None
    # what each variable's gradient does in the first halo
    grads = {"all": "neumann"}

    # which arrays a stage writes, and so what has to come back from the device
    _pull = {
        "euler": ["q"],
        "preDqDxyz": ["q"],
        "postDqDxyz": ["dqdx", "dqdy", "dqdz"],
    }
    # the slices of q that share a rule
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
    # "negate" makes the gradient average to zero on the face, "neumann"
    # carries it over
    _sign = {"negate": -1.0, "neumann": 1.0}
    # the index into a face's qBcVals holding each variable
    _bcIndex = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}

    def __init__(self, adv, spdata):
        self.mb = create(self.bcType, adv, spdata)
        self.blk = self.mb[0]

    def check(self):
        for face in self.blk.faces:
            self.euler(face)
            self.viscous(face)
            self.run(face, "postDqDxyz")
            rules = dict(self.grads)
            if self.blk.ns == 1:
                rules.pop("Y", None)
            self._gradients(face, rules)

    def euler(self, face):
        raise NotImplementedError

    def viscous(self, face):
        """most bcs have nothing to say before the gradients are taken"""
        pass

    def run(self, face, stage):
        face.bcFunc(self.blk, face, self.mb.eos, self.mb.thtrdat, stage, self.mb.tme)
        self.blk.updateHostView(self._pull[stage])

    def q(self, name):
        """the q slice a variable name refers to; a species is "Y3" and so on"""
        if name.startswith("Y") and name[1:].isdigit():
            return self.blk.array["q"][:, :, :, 5 + int(name[1:])]
        return self.blk.array["q"][:, :, :, self._slice[name]]

    def normals(self, face):
        """the unit normal of the face plane, and the sign that makes it point
        out of the block (faces 1, 3, 5 store the inward normal)"""
        d = {1: "i", 2: "i", 3: "j", 4: "j", 5: "k", 6: "k"}[face.nface]
        n = tuple(self.blk.array[f"{d}n{c}"][face.s1_] for c in "xyz")
        return n, (-1.0 if face.nface in (1, 3, 5) else 1.0)

    def _bcVals(self, face, name):
        if name.startswith("Y") and name[1:].isdigit():
            return face.array["qBcVals"][:, :, 5 + int(name[1:])]
        return face.array["qBcVals"][:, :, self._bcIndex[name]]

    def species(self, face, rule):
        """apply a rule to every species the case actually carries"""
        for n in range(self.blk.ns - 1):
            rule(face, f"Y{n}")

    # every rule checks all halo layers; `where` restricts the check to a
    # mask over the face plane

    def mirror(self, face, name, where=None, sign=1.0):
        """halo = +- the first interior cell"""
        a = self.q(name)
        for s0_ in face.s0_:
            self._close(a[s0_], sign * a[face.s1_], where)

    def negate(self, face, name, where=None):
        self.mirror(face, name, where, sign=-1.0)

    def extrapolate(self, face, name, where=None, lo=None, hi=None):
        """halo = linear extrapolation through the first two interior cells"""
        a = self.q(name)
        for s0_, s2_ in zip(face.s0_, face.s2_):
            want = 2.0 * a[face.s1_] - a[s2_]
            if lo is not None or hi is not None:
                want = np.clip(
                    want,
                    lo if lo is not None else -np.inf,
                    hi if hi is not None else np.inf,
                )
            self._close(a[s0_], want, where)

    def imposed(self, face, name, where=None):
        """halo = the value set on the face"""
        a = self.q(name)
        for s0_ in face.s0_:
            self._close(a[s0_], self._bcVals(face, name), where)

    def straddles(self, face, name, where=None):
        """halo = 2 * the face value - interior, so the face sits at the value"""
        a = self.q(name)
        for s0_ in face.s0_:
            self._close(a[s0_], 2.0 * self._bcVals(face, name) - a[face.s1_], where)

    def reflect(self, face, where=None):
        """velocity mirrored about the face plane, so it carries no normal flux"""
        n, _ = self.normals(face)
        velo = [self.q(c) for c in "uvw"]
        uDotn = sum(c[face.s1_] * ni for c, ni in zip(velo, n))
        for s0_ in face.s0_:
            for c, ni in zip(velo, n):
                self._close(c[s0_], c[face.s1_] - 2.0 * uDotn * ni, where)

    def alongNormal(self, face):
        """the halo velocity lies entirely along the face normal"""
        n, _ = self.normals(face)
        velo = [self.q(c) for c in "uvw"]
        for s0_ in face.s0_:
            Vb = sum(c[s0_] * ni for c, ni in zip(velo, n))
            for c, ni in zip(velo, n):
                self._close(c[s0_], Vb * ni)

    def _gradients(self, face, rules):
        """the first halo layer of every gradient, one rule per variable"""
        s0_ = face.s0_[0]
        for d in (self.blk.array[f"dqd{c}"] for c in "xyz"):
            for name, rule in rules.items():
                sl = self._slice[name]
                self._close(d[s0_][..., sl], self._sign[rule] * d[face.s1_][..., sl])

    @staticmethod
    def _close(got, want, where=None):
        if where is not None:
            got, want = got[where], want[where]
        assert np.allclose(got, want)
