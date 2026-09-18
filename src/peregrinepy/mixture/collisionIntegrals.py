"""The Monchick-Mason collision integrals, tabulated against the reduced
temperature and the reduced dipole, read at any (T*, delta): a cubic
spline along ln T* in each dipole column and a quadratic across the three
nearest columns, which is how Cantera reads the same tables."""

import numpy as np

from . import database


class CollisionIntegrals:
    """Omega22* and A* over the tables' range."""

    def __init__(self):
        ci = database("collisionIntegrals")
        self.delta = np.array(ci["delta"], dtype=np.float64)
        self.omega22 = self._columns(ci["tstar22"], ci["omega22"])
        self.astar = self._columns(ci["tstar"], ci["astar"])

    @staticmethod
    def _columns(tstar, table):
        """The natural cubic splines of a table's columns in ln T*: the
        knots, the values, and each column's second derivatives there."""
        # a table's row at T* = 0 is its limit, off the log axis
        keep = np.array(tstar, dtype=np.float64) > 0.0
        x = np.log(np.array(tstar, dtype=np.float64)[keep])
        y = np.array(table, dtype=np.float64)[keep]
        n = len(x)
        h = np.diff(x)
        # the tridiagonal system for the second derivatives, natural ends
        A = np.zeros((n, n))
        rhs = np.zeros((n, y.shape[1]))
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        return x, y, np.linalg.solve(A, rhs)

    @staticmethod
    def _along(column, x, cols):
        """A column spline at x, in the columns :cols: (both arrays)."""
        knots, y, m = column
        i = np.clip(np.searchsorted(knots, x) - 1, 0, len(knots) - 2)
        h = knots[i + 1] - knots[i]
        a, b = (knots[i + 1] - x) / h, (x - knots[i]) / h
        return (
            a * y[i, cols]
            + b * y[i + 1, cols]
            + ((a**3 - a) * m[i, cols] + (b**3 - b) * m[i + 1, cols]) * h**2 / 6.0
        )

    def _at(self, column, tstar, delta):
        """One integral at (T*, delta) arrays of one shape: the three nearest
        dipole columns' splines, then the quadratic through them."""
        x = np.log(np.asarray(tstar, dtype=np.float64))
        delta = np.asarray(delta, dtype=np.float64)
        d = self.delta
        # the three columns about delta, kept inside the table
        j = np.clip(np.searchsorted(d, delta) - 1, 0, len(d) - 3)
        d0, d1, d2 = d[j], d[j + 1], d[j + 2]
        f0, f1, f2 = (self._along(column, x, j + k) for k in range(3))
        return (
            f0 * (delta - d1) * (delta - d2) / ((d0 - d1) * (d0 - d2))
            + f1 * (delta - d0) * (delta - d2) / ((d1 - d0) * (d1 - d2))
            + f2 * (delta - d0) * (delta - d1) / ((d2 - d0) * (d2 - d1))
        )

    def omega22At(self, tstar, delta):
        return self._at(self.omega22, tstar, delta)

    def astarAt(self, tstar, delta):
        return self._at(self.astar, tstar, delta)
