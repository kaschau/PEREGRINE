"""One polynomial in ln T over the case's temperature range: the fit, its
evaluation, and its integrals, for any model that stores a property that way."""

import numpy as np


class PolyFitMixin:
    """For a model that fits a temperature-dependent quantity as one polynomial
    in ln T over the case's range: the fit, its evaluation, and its integrals.
    Nothing here reads the host; every input is an argument."""

    @staticmethod
    def fitLowestDegree(x, y, tolerance, maxDegree, weights=None):
        """The lowest-degree polynomial in x within `tolerance` (max relative
        error) of y, or the best `maxDegree` can do: coefficients ascending,
        and the error achieved."""
        # the tolerance is relative; of no weight, 1/y and 1/y^2, this reaches it at the lowest degree
        weights = 1.0 / y**2 if weights is None else weights
        for degree in range(maxDegree + 1):
            c = np.polyfit(x, y, degree, w=weights)
            error = np.abs(np.polyval(c, x) / y - 1).max()
            if error <= tolerance:
                break
        return np.flip(c), error

    @staticmethod
    def evaluate(poly, x):
        """A polynomial from fitLowestDegree, at x."""
        return np.polyval(np.flip(poly), x)

    @staticmethod
    def integratePolyExp(P):
        """Q such that the integral of P(u) e^u du is e^u Q(u): with u = ln T,
        the integral of P dT is T Q. The constant is the caller's."""
        # by parts: Q = P - P' + P'' - ...
        Q = np.zeros(len(P))
        D = np.flip(P)
        for k in range(len(P)):
            Q[: len(D)] += (-1) ** k * np.flip(D)
            D = np.polyder(D)
        return Q

    @staticmethod
    def integratePoly(P):
        """The integral of P(u) du; the constant is the caller's."""
        return np.flip(np.polyint(np.flip(P)))
