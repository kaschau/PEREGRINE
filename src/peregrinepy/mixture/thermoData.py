"""The formats a species' thermodynamic data can arrive in, each knowing how
to read the cp curve and the h and s references out of its own key."""

import numpy as np

from . import Ru, Tref
from ..misc import subclasses


class BaseThermoData:
    """One format a species' thermodynamic data can arrive in. A format is
    found by being a subclass here, so adding one is adding a class."""

    # the species key it reads
    name = None

    @classmethod
    def findReferenceFormat(cls, sp):
        """The format of the thermodynamic data this species carries."""
        return next(f for f in subclasses(cls) if f.name in sp)

    @staticmethod
    def cp(sp):
        """(T, cp/R) over the data's own range."""
        raise NotImplementedError

    @staticmethod
    def hs(sp):
        """h(Tref)/R and s(Tref)/R, the constants of integration."""
        raise NotImplementedError


class Janaf(BaseThermoData):
    """A NIST-JANAF table: cp on a temperature grid, and the two constants."""

    name = "janaf"

    @staticmethod
    def cp(sp):
        j = sp["janaf"]
        return np.array(j["T"], float), np.array(j["cp"], float) / Ru

    @staticmethod
    def hs(sp):
        j = sp["janaf"]
        return j["dHf298"] / Ru, j["s298"] / Ru


class Nasa7(BaseThermoData):
    """A NASA7 fit: two ranges of a0..a6 with a switch temperature, sampled
    over the range it is good for."""

    name = "NASA7"

    @classmethod
    def cp(cls, sp):
        n7 = sp["NASA7"]
        T = np.linspace(n7["Trange"][0], n7["Trange"][-1], 200)
        return T, np.array([cls.cpR(n7, t) for t in T])

    @classmethod
    def hs(cls, sp):
        n7 = sp["NASA7"]
        return cls.hRT(n7, Tref) * Tref, cls.sR(n7, Tref)

    @staticmethod
    def coefficients(n7, T):
        """a0..a6 of whichever range holds T."""
        return n7["low"] if T <= n7["Trange"][1] else n7["high"]

    @classmethod
    def cpR(cls, n7, T):
        a = cls.coefficients(n7, T)
        return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])))

    @classmethod
    def hRT(cls, n7, T):
        a = cls.coefficients(n7, T)
        return (
            a[0]
            + T * (a[1] / 2 + T * (a[2] / 3 + T * (a[3] / 4 + T * a[4] / 5)))
            + a[5] / T
        )

    @classmethod
    def sR(cls, n7, T):
        a = cls.coefficients(n7, T)
        return (
            a[0] * np.log(T)
            + T * (a[1] + T * (a[2] / 2 + T * (a[3] / 3 + T * a[4] / 4)))
            + a[6]
        )
