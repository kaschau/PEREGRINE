"""The chemistry as a source: the explicit rates, and the substepped
integration that lands the state within the species' bounds -- one
substep within the fastest species' way is the explicit source to the bit;
past it the step is cut and every mass fraction stays in [0, 1]."""

import numpy as np
import pytest

from .cases import at, canterasDelay, cell, mechanism, oursDelay, stoichiometric


def test_oneSubstepWithinTheBoundIsTheExplicitSource(my_setup):
    rng = np.random.default_rng(5)
    Y = rng.random(12)
    Y /= Y.sum()
    explicit = cell(mechanism, "explicit", 10e5, 1800.0, Y)
    omega, rho = at(explicit, "omega"), at(explicit, "Q")[0]
    # the fastest species' way to its bound, in time
    headroom = np.where(omega < 0, Y, 1.0 - Y)
    dt = 0.5 * (rho * headroom / np.abs(omega)).min()
    explicit = cell(mechanism, "explicit", 10e5, 1800.0, Y, dt=dt)
    substepped = cell(mechanism, "substepped", 10e5, 1800.0, Y, dt=dt, maxSubSteps=1)
    assert np.array_equal(at(explicit, "dQ")[5:], at(substepped, "dQ")[5:])


@pytest.mark.parametrize("pseudo", ["rk1", "rk3"])
def test_pointImplicitDualTimeIgnitesWhereExplicitCannot(my_setup, pseudo):
    # at dt 1e-7 the explicit rk3 step breaks (non-finite by 2.4e-6 s);
    # dual time with the source's Jacobian in its pseudo system, each
    # species' own entry and its temperature's, twenty pseudo steps a step
    # and no low-Mach preconditioner, reaches 1900 K at a measured 32.1 us
    # with rk1 and 31.8 us with rk3 pseudo steps, Cantera at 31.75 us
    # (examples/dualTimeIgnition.py has the matrix; the species entries
    # alone broke with rk1 and landed 40 to 65 percent late with the
    # preconditioner on)
    p0, T0, dt = 20e5, 1500.0, 1e-7
    Y = stoichiometric()
    theirs = canterasDelay(p0, T0, Y, 400.0)
    explicit = cell(mechanism, "explicit", p0, T0, Y, dt=dt, cells=2)
    assert oursDelay(explicit, dt, T0, 400.0, 30 * dt) is None
    assert not np.isfinite(at(explicit, "q")).all()
    implicit = cell(
        mechanism,
        "explicit",
        p0,
        T0,
        Y,
        dt=dt,
        cells=2,
        integrator="dualTime",
        pseudoIntegrator=pseudo,
        subIterations=20,
        chemistryJacobian="diagonal",
        lowMach=False,
    )
    ours = oursDelay(implicit, dt, T0, 400.0, 4 * theirs)
    assert ours is not None and abs(ours / theirs - 1.0) < 0.05


@pytest.mark.parametrize("maxSubSteps", [1, 10, 50])
def test_substepsKeepTheStateInBounds(my_setup, maxSubSteps):
    # an ignited mixture over a step far past its chemistry: the explicit
    # source leaves [0, 1] by a lot, the substepped one never
    Y, dt = stoichiometric(), 1e-5
    explicit = cell(mechanism, "explicit", 10e5, 2500.0, Y, dt=dt)
    Q, dQ = at(explicit, "Q"), at(explicit, "dQ")
    after = (Q[5:] + dQ[5:] * dt) / Q[0]
    assert after.min() < -0.1 or after.max() > 1.1
    substepped = cell(
        mechanism, "substepped", 10e5, 2500.0, Y, dt=dt, maxSubSteps=maxSubSteps
    )
    Q, dQ = at(substepped, "Q"), at(substepped, "dQ")
    after = (Q[5:] + dQ[5:] * dt) / Q[0]
    assert np.isfinite(dQ).all()
    assert after.min() > -1e-12 and after.sum() < 1.0 + 1e-12


def test_ignitionFollowsCantera(my_setup):
    # a constant-volume ignition of the stoichiometric mixture at 1500 K
    # and 20 bar: the time to T0 + 400 K against Cantera's reactor
    # (measured: 3.180e-5 s substepped at dt 1e-8, Cantera 3.175e-5 s;
    # explicit at that step is NaN by 2.4e-6 s, and at dt 1e-7 the ten
    # substep cap throttles the source and ignition lands at 7.6e-5 s)
    p0, T0 = 20e5, 1500.0
    theirs = canterasDelay(p0, T0, stoichiometric(), 400.0)
    dt = 1e-8
    mb = cell(mechanism, "substepped", p0, T0, stoichiometric(), dt=dt, maxSubSteps=10)
    ours = oursDelay(mb, dt, T0, 400.0, 4 * theirs)
    assert ours is not None and abs(ours / theirs - 1.0) < 0.01
