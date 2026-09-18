"""The chemistry as a source: the explicit rates, and the substepped
integration that lands the state within the species' bounds -- one
substep within the fastest species' way is the explicit source to the bit;
past it the step is cut and every mass fraction stays in [0, 1]."""

from pathlib import Path

import cantera as ct
import numpy as np
import pytest

from .cases import at, cell

mechanism = "CH4_O2_FFCMY.yaml"
ct.add_directory(
    str(Path(__file__).parent / "../../src/peregrinepy/mixture/database/mechanisms")
)


def stoichiometric():
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(1.0, "CH4", "O2")
    return gas.Y


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
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(1.0, "CH4", "O2")
    gas.TP = T0, p0
    Y = gas.Y
    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])
    t, theirs = 0.0, None
    while theirs is None:
        t = net.step()
        if reactor.T > T0 + 400:
            theirs = t
    dt = 1e-8
    mb = cell(mechanism, "substepped", p0, T0, Y, dt=dt, maxSubSteps=10)
    ours = None
    while ours is None and mb.tme < 4 * theirs:
        mb.integrator.step(dt)
        if at(mb, "q")[1] > T0 + 400:
            ours = mb.tme
    assert ours is not None and abs(ours / theirs - 1.0) < 0.01
