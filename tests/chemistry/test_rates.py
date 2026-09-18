"""The composed production rates against Cantera: the forward rate
constants to rounding (thermo-free), the net rates to the tolerance the
eos's refit thermo sets for the equilibrium constants, and mass
conservation."""

from pathlib import Path

import cantera as ct
import numpy as np
import pytest

from .cases import at, cell, randomState

ct.add_directory(
    str(Path(__file__).parent / "../../src/peregrinepy/mixture/database/mechanisms")
)


@pytest.mark.parametrize(
    "mechanism,within",
    # measured: 5e-6, 4e-4 and 4e-4 of the largest rate, the refit thermo's
    # consequence on the reverse rates
    [
        ("CH4_O2_FFCMY.yaml", 1e-4),
        ("C2H4_Air_Skeletal.yaml", 2e-3),
        ("GRI30.yaml", 2e-3),
    ],
)
def test_theRatesAreCanteras(my_setup, mechanism, within):
    rng = np.random.default_rng(7)
    gas = ct.Solution(mechanism)
    mixture = None
    for trial in range(3):
        if mixture is None:
            mb = cell(
                mechanism,
                "explicit",
                10e5,
                1800.0,
                np.ones(gas.n_species) / gas.n_species,
            )
            mixture = mb.simulation.mixture
        p, T, Y = randomState(mixture, rng)
        mb = cell(mechanism, "explicit", p, T, Y)
        gas.TPY = T, p, Y
        omega = at(mb, "omega")
        theirs = gas.net_production_rates * gas.molecular_weights
        scale = np.abs(theirs).max()
        assert np.abs(omega - theirs).max() < within * scale
        # mass is conserved: the rates sum to nothing
        assert abs(omega.sum()) < 1e-12 * scale
        # the source is the carried species' rates
        assert np.array_equal(at(mb, "dQ")[5:], omega[:-1])
        # thermo-free: every plain reaction's forward rate constant to rounding
        r = mixture.reactionData()
        kept = [i for i, rxn in enumerate(mixture.reactions) if rxn["rate"][0] != 0.0]
        plain = r["type"] == 0
        ours = np.exp(r["logA"] + r["b"] * np.log(T) - r["EaR"] / T)[plain]
        assert (
            np.abs(ours / gas.forward_rate_constants[np.array(kept)][plain] - 1.0).max()
            < 1e-10
        )


def test_singleFollowsDouble(my_setup):
    # measured 1.1e-4 of the largest rate: an exponent rounded at float's
    # ulp; the log-space form is what keeps single finite at all
    rng = np.random.default_rng(11)
    mb = cell("CH4_O2_FFCMY.yaml", "explicit", 10e5, 1800.0, np.ones(12) / 12)
    p, T, Y = randomState(mb.simulation.mixture, rng)
    double = at(cell("CH4_O2_FFCMY.yaml", "explicit", p, T, Y), "omega")
    single = at(
        cell("CH4_O2_FFCMY.yaml", "explicit", p, T, Y, precision="single"), "omega"
    )
    assert np.isfinite(single).all()
    assert np.abs(single - double).max() < 1e-3 * np.abs(double).max()
