"""
PEREGRINE's Cantera reader against Cantera itself.

The reader exists so a mechanism can be read directly rather than translated
ahead of time, which is only worth doing if it agrees with Cantera to the last
bit. Cantera is the oracle here and nowhere else -- the solver does not import
it.
"""

from pathlib import Path

import cantera as ct
import numpy as np
import pytest

from peregrinepy.mixture.cantera import CanteraParser
from peregrinepy.mixture import Ru, kB

mechanisms = ("GRI30.yaml", "CH4_O2_FFCMY.yaml", "C2H4_Air_Skeletal.yaml")


def mechPath(name):
    import peregrinepy.mixture as mix

    return str(Path(mix.__file__).parent / "database" / "mechanisms" / name)


@pytest.fixture(scope="module", params=mechanisms)
def mech(request):
    path = mechPath(request.param)
    parsed = CanteraParser(path)
    ours, reactions = parsed.species, parsed.reactions
    return ours, reactions, ct.Solution(path), request.param


def test_speciesAndReactionCounts(mech):
    ours, reactions, gas, name = mech
    assert list(ours) == list(gas.species_names)
    assert len(reactions) == gas.n_reactions


def test_speciesProperties(mech):
    """Every per-species number the kernels read."""
    ours, _, gas, _ = mech
    for i, name in enumerate(gas.species_names):
        o, s = ours[name], gas.species(i)
        assert o["MW"] == pytest.approx(s.molecular_weight, rel=1e-14)
        # cantera's coeffs are [Tswitch, high a0..a6, low a0..a6]; ours keep
        # the range the fit is good for as well
        c = np.asarray(s.thermo.coeffs)
        n7 = o["NASA7"]
        assert n7["Trange"] == [s.thermo.min_temp, c[0], s.thermo.max_temp]
        assert np.array_equal(np.array(n7["high"]), c[1:8])
        assert np.array_equal(np.array(n7["low"]), c[8:15])
        tr = s.transport
        assert o["well"] / kB == pytest.approx(tr.well_depth / kB, rel=1e-14)
        assert o["diam"] == pytest.approx(tr.diameter, rel=1e-14)
        assert o["dipole"] == pytest.approx(tr.dipole, rel=1e-14, abs=1e-40)
        assert o["polarize"] == pytest.approx(tr.polarizability, rel=1e-14, abs=1e-40)
        assert o["zrot"] == pytest.approx(
            tr.rotational_relaxation, rel=1e-14, abs=1e-40
        )


def test_rateConstants(mech):
    """A, b and Ea of every reaction, including both branches of a falloff."""
    ours, reactions, gas, _ = mech
    for i, o in enumerate(reactions):
        rate = gas.reaction(i).rate
        high = getattr(rate, "high_rate", rate)
        assert o["rate"][0] == pytest.approx(high.pre_exponential_factor, rel=1e-13)
        assert o["rate"][1] == pytest.approx(high.temperature_exponent, rel=1e-13)
        assert o["rate"][2] * Ru == pytest.approx(
            high.activation_energy, rel=1e-13, abs=1e-9
        )
        if "lowRate" in o:
            low = rate.low_rate
            assert o["lowRate"][0] == pytest.approx(
                low.pre_exponential_factor, rel=1e-13
            )
            assert o["lowRate"][1] == pytest.approx(low.temperature_exponent, rel=1e-13)
            assert o["lowRate"][2] * Ru == pytest.approx(
                low.activation_energy, rel=1e-13, abs=1e-9
            )


def test_netStoichiometry(mech):
    """What each reaction does to each species. Cantera reports this already
    cancelled, so a chaperone appearing on both sides drops out of it."""
    ours, reactions, gas, _ = mech
    names = gas.species_names
    reactant = gas.reactant_stoich_coeffs
    net = gas.product_stoich_coeffs - reactant
    for i, o in enumerate(reactions):
        vec = np.zeros(len(names))
        for sp, v in o["reactants"].items():
            vec[names.index(sp)] -= v
        for sp, v in o["products"].items():
            vec[names.index(sp)] += v
        assert np.allclose(vec, net[:, i]), o["equation"]


def test_forwardRatesOfProgress(mech):
    """The reaction orders, checked where it counts rather than against
    Cantera's stoichiometry matrix -- that matrix is the cancelled one, so for
    `H + O2 + H2O <=> HO2 + H2O` it says nothing about the H2O the rate is
    first order in. Reproducing Cantera's own rate of progress is the test
    that catches a dropped chaperone, and it is off by ~2000x when it is
    wrong."""
    ours, reactions, gas, _ = mech
    gas.TPX = 1400.0, 3.0 * ct.one_atm, {n: 1.0 for n in gas.species_names}
    T, c = gas.T, gas.concentrations
    names = gas.species_names
    rop = gas.forward_rates_of_progress
    for i, o in enumerate(reactions):
        # the falloff and third-body forms carry a pressure dependence this
        # simple evaluation does not reproduce
        if o["type"] != "elementary" or o.get("efficiencies"):
            continue
        A, b, EaOverRu = o["rate"]
        k = A * T**b * np.exp(-EaOverRu / T)
        mine = k * np.prod([c[names.index(s)] ** v for s, v in o["fwd"].items()])
        assert mine == pytest.approx(rop[i], rel=1e-10), o["equation"]


def test_reversibilityAndEfficiencies(mech):
    ours, reactions, gas, _ = mech
    for i, o in enumerate(reactions):
        rxn = gas.reaction(i)
        assert o["reversible"] == rxn.reversible, o["equation"]
        theirs = dict(getattr(rxn, "efficiencies", {}) or {})
        if theirs:
            mine = {k: float(v) for k, v in o.get("efficiencies", {}).items()}
            assert mine == pytest.approx(theirs), o["equation"]
