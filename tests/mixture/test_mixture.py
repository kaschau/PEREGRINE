"""
The Mixture class: model selection, what each model needs, and the tables it
hands the kernels.
"""

import pytest

import peregrinepy as pg

from peregrinepy.mixture import BaseEosModel, BaseTransportModel, Mixture, Species
from peregrinepy.misc import subclassWhere
from peregrinepy.mixture.cantera import CanteraParser

air = ["N2", "O2", "AR"]


def mechanismPath(name):
    import peregrinepy.mixture as mix
    from pathlib import Path

    return str(Path(mix.__file__).parent / "database" / "mechanisms" / name)


# the range every fit in these tests is made over
T = (300.0, 3000.0)


def configSect(mixture, **choices):
    """A simulation section as a case would write it: the config's defaults
    with the case's choices over them."""
    sect = dict(pg.files.configFile()["simulation"])
    sect.update({"mixture": mixture, "Trange": T, **choices})
    return sect


def test_speciesListFromLibrary():
    m = Mixture(configSect(air, eos="tpg"))
    assert m.speciesNames == air
    assert m.ns == 3
    assert m.reactions == []
    # molecular weight is derived from the composition, never stored
    assert m.species["N2"]["MW"] == pytest.approx(28.014, rel=1e-4)


def test_everyEosAndTransportIsSelectable():
    for name in ("cpg", "tpg", "realGas"):
        assert subclassWhere(BaseEosModel, name=name).name == name
    for name in ("constantProps", "kineticTheory", "chungDenseGas"):
        assert subclassWhere(BaseTransportModel, name=name).name == name


def test_unknownModelIsRejected():
    with pytest.raises(KeyError):
        Mixture(configSect(air, eos="notAnEos"))
    with pytest.raises(KeyError):
        Mixture(configSect(air, eos="tpg", trans="notATransport"))


def test_missingPropertyNamesTheSpecies():
    """A species the library cannot complete fails by name, not by
    AttributeError somewhere downstream."""
    m = Mixture(configSect(air, eos="tpg"))
    broken = {k: Species(k, dict(sp.data)) for k, sp in m.species.items()}
    broken["N2"].data.pop("janaf")
    with pytest.raises(ValueError, match="janaf|NASA7 for N2"):
        subclassWhere(BaseEosModel, name="tpg")(configSect(air)).check(broken)


def test_mechanismWinsOverLibrary():
    """A property the mechanism gives shadows the library's."""
    m = Mixture(configSect("GRI30.yaml", eos="tpg"))
    lib = Species.referenceData()
    gri = CanteraParser(mechanismPath("GRI30.yaml")).species
    # GRI30 carries its own Lennard-Jones well for H2O, and it differs from
    # the library's; the mechanism's is the one on the species
    assert gri["H2O"]["well"] != lib["H2O"]["well"]
    assert m.species["H2O"]["well"] == gri["H2O"]["well"]


def test_aReactingMixtureNeedsReactions():
    from peregrinepy.mixture import ReactingMixture, getMixture

    sect = configSect("GRI30.yaml", eos="tpg")
    sect["chemistry"] = "explicit"
    m = getMixture(sect)
    assert isinstance(m, ReactingMixture) and "reactions" in m.tables()
    assert "reactions" not in getMixture(configSect("GRI30.yaml", eos="tpg")).tables()
    sect = configSect(air, eos="tpg")
    sect["chemistry"] = "explicit"
    with pytest.raises(ValueError):
        getMixture(sect)
