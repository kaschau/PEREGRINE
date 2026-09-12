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


def cfgsect(mixture, **choices):
    """An mcPhysics section as a case would write it: the config's defaults
    with the case's choices over them."""
    sect = dict(pg.files.configFile()["mcPhysics"])
    sect.update({"mixture": mixture, "Trange": T, **choices})
    return sect


def test_speciesListFromLibrary():
    m = Mixture(cfgsect(air, eos="tpg"))
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
        Mixture(cfgsect(air, eos="notAnEos"))
    with pytest.raises(KeyError):
        Mixture(cfgsect(air, eos="tpg", trans="notATransport"))


def test_missingPropertyNamesTheSpecies():
    """A species the library cannot complete fails by name, not by
    AttributeError somewhere downstream."""
    m = Mixture(cfgsect(air, eos="tpg"))
    broken = {k: Species(k, dict(sp.data)) for k, sp in m.species.items()}
    broken["N2"].data.pop("janaf")
    with pytest.raises(ValueError, match="janaf|NASA7 for N2"):
        subclassWhere(BaseEosModel, name="tpg")(cfgsect(air)).check(broken)


def test_mechanismWinsOverLibrary():
    """A property the mechanism gives shadows the library's."""
    m = Mixture(cfgsect("GRI30.yaml", eos="tpg"))
    lib = Species.referenceData()
    gri = CanteraParser(mechanismPath("GRI30.yaml")).species
    # GRI30 carries its own Lennard-Jones well for H2O, and it differs from
    # the library's; the mechanism's is the one on the species
    assert gri["H2O"]["well"] != lib["H2O"]["well"]
    assert m.species["H2O"]["well"] == gri["H2O"]["well"]
