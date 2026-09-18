"""A plugin may add kernels to the end of a stage: the viscous sponge
scales the viscosity along a line once consistify has made the transport
properties, and changes a strained flow's answer."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.graph import LaunchNode
from peregrinepy.plugins import BasePlugin

from ..simulation.test_flux import sod


def sponge(config):
    config["plugins"]["viscousSponge"] = dict(
        origin=[0.5, 0.0, 0.0], ending=[1.0, 0.0, 0.0], multiplier=20.0
    )


def test_theSpongeFollowsConsistify():
    config = pg.files.configFile()
    sponge(config)
    (plugin,) = pg.plugins.getPlugins(config).values()
    assert plugin.values["start"] == 0.5 and plugin.values["length"] == 0.5
    assert plugin.values["nx"] == 1.0 and plugin.values["mult"] == 20.0
    (node,) = plugin.after()["consistify"]
    assert isinstance(node, LaunchNode) and node.fixed == plugin.values
    assert "viscousSponge" in plugin.declKernels()


class Nowhere(BasePlugin):
    name = "nowhere"

    def after(self):
        return {"nowhere": []}


def test_aPluginMayOnlyFollowAStage(my_setup):
    with pytest.raises(ValueError):
        sod("KEPaEC", configure=lambda c: c["plugins"].update(nowhere={}))


def test_theSpongeActsOnTheTube(my_setup):
    # Sod's tube is strained, so the sponge changes the answer and stays finite
    plain = sod("KEPaEC", physics="navierStokes")
    sponged = sod("KEPaEC", physics="navierStokes", configure=sponge)
    assert np.isfinite(sponged) and sponged != plain
