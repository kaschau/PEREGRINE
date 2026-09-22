"""Gases for tests that need one but are not about it."""

# constants only, nothing from the library
air = {
    "Air": {
        "MW": 28.96,
        "cp0": 1005.0,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}

gases = {"air": air, "CH4_O2": "CH4_O2_FFCMY.yaml"}


def configure(config, gas, simulator="navierStokes"):
    """Fills the mixture section for a named gas -- constants for air, a
    refit thermally perfect gas with kinetic theory transport for a
    mechanism -- and names the simulator."""
    config["simulation"]["simulator"] = simulator
    mixture = config["mixture"]
    mixture["species"] = gases[gas]
    if gas == "air":
        mixture["eos"] = "cpg"
        mixture["trans"] = "constantProps"
    else:
        mixture["eos"] = "tpg"
        mixture["trans"] = "kineticTheory"
        mixture["Trange"] = (300.0, 3500.0)


def primitives(mb, blk):
    """The primitive vector of a block as one host array over every cell,
    (i, j, k, primVars), as the case derives it from the state: what a test
    fills and hands to setPrimitives, and reads back to check."""
    import numpy as np

    data = mb.exportData(blk, mb.primVars)
    return np.stack([data[name] for name in mb.primVars], axis=-1)
