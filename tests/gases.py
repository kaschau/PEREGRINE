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


def configure(config, gas):
    """The mcPhysics section for a named gas: constants for air, a refit
    thermally perfect gas with kinetic theory transport for a mechanism."""
    config["mcPhysics"]["mixture"] = gases[gas]
    if gas == "air":
        config["mcPhysics"]["eos"] = "cpg"
        config["mcPhysics"]["trans"] = "constantProps"
    else:
        config["mcPhysics"]["eos"] = "tpg"
        config["mcPhysics"]["trans"] = "kineticTheory"
        config["mcPhysics"]["Trange"] = (300.0, 3500.0)
