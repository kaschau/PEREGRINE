# The mixture

The gas a case solves: its species, and the models that give their
properties. The `mixture` section of the [config](config.md#mixture) names
them; this page says what each choice needs and which go together.

## Where the species come from

`mixture.species` is one of three things.

**A Cantera mechanism.** A yaml file, given as a path or as the name of one
shipped in [mixture/database/mechanisms](../src/peregrinepy/mixture/database/mechanisms):
`C2H4_Air_Red22.yaml`, `C2H4_Air_Skeletal.yaml`, `CH4_O2_FFCMY.yaml`,
`GRI30.yaml`. The file is read without Cantera installed. It brings the
species in its own order, their thermodynamic and transport data, and the
reactions, which is the only way a case gets chemistry.

```yaml
mixture:
  species: CH4_O2_FFCMY.yaml
  eos: tpg
  Trange: [300.0, 3500.0]
```

**A list of library names.** The species library,
[speciesLibrary.yaml](../src/peregrinepy/mixture/database/speciesLibrary.yaml),
carries curated, cited data for every species PEREGRINE knows: NIST-JANAF
thermodynamics, Lennard-Jones transport parameters, a critical point. Name
the ones the case has, in the order wanted.

```yaml
mixture:
  species: [O2, N2]
  eos: tpg
  trans: kineticTheory
  Trange: [200.0, 1000.0]
```

**A dict of name to data.** A gas stated in full, for one the library does
not carry: a single-component air, say. The data are whatever the models
need.

```python
config["mixture"]["species"] = {
    "Air": {"MW": 28.97, "cp0": 1002.8, "mu0": 1.859e-05, "kappa0": 0.02625}
}
```

Data are taken in this order of precedence: what the case states, then what
the mechanism file carries, then the library. Nothing overrides what came
before it. A species that can answer for its composition (`comp`, a dict of
element to count) needs no molecular weight.

Everywhere in PEREGRINE the species are in this order, and the last one is
what the others leave: it is not a solved variable, and a result writes it
as one minus the rest.

## The models

Each model declares what it needs from every species, and the mixture
refuses a species that cannot answer, by name and quantity. Units are SI
with kmol: molecular weights in kg/kmol, energies in J/kmol.

### Equation of state, `eos`

| name | what it is | needs per species |
|---|---|---|
| `cpg` | Calorically perfect: one cp per species, so enthalpy and entropy are closed form. | `MW` or `comp`; `cp0` in J/kg/K; for a reacting case `dHf298` and `s298`. |
| `tpg` | Thermally perfect: cp against temperature refit from the species' data. | `MW` or `comp`; thermodynamic data in one of the known formats: a JANAF table (`janaf`) or a NASA7 polynomial pair (`NASA7`). |
| `realGas` | A cubic equation of state, with the thermally perfect gas as its ideal reference. | Everything `tpg` needs, and a critical point: `Tcrit`, `pcrit`, `Vcrit`, `acentric`, or `well` and `diam` to derive one from. |

### Transport, `trans`

Momentum and heat: each species' viscosity and conductivity. A viscous case
picks one; an Euler case has none.

| name | what it is | needs per species | goes with |
|---|---|---|---|
| `constantProps` | One viscosity and one conductivity per species, stated by the case. | `mu0` in Pa s, `kappa0` in W/m/K. | any eos |
| `kineticTheory` | Chapman-Enskog from Lennard-Jones parameters, refit against temperature. | `well`, `diam`, `dipole`, `polarize`, `zrot`, `geometry` | `tpg` or `realGas` |
| `chungDenseGas` | Chung's high-pressure correlation, written around the critical point. | `dipole`, and the critical point | `realGas` |

### Species diffusion, `diffusion`

A separate choice from how momentum and heat diffuse, and only made in a
viscous case.

| name | what it is | needs per species | goes with |
|---|---|---|---|
| `lewis` | D = kappa / (rho cp Le), with a Lewis number of one unless the species states `lewis`. | nothing more | any transport model |
| `binary` | Every pair's diffusion coefficient from the collision integrals, refit against temperature. | `well`, `diam`, `dipole`, `polarize` | `kineticTheory` |

### Mixing rule, `mixingRule`

How the species' viscosities and conductivities combine into the mixture's:
`wilke` or `herning`. It is a switch baked into the transport kernel and
needs nothing of the species.

## The fits

Every temperature-dependent property the kernels read is a polynomial in
ln T over `Trange`, refit from the species' data. Each refit takes the lowest
degree whose relative error is within `reFitTol`, or the best it can do at
`reFitMaxDegree`. The range is required whenever a fit is made, which is for
`tpg`, `realGas`, `kineticTheory` and `binary`; a calorically perfect gas with
constant transport properties has no fit and needs none. Where the range
reaches past the source data, the data is extended.

The fit's own error is kept, so a check of the equation of state against
its source data can ask for it. The tests set the range to what they need;
there is no clipping to a range in the kernels.

## Chemistry

Reactions come from the mechanism: elementary, three-body, Lindemann and
Troe falloff reactions, with their rates composed per reaction in log space
from tables the compiler bakes into the chemistry kernels. The
[chemistry section](config.md#chemistry) of the config chooses how the
source enters the right-hand side. A mixture given as a list or a dict has no
reactions, and a chemistry source on it is refused.

## Reading the mixture from a script

The mixture a case built is on the solver, and can be made on its own from a
config for a look at what it holds:

```python
mb = pg.multiBlock.solver(config, mesh)
m = mb.simulator.mixture
m.speciesNames, m.ns, m.eos.name, m.species["O2"]["MW"]

from peregrinepy.mixture import getMixture
m = getMixture(config)
```
