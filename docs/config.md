# The config file

A case is described by one config: in a script, a `pg.files.configFile()`
whose entries you set; in executable mode, a yaml file with the same
sections, read over the defaults by `pg.readers.readConfigFile`. Every
section and key below exists in the defaults, so a misspelled key is refused
when it is set, and a missing one takes its default. The defaults live in
[configFile.py](../src/peregrinepy/files/configFile.py).

A complete file, with every default:

```yaml
simulation:
  simulator: navierStokes   # euler or navierStokes
  precision: double         # double or single
  niter: 1                  # steps a run takes
  controller: fixed         # fixed or cfl
  dt: 1.0e-3                # s: every step, or the first
  maxDt: 1.0e-3             # s: the cfl controller's ceiling
  maxCFL: 0.1               # the cfl controller's target

mixture:
  species: null             # a mechanism file, a list of library names, or a dict
  eos: cpg                  # cpg, tpg or realGas
  trans: null               # constantProps, kineticTheory or chungDenseGas
  diffusion: lewis          # lewis or binary
  mixingRule: wilke         # wilke or herning
  Trange: null              # [Tlow, Thigh] every fit is made over
  reFitTol: 1.0e-3          # relative error a refit polynomial meets
  reFitMaxDegree: 6         # and the degree it may go to

chemistry:
  source: null              # null, explicit or substepped
  maxSubSteps: 200
  entropyBisections: 12

initialConditions:
  p: 101325.0
  u: 0.0
  v: 0.0
  w: 0.0
  T: 300.0
  Y: {}                     # mass fraction by species name

timeIntegration:
  integrator: rk3           # rk1, rk2, rk3, rk34, rk4, maccormack or dualTime
  pseudoIntegrator: rk3     # dual time: any Runge-Kutta integrator
  subIterations: 20         # dual time: pseudo steps per physical step
  lowMach: true             # dual time: the low-Mach preconditioner
  chemistryJacobian: null   # dual time: null or diagonal

RHS:
  primaryAdvFlux: KEPaEC
  secondaryAdvFlux: null
  switchAdvFlux: null
  switchValues: {}

backend-serial:  {tileSize: 128, tileElements: 1024}
backend-openmp:  {tileSize: 128, tileElements: 1024}
backend-cuda:    {tileSize: 128, tileElements: 1024, launchThreads: 256, launchWaves: 1}
backend-hip:     {tileSize: 128, tileElements: 1024, launchThreads: 256, launchWaves: 2}

haloExchange:
  kind: hostStaged          # hostStaged or device

bcValues: {}                # by boundary name; see boundaryConditions.md
plugins: {}                 # by plugin name; see plugins.md
```

## simulation

What is simulated, in what precision, for how long, and how each step is
sized.

| key | default | what it does |
|---|---|---|
| `simulator` | `navierStokes` | The equations solved. `euler` is inviscid: no transport model, no gradients, no diffusive flux. `navierStokes` adds the transport properties, the gradients of velocity, temperature and mass fractions, and the diffusive flux, and needs a transport model in `mixture.trans`. |
| `precision` | `double` | What every array and kernel value is: `double` or `single`. The kernels are compiled for it. Results are written in the writer plugin's own precision, single unless it says otherwise. |
| `niter` | `1` | How many steps `run()` takes. A script stepping by hand ignores it. |
| `controller` | `fixed` | How each step is sized. `fixed` takes `dt` every step. `cfl` takes the largest step `maxCFL` allows, from the largest acoustic plus convective speed on any rank, capped at `maxDt`. Dual time takes `fixed` only. |
| `dt` | `1e-3` | The step, in seconds, for the fixed controller. Cast to a float when the config is validated, so `1e-7` in yaml is fine. |
| `maxDt` | `1e-3` | The cfl controller's ceiling, in seconds. |
| `maxCFL` | `0.1` | The cfl controller's target CFL number, on the combined acoustic and convective speed. |

## mixture

The gas: its species, and the models that give their properties. The
[mixture page](mixture.md) says what each choice needs of a species and
which choices go together.

| key | default | what it does |
|---|---|---|
| `species` | `None` | Where the species come from. A Cantera yaml mechanism (a path, or the name of one shipped in `mixture/database/mechanisms`), which also brings the reactions; a list of names from the species library; or a dict of name to data, for a gas stated in full. The order is the species order everywhere, and the last species takes what the others leave. |
| `eos` | `cpg` | The equation of state: `cpg` calorically perfect, `tpg` thermally perfect, `realGas` cubic with the thermally perfect gas as its ideal reference. |
| `trans` | `None` | How momentum and heat diffuse: each species' viscosity and conductivity by `constantProps`, `kineticTheory` (Chapman-Enskog from Lennard-Jones parameters) or `chungDenseGas` (Chung's high-pressure correlation). A viscous case picks one; an Euler case leaves it `None`. |
| `diffusion` | `lewis` | How species diffuse, only in a viscous case: `lewis` gives each species a diffusivity from the conductivity and a Lewis number, one unless the species states its own; `binary` fits every pair's coefficient from the collision integrals and needs `kineticTheory`. |
| `mixingRule` | `wilke` | How the species' viscosities and conductivities are mixed: `wilke` or `herning`. Baked into the transport kernel. |
| `Trange` | `None` | `[Tlow, Thigh]`, the temperature range every temperature-dependent property is fit over. Required whenever a property is fit: `tpg`, `realGas`, `kineticTheory` and `binary`. The source data is extended past its ends where the range goes further. |
| `reFitTol` | `1e-3` | The relative error each refit polynomial (in ln T) has to meet. |
| `reFitMaxDegree` | `6` | The degree a refit may go to: the lowest degree within the tolerance is taken, or the best at the cap. Seven terms is what the source data has and what a polynomial in ln T stays well conditioned at. |

## chemistry

Finite-rate chemistry, from the reactions the mechanism carries. A species
list or dict has no reactions and cannot react.

| key | default | what it does |
|---|---|---|
| `source` | `None` | `None` for no chemistry. `explicit` adds the production rates of the state to the right-hand side as a source. `substepped` integrates the source over the step in forward Euler substeps sized by the fastest species' headroom, and adds the mean rate; it survives stiffness the explicit source does not. |
| `maxSubSteps` | `200` | `substepped`: the most substeps one step may take. |
| `entropyBisections` | `12` | `substepped`: each substep is capped where the mixture's entropy stops rising along it, located by this many bisections. `0` turns the cap off. |

Point-implicit chemistry in dual time is `timeIntegration.chemistryJacobian`.

## initialConditions

The uniform state a case starts from when it does not restart. A script may
overwrite it afterwards with `setPrimitives`.

| key | default | what it does |
|---|---|---|
| `p` | `101325.0` | Pressure, Pa. |
| `u`, `v`, `w` | `0.0` | Velocity, m/s. |
| `T` | `300.0` | Temperature, K. |
| `Y` | `{}` | Mass fraction by species name. A species not named is zero, and the last species of the mixture takes the remainder. |

## timeIntegration

The integrator, and the settings that are its own.

| key | default | what it does |
|---|---|---|
| `integrator` | `rk3` | `rk1`, `rk2`, `rk3`, `rk34` and `rk4` are explicit Runge-Kutta schemes of that order; `maccormack` is the predictor-corrector; `dualTime` converges each physical step of a second-order backward difference in pseudo time. |
| `pseudoIntegrator` | `rk3` | Dual time: the pseudo-time scheme, any of the Runge-Kutta integrators. |
| `subIterations` | `20` | Dual time: pseudo steps per physical step. A positive integer. |
| `lowMach` | `true` | Dual time: the Weiss and Smith low-Mach preconditioner on the per-cell pseudo system. Off, the pseudo system is the plain time derivative. |
| `chemistryJacobian` | `None` | Dual time with a chemistry source: `diagonal` adds each species' own production-rate derivative to the pseudo system, making the chemistry point implicit. Refused without a chemistry source or with another integrator. |

Dual time keeps the state one step back, and a result written by the writer
plugin carries it, so a restart continues exactly. Restarting from a result
without it starts the scheme afresh.

## RHS

The advective flux, composed by the compiler from the scheme's name, and
shock capturing.

| key | default | what it does |
|---|---|---|
| `primaryAdvFlux` | `KEPaEC` | The flux at every cell face. A formula alone takes the cells as they are: `KEPaEC` (kinetic energy and pressure equilibrium conserving, two-point), `fourthOrderKEPaEC` (the same over four cells), or a Riemann solver, `rusanov`, `hllc`, `ausmPlusUp`. A Riemann solver may take a reconstruction and a limiter as `reconstruct-limiter-formula`: `muscl-vanLeer-hllc`, with limiters `minmod`, `vanLeer`, `mc`, `superbee`. |
| `secondaryAdvFlux` | `None` | Shock capturing: a second formula, typically `rusanov`, blended into the primary by the switch's weight. Given with a switch or not at all. |
| `switchAdvFlux` | `None` | What weights the secondary flux: `jamesonPressure`, the second difference of the pressure along the face normal; or `ducros`, the dilatation over dilatation plus vorticity, which reads the velocity gradients and is the viscous case's. |
| `switchValues` | `{}` | What the switch takes, baked into the kernel: `jamesonPressure` a `gain`; `ducros` a `nu` and a `floor`. |

## backend-serial, backend-openmp, backend-cuda, backend-hip

How the kernels are launched, one section per backend the runtime may have
been built for. A run reads the section of the one it was built for and
ignores the rest.

| key | default | what it does |
|---|---|---|
| `tileSize` | `128` | The cells of one block a team does, for a launch over cells or cell faces. On a device a team is its tile. |
| `tileElements` | `1024` | The elements a team does for a launch whose item is one element: a copy, a launch over cells and components. |
| `launchThreads` | `256` (device) | The launch bound the device kernels are compiled with: threads per block. At least `tileSize`. |
| `launchWaves` | `1` cuda, `2` hip | The floor of resident waves per multiprocessor the compiler is told to fit. This is the register budget. `2` measured best on an MI100, where the unbounded floor of `4` spilled the viscous flux and transport kernels; `1` keeps nvcc's own budget until a CUDA card measures otherwise. |

Every value is a positive integer.

## haloExchange

| key | default | what it does |
|---|---|---|
| `kind` | `hostStaged` | How a halo message travels between ranks. `hostStaged` copies through pinned host memory. `device` sends straight from the device buffers, which needs a GPU-aware MPI; the installation knows whether it has one, the runtime cannot tell. |

## bcValues

What each named boundary is and reads, by the name the grid gives the face.
The keys here are the case's own and are not checked against a list. See
[boundary conditions](boundaryConditions.md).

## plugins

What runs alongside the stepping, by plugin name, each with its own options
and how often it acts. Without the `report` plugin a run prints nothing. See
[plugins](plugins.md).

## What is refused

The config is validated when it is made from a dict (`fromDict`,
`readConfigFile`) or when a script calls `validateConfig()`, and the
simulator refuses the rest when the solver is built:

- a secondary flux without a switch, or a switch without a secondary flux;
- `primaryAdvFlux` of `None`;
- a `navierStokes` case with no transport model;
- `dualTime` with a controller other than `fixed`, or with itself as the
  pseudo integrator;
- a `chemistryJacobian` without a chemistry source or outside dual time;
- an `eos`, `chemistry.source` or `chemistryJacobian` that is not one of the
  names above; `lowMach` that is not a boolean; `entropyBisections` that is
  not a count;
- a backend value that is not a positive integer, or a `tileSize` above
  `launchThreads`;
- `subIterations` that is not a positive integer.

A mixture whose models do not go together, or whose species lack what a
model needs, is refused when the mixture is built, naming the species and
the quantity.
