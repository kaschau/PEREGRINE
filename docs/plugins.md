# Plugins

A plugin is something a run does alongside the stepping: writing results,
printing, checking the state, sampling points. The `plugins` section of the
[config](config.md#plugins) names the ones a run has, each with its own
options and how often it acts. A run with no plugins steps in silence and
writes nothing.

```yaml
plugins:
  report:
    everyIter: 10
  writer:
    everyTime: 1.0e-4
    dir: Results
    precision: double
  nanCheck: {}
```

## How often

Every plugin takes one of two keys. Neither given means every step.

| key | what it does |
|---|---|
| `everyIter` | Act after every so many steps. |
| `everyTime` | Act whenever the simulated time crosses a multiple of this many seconds. |

## report

Everything a run prints. Without it, nothing is printed.

At the start: the banner and the case, as `print(mb)` would show it. After
every step it is due on: the step number, the time, the step size, and the
largest CFL numbers on any rank, acoustic, convective and combined. Dual
time adds its pseudo-time residuals. At the end: the wall time and the
seconds per step per cell. No options.

## writer

Results into a directory, as numbered pairs of `.h5` and `.xmf` files. See
[files](files.md#results) for what a result holds.

| key | default | what it does |
|---|---|---|
| `dir` | `.` | The directory, made if need be. |
| `precision` | `single` | What the file holds: `single` or `double`, whatever the case computes in. |
| `basename` | `q.{n:08d}` | The file name, formatted from the step `n` and the time `t`. |

The `.xmf` points at the grid file the case came from. A case meshed in a
script has no grid file, so one is written into the directory first. A
result carries what the integrator keeps beyond the state, so a restart from
it continues exactly.

## nanCheck

Stops the run on a non-finite conserved value. Every rank writes where its
are to `nans_<block>.log`, the state is written if there is a writer, and the
run raises. No options.

## trace

The primitive variables at chosen cells, appended to one csv per point each
time it acts, with the time in the first column.

| key | default | what it does |
|---|---|---|
| `points` | required | A `.npy` file: an array of `(block, i, j, k)` rows, then an array of tags, one per row. `utilities/generateTracePoints.py` makes one from coordinates. |
| `dir` | `Trace` | Where the csv files go, one per point, named `<tag>_<x>_<y>_<z>.csv`. |

## viscousSponge

Raises the viscosity along a line, from `origin` to `ending`, to
`multiplier` times itself, after the transport properties are made each
step. A viscous case only.

| key | what it does |
|---|---|
| `origin` | `[x, y, z]` where the ramp starts. |
| `ending` | `[x, y, z]` where it reaches the full multiplier. |
| `multiplier` | What the viscosity is multiplied by at the end. |

## catalyst

ParaView Catalyst in situ processing, driven by a pipeline script. ParaView
has to be importable in the case's Python; only a case that asks for this
plugin imports it.

| key | what it does |
|---|---|
| `script` | The Catalyst pipeline script. |

## Writing one

A plugin is a subclass of `BasePlugin` in
[plugins/base.py](../src/peregrinepy/plugins/base.py) with a `name`. It is
made from its config section before the solver is built, so it can declare
arrays and kernels the way the simulator does, and it can add launches to
the end of a stage's graphs. It sees the solver when the run starts, before
each step, after each step it is due on, and once at the end. The sponge is
the smallest example that adds a kernel; the writer the smallest that only
acts.
