# Plugins

A plugin is something a run does alongside the stepping: writing results,
printing, checking the state, sampling points. The `plugins` section of the
[config](config.md#plugins) names the ones a run has, each with its own
options and how often it acts. A run with no plugins steps in silence and
writes nothing.

```yaml
plugins:
  report:
    niterOut: 10
  writer:
    dtOut: 1.0e-4
    dir: Results
    precision: double
  nanCheck: {}
```

## How often

Every plugin takes one of two keys, and neither given means every step. A
trace may name its own, so one section can trace at several cadences.

| key | what it does |
|---|---|
| `niterOut` | Act after every so many steps. |
| `dtOut` | Act at every multiple of this many seconds. The cfl controller shortens a step to land on the multiple; with a fixed step the plugin acts on the first step past it. |

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

The primitive variables at the cells nearest chosen points, appended to one
csv per trace each time it acts. A trace is a point, a line of `n` points
from `p0` to `p1`, or a plane of `n01` by `n02` points spanned from `p0` by
`p1` and `p2`, and names the file it writes. Each row is the time, the cell
center, and the primitive variables there; the header names them.

```yaml
plugins:
  trace:
    niterOut: 10
    traces:
      probe:
        type: point
        p0: [0.5, 0.0, 0.0]
        file: probe.csv
        niterOut: 1           # its own cadence; the others act on the section's
      centerline:
        type: line
        p0: [0.0, 0.0, 0.0]
        p1: [1.0, 0.0, 0.0]
        n: 50
        file: centerline.csv
      inlet:
        type: plane
        p0: [0.0, 0.0, 0.0]
        p1: [0.0, 1.0, 0.0]
        p2: [0.0, 0.0, 1.0]
        n01: 20
        n02: 20
        file: inlet.csv
```

A point is traced at the cell whose center is nearest it, on whichever rank
holds that cell, so a point outside the grid lands on the nearest cell there
is. A file that exists is appended to, so a restart continues it.

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
