# Boundary conditions

A face's boundary condition is split between the grid and the case, by what
each one owns. The grid gives a face a **name**, and for a periodic face the
transform that reaches its partner: the shape of the grid, which does not
change from run to run. The case says what each name **is** and what it
**reads**, in the `bcValues` section of the [config](config.md#bcvalues).
So one grid runs as a wall on one case and an inlet on the next without
being rewritten.

```yaml
bcValues:
  inlet:
    bcType: constantVelocitySubsonicInlet
    u: 12.5
    v: 0.0
    w: 0.0
    T: 415.0
    CH4: 0.055        # an inlet carries the composition, by species name
  outlet:
    bcType: constantPressureSubsonicExit
    p: 101325.0
  walls:
    bcType: adiabaticNoSlipWall
```

Every name the grid carries needs an entry, and a name the config does not
know is refused when the solver is built, naming what it does know. A
condition that reads no values needs only its `bcType`.

## Where a name comes from

A grid file carries the name of every named face in its connectivity, put
there by the mesh translator or by hand. A mesher names the outside faces
of its lattice by side: `boundaryNames={3: "still", 4: "moving"}` names the
low and high j sides, with 1 and 2 the i sides and 5 and 6 the k sides.

A face the grid leaves unnamed says what it is by itself: one with a
neighbor is interior, one with a transform is periodic, and one with neither
is an adiabatic slip wall. A periodic face is not a boundary condition: what
it does to its halo, the halo exchange does as the halo lands.

## The conditions

What a simulator may use is what is declared under its base in
[euler/boundaries.py](../src/peregrinepy/simulator/euler/boundaries.py) and
[navierStokes/boundaries.py](../src/peregrinepy/simulator/navierStokes/boundaries.py),
and nothing else. The Navier-Stokes simulator has every condition Euler has,
with gradient bodies as well, and adds the walls the flow sticks to. An inlet
also reads a mass fraction for any species named in its entry; a species not
named is zero, and the last species takes the remainder.

| bcType | reads | euler | navierStokes |
|---|---|---|---|
| `adiabaticSlipWall` | nothing | yes | yes |
| `isoTSlipWall` | `T` | yes | yes |
| `constantVelocitySubsonicInlet` | `u`, `v`, `w`, `T`, species | yes | yes |
| `supersonicInlet` | `p`, `u`, `v`, `w`, `T`, species | yes | yes |
| `stagnationSubsonicInlet` | `pt`, `Tt`, species | yes | yes |
| `constantMassFluxSubsonicInlet` | `mDotPerUnitArea`, `T`, species | yes | yes |
| `constantPressureSubsonicExit` | `p` | yes | yes |
| `supersonicExit` | nothing | yes | yes |
| `adiabaticNoSlipWall` | nothing | | yes |
| `adiabaticMovingWall` | `u`, `v`, `w` | | yes |
| `isoTNoSlipWall` | `T` | | yes |
| `isoTMovingWall` | `u`, `v`, `w`, `T` | | yes |

Velocities are in m/s, temperatures in K, pressures in Pa, the mass flux in
kg/m^2/s. The mass flux inlet turns its flux into a momentum along the face
normal, into the block.

## Profiles

A condition that reads values may take them from a file instead of
constants, one plane per block face, with `profile: <directory>` in place
of the values:

```yaml
bcValues:
  inlet:
    bcType: constantVelocitySubsonicInlet
    profile: Input/inletProfile
```

The file for each face is `<directory>/<bcName>_<block>_<face>.npy`, holding
two arrays saved one after the other with `numpy.save`: the plane of
primitive values, then the plane of conserved values, each shaped like the
face's value plane. A script that builds the case can write them from the
solver's faces.

## Adding one

A condition is a subclass under the simulator's base, with its `bcType` and
the `values` it reads, and a header of the same name in
[compute/boundaryConditions](../src/compute/boundaryConditions) with one
body per hook it acts at: `euler` for the state in the halo, `preDqDxyz`
and `postDqDxyz` around the gradients in a viscous case. The gradient rules
live in the C++ body alone; nothing on the Python side repeats them.
