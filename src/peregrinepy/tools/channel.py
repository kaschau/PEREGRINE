"""A channel or boundary layer grid of one block, its wall-normal spacing
grown in wall units from the viscous sublayer out: a yaml of the geometry,
the flow and the resolution, and the grid file it makes."""

from pathlib import Path

import numpy as np
import yaml

import peregrinepy as pg

name = "channel"
help = "make a channel or boundary layer grid, spaced in wall units"

template = """---
domainType: channel        # channel, or bl for a boundary layer

# Geometry
delta: 0.05                # channel half height or boundary layer thickness, m
zWidth: 3                  # width, in delta
yHeight: 2                 # height, in delta; a channel is two delta high
xLength: 6                 # length, in delta

# Flow
viscosity: 1.81e-5         # kinematic viscosity, m^2/s
ReTau: 587.19              # friction Reynolds number

# Resolution, in wall units; a default is the value in the comment
nVSL: default              # points across the viscous sublayer, y+ < 5 (6)
buffGR: default            # growth rate through the buffer layer (1.1)
logGR: default             # growth rate through the log layer (1.1)
dYplusCore: default        # wall-normal spacing in the core or free stream (20)
dXplus: default            # streamwise spacing (20)
dZplus: default            # spanwise spacing (20)
"""

defaults = {
    "nVSL": 6,
    "buffGR": 1.1,
    "logGR": 1.1,
    "dYplusCore": 20.0,
    "dXplus": 20.0,
    "dZplus": 20.0,
}


def addArguments(parser):
    parser.add_argument("spec", nargs="?", help="the yaml describing the grid")
    parser.add_argument("out", nargs="?", help="the grid file to write")
    parser.add_argument(
        "-template", metavar="FILE", help="write a template yaml to FILE and stop"
    )


def ptsWithGR(start, stop, gr, startDy):
    assert stop > start > 0.0 and gr > 1.0
    pts = [start]
    dY = startDy * gr
    while pts[-1] < stop:
        pts.append(pts[-1] + dY)
        dY *= gr
    if abs(pts[-1] - stop) > abs(pts[-2] - stop):
        pts = pts[0:-1]
    pts = np.array(pts)
    return start + (pts - start) / (stop - start) * stop


def growToDy(startDy, endDy, gr):
    assert 0.0 < startDy < endDy and gr > 1.0
    pts = [0]
    dY = startDy * gr
    while dY < endDy:
        pts.append(pts[-1] + dY)
        dY *= gr
    return np.array(pts)


def wallNormalPoints(inp):
    """The wall-normal coordinates in wall units, from the viscous
    sublayer through the buffer and log layers to the core; a channel gets
    the mirror image on top."""

    def setting(key):
        given = inp[key]
        return defaults[key] if given == "default" else type(defaults[key])(given)

    nVSL = setting("nVSL")
    assert nVSL > 2
    yVSL = np.linspace(0, 5, nVSL)
    yBuff = ptsWithGR(5, 30, setting("buffGR"), yVSL[-1] - yVSL[-2])
    dYplusCore = setting("dYplusCore")
    yLog = yBuff[-1] + growToDy(yBuff[-1] - yBuff[-2], dYplusCore, setting("logGR"))

    delta, viscosity, ReTau = (float(inp[k]) for k in ("delta", "viscosity", "ReTau"))
    yp1 = viscosity / (ReTau * viscosity / delta)
    if inp["domainType"] == "channel":
        endYplus = delta / yp1
    elif inp["domainType"] == "bl":
        endYplus = float(inp["yHeight"]) * delta / yp1
    else:
        raise ValueError(f"domainType is channel or bl, not {inp['domainType']!r}")
    nCore = int((endYplus - yLog[-1]) / dYplusCore)
    yCore = np.linspace(yLog[-1], endYplus, nCore)

    ys = np.concatenate((yVSL, yBuff[1:-1], yLog[0:-1], yCore))
    if inp["domainType"] == "channel":
        ys = np.concatenate((ys, 2 * ys[-1] - np.flip(ys)[1::]))
    return ys, yp1, (len(yVSL) - 1, len(yBuff) - 1)


def main(args):
    if args.template:
        Path(args.template).write_text(template)
        return
    if not (args.spec and args.out):
        raise SystemExit("peregrine channel: the yaml and the grid file to write")
    with open(args.spec) as f:
        inp = yaml.safe_load(f)

    ys, yp1, (nVSL, nBuff) = wallNormalPoints(inp)
    delta = float(inp["delta"])
    dXplus = defaults["dXplus"] if inp["dXplus"] == "default" else float(inp["dXplus"])
    dZplus = defaults["dZplus"] if inp["dZplus"] == "default" else float(inp["dZplus"])
    Lx, Ly, Lz = (
        float(inp["xLength"]) * delta,
        ys[-1] * yp1,
        float(inp["zWidth"]) * delta,
    )
    nx, ny, nz = int(Lx / (dXplus * yp1)), len(ys), int(Lz / (dZplus * yp1))

    grid = pg.multiBlock.grid()
    pg.mesher.CubeMesher(lengths=[Lx, Ly, Lz], dimsPerBlock=[nx, ny, nz]).fill(grid)
    # the box's even spacing in y replaced by the wall-unit spacing
    blk = grid.blocks[0]
    nodes = blk.nodes.get()
    nodes[..., 1] = ys[np.newaxis, :, np.newaxis] * yp1
    blk.nodes.set(nodes)

    inLog = np.flatnonzero((ys > 30.0) & (ys * yp1 / delta < 0.2))
    print(
        "Summary:\n"
        f"Domain type: {inp['domainType']}\n"
        f"{nx=}, {ny=}, {nz=}\n"
        f"Total Cells: {(nx - 1) * (ny - 1) * (nz - 1)}\n"
        f"Viscous Sub Layer Cells: {nVSL}\n"
        f"Buffer Layer Cells: {nBuff}\n"
        f"Log Layer Cells: {len(inLog)}\n"
    )
    pg.writers.GridWriter(grid, args.out, quiet=False).write(grid)
