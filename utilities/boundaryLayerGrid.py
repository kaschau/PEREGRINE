#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility creates a channel grid given some specifications

The utility will output a g.xmf grid file that can be cut.

Run

./channelGrid.py -genInp

to output a template input file. Otherwise

./channelGrid.py <inputfile.yaml> to create the grid

"""

import argparse
import peregrinepy as pg
import numpy as np
import yaml

inpFileTemplate = """---

domainType: channel        # 'channel' or 'bl'


# Geometry
# ----------
delta: 0.05                # Channel half height or BL thicknes [m]
zWidth: 3                  # Geometry width [delta]
yHeight: 2                 # Geometry height [delta] (ignored for channels)
xLength: 6                 # Geometry length [delta]

# Flow
# ------
viscosity: 1.81e-5         # Kinematic viscosity [m^2/s]
ReTau: 587.19              # Friction velocity Reynolds number

# Resolution
# ----------
nVSL: default              # Number of points in viscous sub layer (default = 6)
buffGR: default            # Buffer Layer Growth Rate (default = 1.1)
logGR: default             # Number of points in log layer (default = 1.1)
dYplusCore: default        # Resolution in wall normal direction in channel core (or free stream) (default = 20)
dXplus: default            # Resolution in streamwise direction (default = 20)
dZplus: default            # Resolution in spanwise direction (default = 20)

"""


def ptsWithGR(start, stop, gr, startDy):
    assert stop > start
    assert start > 0.0
    assert gr > 1.0

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
    assert startDy > 0.0
    assert startDy < endDy
    assert gr > 1.0

    pts = [0]
    dY = startDy * gr
    while dY < endDy:
        pts.append(pts[-1] + dY)
        dY *= gr

    return np.array(pts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Utility to create a boundary layer grid."
    )
    parser.add_argument(
        "-genInp",
        "--generateInput",
        action="store_true",
        dest="genInp",
    )
    parser.add_argument("inputFile", nargs="?", help="Input yaml file.", default=None)

    args = parser.parse_args()
    args.genInp
    args.inputFile

    if args.genInp:
        with open("channel.yaml", "w") as f:
            f.write(inpFileTemplate)
        raise SystemExit(0)

    with open(args.inputFile, "r") as connFile:
        inp = yaml.load(connFile, Loader=yaml.FullLoader)
        inp = argparse.Namespace(**inp)

    # Default values
    defNvsl = 6
    defBuffGR = 1.1
    defLogGR = 1.1
    defXplus = 20.0
    defYplusCore = 20.0
    defZplus = 20.0

    # height of y+ = 1
    viscosity = float(inp.viscosity)
    delta = float(inp.delta)
    ReTau = float(inp.ReTau)
    utau = ReTau * viscosity / delta
    yp1 = viscosity / utau

    # Build boundary layer in terms of yPlus units

    # Cells in the between y+=[0,5] aka the viscous sub layer
    nVSL = defNvsl if inp.nVSL == "default" else int(inp.nVSL)
    assert nVSL > 2
    yVSL = np.linspace(0, 5, nVSL)

    # Grow through the buffer layer
    buffGR = defBuffGR if inp.buffGR == "default" else float(inp.buffGR)
    assert buffGR > 1.0

    startDy = yVSL[-1] - yVSL[-2]
    yBuff = ptsWithGR(5, 30, buffGR, startDy)

    # Grow through the log layer
    logGR = defLogGR if inp.logGR == "default" else float(inp.logGR)

    startDy = yBuff[-1] - yBuff[-2]
    if inp.domainType == "channel":
        dYplusCore = (
            defYplusCore if inp.dYplusCore == "default" else float(inp.dYplusCore)
        )
    elif inp.domainType == "bl":
        dYplusCore = (
            defYplusCore if inp.dYplusCore == "default" else float(inp.dYplusCore)
        )
    else:
        raise ValueError

    yLog = yBuff[-1] + growToDy(startDy, dYplusCore, logGR)

    # Grow through the core to end of domain
    delta = float(inp.delta)
    yHeight = float(inp.yHeight)
    if inp.domainType == "channel":
        endYplus = delta / yp1
    elif inp.domainType == "bl":
        endYplus = yHeight / yp1

    nCore = int((endYplus - yLog[-1]) / dYplusCore)

    yCore = np.linspace(yLog[-1], endYplus, nCore)

    # Put them together, convert from y+ to dimensional units
    totalYs = np.concatenate((yVSL, yBuff[1:-1], yLog[0:-1], yCore))

    endY = endYplus * yp1
    if inp.domainType == "channel":
        totalYs = np.concatenate((totalYs, 2 * totalYs[-1] - np.flip(totalYs)[1::]))
        endY *= 2

    totalYs

    # Create the grid
    xLength = float(inp.xLength)
    zWidth = float(inp.zWidth)
    dXplus = defXplus if inp.dXplus == "default" else float(inp.dXplus)
    dZplus = defZplus if inp.dZplus == "default" else float(inp.dZplus)

    grid = pg.multiBlock.grid(1)
    Lx, Ly, Lz = xLength * delta, endY, zWidth * delta
    nx, ny, nz = int(Lx / (dXplus * yp1)), len(totalYs), int(Lz / (dZplus * yp1))

    pg.grid.create.multiBlockCube(grid, lengths=[Lx, Ly, Lz], dimsPerBlock=[nx, ny, nz])

    # Overwrite the y values with the BL values
    grid[0].array["y"][:, :, :] = totalYs[np.newaxis, :, np.newaxis] * yp1

    string = "Summary:\n"
    string += f"Domain type: {inp.domainType}\n"
    string += f"{nx=}, {ny=}, {nz=}\n"
    string += f"Total Cells: {(nx-1)*(ny-1)*(nz-1)}\n"
    string += f"Viscous Sub Layer Cells: {len(yVSL)-1}\n"
    string += f"Buffer Layer Cells: {len(yBuff)-1}\n"
    string += f"Log Layer Cells: {len(np.where((totalYs > 30.0) & (totalYs*yp1/delta < 0.2))[0])}\n"
    print(string)

    pg.writers.writeGrid(grid)
    pg.writers.writeConnectivity(grid)
