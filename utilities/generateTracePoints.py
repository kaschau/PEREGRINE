#!/usr/bin/env python3
"""
A utility to generate trace point file for use in peregrine simulation.
"""

import peregrinepy as pg
import numpy as np


def generateTracePoints(mb, points, tags):

    indexes = np.zeros((points.shape[0], 4), dtype=int)
    tags = np.array(tags)

    found = [False for _ in range(points.shape[0])]
    maxDist = 0.0
    for blk in mb:
        # pg.misc.progressBar(blk.nblki, nb, "Searching blocks")
        inside = pg.interpolation.bounds.ptsInBlkBounds(blk, points)
        for index in np.where(inside)[0]:
            x = points[index][0]
            y = points[index][1]
            z = points[index][2]
            if found[index]:
                print("Duplicate detected")
                print(x, y, z)
                print("Detected with nblki = ", blk.nblki)
                print("But was also found in nblki = ", indexes[index, 0])
                continue
            found[index] = True
            indexes[index, 0] = blk.nblki

            blk.computeMetrics(2, xcOnly=True)
            dists = np.sqrt(
                (blk.array["xc"] - x) ** 2
                + (blk.array["yc"] - y) ** 2
                + (blk.array["zc"] - z) ** 2
            )
            minIndex = np.where((dists == np.min(dists)))
            minIndex = tuple([minIndex[0][0], minIndex[1][0], minIndex[2][0]])
            maxDist = max(maxDist, dists[minIndex])

            indexes[index, 1] = minIndex[0]
            indexes[index, 2] = minIndex[1]
            indexes[index, 3] = minIndex[2]

            indexes[index, 1] = minIndex[0][0]
            indexes[index, 2] = minIndex[1][0]
            indexes[index, 3] = minIndex[2][0]

    if False in found:
        print(
            f"\nWarning!!! Looking for {points.shape[0]} trace locations,\n"
            f"but only found {sum(found)}"
        )
        for i, f in enumerate(found):
            if not f:
                print("Could not find a home for:", points[i])
        indexes = indexes[found]
        tags = tags[found]
    else:
        print(f"Maximum distance from cell center to requested point = {maxDist})")

    assert indexes.shape[0] == len(tags)

    with open("tracePoints.npy", "wb") as f:
        np.save(f, indexes)
        np.save(f, tags)


if __name__ == "__main__":
    import argparse
    import os

    # Create a list of points to trace, points is a numpy array
    # Also create "tags" for each point that will prepend
    # the output file name to make sorting through data easier.
    # So create a numpy array of (npts,3) in size, and a list of
    # tag names that is [npts] long

    # Circle
    points = np.empty((10, 3))
    radius = 38.05e-3
    radians = np.linspace(0, 2 * np.pi / 4.0, points.shape[0] + 1)[0:-1]
    points[:, 0] = radius * np.cos(radians)
    points[:, 1] = 8.9e-3
    points[:, 2] = -radius * np.sin(radians)
    tags = [f"Circle{i}" for i in range(points.shape[0])]

    # Probe 2
    points = np.append(points, np.array([[radius, 28.6e-3, 0.0]]), axis=0)
    tags.append("Probe2")
    # Probe 3
    points = np.append(points, np.array([[radius, 65.3e-3, 0.0]]), axis=0)
    tags.append("Probe3")

    ##########################################################
    # End edit region
    ##########################################################

    parser = argparse.ArgumentParser(
        description="Utility to generate the tracePoints.npy binary file used in peregrine to trace point data in situ."
    )
    parser.add_argument(
        "-gpath",
        action="store",
        metavar="<gridPath>",
        dest="gridPath",
        default="./",
        help="Path to grid files",
        type=str,
    )

    args = parser.parse_args()

    gp = args.gridPath
    nblks = len([i for i in os.listdir(gp) if i.startswith("g.") and i.endswith(".h5")])
    assert nblks > 0

    mb = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(mb, gp)

    generateTracePoints(mb, points, tags)
