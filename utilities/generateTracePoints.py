#!/usr/bin/env python3
"""
A utility to generate trace point file for use in peregrine simulation.
"""

import numpy as np
import peregrinepy as pg
import yaml

inpFileTemplate = """---

# Template for input file to generate trace points

# The level 1 headings are the tags given for the collection
# so rename them to be distinct if you wang

myPoint:
  type: point
  p0:
    - 0.0
    - 0.0
    - 0.0

myLine:
  type: line
  p0:
    - 0.0
    - 0.0
    - 0.0
  p1:
    - 1.0
    - 0.0
    - 0.0
  n: 10

myPlane:
  type: plane
  p0:
    - 0.0
    - 0.0
    - 0.0
  p1:
    - 1.0
    - 0.0
    - 0.0
  p2:
    - 0.0
    - 1.0
    - 0.0
  n01: 10
  n02: 10

# A custom user defined function in udf.py
myCustomFunction:
  type: userFunction"""

udf = """
def udf():
    # generate your points here
    return pts,tags"""


def createPoint(pdict):
    x = pdict["p0"][0]
    y = pdict["p0"][1]
    z = pdict["p0"][2]

    pts = np.array([[x, y, z]])
    return pts


def createLine(pdict):
    x0 = pdict["p0"][0]
    y0 = pdict["p0"][1]
    z0 = pdict["p0"][2]
    x1 = pdict["p1"][0]
    y1 = pdict["p1"][1]
    z1 = pdict["p1"][2]

    n = pdict["n"]

    pts = np.empty((n, 3))
    pts[:, 0] = np.linspace(x0, x1, n)
    pts[:, 1] = np.linspace(y0, y1, n)
    pts[:, 2] = np.linspace(z0, z1, n)
    return pts


def createPlane(pdict):
    x0 = pdict["p0"][0]
    y0 = pdict["p0"][1]
    z0 = pdict["p0"][2]
    p0 = np.array([x0, y0, z0])
    x1 = pdict["p1"][0]
    y1 = pdict["p1"][1]
    z1 = pdict["p1"][2]
    p1 = np.array([x1, y1, z1])
    x2 = pdict["p2"][0]
    y2 = pdict["p2"][1]
    z2 = pdict["p2"][2]
    p2 = np.array([x2, y2, z2])

    n01 = pdict["n01"]
    n02 = pdict["n02"]

    v01 = p1 - p0
    v02 = p2 - p0
    l01 = np.linalg.norm(v01)
    l02 = np.linalg.norm(v02)
    norm01 = v01 / l01
    norm02 = v02 / l02

    d01 = np.linspace(0, 1, n01)
    d02 = np.linspace(0, 1, n02)
    xx = np.zeros((n01, n02))
    yy = np.zeros((n01, n02))
    zz = np.zeros((n01, n02))

    for j in range(n02):
        for i in range(n01):
            xx[i, j] = x0 + norm01[0] * d01[i] + norm02[0] * d02[j]
            yy[i, j] = y0 + norm01[1] * d01[i] + norm02[1] * d02[j]
            zz[i, j] = z0 + norm01[2] * d01[i] + norm02[2] * d02[j]
    pts = np.zeros((n01 * n02, 3))
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    pts[:, 2] = zz.ravel()

    return pts


def createCircle(pdict):
    pass


def getPointsTagsFromInput(inp):
    # Create a list of points to trace, points is a numpy array
    # Also create "tags" for each point that will prepend
    # the output file name to make sorting through data easier.
    # So create a numpy array of (npts,3) in size, and a list of
    # tag names that is [npts] long

    pts = np.empty((0, 3))
    tags = []
    for item in inp:
        pdict = inp[item]
        ptype = pdict["type"]
        if ptype == "point":
            tpts = createPoint(pdict)
            ttags = [item for _ in range(tpts.shape[0])]
        elif ptype == "line":
            tpts = createLine(pdict)
            ttags = [item for _ in range(tpts.shape[0])]
        elif ptype == "plane":
            tpts = createPlane(pdict)
            ttags = [item for _ in range(tpts.shape[0])]
        elif ptype == "circle":
            tpts = createCircle(pdict)
            ttags = [item for _ in range(tpts.shape[0])]
        elif ptype == "userFunction":
            from .udf import udf

            tpts, ttags = udf()
        else:
            raise TypeError(f"Unknown template type {item}")

        pts = np.append(pts, tpts, axis=0)
        tags += ttags

    return pts, tags


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

    parser = argparse.ArgumentParser(
        description="Utility to generate the tracePoints.npy binary file used in peregrine to trace point data in situ."
    )
    parser.add_argument("inputFile", nargs="?", help="Input yaml file.", default=None)
    parser.add_argument(
        "-genInp",
        "--generateInput",
        action="store_true",
        dest="genInp",
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
    if args.genInp:
        with open("tracePoints.yaml", "w") as f:
            f.write(inpFileTemplate)
        with open("udf.py", "w") as f:
            f.write(udf)
        raise SystemExit(0)

    with open(args.inputFile, "r") as connFile:
        inp = yaml.load(connFile, Loader=yaml.FullLoader)

    # gp = args.gridPath
    # nblks = len([i for i in os.listdir(gp) if i.startswith("g.") and i.endswith(".h5")])
    # assert nblks > 0

    # mb = pg.multiBlock.grid(nblks)
    # pg.readers.readGrid(mb, gp)

    mb = []
    points, tags = getPointsTagsFromInput(inp)
    print(points)
    print(tags)
    # generateTracePoints(mb, points, tags)
