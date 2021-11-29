#!/usr/bin/env python3
"""
A utility to create groups of blocks to balance the computational load among
MPI ranks.

"""

import numpy as np
import peregrinepy as pg


def getSortedBlockSizes(mb):
    sizes = np.empty(mb.nblks, dtype=np.int32)
    nblkis = np.empty(mb.nblks, dtype=np.int32)
    for i, blk in enumerate(mb):
        sizes[i] = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        nblkis[i] = blk.nblki

    perm = sizes.argsort()
    return sizes[perm], nblkis[perm]


def analyzeLoad(procSizes, procGroups):
    assert len(procSizes) > 0
    assert len(procGroups) > 0

    maxBlksForProc = 0
    maxLoad = np.max(procSizes)

    for size, group in zip(procSizes, procGroups):
        maxBlksForProc = len(group) if len(group) > maxBlksForProc else maxBlksForProc

    perfectLoad = np.mean(procSizes)
    efficiency = perfectLoad / maxLoad * 100

    return efficiency, maxBlksForProc


def allBlocksAssigned(mb, procGroups):

    sum = 0
    for nblki in mb.blockList:
        found = False
        for group in procGroups:
            sum += len(group)
            if nblki in group:
                found = True
                break
            else:
                pass
        if not found:
            print(f"Block {nblki} not found in any processor groups.")
            break
    if sum == mb.nblks:
        print("Number of blocks in groups does not equal number of total blocks")
        found = False

    return found


if __name__ == "__main__":
    import os

    import argparse

    parser = argparse.ArgumentParser(
        description="Create blocksForProcs.txt file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-from",
        "--fromDir",
        action="store",
        metavar="<fromDir>",
        dest="fromDir",
        default="./",
        help="Directory containing the gv.* and conn.yaml files. Default is ./",
        type=str,
    )
    parser.add_argument(
        "-minProcs",
        "--minimizeProcessors",
        action="store_true",
        dest="minProcs",
        help="Attempt to minimize the number or required processors",
    )
    parser.add_argument(
        "-minBlksPerProc",
        "--minimizeBlocksPerProc",
        action="store_true",
        dest="minBlksPerProc",
        help="Attampt to minimize the maximum number of blocks in on an individual processor",
    )
    parser.add_argument(
        "-blksPerProcLimit",
        "--blocksPerProcLimit",
        dest="blksPerProcLimit",
        default=float("inf"),
        help="Maximum allowable blocks per processor",
        type=float,
    )

    args = parser.parse_args()

    fromDir = args.fromDir
    minProcs = args.minProcs
    minBlksPerProc = args.minBlksPerProc
    blksPerProcLimit = args.blksPerProcLimit

    nblks = len([f for f in os.listdir(f"{fromDir}") if f.endswith(".h5")])
    mb = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(mb, fromDir)
    pg.readers.readConnectivity(mb, fromDir)

    sizes, nblkis = getSortedBlockSizes(mb)

    maxSize = np.max(sizes)
    procGroups = []
    procLoad = []

    if minProcs:
        currentProc = []
        currentSize = 0
        for size, nblki in zip(sizes, nblkis):
            if currentProc == []:
                currentProc.append(nblki)
                currentSize += size
                continue

            tempSize = currentSize + size
            if tempSize >= maxSize or len(currentProc) == blksPerProcLimit:
                procGroups.append(currentProc)
                procLoad.append(currentSize)
                currentProc = [nblki]
                currentSize = size
            else:
                currentProc.append(nblki)
                currentSize += size
        procGroups.append(currentProc)
        procLoad.append(currentSize)

    elif minBlksPerProc:
        remainingIndex = [i for i in range(nblks)]
        while len(remainingIndex) > 0:
            largeIndex = remainingIndex[-1]
            currentSize = sizes[largeIndex]
            currentProc = [nblkis[largeIndex]]
            remainingIndex.pop()

            while currentSize <= maxSize and len(remainingIndex) > 1:
                smallIndex = remainingIndex[0]
                smallSize = sizes[smallIndex]
                if (
                    currentSize + smallSize > maxSize
                    or len(currentProc) == blksPerProcLimit
                ):
                    break
                else:
                    currentProc.append(nblkis[smallIndex])
                    currentSize += smallSize
                    remainingIndex.pop(0)

            procGroups.append(currentProc)
            procLoad.append(currentSize)

    assert allBlocksAssigned(mb, procGroups)
    efficiency, maxBlksForProcs = analyzeLoad(procLoad, procGroups)

    with open("blocksForProcs.inp", "w") as f:
        for group in procGroups:
            for nblki in group:
                f.write(f"{nblki}, ")
            f.write("\n")

    print(
        "Results of Load Balancing:\n",
        f"Total Number of blocks = {mb.nblks}\n",
        f"Total Number of cells  = {np.sum(procLoad)}\n\n",
        f"Required numer of processors = {len(procGroups)}\n",
        f"Maximum blocks on processor = {maxBlksForProcs}\n",
        f"Eficiency = {efficiency}\n",
    )
