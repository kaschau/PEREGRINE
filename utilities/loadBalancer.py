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
    return list(sizes[perm]), list(nblkis[perm])


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

    for nblki in mb.blockList:
        found = False
        for group in procGroups:
            if nblki in group:
                found = True
                break
            else:
                pass
        if not found:
            print(f"Block {nblki} not found in any processor groups.")
            break

    summ = 0
    for group in procGroups:
        summ += len(group)
    if summ != mb.nblks:
        print("Number of blocks in groups does not equal number of total blocks")
        found = False

    return found


if __name__ == "__main__":
    import argparse
    import os

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
        help="Directory containing the g.* files. Default is ./",
        type=str,
    )
    parser.add_argument(
        "-minProcs",
        "--minimizeProcessors",
        action="store_true",
        dest="minProcs",
        default=False,
        help="Attempt to minimize the number or required processors",
    )
    parser.add_argument(
        "-minBlksPerProc",
        "--minimizeBlocksPerProc",
        action="store_true",
        dest="minBlksPerProc",
        default=False,
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
    parser.add_argument(
        "-numProcs",
        "--numberOfProcs",
        dest="numProcs",
        help="Number of processors (groups) used",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    fromDir = args.fromDir
    minProcs = args.minProcs
    minBlksPerProc = args.minBlksPerProc
    blksPerProcLimit = args.blksPerProcLimit
    numProcs = args.numProcs

    nblks = len([f for f in os.listdir(f"{fromDir}") if f.endswith(".h5")])
    mb = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(mb, fromDir)

    sizesS_L, nblkisS_L = getSortedBlockSizes(mb)
    sizesL_S = sizesS_L[::-1]
    nblkisL_S = nblkisS_L[::-1]

    maxSize = np.max(sizesS_L)
    procGroups = []
    procLoad = []

    if minProcs:
        currentProc = []
        currentSize = 0
        for size, nblki in zip(sizesS_L, nblkisS_L):
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
            currentSize = sizesS_L[largeIndex]
            currentProc = [nblkisS_L[largeIndex]]
            remainingIndex.pop()

            while currentSize <= maxSize and len(remainingIndex) > 1:
                smallIndex = remainingIndex[0]
                smallSize = sizesS_L[smallIndex]
                if (
                    currentSize + smallSize > maxSize
                    or len(currentProc) == blksPerProcLimit
                ):
                    break
                else:
                    currentProc.append(nblkisS_L[smallIndex])
                    currentSize += smallSize
                    remainingIndex.pop(0)

            procGroups.append(currentProc)
            procLoad.append(currentSize)
    elif numProcs > 0:
        procGroups = [[i] for i in nblkisL_S[0:numProcs]]
        procLoad = [i for i in sizesL_S[0:numProcs]]
        tbaProcs = nblkisL_S[numProcs::]
        tbaLoads = sizesL_S[numProcs::]

        while len(tbaProcs) > 0:
            assignProc = tbaProcs[0]
            assignSize = tbaLoads[0]

            ncellWeight = 0.5
            nBlocksWeight = 1.0 - ncellWeight
            normLoad = np.array(procLoad) / max(procLoad)
            normBlockLoad = np.array([len(i) for i in procGroups]) / max(
                [len(i) for i in procGroups]
            )

            loadScore = ncellWeight * normLoad + nBlocksWeight * normBlockLoad

            lightProc = np.argmin(loadScore)
            procGroups[lightProc].append(assignProc)
            procLoad[lightProc] += assignSize

            tbaProcs.pop(0)
            tbaLoads.pop(0)

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
        f"Maximum load on processor = {max(procLoad)}\n",
        f"Eficiency = {efficiency}\n",
    )

    if os.name == "posix" and "DISPLAY" in os.environ:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel("Processor")
        ax.set_ylabel("Number of Cells")
        ax.plot(procLoad, label="ncells", color="k")
        ax.set_ylim(bottom=0, top=None)

        ax1 = ax.twinx()
        ax1.set_ylabel("Number of Blocks")
        ax1.plot(np.array([len(i) for i in procGroups]), label="nBlocks", color="r")
        ax1.set_ylim(bottom=0, top=None)

        h1, la1 = ax.get_legend_handles_labels()
        h2, la2 = ax1.get_legend_handles_labels()

        ax.legend(h1 + h2, la1 + la2)
        plt.show()
