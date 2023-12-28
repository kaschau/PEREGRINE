import numpy as np
from peregrinepy.mpiComm.mpiUtils import getCommRankSize
from os.path import isdir, isfile


class trace:
    def __init__(self, fileName, nblki, i, j, k):
        self.nblki = nblki
        self.i = i
        self.j = j
        self.k = k

        self.fileName = fileName


class tracePointsCoprocessor:
    def __init__(self, mb):
        # First just make sure the folder is there
        comm, rank, size = getCommRankSize()
        path = f"{mb.config['io']['archiveDir']}/Trace/"
        if rank == 0:
            if not isdir(path):
                from os import mkdir

                mkdir(path)
        comm.Barrier()

        # For now, we provide a numpy array of ints, where each row is
        #   nblki,  i, j, k
        #
        # To specify the points and a second numpy array of strings to specify the
        # tags
        with open(f"{mb.config['io']['inputDir']}/tracePoints.npy", "rb") as f:
            points = np.load(f)
            tags = np.load(f)

        self.traces = []

        for blk in mb:
            ng = blk.ng
            match = np.where((points[:, 0] == blk.nblki))[0]
            if len(match) == 0:
                continue
            for m in match:
                i = points[m, 1] + ng
                j = points[m, 2] + ng
                k = points[m, 3] + ng

                xc = blk.array["xc"][i, j, k]
                yc = blk.array["yc"][i, j, k]
                zc = blk.array["zc"][i, j, k]

                fileName = path + f"{tags[m]}_{xc:.6f}_{yc:.6f}_{zc:.6f}.csv"
                self.traces.append(trace(fileName, blk.nblki, i, j, k))
                strings = "".join(
                    [
                        "Time (s), ",
                        "p [Pa], ",
                        "u [m/s], ",
                        "v [m/s], ",
                        "w [m/s], ",
                        "T [K]",
                    ]
                )
                if len(blk.speciesNames) > 1:
                    strings += ", "
                    strings += "".join(
                        [f"{s}, " for s in blk.speciesNames[0:-2]]
                        + [f"{s}\n" for s in [blk.speciesNames[-2]]]
                    )
                else:
                    strings += "\n"
                if not isfile(fileName):
                    with open(fileName, "w") as f:
                        f.write(strings)

    def __call__(self, mb):
        # Check if we need to get trace data or not
        if mb.nrt % mb.config["coprocess"]["niterTrace"] != 0:
            return

        # TODO: this may be a redundant copy
        for nblki in list(set([trc.nblki for trc in self.traces])):
            blk = mb.getBlock(nblki)
            blk.updateHostView(["q"])

        for trc in self.traces:
            blk = mb.getBlock(trc.nblki)
            i = trc.i
            j = trc.j
            k = trc.k
            arr = np.concatenate((np.array([mb.tme]), blk.array["q"][i, j, k, :]))
            with open(trc.fileName, "a") as f:
                np.savetxt(
                    f,
                    [arr],
                    fmt="%.8e",
                    delimiter=",",
                )
