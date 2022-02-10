def readBlocksForProcs(pathToFile="./Input", parallel=False):

    fileName = f"{pathToFile}/blocksForProcs.inp"
    if not parallel:
        try:
            with open(fileName) as f:
                lines = [
                    i.strip().split(",")
                    for i in f.readlines()
                    if not i.strip().startswith("#")
                ]
        except FileNotFoundError:
            return None
    else:
        from mpi4py import MPI
        from ..mpiComm.mpiUtils import getCommRankSize

        try:
            comm, rank, size = getCommRankSize()
            f = MPI.File.Open(comm, fileName, amode=MPI.MODE_RDONLY)
            ba = bytearray(f.Get_size())
            # read the contents into a byte array
            req = f.Iread(ba)
            MPI.Request.Wait(req)
            f.Close()
        except MPI.Exception:
            return None

        dataString = ba.decode("utf-8")
        lines = [
            i.strip().split(",")
            for i in dataString.splitlines(keepends=False)
            if not i.strip().startswith("#")
        ]

    blocks4procs = []
    for line in lines:
        if line[-1] == "":
            line = line[0:-1]
        blocks4procs.append([int(i) for i in line])

    return blocks4procs
