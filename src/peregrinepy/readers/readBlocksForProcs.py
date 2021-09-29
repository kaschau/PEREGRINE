def readBlocksForProcs(pathToFile):

    with open(f"{pathToFile}/blocksForProcs.inp") as f:
        lines = [
            i.strip().split(",") for i in f.readlines() if not i.strip().startswith("#")
        ]

        blocks4procs = []
        for line in lines:
            if line[-1] == "":
                line = line[0:-1]
            blocks4procs.append([int(i) for i in line])

    return blocks4procs
