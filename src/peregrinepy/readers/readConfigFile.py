import yaml
from ..files import configFile


def readConfigFile(filePath="./peregrine.yaml", parallel=False):

    if not parallel:
        rank = 0
        with open(filePath, "r") as connFile:
            connIn = yaml.load(connFile, Loader=yaml.FullLoader)
    else:
        from mpi4py import MPI
        from ..mpiComm.mpiUtils import getCommRankSize

        comm, rank, size = getCommRankSize()
        connFile = MPI.File.Open(comm, filePath, amode=MPI.MODE_RDONLY)
        ba = bytearray(connFile.Get_size())
        # read the contents into a byte array
        req = connFile.Iread(ba)
        MPI.Request.Wait(req)
        # close the file
        connFile.Close()
        dataString = ba.decode("utf-8")
        connIn = yaml.safe_load(dataString)

    config = configFile()

    for k1 in connIn.keys():
        for k2 in connIn[k1].keys():
            config[k1][k2] = connIn[k1][k2]

    config.validateConfig()

    return config
