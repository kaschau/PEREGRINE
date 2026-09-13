import yaml

from ..files import configFile
from ..mpiComm.mpiUtils import getCommRankSize


def readConfigFile(filePath="./peregrine.yaml"):
    """A config from its yaml, over the defaults. One rank reads the file and
    every rank gets the text, so a run never opens one small file many times
    over; on one rank that is just a read."""
    comm, rank, size = getCommRankSize()
    text = open(filePath).read() if rank == 0 else None
    return configFile.fromDict(yaml.safe_load(comm.bcast(text, root=0)))
