import yaml
from ..files import configFile


def readConfigFile(file_path="./"):

    # only the zeroth block reads in the file
    with open(f"{file_path}", "r") as connFile:
        connIn = yaml.load(connFile, Loader=yaml.FullLoader)

    config = configFile()

    for k1 in connIn.keys():
        for k2 in connIn[k1].keys():
            config[k1][k2] = connIn[k1][k2]

    # ensure common problem variables are typed correctly
    config["simulation"]["dt"] = float(config["simulation"]["dt"])

    return config
