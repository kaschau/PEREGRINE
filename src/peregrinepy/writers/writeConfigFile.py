# -*- coding: utf-8 -*-
import yaml


class myDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def writeLineBreak(self, data=None):
        super().writeLineBreak(data)

        if len(self.indents) == 1:
            super().writeLineBreak()


def writeConfigFile(config, filePath="./"):

    """This function write a RAPTOR input file dtms.inp from a raptorpy.files.input_file object

    Parameters
    ----------

    config : raptorpy.files.configut_file

    filePath : str
       Path of location to write output RAPTOR input file to

    Returns
    -------
    None

    """

    connOut = {}
    for k1 in config.keys():
        connOut[k1] = {}
        for k2 in config[k1].keys():
            connOut[k1][k2] = config[k1][k2]

    with open(f"{filePath}/peregrine.yaml", "w") as f:
        yaml.dump(connOut, f, Dumper=myDumper, sort_keys=False)
