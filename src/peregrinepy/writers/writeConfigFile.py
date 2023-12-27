import yaml


class myDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def writeConfigFile(config, filePath="./"):
    """This function write a PEREGRINE input file dtms.inp from a peregrinepy.files.inputFile object

    Parameters
    ----------

    config : peregrinepy.files.configFile

    filePath : str
       Path of location to write output PEREGRINE input file to

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
