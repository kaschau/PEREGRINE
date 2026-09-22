import yaml


class myDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def writeConfigFile(config, filePath="./"):
    """Writes a config as the yaml a run reads."""

    connOut = {}
    for k1 in config.keys():
        connOut[k1] = {}
        for k2 in config[k1].keys():
            connOut[k1][k2] = config[k1][k2]

    with open(f"{filePath}/peregrine.yaml", "w") as f:
        yaml.dump(connOut, f, Dumper=myDumper, sort_keys=False)
