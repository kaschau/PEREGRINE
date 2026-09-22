import yaml


class myDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def writeConfigFile(config, fileName):
    """Writes a config as the yaml a run reads."""
    with open(fileName, "w") as f:
        yaml.dump(config.toDict(), f, Dumper=myDumper, sort_keys=False)
