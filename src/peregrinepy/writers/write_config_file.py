# -*- coding: utf-8 -*-
import yaml


class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def write_config_file(config, file_path="./"):

    """This function write a RAPTOR input file dtms.inp from a raptorpy.files.input_file object

    Parameters
    ----------

    config : raptorpy.files.configut_file

    file_path : str
       Path of location to write output RAPTOR input file to

    Returns
    -------
    None

    """

    connout = {}
    for k1 in config.keys():
        connout[k1] = {}
        for k2 in config[k1].keys():
            connout[k1][k2] = config[k1][k2]

    with open(f"{file_path}/peregrine.yaml", "w") as f:
        yaml.dump(connout, f, Dumper=MyDumper, sort_keys=False)
