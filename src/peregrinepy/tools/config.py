"""A config file of every default, to start a case from."""

import peregrinepy as pg

name = "config"
help = "write a config file of every default, to start a case from"


def addArguments(parser):
    parser.add_argument("out", help="the yaml to write")


def main(args):
    pg.writers.writeConfigFile(pg.files.configFile(), args.out)
    print(f"Wrote the defaults to {args.out}; docs/config.md says what each key does.")
