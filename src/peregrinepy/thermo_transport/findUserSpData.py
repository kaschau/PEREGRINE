import yaml


def findUserSpData(config):
    from pathlib import Path

    # list of places to check, in order, for the spdata yaml input
    relpath = str(Path(__file__).parent)
    # First place we look for the file is in the input folder
    # then we look in the local directory
    # then we look in the PEREGRINE data base
    # then we see if we have an absolute path
    spdataLocs = [
        f'{config["io"]["inputdir"]}/{config["thermochem"]["spdata"]}',
        f'./{config["thermochem"]["spdata"]}',
        f'{relpath}/database/{config["thermochem"]["spdata"]}',
        config["thermochem"]["spdata"],
    ]
    for loc in spdataLocs:
        try:
            with open(loc, "r") as f:
                usersp = yaml.load(f, Loader=yaml.SafeLoader)
                break
        except FileNotFoundError:
            pass
    else:
        raise ValueError(
            f"Not able to find your spdata yaml input file. Tried {spdataLocs}"
        )
    return usersp
