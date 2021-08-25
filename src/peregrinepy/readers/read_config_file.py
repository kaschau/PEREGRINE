import yaml
from ..files import config_file
from ..mpicomm import mpiutils

def read_config_file(file_path='./'):

    comm,rank,size = mpiutils.get_comm_rank_size()

    #only the zeroth block reads in the file
    with open(f'{file_path}', 'r') as conn_file:
        connin = yaml.load(conn_file, Loader=yaml.FullLoader)

    config = config_file()

    for k1 in connin.keys():
        for k2 in connin[k1].keys():
            config[k1][k2] = connin[k1][k2]

    return config
