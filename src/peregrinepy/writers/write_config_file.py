# -*- coding: utf-8 -*-


def write_config_file(config,file_path='./'):

    '''This function write a RAPTOR input file dtms.inp from a raptorpy.files.input_file object

    Parameters
    ----------

    config : raptorpy.files.configut_file

    file_path : str
       Path of location to write output RAPTOR input file to

    Returns
    -------
    None

    '''

    with open(f'{file_path}/peregrine.inp', 'w') as f:

        for key in config.keys():

            f.write('{}\n'.format(key))

            for key2 in config[key]:

                f.write('{0: >8} = '.format(key2))

                value = config[key][key2]

                f.write('{}'.format(value))

                f.write('\n')

            f.write('\n')
