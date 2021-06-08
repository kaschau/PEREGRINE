# -*- coding: utf-8 -*-


def write_config_file(config, file_path = './'):

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

    with open('{}/dtms.inp'.format(file_path), 'w') as f:

        for key in config.keys():

            f.write(' &{}\n'.format(key))

            for key2 in config[key]:

                f.write('{0: >8} = '.format(key2))

                value = config[key][key2]

                if isinstance(value, str):
                    try :
                        int(value)
                        f.write('{}'.format(value))
                    except ValueError:
                        f.write('\'{}\''.format(value))

                elif isinstance(value, float):
                    f.write('{:1.16e}'.format(value))

                elif isinstance(value, int):
                    f.write('{}'.format(value))

                else:
                    print('Error, unrecognized value in config file: {}'.format(value))

                f.write(',\n')

            f.write(' /\n')
