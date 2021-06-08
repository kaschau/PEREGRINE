# -*- coding: utf-8 -*-


def write_connectivity(mb, path='./'):
    '''This function produces RAPTOR grid connectivity file (conn.inp) files from a raptorpy.multiblock.grid (or a descendant)

    Parameters
    ----------

    mb : raptorpy.multiblock.grid (or a descendant)

    file_path : str
       Path of location to write output RAPTOR grid connectivity file to

    Returns
    -------
    None


    '''

    with open('{}/conn.inp'.format(path), 'w') as conn_file:

        conn_file.write('{:>10d}\n'.format(len(mb)))

        for blk in mb:
            conn_file.write('B {:>8d} '.format(blk.nblki))
            for j in range(6):
                bc = blk.connectivity[str(j+1)]['bc'].replace(' ','')

                connection = int(blk.connectivity[str(j+1)]['connection'])
                orientation = blk.connectivity[str(j+1)]['orientation']
                face_string = '{} {} {:>8d} {} '.format(bc[0], bc[1], connection, orientation)

                conn_file.write(face_string)
            conn_file.write('\n')
