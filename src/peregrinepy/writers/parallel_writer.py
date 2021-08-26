# -*- coding: utf-8 -*-

import h5py
import numpy as np
from lxml import etree
from copy import deepcopy
from ..mpicomm.mpiutils import get_comm_rank_size

def register_parallel_xdmf(mb, path='./', grid_path='./'):

    comm,rank,size = get_comm_rank_size()
    #the mb with BLock0 must get a list of all other block's ni,nj,nj
    my_block_list = [blk.nblki for blk in mb]
    my_ni_list = [[blk.ni, blk.nj, blk.nk] for blk in mb]

    if rank == 0:
        total_block_list = [[i for i in my_block_list]]
        total_ni_list = [[i for i in my_ni_list]]
    else:
        total_block_list = None
        total_ni_list = None

    for sendrank in range(1,size):
        #Send block list
        if rank == sendrank:
            tag = int(f'1{rank}201')
            comm.send(my_block_list, dest=0, tag=tag)
        #recv block list
        elif rank == 0:
            tag = int(f'1{sendrank}201')
            recv_block_list = comm.recv(source=sendrank, tag=tag)
            total_block_list.append(recv_block_list)
        else:
            pass

        #send ni list
        if rank == sendrank:
            tag = int(f'1{rank}201')
            comm.send(my_ni_list, dest=0, tag=tag)
        #recv ni list
        elif rank == 0:
            tag = int(f'1{sendrank}201')
            recv_ni_list = comm.recv(source=sendrank, tag=tag)
            total_ni_list.append(recv_ni_list)
        else:
            pass

    #Flatten the list, then sort in block order
    if rank == 0:
        total_block_list = [nblki for l in total_block_list for nblki in l]
        total_ni_list = [ni for l in total_ni_list for ni in l]
        total_block_list, total_ni_list = (list(t) for t in zip(*sorted(zip(total_block_list,
                                                                            total_ni_list))))

    # Create the xml for all the blocks
    if rank == 0:
        xdmf_elem = etree.Element('Xdmf')
        xdmf_elem.set('Version', '2')

        domain_elem = etree.SubElement(xdmf_elem, 'Domain')

        grid_elem = etree.SubElement(domain_elem, 'Grid')
        grid_elem.set('Name','PEREGRINE Output')
        grid_elem.set('GridType', 'Collection')
        grid_elem.set('CollectionType','Spatial')

        for nblki,n in zip(total_block_list, total_ni_list):
            ni = n[0]
            nj = n[1]
            nk = n[2]

            extent = ni*nj*nk
            extent_cc = (ni-1)*(nj-1)*(nk-1)

            block_elem = etree.Element('Grid')
            block_elem.set('Name',f'B{nblki:06d}')

            time_elem = etree.SubElement(block_elem, 'Time')
            time_elem.set('Value', str(mb.tme))

            topology_elem = etree.SubElement(block_elem, 'Topology')
            topology_elem.set('TopologyType', '3DSMesh')
            topology_elem.set('NumberOfElements', f'{nk} {nj} {ni}')

            geometry_elem = etree.SubElement(block_elem, 'Geometry')
            geometry_elem.set('GeometryType', 'X_Y_Z')

            data_x_elem = etree.SubElement(geometry_elem, 'DataItem')
            data_x_elem.set('ItemType', 'Hyperslab')
            data_x_elem.set('Dimensions', f'{nk} {nj} {ni}')
            data_x_elem.set('Type', 'HyperSlab')
            data_x1_elem = etree.SubElement(data_x_elem, 'DataItem')
            data_x1_elem.set('DataType', 'Int')
            data_x1_elem.set('Dimensions', '3')
            data_x1_elem.set('Format', 'XML')
            data_x1_elem.text = f'0 1 {extent}'
            data_x2_elem = etree.SubElement(data_x_elem, 'DataItem')
            data_x2_elem.set('NumberType', 'Float')
            data_x2_elem.set('ItemType', 'Uniform')
            data_x2_elem.set('Dimensions', f'{extent}')
            data_x2_elem.set('Precision', '4')
            data_x2_elem.set('Format', 'HDF')
            data_x2_elem.text = f'{grid_path}/gv.{nblki:06d}.h5:/coordinates/x'

            geometry_elem.append(deepcopy(data_x_elem))
            geometry_elem[-1][1].text = f'{grid_path}/gv.{nblki:06d}.h5:/coordinates/y'

            geometry_elem.append(deepcopy(data_x_elem))
            geometry_elem[-1][1].text = f'{grid_path}/gv.{nblki:06d}.h5:/coordinates/z'

            #Only solvers will call this
            names = ['rho','p','u','v','w','T']+mb[0].species_names

            name = names[0]
            #Attributes
            attribute_elem = etree.SubElement(block_elem, 'Attribute')
            attribute_elem.set('Name', name)
            attribute_elem.set('AttributeType', 'Scalar')
            attribute_elem.set('Center', 'Cell')
            data_res_elem = etree.SubElement(attribute_elem, 'DataItem')
            data_res_elem.set('ItemType', 'Hyperslab')
            data_res_elem.set('Dimensions', f'{nk-1} {nj-1} {ni-1}')
            data_res_elem.set('Type', 'HyperSlab')
            data_res1_elem = etree.SubElement(data_res_elem, 'DataItem')
            data_res1_elem.set('DataType', 'Int')
            data_res1_elem.set('Dimensions', '3')
            data_res1_elem.set('Format', 'XML')
            data_res1_elem.text = f'0 1 {extent_cc}'
            data_res2_elem = etree.SubElement(data_res_elem, 'DataItem')
            data_res2_elem.set('NumberType', 'Float')
            data_res2_elem.set('Dimensions', f'{extent_cc}')
            data_res2_elem.set('Precision', '4')
            data_res2_elem.set('Format', 'HDF')

            text = f'q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}'
            data_res2_elem.text = text

            for name in names[1::]:
                block_elem.append(deepcopy(attribute_elem))
                block_elem[-1].set('Name', name)
                text = f'q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}'
                block_elem[-1][0][1].text = text

            grid_elem.append(deepcopy(block_elem))


        #We add the et to the zeroth ranks mb object
        mb.parallel_xmf = etree.ElementTree(xdmf_elem)
    else:
        mb.parallel_xmf = None


def parallel_write_restart(mb, path='./', grid_path='./', precision='double'):

    comm,rank,size = get_comm_rank_size()

    if precision == 'double':
        fdtype = 'float64'
    else:
        fdtype = 'float32'

    for blk in mb:

        extent = blk.ni*blk.nj*blk.nk
        extent_cc = (blk.ni-1)*(blk.nj-1)*(blk.nk-1)

        file_name = f'{path}/q.{mb.nrt:08d}.{blk.nblki:06d}.h5'

        with h5py.File(file_name, 'w') as qf:

            qf.create_group('iter')
            qf['iter'].create_dataset('nrt', shape=(1,), dtype='int32')
            qf['iter'].create_dataset('tme', shape=(1,), dtype='float64')

            dset = qf['iter']['nrt']
            dset[0] = blk.nrt
            dset = qf['iter']['tme']
            dset[0] = blk.tme

            qf.create_group('results')

            if blk.block_type == 'solver':
                dset_name = 'rho'
                qf['results'].create_dataset(dset_name, shape=(extent_cc,), dtype=fdtype)
                dset = qf['results'][dset_name]
                dset[:] = blk.array['Q'][1:-1,1:-1,1:-1,0].ravel(order='F')
            names = ['p','u','v','w','T']+blk.species_names[0:-1]
            for j in range(len(names)):
                dset_name = names[j]
                qf['results'].create_dataset(dset_name, shape=(extent_cc,), dtype=fdtype)
                dset = qf['results'][dset_name]
                dset[:] = blk.array['q'][1:-1,1:-1,1:-1,j].ravel(order='F')
            #Compute the nth species here
            dset_name = blk.species_names[-1]
            qf['results'].create_dataset(dset_name, shape=(extent_cc,), dtype=fdtype)
            dset = qf['results'][dset_name]
            if blk.ns > 1:
                dset[:] = 1.0 - np.sum(blk.array['q'][1:-1,1:-1,1:-1,5::], axis=-1).ravel()
            elif blk.ns == 1:
                dset[:] = 1.0

    #Write out xmf
    if rank == 0:
        et = mb.parallel_xmf

        for grid in et.getroot()[0][0]:
            nblki = int(grid.get('Name')[1::])
            time = grid.find('Time')
            time.set('Value', str(mb.tme))
            time = grid.find('Time')
            for var in grid.findall('Attribute'):
                name = var.get('Name')
                text = f'q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}'
                var[0][-1].text = text

        save_file = f'{path}/q.{mb.nrt:08d}.xmf'
        et.write(save_file, pretty_print=True, encoding="UTF-8", xml_declaration=True)
