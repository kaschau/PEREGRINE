# -*- coding: utf-8 -*-

import h5py
from lxml import etree
from copy import deepcopy


def write_restart(mb, path='./', precision='single',ishot=0):
    '''This function produces an hdf5 file from a raptorpy.multiblock.restart for viewing in Paraview.

    Parameters
    ----------

    mb : raptorpy.multiblock.restart

    file_path : str
        Path to location to write output files

    precision : str
        Options - 'single' for single precision
                  'double' for double precision

    Returns
    -------
    None

    '''

    if precision == 'single':
        fdtype = '<f4'
    else:
        fdtype = '<f8'

    xdmf_elem = etree.Element('Xdmf')
    xdmf_elem.set('Version', '2')

    domain_elem = etree.SubElement(xdmf_elem, 'Domain')

    grid_elem = etree.SubElement(domain_elem, 'Grid')
    grid_elem.set('Name','RAPTOR Output')
    grid_elem.set('GridType', 'Collection')
    grid_elem.set('CollectionType','Spatial')

    for blk in mb:

        extent = blk.nx*blk.ny*blk.nz
        extent_cc = (blk.nx-1)*(blk.ny-1)*(blk.nz-1)

        if ishot < 0:
            file_name = '{}/qv.{:06d}.h5'.format(path, blk.nblki)
        else:
            file_name = '{}/qv.{:08d}.{:06d}.h5'.format(path, mb.nrt, blk.nblki)
            
        with h5py.File(file_name, 'w') as qf:
            qf.create_group('results')
            qf.create_group('dimensions')

            qf['dimensions'].create_dataset('nx', shape=(1,), dtype='int32')
            qf['dimensions'].create_dataset('ny', shape=(1,), dtype='int32')
            qf['dimensions'].create_dataset('nz', shape=(1,), dtype='int32')

            dset = qf['dimensions']['nx']
            dset[0] = blk.nx
            dset = qf['dimensions']['ny']
            dset[0] = blk.ny
            dset = qf['dimensions']['nz']
            dset[0] = blk.nz


            for j,res in enumerate(blk.qv):
                dset_name = 'qv_{:02d}'.format(j+1)
                qf['results'].create_dataset(dset_name, shape=(extent_cc,), dtype=fdtype)
                dset = qf['results'][dset_name]
                dset[:] = res[1:-1,1:-1,1:-1].ravel()

        block_elem = etree.Element('Grid')
        block_elem.set('Name','B{:06d}'.format(blk.nblki))

        topology_elem = etree.SubElement(block_elem, 'Topology')
        topology_elem.set('TopologyType', '3DSMesh')
        topology_elem.set('NumberOfElements', '{} {} {}'.format(blk.nz,blk.ny,blk.nx))

        geometry_elem = etree.SubElement(block_elem, 'Geometry')
        geometry_elem.set('GeometryType', 'X_Y_Z')

        data_x_elem = etree.SubElement(geometry_elem, 'DataItem')
        data_x_elem.set('ItemType', 'Hyperslab')
        data_x_elem.set('Dimensions', '{} {} {}'.format(blk.nz,blk.ny,blk.nx))
        data_x_elem.set('Type', 'HyperSlab')
        data_x1_elem = etree.SubElement(data_x_elem, 'DataItem')
        data_x1_elem.set('DataType', 'Int')
        data_x1_elem.set('Dimensions', '3')
        data_x1_elem.set('Format', 'XML')
        data_x1_elem.text = '0 1 {}'.format(extent)
        data_x2_elem = etree.SubElement(data_x_elem, 'DataItem')
        data_x2_elem.set('NumberType', 'Float')
        data_x2_elem.set('ItemType', 'Uniform')
        data_x2_elem.set('Dimensions', '{}'.format(extent))
        data_x2_elem.set('Precision', '4')
        data_x2_elem.set('Format', 'HDF')
        data_x2_elem.text = 'gv.{:06d}.h5:/coordinates/x'.format(blk.nblki)

        geometry_elem.append(deepcopy(data_x_elem))
        geometry_elem[-1][1].text = 'gv.{:06d}.h5:/coordinates/y'.format(blk.nblki)

        geometry_elem.append(deepcopy(data_x_elem))
        geometry_elem[-1][1].text = 'gv.{:06d}.h5:/coordinates/z'.format(blk.nblki)

        #Attributes
        attribute_elem = etree.SubElement(block_elem, 'Attribute')
        attribute_elem.set('Name', 'qv_01')
        attribute_elem.set('AttributeType', 'Scalar')
        attribute_elem.set('Center', 'Cell')
        data_res_elem = etree.SubElement(attribute_elem, 'DataItem')
        data_res_elem.set('ItemType', 'Hyperslab')
        data_res_elem.set('Dimensions', '{} {} {}'.format(blk.nz-1,blk.ny-1,blk.nx-1))
        data_res_elem.set('Type', 'HyperSlab')
        data_res1_elem = etree.SubElement(data_res_elem, 'DataItem')
        data_res1_elem.set('DataType', 'Int')
        data_res1_elem.set('Dimensions', '3')
        data_res1_elem.set('Format', 'XML')
        data_res1_elem.text = '0 1 {}'.format(extent_cc)
        data_res2_elem = etree.SubElement(data_res_elem, 'DataItem')
        data_res2_elem.set('NumberType', 'Float')
        data_res2_elem.set('Dimensions', '{}'.format(extent_cc))
        data_res2_elem.set('Precision', '4')
        data_res2_elem.set('Format', 'HDF')
        if ishot < 0:
            text = 'qv.{:06d}.h5:/results/qv_01'.format(blk.nblki)
        else:
            text = 'qv.{:08d}.{:06d}.h5:/results/qv_01'.format(mb.nrt, blk.nblki)
        data_res2_elem.text = text

        for j in range(2,len(blk.qv)+1):
            block_elem.append(deepcopy(attribute_elem))
            block_elem[-1].set('Name', 'qv_{:02d}'.format(j))
            if ishot < 0:
                text = 'qv.{:06d}.h5:/results/qv_{:02d}'.format(blk.nblki, j)
            else:
                text = 'qv.{:08d}.{:06d}.h5:/results/qv_{:02d}'.format(mb.nrt, blk.nblki, j)
            block_elem[-1][0][1].text = text

        grid_elem.append(deepcopy(block_elem))

        progress_bar(blk.nblki, len(mb), 'Writing out block {}'.format(blk.nblki))

    et = etree.ElementTree(xdmf_elem)
    save_file = '{}/qv.xmf'.format(path)
    et.write(save_file, pretty_print=True, encoding="UTF-8", xml_declaration=True)
