# -*- coding: utf-8 -*-
import h5py
from lxml import etree
from copy import deepcopy

def write_grid(mb_data, path='./', precision='single'):
    '''This function produces an hdf5 file from a raptorpy.multiblock.grid (or a descendant) for viewing in Paraview.

    Parameters
    ----------

    mb_data : raptorpy.multiblock.grid (or a descendant)

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

    for blk in mb_data:

        with h5py.File('{}/gv.{:06d}.h5'.format(path, blk.nblki), 'w') as f:
            f.create_group('coordinates')
            f.create_group('dimensions')

            f['dimensions'].create_dataset('nx', shape=(1,), dtype='int32')
            f['dimensions'].create_dataset('ny', shape=(1,), dtype='int32')
            f['dimensions'].create_dataset('nz', shape=(1,), dtype='int32')

            dset = f['dimensions']['nx']
            dset[0] = blk.nx
            dset = f['dimensions']['ny']
            dset[0] = blk.ny
            dset = f['dimensions']['nz']
            dset[0] = blk.nz

            extent = blk.nx*blk.ny*blk.nz
            f['coordinates'].create_dataset('x', shape=(extent,), dtype=fdtype)
            f['coordinates'].create_dataset('y', shape=(extent,), dtype=fdtype)
            f['coordinates'].create_dataset('z', shape=(extent,), dtype=fdtype)

            dset = f['coordinates']['x']
            dset[:] = blk.x.ravel()
            dset = f['coordinates']['y']
            dset[:] = blk.y.ravel()
            dset = f['coordinates']['z']
            dset[:] = blk.z.ravel()

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

        grid_elem.append(deepcopy(block_elem))

        progress_bar(blk.nblki, len(mb_data), 'Writing out block {}'.format(blk.nblki))

    et = etree.ElementTree(xdmf_elem)
    save_file = '{}/gv.xmf'.format(path)
    et.write(save_file, pretty_print=True, encoding="UTF-8", xml_declaration=True)
