import unittest
import pytest
import peregrinepy as pg
import numpy as np

np.random.seed(111)

class twoblock123:
    def __init__(self,stack):
       self.config = pg.files.config_file()
       self.config['RunTime']['ngls'] = 1
       self.mb = pg.multiblock(2,self.config)

       if stack == 'ii':
           mb_dim = [2,1,1]
       elif stack == 'jj':
           mb_dim = [1,2,1]
       elif stack == 'kk':
           mb_dim = [1,1,2]

       pg.grid.create.multiblock_cube(self.mb, mb_dimensions=mb_dim,
                                      dimensions_perblock=[2,2,2])

       self.mb[0].connectivity['2']['comm_rank'] = 0
       self.mb[1].connectivity['1']['comm_rank'] = 0

       self.shape = self.mb[0].array['x'].shape

def compare_arrays(a1,a2):
    return np.all(np.equal(a1,a2))


##################################################################################
##### Test for all positive i aligned orientations
##################################################################################
def test_123():
    tb = twoblock123('ii')

    for blk in tb.mb:
        blk.array['x'] = np.random.random((tb.shape))
        blk.array['y'] = np.random.random((tb.shape))
        blk.array['z'] = np.random.random((tb.shape))

    #Reorient and update communication info
    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)
    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_s3['2']],
                                       tb.mb[1].array[var][tb.mb[1].slice_r3['1']]))
        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_r3['2']],
                                       tb.mb[1].array[var][tb.mb[1].slice_s3['1']]))

    assert (False not in passfail)

def test_135():
    tb = twoblock123('ii')

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '135'
    tb.mb[1].connectivity['1']['orientation'] = '162'
    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    #Un-reorient to compare
    for var in ['x','y','z']:
        tb.mb[1].array[var][:,:,:] = np.rot90(tb.mb[1].array[var][:,:,:], 1, (1,2))

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_s3['2']],
                                       tb.mb[1].array[var][tb.mb[1].slice_r3['1']]))
        passfail.append(compare_arrays(tb.mb[1].array[var][tb.mb[1].slice_s3['1']],
                                       tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

#def test_156():
#    tb = twoblock123('ii')
#
#    for blk in tb.mb:
#        blk.array['x'][:,:,:] = np.random.random((tb.shape))
#        blk.array['y'][:,:,:] = np.random.random((tb.shape))
#        blk.array['z'][:,:,:] = np.random.random((tb.shape))
#
#    #Reorient second block and update communication info
#    tb.mb[0].connectivity['2']['orientation'] = '135'
#    tb.mb[1].connectivity['1']['orientation'] = '162'
#    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)
#
#    #Execute communication
#    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])
#
#    #Un-reorient to compare
#    for var in ['x','y','z']:
#        tb.mb[1].array[var][:,:,:] = np.rot90(tb.mb[1].array[var][:,:,:], 1, (1,2))
#
#    passfail = []
#    for var in ['x','y','z']:
#        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_s3['2']],
#                                       tb.mb[1].array[var][tb.mb[1].slice_r3['1']]))
#        passfail.append(compare_arrays(tb.mb[1].array[var][tb.mb[1].slice_s3['1']],
#                                       tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))
#
#    assert (False not in passfail)
