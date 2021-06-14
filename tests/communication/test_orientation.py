import unittest
import pytest
import peregrinepy as pg
import numpy as np

np.random.seed(111)

class twoblock123:
    def __init__(self):
       self.config = pg.files.config_file()
       self.config['RunTime']['ngls'] = 2
       self.mb = pg.multiblock(2,self.config)

       pg.grid.create.multiblock_cube(self.mb, mb_dimensions=[2,1,1],
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

    tb = twoblock123()

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
    tb = twoblock123()

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
        tb.mb[1].array[var][:,:,:] = np.rot90(tb.mb[1].array[var], 1, (1,2))

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_s3['2']],
                                       tb.mb[1].array[var][tb.mb[1].slice_r3['1']]))
        passfail.append(compare_arrays(tb.mb[1].array[var][tb.mb[1].slice_s3['1']],
                                       tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

def test_156():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '156'
    tb.mb[1].connectivity['1']['orientation'] = '156'
    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    #Un-reorient to compare
    for var in ['x','y','z']:
        tb.mb[1].array[var][:,:,:] = np.rot90(tb.mb[1].array[var][:,:,:], 2, (1,2))

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_s3['2']],
                                       tb.mb[1].array[var][tb.mb[1].slice_r3['1']]))
        passfail.append(compare_arrays(tb.mb[1].array[var][tb.mb[1].slice_s3['1']],
                                       tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

def test_162():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '162'
    tb.mb[1].connectivity['1']['orientation'] = '135'
    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    #Un-reorient to compare
    for var in ['x','y','z']:
        tb.mb[1].array[var][:,:,:] = np.rot90(tb.mb[1].array[var][:,:,:], 1, (2,1))

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].array[var][tb.mb[0].slice_s3['2']],
                                       tb.mb[1].array[var][tb.mb[1].slice_r3['1']]))
        passfail.append(compare_arrays(tb.mb[1].array[var][tb.mb[1].slice_s3['1']],
                                       tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

##################################################################################
##### Test for all positive j aligned orientations
##################################################################################
def test_231():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '231'

    tb.mb[1].connectivity['1']['neighbor'] = None
    tb.mb[1].connectivity['1']['bc'] = 's1'
    tb.mb[1].connectivity['1']['orientation'] = None
    tb.mb[1].connectivity['1']['comm_rank'] = None

    tb.mb[1].connectivity['3']['neighbor'] = 0
    tb.mb[1].connectivity['3']['bc'] = 'b0'
    tb.mb[1].connectivity['3']['orientation'] = '312'
    tb.mb[1].connectivity['3']['comm_rank'] = 0

    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['3']]))

        passfail.append(compare_arrays(tb.mb[1].orient['3'](tb.mb[1].array[var][tb.mb[1].slice_s3['3']]),
                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

##################################################################################
##### Test for all positive k aligned orientations
##################################################################################
def test_321():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '312'

    tb.mb[1].connectivity['1']['neighbor'] = None
    tb.mb[1].connectivity['1']['bc'] = 's1'
    tb.mb[1].connectivity['1']['orientation'] = None
    tb.mb[1].connectivity['1']['comm_rank'] = None

    tb.mb[1].connectivity['5']['neighbor'] = 0
    tb.mb[1].connectivity['5']['bc'] = 'b0'
    tb.mb[1].connectivity['5']['orientation'] = '231'
    tb.mb[1].connectivity['5']['comm_rank'] = 0

    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['5']]))

        passfail.append(compare_arrays(tb.mb[1].orient['5'](tb.mb[1].array[var][tb.mb[1].slice_s3['5']]),
                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

##################################################################################
##### Test for all negative i aligned orientations
##################################################################################
def test_432():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '432'

    tb.mb[1].connectivity['1']['neighbor'] = None
    tb.mb[1].connectivity['1']['bc'] = 's1'
    tb.mb[1].connectivity['1']['orientation'] = None
    tb.mb[1].connectivity['1']['comm_rank'] = None

    tb.mb[1].connectivity['2']['neighbor'] = 0
    tb.mb[1].connectivity['2']['bc'] = 'b0'
    tb.mb[1].connectivity['2']['orientation'] = '432'
    tb.mb[1].connectivity['2']['comm_rank'] = 0

    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['2']]))

        passfail.append(compare_arrays(tb.mb[1].orient['2'](tb.mb[1].array[var][tb.mb[1].slice_s3['2']]),
                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

##################################################################################
##### Test for all negative j aligned orientations
##################################################################################
def test_513():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '513'

    tb.mb[1].connectivity['1']['neighbor'] = None
    tb.mb[1].connectivity['1']['bc'] = 's1'
    tb.mb[1].connectivity['1']['orientation'] = None
    tb.mb[1].connectivity['1']['comm_rank'] = None

    tb.mb[1].connectivity['4']['neighbor'] = 0
    tb.mb[1].connectivity['4']['bc'] = 'b0'
    tb.mb[1].connectivity['4']['orientation'] = '243'
    tb.mb[1].connectivity['4']['comm_rank'] = 0

    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['4']]))

        passfail.append(compare_arrays(tb.mb[1].orient['4'](tb.mb[1].array[var][tb.mb[1].slice_s3['4']]),
                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)

##################################################################################
##### Test for all negative k aligned orientations
##################################################################################
def test_621():
    tb = twoblock123()

    for blk in tb.mb:
        blk.array['x'][:,:,:] = np.random.random((tb.shape))
        blk.array['y'][:,:,:] = np.random.random((tb.shape))
        blk.array['z'][:,:,:] = np.random.random((tb.shape))

    #Reorient second block and update communication info
    tb.mb[0].connectivity['2']['orientation'] = '621'

    tb.mb[1].connectivity['1']['neighbor'] = None
    tb.mb[1].connectivity['1']['bc'] = 's1'
    tb.mb[1].connectivity['1']['orientation'] = None
    tb.mb[1].connectivity['1']['comm_rank'] = None

    tb.mb[1].connectivity['6']['neighbor'] = 0
    tb.mb[1].connectivity['6']['bc'] = 'b0'
    tb.mb[1].connectivity['6']['orientation'] = '324'
    tb.mb[1].connectivity['6']['comm_rank'] = 0

    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    passfail = []
    for var in ['x','y','z']:
        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['6']]))

        passfail.append(compare_arrays(tb.mb[1].orient['6'](tb.mb[1].array[var][tb.mb[1].slice_s3['6']]),
                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))

    assert (False not in passfail)
