import unittest
import pytest
import peregrinepy as pg
import numpy as np

#np.random.seed(111)

class twoblock123:
    def __init__(self):
       self.config = pg.files.config_file()
       self.mb = pg.multiblock.solver(2,self.config)

       pg.grid.create.multiblock_cube(self.mb,
                                      mb_dimensions=[2,1,1],
                                      dimensions_perblock=[6,4,2],
                                      lengths=[2,1,1])

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
    pg.mpicomm.blockcomm.set_block_communication(tb.mb)
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

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ['x','y','z']:
        blk1.array[var] = np.moveaxis(np.flip(blk1.array[var],axis=2),(0,1,2),(0,2,1))
    blk1.nj = tb.shape[2]-2
    blk1.nk = tb.shape[1]-2

    #Reorient second block and update communication info
    blk0.connectivity['2']['orientation'] = '135'
    blk1.connectivity['1']['orientation'] = '162'

    pg.mpicomm.blockcomm.set_block_communication(tb.mb)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    check0 = True
    check1 = True
    passfail=[]
    for var in ['x','y','z']:
        for k in range(tb.shape[2]):
            for j in range(tb.shape[1]):
                check0 = np.equal(blk0.array[var][-3,j,k],
                                    blk1.array[var][ 0,-(k+1),j])
                check1 = np.equal(blk0.array[var][-1,j,k],
                                    blk1.array[var][ 2,-(k+1),j])
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert (False not in passfail)

###################################################################################
###### Test for all positive j aligned orientations
###################################################################################
def test_231():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ['x','y','z']:
        blk1.array[var] = np.moveaxis(blk1.array[var],(0,1,2),(1,2,0))
    blk1.ni = tb.shape[2]-2
    blk1.nj = tb.shape[0]-2
    blk1.nk = tb.shape[1]-2

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

    pg.mpicomm.blockcomm.set_block_communication(tb.mb)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    check0 = True
    check1 = True
    passfail=[]
    for var in ['x','y','z']:
        for k in range(tb.shape[2]):
            for j in range(tb.shape[1]):
                check0 = np.equal(blk0.array[var][-3,j,k],
                                    blk1.array[var][k,0,j])
                check1 = np.equal(blk0.array[var][-1,j,k],
                                    blk1.array[var][k,2,j])
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert (False not in passfail)

###################################################################################
###### Test for all positive k aligned orientations
###################################################################################
def test_321():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ['x','y','z']:
        blk1.array[var] = np.moveaxis(blk1.array[var],(0,1,2),(2,0,1))
    blk1.ni = tb.shape[1]-2
    blk1.nj = tb.shape[2]-2
    blk1.nk = tb.shape[0]-2

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

    pg.mpicomm.blockcomm.set_block_communication(tb.mb)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])

    check0 = True
    check1 = True
    passfail=[]
    for var in ['x','y','z']:
        for k in range(tb.shape[2]):
            for j in range(tb.shape[1]):
                check0 = np.equal(blk0.array[var][-3,j,k],
                                    blk1.array[var][j,k,0])
                check1 = np.equal(blk0.array[var][-1,j,k],
                                    blk1.array[var][j,k,2])
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)
    passfail = []

    assert (False not in passfail)

###################################################################################
###### Test for all negative i aligned orientations
###################################################################################
#def test_432():
#    tb = twoblock123()
#
#    for blk in tb.mb:
#        blk.array['x'][:,:,:] = np.random.random((tb.shape))
#        blk.array['y'][:,:,:] = np.random.random((tb.shape))
#        blk.array['z'][:,:,:] = np.random.random((tb.shape))
#
#    #Reorient second block and update communication info
#    tb.mb[0].connectivity['2']['orientation'] = '432'
#
#    tb.mb[1].connectivity['1']['neighbor'] = None
#    tb.mb[1].connectivity['1']['bc'] = 's1'
#    tb.mb[1].connectivity['1']['orientation'] = None
#    tb.mb[1].connectivity['1']['comm_rank'] = None
#
#    tb.mb[1].connectivity['2']['neighbor'] = 0
#    tb.mb[1].connectivity['2']['bc'] = 'b0'
#    tb.mb[1].connectivity['2']['orientation'] = '432'
#    tb.mb[1].connectivity['2']['comm_rank'] = 0
#
#    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)
#
#    #Execute communication
#    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])
#
#    passfail = []
#    for var in ['x','y','z']:
#        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
#                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['2']]))
#
#        passfail.append(compare_arrays(tb.mb[1].orient['2'](tb.mb[1].array[var][tb.mb[1].slice_s3['2']]),
#                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))
#
#    assert (False not in passfail)
#
###################################################################################
###### Test for all negative j aligned orientations
###################################################################################
#def test_513():
#    tb = twoblock123()
#
#    for blk in tb.mb:
#        blk.array['x'][:,:,:] = np.random.random((tb.shape))
#        blk.array['y'][:,:,:] = np.random.random((tb.shape))
#        blk.array['z'][:,:,:] = np.random.random((tb.shape))
#
#    #Reorient second block and update communication info
#    tb.mb[0].connectivity['2']['orientation'] = '513'
#
#    tb.mb[1].connectivity['1']['neighbor'] = None
#    tb.mb[1].connectivity['1']['bc'] = 's1'
#    tb.mb[1].connectivity['1']['orientation'] = None
#    tb.mb[1].connectivity['1']['comm_rank'] = None
#
#    tb.mb[1].connectivity['4']['neighbor'] = 0
#    tb.mb[1].connectivity['4']['bc'] = 'b0'
#    tb.mb[1].connectivity['4']['orientation'] = '243'
#    tb.mb[1].connectivity['4']['comm_rank'] = 0
#
#    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)
#
#    #Execute communication
#    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])
#
#    passfail = []
#    for var in ['x','y','z']:
#        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
#                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['4']]))
#
#        passfail.append(compare_arrays(tb.mb[1].orient['4'](tb.mb[1].array[var][tb.mb[1].slice_s3['4']]),
#                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))
#
#    assert (False not in passfail)
#
###################################################################################
###### Test for all negative k aligned orientations
###################################################################################
#def test_621():
#    tb = twoblock123()
#
#    for blk in tb.mb:
#        blk.array['x'][:,:,:] = np.random.random((tb.shape))
#        blk.array['y'][:,:,:] = np.random.random((tb.shape))
#        blk.array['z'][:,:,:] = np.random.random((tb.shape))
#
#    #Reorient second block and update communication info
#    tb.mb[0].connectivity['2']['orientation'] = '621'
#
#    tb.mb[1].connectivity['1']['neighbor'] = None
#    tb.mb[1].connectivity['1']['bc'] = 's1'
#    tb.mb[1].connectivity['1']['orientation'] = None
#    tb.mb[1].connectivity['1']['comm_rank'] = None
#
#    tb.mb[1].connectivity['6']['neighbor'] = 0
#    tb.mb[1].connectivity['6']['bc'] = 'b0'
#    tb.mb[1].connectivity['6']['orientation'] = '324'
#    tb.mb[1].connectivity['6']['comm_rank'] = 0
#
#    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)
#
#    #Execute communication
#    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z'])
#
#    passfail = []
#    for var in ['x','y','z']:
#        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
#                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['6']]))
#
#        passfail.append(compare_arrays(tb.mb[1].orient['6'](tb.mb[1].array[var][tb.mb[1].slice_s3['6']]),
#                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))
#
#    assert (False not in passfail)
