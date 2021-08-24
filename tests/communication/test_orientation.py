import unittest
import pytest
import peregrinepy as pg
import numpy as np

#np.random.seed(111)

class twoblock123:
    def __init__(self):
        self.config = pg.files.config_file()
        self.mb = pg.multiblock.solver(2,['Air'])

        pg.grid.create.multiblock_cube(self.mb,
                                       mb_dimensions=[2,1,1],
                                       dimensions_perblock=[6,3,2],
                                       lengths=[2,1,1])

        blk0 = self.mb[0]
        blk1 = self.mb[1]
        blk0.get_face(2).comm_rank = 0
        blk1.get_face(1).comm_rank = 0

        self.xshape = self.mb[0].array['x'].shape
        self.qshape = self.mb[0].array['q'].shape

        for blk in self.mb:
            blk.array['x'][:] = np.random.random((self.xshape))
            blk.array['y'][:] = np.random.random((self.xshape))
            blk.array['z'][:] = np.random.random((self.xshape))
            blk.array['q'][:] = np.random.random((self.qshape))


##################################################################################
##### Test for all positive i aligned orientations
##################################################################################
def test_123():

    tb = twoblock123()
    blk0 = tb.mb[0]
    blk1 = tb.mb[1]

    #Reorient and update communication info
    pg.mpicomm.blockcomm.set_block_communication(tb.mb)
    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z','q'])

    passfail=[]
    for var,shape,off in zip(['x','y','z','q'],
                              [tb.xshape,tb.xshape,tb.xshape,tb.qshape],
                              [0,0,0,1]):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(blk0.array[var][-3+off,j,k] == blk1.array[var][ 0,j,k])
                check1 = np.all(blk0.array[var][-1,j,k] == blk1.array[var][2-off,j,k])
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert (False not in passfail)

def test_135():
    tb = twoblock123()
    blk0 = tb.mb[0]
    blk1 = tb.mb[1]


    for var in ['x','y','z','q']:
        blk1.array[var] = np.moveaxis(np.flip(blk1.array[var],axis=2),(0,1,2),(0,2,1))
    blk1.nj = tb.xshape[2]-2
    blk1.nk = tb.xshape[1]-2

    #Reorient second block and update communication info
    blk0.get_face_conn(2)['orientation'] = '135'
    blk1.get_face_conn(1)['orientation'] = '162'

    pg.mpicomm.blockcomm.set_block_communication(tb.mb)

    #Execute communication
    pg.mpicomm.blockcomm.communicate(tb.mb,['x','y','z','q'])

    passfail=[]
    for var,shape,off in zip(['x','y','z','q'],
                              [tb.xshape,tb.xshape,tb.xshape,tb.qshape],
                              [0,0,0,1]):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(blk0.array[var][-3+off,j,k] == blk1.array[var][ 0,-(k+1),j])
                check1 = np.all(blk0.array[var][-1,j,k] == blk1.array[var][2-off,-(k+1),j])
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert (False not in passfail)

# ###################################################################################
# ###### Test for all positive j aligned orientations
# ###################################################################################
# def test_231():
#     tb = twoblock123()
#
#     blk0 = tb.mb[0]
#     blk1 = tb.mb[1]
#     for var in ['x','y','z']:
#         blk1.array[var] = np.moveaxis(blk1.array[var],(0,1,2),(1,2,0))
#     blk1.ni = tb.shape[2]-2
#     blk1.nj = tb.shape[0]-2
#     blk1.nk = tb.shape[1]-2
#
#     #Reorient second block and update communication info
#     blk0.get_face_conn(2)['orientation'] = '231'
#
#     blk1.get_face_conn(1)['neighbor'] = None
#     blk1.get_face_conn(1)['bctype'] = 's1'
#     blk1.get_face_conn(1)['orientation'] = None
#     blk1.get_face(1).comm_rank = None
#
#     blk1.get_face_conn(3)['neighbor'] = 0
#     blk1.get_face_conn(3)['bctype'] = 'b0'
#     blk1.get_face_conn(3)['orientation'] = '312'
#     blk1.get_face(3).comm_rank = 0
#
#     pg.mpicomm.blockcomm.set_block_communication(tb.mb)
#
#     #Execute communication
#     pg.mpicomm.blockcomm.communicate3(tb.mb,['x','y','z'])
#
#     check0 = True
#     check1 = True
#     passfail=[]
#     for var in ['x','y','z']:
#         for k in range(tb.shape[2]):
#             for j in range(tb.shape[1]):
#                 check0 = np.equal(blk0.array[var][-3,j,k],
#                                   blk1.array[var][k,0,j])
#                 check1 = np.equal(blk0.array[var][-1,j,k],
#                                   blk1.array[var][k,2,j])
#                 if not check0 or not check1:
#                     print(blk0.array[var][-3,j,k], blk1.array[var][k,0,j])
#                     break
#             if not check0 or not check1:
#                 print(blk0.array[var][-1,j,k], blk1.array[var][k,2,j])
#                 break
#         passfail.append(check0)
#         passfail.append(check1)
#
#     assert (False not in passfail)
#
# ###################################################################################
# ###### Test for all positive k aligned orientations
# ###################################################################################
# def test_321():
#     tb = twoblock123()
#
#     blk0 = tb.mb[0]
#     blk1 = tb.mb[1]
#     for var in ['x','y','z']:
#         blk1.array[var] = np.moveaxis(blk1.array[var],(0,1,2),(2,0,1))
#     blk1.ni = tb.shape[1]-2
#     blk1.nj = tb.shape[2]-2
#     blk1.nk = tb.shape[0]-2
#
#     #Reorient second block and update communication info
#     blk0.get_face_conn(2)['orientation'] = '312'
#
#     blk1.get_face_conn(1)['neighbor'] = None
#     blk1.get_face_conn(1)['bctype'] = 's1'
#     blk1.get_face_conn(1)['orientation'] = None
#     blk1.get_face(1).comm_rank = None
#
#     blk1.get_face_conn(5)['neighbor'] = 0
#     blk1.get_face_conn(5)['bctype'] = 'b0'
#     blk1.get_face_conn(5)['orientation'] = '231'
#     blk1.get_face(5).comm_rank = 0
#
#     pg.mpicomm.blockcomm.set_block_communication(tb.mb)
#
#     #Execute communication
#     pg.mpicomm.blockcomm.communicate3(tb.mb,['x','y','z'])
#
#     check0 = True
#     check1 = True
#     passfail=[]
#     for var in ['x','y','z']:
#         for k in range(tb.shape[2]):
#             for j in range(tb.shape[1]):
#                 check0 = np.equal(blk0.array[var][-3,j,k],
#                                   blk1.array[var][j,k,0])
#                 check1 = np.equal(blk0.array[var][-1,j,k],
#                                   blk1.array[var][j,k,2])
#                 if not check0 or not check1:
#                     print(blk0.array[var][-3,j,k], blk1.array[var][j,k,0])
#                     break
#             if not check0 or not check1:
#                 print(blk0.array[var][-1,j,k], blk1.array[var][j,k,2])
#                 break
#         passfail.append(check0)
#         passfail.append(check1)
#     passfail = []
#
#     assert (False not in passfail)
#
# ##################################################################################
# ##### Test for all negative i aligned orientations
# ##################################################################################
# def test_432():
#     tb = twoblock123()
#
#     blk0 = tb.mb[0]
#     blk1 = tb.mb[1]
#     for var in ['x','y','z']:
#         blk1.array[var] = np.moveaxis(blk1.array[var],(0,1,2),(0,2,1))
#     blk1.ni = tb.shape[0]-2
#     blk1.nj = tb.shape[2]-2
#     blk1.nk = tb.shape[1]-2
#
#     #Reorient second block and update communication info
#     blk0.get_face_conn(2)['orientation'] = '432'
#
#     blk1.get_face_conn(1)['neighbor'] = None
#     blk1.get_face_conn(1)['bctype'] = 's1'
#     blk1.get_face_conn(1)['orientation'] = None
#     blk1.get_face(1).comm_rank = None
#
#     blk1.get_face_conn(2)['neighbor'] = 0
#     blk1.get_face_conn(2)['bctype'] = 'b0'
#     blk1.get_face_conn(2)['orientation'] = '432'
#     blk1.get_face(2).comm_rank = 0
#
#     pg.mpicomm.blockcomm.set_block_communication(tb.mb)
#
#     #Execute communication
#     pg.mpicomm.blockcomm.communicate3(tb.mb,['x','y','z'])
#
#     check0 = True
#     check1 = True
#     passfail=[]
#
#     for var in ['x','y','z']:
#         for k in range(tb.shape[2]):
#             for j in range(tb.shape[1]):
#                 check0 = np.equal(blk0.array[var][-3,j,k],
#                                   blk1.array[var][-1,k,j])
#                 check1 = np.equal(blk0.array[var][-1,j,k],
#                                   blk1.array[var][-3,k,j])
#                 if not check0 or not check1:
#                     print(blk0.array[var][-3,j,k], blk1.array[var][-1,k,j])
#                     break
#             if not check0 or not check1:
#                 print(blk0.array[var][-1,j,k], blk1.array[var][-3,k,j])
#                 break
#         passfail.append(check0)
#         passfail.append(check1)
#
#     assert (False not in passfail)
#
# ##################################################################################
# ##### Test for all negative j aligned orientations
# ##################################################################################
# def test_513():
#     tb = twoblock123()
#
#     blk0 = tb.mb[0]
#     blk1 = tb.mb[1]
#     for var in ['x','y','z']:
#         blk1.array[var] = np.moveaxis(blk1.array[var],(0,1,2),(1,0,2))
#     blk1.ni = tb.shape[1]-2
#     blk1.nj = tb.shape[0]-2
#     blk1.nk = tb.shape[2]-2
#
#     #Reorient second block and update communication info
#     blk0.get_face_conn(2)['orientation'] = '513'
#
#     blk1.get_face_conn(1)['neighbor'] = None
#     blk1.get_face_conn(1)['bctype'] = 's1'
#     blk1.get_face_conn(1)['orientation'] = None
#     blk1.get_face(1).comm_rank = None
#
#     blk1.get_face_conn(4)['neighbor'] = 0
#     blk1.get_face_conn(4)['bctype'] = 'b0'
#     blk1.get_face_conn(4)['orientation'] = '243'
#     blk1.get_face(4).comm_rank = 0
#
#     pg.mpicomm.blockcomm.set_block_communication(tb.mb)
#
#     #Execute communication
#     pg.mpicomm.blockcomm.communicate3(tb.mb,['x','y','z'])
#
#     check0 = True
#     check1 = True
#     passfail=[]
#
#     for var in ['x','y','z']:
#         for k in range(tb.shape[2]):
#             for j in range(tb.shape[1]):
#                 check0 = np.equal(blk0.array[var][-3, j,k],
#                                   blk1.array[var][ j,-1,k])
#                 check1 = np.equal(blk0.array[var][-1, j,k],
#                                   blk1.array[var][ j,-3,k])
#                 if not check0 or not check1:
#                     print(blk0.array[var][-3,j,k], blk1.array[var][j,-1,k])
#                     break
#             if not check0 or not check1:
#                 print(blk0.array[var][-1,j,k], blk1.array[var][j,-3,k])
#                 break
#         passfail.append(check0)
#         passfail.append(check1)
#
#     assert (False not in passfail)
#
# ###################################################################################
# ###### Test for all negative k aligned orientations
# ###################################################################################
# #def test_621():
# #    tb = twoblock123()
# #
# #    for blk in tb.mb:
# #        blk.array['x'][:,:,:] = np.random.random((tb.shape))
# #        blk.array['y'][:,:,:] = np.random.random((tb.shape))
# #        blk.array['z'][:,:,:] = np.random.random((tb.shape))
# #
# #    #Reorient second block and update communication info
# #    tb.mb[0].connectivity['2']['orientation'] = '621'
# #
# #    tb.mb[1].connectivity['1']['neighbor'] = None
# #    tb.mb[1].connectivity['1']['bc'] = 's1'
# #    tb.mb[1].connectivity['1']['orientation'] = None
# #    tb.mb[1].connectivity['1']['comm_rank'] = None
# #
# #    tb.mb[1].connectivity['6']['neighbor'] = 0
# #    tb.mb[1].connectivity['6']['bc'] = 'b0'
# #    tb.mb[1].connectivity['6']['orientation'] = '324'
# #    tb.mb[1].connectivity['6']['comm_rank'] = 0
# #
# #    pg.mpicomm.blockcomm.set_block_communication(tb.mb,tb.config)
# #
# #    #Execute communication
# #    pg.mpicomm.blockcomm.communicate3(tb.mb,['x','y','z'])
# #
# #    passfail = []
# #    for var in ['x','y','z']:
# #        passfail.append(compare_arrays(tb.mb[0].orient['2'](tb.mb[0].array[var][tb.mb[0].slice_s3['2']]),
# #                                                            tb.mb[1].array[var][tb.mb[1].slice_r3['6']]))
# #
# #        passfail.append(compare_arrays(tb.mb[1].orient['6'](tb.mb[1].array[var][tb.mb[1].slice_s3['6']]),
# #                                                            tb.mb[0].array[var][tb.mb[0].slice_r3['2']]))
# #
# #    assert (False not in passfail)
