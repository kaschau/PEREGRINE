from .mpiutils import get_comm_rank_size
from .reorient import *
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from numpy import s_


def communicate(mb,varis):
    if not isinstance(varis,list):
        varis = [varis]

    comm,rank,size = get_comm_rank_size()

    for var in varis:
        #Communicate 3 times, first updates faces, second updates edges, third updates corners
        for _ in range(3):
            #Post sends
            for blk in mb:
                for face in ['1','2','3','4','5','6']:
                    neighbor = blk.connectivity[face]['neighbor']
                    if neighbor is None:
                        continue
                    comm_rank = blk.connectivity[face]['comm_rank']
                    orientation = blk.connectivity[face]['orientation']
                    tag = int(f'1{blk.nblki}020{neighbor}1')
                    blk.sendbuffer[face][:] = blk.orient[face](blk.array[var][blk.slice_s3[face]])
                    comm.Isend([blk.sendbuffer[face], MPIDOUBLE], dest=comm_rank, tag=tag)

            #Post recieves
            for blk in mb:
                for face in ['1','2','3','4','5','6']:
                    neighbor = blk.connectivity[face]['neighbor']
                    if neighbor is None:
                        continue
                    comm_rank = blk.connectivity[face]['comm_rank']
                    orientation = blk.connectivity[face]['orientation']
                    comm_rank   = blk.connectivity[face]['comm_rank']
                    tag = int(f'1{neighbor}020{blk.nblki}1')
                    comm.Recv([blk.recvbuffer[face][:], MPIDOUBLE], source=comm_rank, tag=tag)
                    blk.array[var][blk.slice_r3[face]] = blk.recvbuffer[face][:]

            comm.Barrier()

def set_block_communication(mb,config):

    def orient123(temp):
        return temp
    def orient135(temp):
        return mb.np.moveaxis(mb.np.flip(temp,axis=1),(0,1),(1,0))
    def orient156(temp):
        return mb.np.rot90(temp,2)
    def orient162(temp):
        return mb.np.moveaxis(mb.np.flip(temp,axis=0),(0,1),(1,0))

    def orient231(temp):
        return mb.np.moveaxis(temp,(0,1,2),(1,2,0))
    def orient216(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(1,0)),2,(2,0))
    def orient264(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(1,0)),1,(2,0))
    def orient243(temp):
        return mb.np.rot90(temp,1,(1,0))

    def orient312(temp):
        return mb.np.moveaxis(temp,(0,1,2),(2,0,1))
    def orient324(temp):
        return mb.np.rot90(temp,1,(2,0))
    def orient345(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(2,0)),1,(1,0))
    def orient351(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(2,0)),2,(1,0))

    def orient432(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(2,1)),2,(1,0))
    def orient426(temp):
        return mb.np.rot90(temp,1,(2,0))
    def orient465(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(1,2)),2,(0,1))
    def orient453(temp):
        return mb.np.rot90(temp,2,(0,1))

    def orient513(temp):
        return mb.np.rot90(temp,1,(0,1))
    def orient534(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(2,0)),1,(2,1))
    def orient546(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(1,0)),2,(1,2))
    def orient561(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(1,0)),1,(0,2))

    def orient621(temp):
        return mb.np.rot90(temp,1,(0,2))
    def orient615(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(1,0)),1,(2,1))
    def orient654(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(0,2)),2,(0,1))
    def orient642(temp):
        return mb.np.rot90(mb.np.rot90(temp,1,(0,2)),1,(1,0))

    ##########################################################
    ### This chunk predefines the slice extents for each block
    ### face send and recieves.
    ##########################################################
    for blk in mb:
        slice_s3 = {}
        commfaceshape = {}
        slice_r3 = {}

        slice_s3['1']      = s_[ 2 , :, :]
        commfaceshape['1'] =   ( blk.nj+2, blk.nk+2)
        slice_r3['1']      = s_[0,:,:]

        slice_s3['2'] = s_[-3 , :, :]
        commfaceshape['2'] = commfaceshape['1']
        slice_r3['2']      = s_[-1,:,:]

        slice_s3['3'] = s_[:,2,:]
        commfaceshape['3'] =   ( blk.ni+2, blk.nk+2)
        slice_r3['3'] = s_[:,0,:]

        slice_s3['4'] = s_[:,-3,:]
        commfaceshape['4'] = commfaceshape['3']
        slice_r3['4'] = s_[:,-1,:]

        slice_s3['5'] = s_[:,:,2]
        commfaceshape['5'] =   ( blk.ni+2, blk.nj+2)
        slice_r3['5'] = s_[:,:,0]

        slice_s3['6'] = s_[:,:,-3]
        commfaceshape['6'] = commfaceshape['5']
        slice_r3['6'] = s_[:,:,-1]

        for face in ['1','2','3','4','5','6']:
            neighbor = blk.connectivity[face]['neighbor']
            if neighbor is None:
                blk.slice_s3[face] = None
                blk.slice_r3[face] = None
                blk.sendbuffer[face] = None
                blk.recvbuffer[face] = None
            else:
                blk.slice_s3[face] = slice_s3[face]
                blk.slice_r3[face] = slice_r3[face]

                orientation = blk.connectivity[face]['orientation']
                if orientation == '123':
                    blk.orient[face] = orient123
                elif orientation == '135':
                    blk.orient[face] = orient135
                elif orientation == '156':
                    blk.orient[face] = orient156
                elif orientation == '162':
                    blk.orient[face] = orient162

                elif orientation == '231':
                    blk.orient[face] = orient231
                elif orientation == '216':
                    blk.orient[face] = orient216
                elif orientation == '264':
                    blk.orient[face] = orient264
                elif orientation == '243':
                    blk.orient[face] = orient243

                elif orientation == '312':
                    blk.orient[face] = orient312
                elif orientation == '324':
                    blk.orient[face] = orient324
                elif orientation == '345':
                    blk.orient[face] = orient345
                elif orientation == '351':
                    blk.orient[face] = orient351

                elif orientation == '432':
                    blk.orient[face] = orient432
                elif orientation == '426':
                    blk.orient[face] = orient426
                elif orientation == '465':
                    blk.orient[face] = orient465
                elif orientation == '453':
                    blk.orient[face] = orient453

                elif orientation == '513':
                    blk.orient[face] = orient513
                elif orientation == '534':
                    blk.orient[face] = orient534
                elif orientation == '546':
                    blk.orient[face] = orient546
                elif orientation == '561':
                    blk.orient[face] = orient561

                elif orientation == '621':
                    blk.orient[face] = orient621
                elif orientation == '615':
                    blk.orient[face] = orient615
                elif orientation == '654':
                    blk.orient[face] = orient654
                elif orientation == '642':
                    blk.orient[face] = orient642

                # We send the data in the correct shape already
                temp = blk.orient[face](mb.np.empty(commfaceshape[face]))
                blk.sendbuffer[face] = mb.np.ascontiguousarray(temp)
                # We revieve the data in the correct shape already
                temp = mb.np.empty(commfaceshape[face])
                blk.recvbuffer[face] = mb.np.ascontiguousarray(mb.np.empty(commfaceshape[face]))
