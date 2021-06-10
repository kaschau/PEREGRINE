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
                    comm_rank   = blk.connectivity[face]['comm_rank']
                    orientation = blk.connectivity[face]['orientation']
                    tag = int(f'1{blk.nblki}020{neighbor}1')
                    sendbuffer = blk.array[var][blk.slice_s3[face]]
                    blk.orientB4send[face](sendbuffer)
                    comm.Isend([sendbuffer, MPIDOUBLE], dest=comm_rank, tag=tag)

            comm.Barrier()

            #Post recieves
            for blk in mb:
                for face in ['1','2','3','4','5','6']:
                    neighbor    = blk.connectivity[face]['neighbor']
                    if neighbor is None:
                        continue
                    comm_rank   = blk.connectivity[face]['comm_rank']
                    orientation = blk.connectivity[face]['orientation']
                    comm_rank   = blk.connectivity[face]['comm_rank']
                    tag = int(f'1{neighbor}020{blk.nblki}1')
                    comm.Irecv([blk.array[var][blk.slice_r3[face]], MPIDOUBLE], source=comm_rank, tag=tag)

            comm.Barrier()


def set_block_communication(mb,config):

    comm,rank,size = get_comm_rank_size()

    ##########################################################
    ### This chunk defines the reorientation functions that
    ### each block will execute on its halo buffer BEFORE
    ### it sends it, so that when it arrives at it's
    ### neighbor, it is able to be put in place.
    ##########################################################

    #Positive i aligned blocks
    def reorient123(temp):
        pass
    def reorient135(temp):
        mb.np.rot90(temp,-1,(1,2))
    def reorient156(temp):
        mb.np.rot90(temp,-2,(1,2))
    def reorient162(temp):
        mb.np.rot90(temp, 1,(1,2))

    #Positive j aligned blocks
    def reorient231(temp):
        mb.np.rot90(temp,-1,(0,1))
        mb.np.rot90(temp,-1,(1,2))
    def reorient216(temp):
        mb.np.rot90(temp,-1,(0,1))
        mb.np.rot90(temp,-2,(1,2))
    def reorient264(temp):
        mb.np.rot90(temp,-1,(0,1))
        mb.np.rot90(temp, 1,(1,2))
    def reorient243(temp):
        mb.np.rot90(temp,-1,(0,1))

    #Positive k aligned blocks
    def reorient312(temp):
        mb.np.rot90(temp, 1,(0,2))
        mb.np.rot90(temp, 1,(1,2))
    def reorient324(temp):
        mb.np.rot90(temp, 1,(0,2))
    def reorient345(temp):
        mb.np.rot90(temp, 1,(0,2))
        mb.np.rot90(temp,-1,(1,2))
    def reorient351(temp):
        mb.np.rot90(temp, 1,(0,2))
        mb.np.rot90(temp,-2,(1,2))

    #Negative i aligned blocks
    def reorient432(temp):
        mb.np.rot90(temp,-2,(0,1))
        mb.np.rot90(temp,-1,(1,2))
    def reorient453(temp):
        mb.np.rot90(temp,-2,(0,1))
    def reorient465(temp):
        mb.np.rot90(temp,-2,(0,1))
        mb.np.rot90(temp, 1,(1,2))
    def reorient426(temp):
        mb.np.rot90(temp,-2,(0,2))

    #Negative j aligned blocks
    def reorient513(temp):
        mb.np.rot90(temp, 1,(0,1))
    def reorient561(temp):
        mb.np.rot90(temp, 1,(0,1))
        mb.np.rot90(temp, 1,(1,2))
    def reorient546(temp):
        mb.np.rot90(temp, 1,(0,1))
        mb.np.rot90(temp,-2,(1,2))
    def reorient534(temp):
        mb.np.rot90(temp, 1,(0,1))
        mb.np.rot90(temp,-1,(1,2))

    #Negative k aligned blocks
    def reorient621(temp):
        mb.np.rot90(temp,-1,(0,2))
    def reorient642(temp):
        mb.np.rot90(temp,-1,(0,2))
        mb.np.rot90(temp, 1,(1,2))
    def reorient654(temp):
        mb.np.rot90(temp,-1,(0,2))
        mb.np.rot90(temp,-2,(1,2))
    def reorient615(temp):
        mb.np.rot90(temp,-1,(0,2))
        mb.np.rot90(temp, 1,(1,2))

    ##########################################################
    ### This chunk predefines the slice extents for each block
    ### face send and recieves.
    ##########################################################
    for blk in mb:
        blk.slice_s3['1'] = s_[ blk.ngls  +1:2*blk.ngls+1,:,:]
        blk.slice_s3['2'] = s_[-blk.ngls*2-1::           ,:,:]

        blk.slice_s3['3'] = s_[:,blk.ngls+1:2*blk.ngls+1,:]
        blk.slice_s3['4'] = s_[:,-blk.ngls*2-1::,:]

        blk.slice_s3['5'] = s_[:,:,blk.ngls+1:2*blk.ngls+1]
        blk.slice_s3['6'] = s_[:,:,-blk.ngls*2-1::]

        blk.slice_r3['1'] = s_[0:blk.ngls,:,:]
        blk.slice_r3['2'] = s_[-blk.ngls::,:,:]

        blk.slice_r3['3'] = s_[:,0:blk.ngls,:]
        blk.slice_r3['4'] = s_[:,-blk.ngls::,:]

        blk.slice_r3['5'] = s_[:,:,0:blk.ngls]
        blk.slice_r3['6'] = s_[:,:,-blk.ngls::]

        for face in ['1','2','3','4','5','6']:
            neighbor = blk.connectivity[face]['neighbor']
            if neighbor is not None:
                orientation = blk.connectivity[face]['orientation']
                #Positive i aligned blocks
                if   orientation == '123':
                    blk.orientB4send[face] = reorient123
                elif orientation == '135':
                    blk.orientB4send[face] = reorient135
                elif orientation == '156':
                    blk.orientB4send[face] = reorient156
                elif orientation == '162':
                    blk.orientB4send[face] = reorient162
                #Positive j aligned blocks
                elif orientation == '231':
                    blk.orientB4send[face] = reorient231
                elif orientation == '216':
                    blk.orientB4send[face] = reorient216
                elif orientation == '264':
                    blk.orientB4send[face] = reorient264
                elif orientation == '243':
                    blk.orientB4send[face] = reorient243
                #Positive k aligned blocks
                elif orientation == '312':
                    blk.orientB4send[face] = reorient312
                elif orientation == '324':
                    blk.orientB4send[face] = reorient324
                elif orientation == '345':
                    blk.orientB4send[face] = reorient345
                elif orientation == '351':
                    blk.orientB4send[face] = reorient351

                #Negative i aligned blocks
                elif orientation == '432':
                    blk.orientB4send[face] = reorient432
                elif orientation == '453':
                    blk.orientB4send[face] = reorient453
                elif orientation == '465':
                    blk.orientB4send[face] = reorient465
                elif orientation == '426':
                    blk.orientB4send[face] = reorient426
                #Negative j aligned blocks
                elif orientation == '513':
                    blk.orientB4send[face] = reorient513
                elif orientation == '561':
                    blk.orientB4send[face] = reorient561
                elif orientation == '546':
                    blk.orientB4send[face] = reorient546
                elif orientation == '534':
                    blk.orientB4send[face] = reorient534
                #Negative k aligned blocks
                elif orientation == '621':
                    blk.orientB4send[face] = reorient621
                elif orientation == '642':
                    blk.orientB4send[face] = reorient642
                elif orientation == '654':
                    blk.orientB4send[face] = reorient654
                elif orientation == '615':
                    blk.orientB4send[face] = reorient615
