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
                    orientation = blk.connectivity[face]['orientation']
                    comm_rank = blk.connectivity[face]['comm_rank']
                    tag = int(f'1{blk.nblki}020{neighbor}1')
                    blk.sendbuffer[face][:] = blk.orient[face](blk.array[var][blk.slice_s3[face]])
                    comm.Isend([blk.sendbuffer[face], MPIDOUBLE], dest=comm_rank, tag=tag)

            #Post recieves
            for blk in mb:
                for face in ['1','2','3','4','5','6']:
                    neighbor = blk.connectivity[face]['neighbor']
                    if neighbor is None:
                        continue
                    orientation = blk.connectivity[face]['orientation']
                    comm_rank   = blk.connectivity[face]['comm_rank']
                    tag = int(f'1{neighbor}020{blk.nblki}1')
                    comm.Recv([blk.recvbuffer[face][:], MPIDOUBLE], source=comm_rank, tag=tag)
                    blk.array[var][blk.slice_r3[face]] = blk.recvbuffer[face][:]

            comm.Barrier()

def set_block_communication(mb,config):

    def orient_T(temp):
        return temp.T
    def orient_Tf0(temp):
        return mb.np.flip(temp.T,0)
    def orient_Tf1(temp):
        return mb.np.flip(temp.T,1)
    def orient_Tf0f1(temp):
        return mb.np.flip(mb.np.flip(temp,0),1)
    def orient_f0(temp):
        return mb.np.flip(temp,0)
    def orient_f0f1(temp):
        return mb.np.flip(mb.np.flip(temp,0),1)
    def orient_f1(temp):
        return mb.np.flip(temp,1)
    def orient_na(temp):
        return temp

    def get_neighbor_face(nface, orientation):

        direction = orientation[int(face_to_orient_place_mapping[nface])]

        if nface in ['2','4','6']:
            nface2 = orient_to_large_face_mapping[direction]
        elif nface in ['1','3','5']:
            nface2 = orient_to_small_face_mapping[direction]

        return nface2
    ##########################################################
    ### This chunk predefines the slice extents for each block
    ### face send and recieves.
    ##########################################################
    face_to_orient_place_mapping = { '1':'0', '2':'0', '3':'1', '4':'1', '5':'2', '6':'2'}
    orient_to_small_face_mapping = { '1':'2', '2':'4', '3':'6', '4':'1', '5':'3', '6':'5'}
    orient_to_large_face_mapping = { '1':'1', '2':'3', '3':'5', '4':'2', '5':'4', '6':'6'}

    large_index_mapping = {0:'k', 1:'k', 2:'j'}
    need_to_transpose = {'k':{'k':[1,2,4,5], 'j':[1,4]},
                         'j':{'k':[1,2,4,5], 'j':[1,4]}}

    #Get the neighbor orientation opposite of each face
    comm,rank,size = get_comm_rank_size()
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            orientation = blk.connectivity[face]['orientation']
            comm_rank   = blk.connectivity[face]['comm_rank']
            tag = int(f'1{blk.nblki}020{neighbor}1')
            comm.isend(orientation, dest=comm_rank, tag=tag)
    neighbor_orientations = []
    for blk in mb:
        neighbor_orientations.append({})
        for face in ['1','2','3','4','5','6']:
            neighbor = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            orientation = blk.connectivity[face]['orientation']
            comm_rank   = blk.connectivity[face]['comm_rank']
            tag = int(f'1{neighbor}020{blk.nblki}1')
            neighbor_orientations[-1][face] = comm.recv(source=comm_rank, tag=tag)

    comm.Barrier()

    for blk,no in zip(mb,neighbor_orientations):
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
                neighbor = int(blk.connectivity[face]['neighbor'])

                face2 = get_neighbor_face(face, orientation)
                orientation2 = no[face]

                face_orientations = [i for j,i in enumerate(orientation) if j != int(face_to_orient_place_mapping[face])]
                normal_index = [j for j in range(3) if j == int(face_to_orient_place_mapping[face])][0]
                face_orientations2 = [i for j,i in enumerate(orientation2) if j != int(face_to_orient_place_mapping[face2])]
                normal_index2 = [j for j in range(3) if j == int(face_to_orient_place_mapping[face2])][0]

                big_index = large_index_mapping[normal_index]
                big_index2 = large_index_mapping[normal_index2]

                print('!!!!!!')
                print(blk.nblki)
                print(int(face_orientations[1]) in need_to_transpose[big_index][big_index2])
                print(face_orientations[1] in ['4','5','6'])
                print(face_orientations[0] in ['4','5','6'])
                print('!!!!!!')

                #Do we need to transpoze?
                if int(face_orientations[1]) in need_to_transpose[big_index][big_index2]:
                    #Do we need to flip along 0 axis?
                    if face_orientations[1] in ['4','5','6']:
                        #Do we need to flip along 1 axis?
                        if face_orientations[0] in ['4','5','6']:
                            # Then do all three
                            blk.orient[face] = orient_Tf0f1
                        else:
                            # Then do just T and flip0
                            blk.orient[face] = orient_Tf0
                    else:
                        #Do we need to flip along 1 axis?
                        if face_orientations[0] in ['4','5','6']:
                            # Then do just T and flip1
                            blk.orient[face] = orient_Tf1
                        else:
                            # Then do just T
                            blk.orient[face] = orient_T
                else:
                    #Do we need to flip along 0 axis?
                    if face_orientations[1] in ['4','5','6']:
                        #Do we need to flip along 1 axis?
                        if face_orientations[0] in ['4','5','6']:
                            # Then do just flip0 and flip1
                            blk.orient[face] = orient_f0f1
                        else:
                            # Then do just flip0
                            blk.orient[face] = orient_f0
                    elif face_orientations[0] in ['4','5','6']:
                            # Then do just flip1
                        blk.orient[face] = orient_f1
                    else:
                        # Then do nothing
                        blk.orient[face] = orient_na

                # We send the data in the correct shape already
                temp = blk.orient[face](mb.np.empty(commfaceshape[face]))
                blk.sendbuffer[face] = mb.np.ascontiguousarray(temp)
                # We revieve the data in the correct shape already
                temp = mb.np.empty(commfaceshape[face])
                blk.recvbuffer[face] = mb.np.ascontiguousarray(mb.np.empty(commfaceshape[face]))
