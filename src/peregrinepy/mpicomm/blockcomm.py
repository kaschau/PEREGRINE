from .mpiutils import get_comm_rank_size
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request
import numpy as np

#Communicate 3D arrays
def communicate3(mb,varis):
    if not isinstance(varis,list):
        varis = [varis]

    comm,rank,size = get_comm_rank_size()

    for var in varis:
        reqs = []
        #Post non-blocking recieves
        for blk in mb:
            for face in blk.faces:
                neighbor = face.connectivity['neighbor']
                if neighbor is None:
                    continue
                orientation = face.connectivity['orientation']
                comm_rank   = face.comm_rank
                tag = int(f'1{neighbor}2{blk.nblki}1{face.nface}')
                ssize = face.recvbuffer3.size
                reqs.append(comm.Irecv([face.recvbuffer3[:], ssize, MPIDOUBLE], source=comm_rank, tag=tag))

        #Post non-blocking sends
        for blk in mb:
            assert (len(blk.array[var].shape) == 3), 'Trying to communicate the wrong size array with wrong communicate3'
            for face in blk.faces:
                neighbor = face.connectivity['neighbor']
                if neighbor is None:
                    continue
                orientation = face.connectivity['orientation']
                comm_rank = face.comm_rank
                tag = int(f'1{blk.nblki}2{neighbor}1{face.neighbor_face}')
                face.sendbuffer3[:] = face.orient(blk.array[var][face.slice_s3])
                ssize = face.sendbuffer3.size
                comm.Send([face.sendbuffer3, ssize, MPIDOUBLE], dest=comm_rank, tag=tag)

        #wait and assign
        Request.Waitall(reqs)
        for blk in mb:
            for face in blk.faces:
                neighbor = face.connectivity['neighbor']
                if neighbor is None:
                    continue
                blk.array[var][face.slice_r3] = face.recvbuffer3[:]

    comm.Barrier()

#Communicate 4D arrays
def communicate4(mb,varis):
    if not isinstance(varis,list):
        varis = [varis]

    comm,rank,size = get_comm_rank_size()

    for var in varis:
        reqs = []
        #Post non-blocking recieves
        for blk in mb:
            for face in blk.faces:
                neighbor = face.connectivity['neighbor']
                if neighbor is None:
                    continue
                orientation = face.connectivity['orientation']
                comm_rank   = face.comm_rank
                tag = int(f'1{neighbor}2{blk.nblki}1{face.nface}')
                ssize = face.recvbuffer4.size
                reqs.append(comm.Irecv([face.recvbuffer4[:], ssize, MPIDOUBLE], source=comm_rank, tag=tag))

        #Post non-blocking sends
        for blk in mb:
            assert (len(blk.array[var].shape) == 4), 'Trying to communicate the wrong size array with communicate4'
            for face in blk.faces:
                neighbor = face.connectivity['neighbor']
                if neighbor is None:
                    continue
                orientation = face.connectivity['orientation']
                comm_rank = face.comm_rank
                tag = int(f'1{blk.nblki}2{neighbor}1{face.neighbor_face}')
                face.sendbuffer4[:] = face.orient(blk.array[var][face.slice_s4])
                ssize = face.sendbuffer4.size
                comm.Send([face.sendbuffer4, ssize, MPIDOUBLE], dest=comm_rank, tag=tag)

        #wait and assign
        Request.Waitall(reqs)
        for blk in mb:
            for face in blk.faces:
                neighbor = face.connectivity['neighbor']
                if neighbor is None:
                    continue
                blk.array[var][face.slice_r4] = face.recvbuffer4[:]

    comm.Barrier()

def set_block_communication(mb):
    assert (0 not in [mb[0].ni, mb[0].nj, mb[0].nk]), 'Must get grid before setting block communicaitons.'

    from numpy import s_

    ##########################################################
    ### Define the possible reorientation routines
    ##########################################################
    def orient_T(temp):
        return temp.T
    def orient_Tf0(temp):
        return np.flip(temp.T,0)
    def orient_Tf1(temp):
        return np.flip(temp.T,1)
    def orient_Tf0f1(temp):
        return np.flip(np.flip(temp,0),1)
    def orient_f0(temp):
        return np.flip(temp,0)
    def orient_f0f1(temp):
        return np.flip(np.flip(temp,0),1)
    def orient_f1(temp):
        return np.flip(temp,1)
    def orient_na(temp):
        return temp


    ##########################################################
    ### Define the mapping based on orientation
    ##########################################################
    face_to_orient_place_mapping = { 1:0, 2:0, 3:1, 4:1, 5:2, 6:2}
    orient_to_small_face_mapping = { 1:2, 2:4, 3:6, 4:1, 5:3, 6:5}
    orient_to_large_face_mapping = { 1:1, 2:3, 3:5, 4:2, 5:4, 6:6}

    large_index_mapping = {0:'k', 1:'k', 2:'j'}
    need_to_transpose = {'k':{'k':[1,2,4,5], 'j':[1,4]},
                         'j':{'k':[1,2,4,5], 'j':[1,4]}}

    def get_neighbor_face(nface, orientation):
        assert (1<=nface<=6), 'nface must be between (1,6)'

        direction = int(orientation[face_to_orient_place_mapping[nface]])

        if nface in [2,4,6]:
            nface2 = orient_to_large_face_mapping[direction]
        elif nface in [1,3,5]:
            nface2 = orient_to_small_face_mapping[direction]

        return nface2

    ##########################################################
    ### Get the neighbor orientation opposite of each face
    ##########################################################
    comm,rank,size = get_comm_rank_size()
    for blk in mb:
        for face in blk.faces:
            neighbor = face.connectivity['neighbor']
            if neighbor is None:
                continue
            orientation = face.connectivity['orientation']
            comm_rank   = face.comm_rank
            face.neighbor_face = get_neighbor_face(face.nface, orientation)
            tag = int(f'1{blk.nblki}2{neighbor}1{face.neighbor_face}')
            comm.isend(orientation, dest=comm_rank, tag=tag)
    neighbor_orientations = []
    for blk in mb:
        neighbor_orientations.append({})
        for face in blk.faces:
            neighbor = face.connectivity['neighbor']
            if neighbor is None:
                continue
            orientation = face.connectivity['orientation']
            nface = face.neighbor_face
            comm_rank   = face.comm_rank
            tag = int(f'1{neighbor}2{blk.nblki}1{face.nface}')
            neighbor_orientations[-1][face.nface] = comm.recv(source=comm_rank, tag=tag)

    comm.Barrier()

    for blk,no in zip(mb,neighbor_orientations):
        slice_sfp = {}
        slice_sc  = {}
        commfpshape = {}
        commcshape = {}
        slice_rfp = {}
        slice_rc = {}

        slice_sfp[1]      = s_[ 2 , :, :]
        slice_sc[1]       = s_[ 1 , :, :, :]
        commfpshape[1] =   ( blk.nj+2, blk.nk+2)
        commcshape[1]  =   ( blk.nj+1, blk.nk+1, blk.ne)
        slice_rfp[1]      = s_[0,:,:]
        slice_rc[1]       = s_[0,:,:,:]

        slice_sfp[2]      = s_[-3 , :, :]
        slice_sc[2]       = s_[-2 , :, :, :]
        commfpshape[2] = commfpshape[1]
        commcshape[2]  = commcshape[1]
        slice_rfp[2]      = s_[-1,:,:]
        slice_rc[2]      = s_[-1,:,:,:]

        slice_sfp[3]      = s_[:,2,:]
        slice_sc[3]       = s_[:,1,:,:]
        commfpshape[3] =   ( blk.ni+2, blk.nk+2)
        commcshape[3]  =   ( blk.ni+1, blk.nk+1, blk.ne)
        slice_rfp[3]      = s_[:,0,:]
        slice_rc[3]       = s_[:,0,:,:]

        slice_sfp[4]      = s_[:,-3,:]
        slice_sc[4]       = s_[:,-2,:,:]
        commfpshape[4] = commfpshape[3]
        commcshape[4]  = commcshape[3]
        slice_rfp[4]      = s_[:,-1,:]
        slice_rc[4]       = s_[:,-1,:,:]

        slice_sfp[5]      = s_[:,:,2]
        slice_sc[5]       = s_[:,:,1]
        commfpshape[5] =   ( blk.ni+2, blk.nj+2)
        commcshape[5]  =   ( blk.ni+1, blk.nj+1, blk.ne)
        slice_rfp[5]      = s_[:,:,0]
        slice_rc[5]       = s_[:,:,0,:]

        slice_sfp[6]      = s_[:,:,-3]
        slice_sc[6]       = s_[:,:,-2,:]
        commfpshape[6] = commfpshape[5]
        commcshape[6]  = commcshape[5]
        slice_rfp[6]      = s_[:,:,-1]
        slice_rc[6]       = s_[:,:,-1,:]

        for face in blk.faces:
            neighbor = face.connectivity['neighbor']
            if neighbor is None:
                pass
            else:
                face.slice_s3 = slice_sfp[face.nface]
                face.slice_r3 = slice_rfp[face.nface]

                face.slice_s4 = slice_sc[face.nface]
                face.slice_r4 = slice_rc[face.nface]

                orientation = face.connectivity['orientation']
                neighbor = face.connectivity['neighbor']

                face2 = face.neighbor_face
                orientation2 = no[face.nface]

                face_orientations = [int(i) for j,i in enumerate(orientation) if j != face_to_orient_place_mapping[face.nface]]
                normal_index = [j for j in range(3) if j == face_to_orient_place_mapping[face.nface]][0]
                face_orientations2 = [int(i) for j,i in enumerate(orientation2) if j != face_to_orient_place_mapping[face2]]
                normal_index2 = [j for j in range(3) if j == face_to_orient_place_mapping[face2]][0]

                big_index = large_index_mapping[normal_index]
                big_index2 = large_index_mapping[normal_index2]

                #Do we need to transpoze?
                if face_orientations[1] in need_to_transpose[big_index][big_index2]:
                    #Do we need to flip along 0 axis?
                    if face_orientations[1] in [4,5,6]:
                        #Do we need to flip along 1 axis?
                        if face_orientations[0] in [4,5,6]:
                            # Then do all three
                            face.orient = orient_Tf0f1
                        else:
                            # Then do just T and flip0
                            face.orient = orient_Tf0
                    else:
                        #Do we need to flip along 1 axis?
                        if face_orientations[0] in [4,5,6]:
                            # Then do just T and flip1
                            face.orient = orient_Tf1
                        else:
                            # Then do just T
                            face.orient = orient_T
                else:
                    #Do we need to flip along 0 axis?
                    if face_orientations[1] in [4,5,6]:
                        #Do we need to flip along 1 axis?
                        if face_orientations[0] in [4,5,6]:
                            # Then do just flip0 and flip1
                            face.orient = orient_f0f1
                        else:
                            # Then do just flip0
                            face.orient = orient_f0
                    elif face_orientations[0] in [4,5,6]:
                            # Then do just flip1
                        face.orient = orient_f1
                    else:
                        # Then do nothing
                        face.orient = orient_na

                # We send the data in the correct shape already
                # Face and point shape
                temp = face.orient(np.empty(commfpshape[face.nface]))
                face.sendbuffer3 = np.ascontiguousarray(temp)
                # We revieve the data in the correct shape already
                temp = np.empty(commfpshape[face.nface])
                face.recvbuffer3 = np.ascontiguousarray(np.empty(commfpshape[face.nface]))

                # Cell
                temp = face.orient(np.empty(commcshape[face.nface]))
                face.sendbuffer4 = np.ascontiguousarray(temp)
                # We revieve the data in the correct shape already
                temp = np.empty(commcshape[face.nface])
                face.recvbuffer4 = np.ascontiguousarray(np.empty(commcshape[face.nface]))
