from .mpiutils import get_comm_rank_size
import numpy as np
from mpi4py.MPI import DOUBLE as MPIDOUBLE

def reshape(temp,orientation):
    return temp

face_to_orient_place_mapping = { '1':'0', '2':'0', '3':'1', '4':'1', '5':'2', '6':'2'}
orient_to_small_face_mapping = { '1':'2', '2':'4', '3':'6', '4':'1', '5':'3', '6':'5'}
orient_to_large_face_mapping = { '1':'1', '2':'3', '3':'5', '4':'2', '5':'4', '6':'6'}

def halo_grid(mb,config):

    comm,rank,size = get_comm_rank_size()
    block_list = mb.block_list
    ngls = mb[0].ngls

    for blk in mb:
        slices_s = {}
        slices_s['1'] = np.s_[ ngls+1:2*ngls+1,:,:]
        slices_s['2'] = np.s_[-ngls*2-1::     ,:,:]

        slices_s['3'] = np.s_[:,ngls+1:2*ngls+1,:]
        slices_s['4'] = np.s_[:,-ngls*2-1::,:]

        slices_s['5'] = np.s_[:,:,ngls+1:2*ngls+1]
        slices_s['6'] = np.s_[:,:,-ngls*2-1::]

        slices_r = {}
        slices_r['1'] = np.s_[0:blk.ngls,:,:]
        slices_r['2'] = np.s_[-blk.ngls:,:,:]

        slices_r['3'] = np.s_[:,0:blk.ngls,:]
        slices_r['4'] = np.s_[:,-blk.ngls:,:]

        slices_r['5'] = np.s_[:,:,0:blk.ngls]
        slices_r['6'] = np.s_[:,:,-blk.ngls:]

        temp = {}
        for face in ['1','2','3','4','5','6']:
            temp[face] = np.empty(blk.array['x'][slices_r[face]].shape)


    #Post sends
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            comm_rank   = blk.connectivity[face]['comm_rank']
            if comm_rank == rank:
                pass #do transfer in receive
            else:
                orientation = blk.connectivity[face]['orientation']
                tag = int(f'{blk.nblki+1}2{neighbor+1}')
                comm.Isend([blk.array['x'][slices_s[face]], MPIDOUBLE], dest=comm_rank, tag=tag)

    comm.Barrier()

    #Post recieves
    for blk in mb:
        print(blk.nblki)
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            comm_rank   = blk.connectivity[face]['comm_rank']
            orientation = blk.connectivity[face]['orientation']
            if comm_rank == rank:
                index=mb.index_by_nblki(neighbor)
                direction = orientation[int(face_to_orient_place_mapping[face])]
                if face in ['1','3','5']:
                    neignbor_face = orient_to_small_face_mapping[direction]
                elif face in ['2','4','6']:
                    neighbor_face = orient_to_large_face_mapping[direction]
                temp[face] = mb[index].array["x"][slices_s[neighbor_face]]
            else:
                comm_rank   = blk.connectivity[face]['comm_rank']
                tag = int(f'{neighbor+1}2{blk.nblki+1}')
                comm.Irecv([temp[face], MPIDOUBLE], source=neighbor, tag=tag)

    comm.Barrier()

    #Reorient recieves
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue

            orientation = blk.connectivity[face]['orientation']
            blk.array['x'][slices_r[face]] = reshape(temp[face],orientation)
