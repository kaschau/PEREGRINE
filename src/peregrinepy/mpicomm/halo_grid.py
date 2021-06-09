from .mpiutils import get_comm_rank_size
import numpy as np
from mpi4py.MPI import DOUBLE as MPIDOUBLE

def reshape(temp,orientation):
    return temp

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
            neighbor    = blk.connectivity[face]['neighbor']
            comm_rank   = blk.connectivity[face]['comm_rank']
            if neighbor is None:
                continue
            if comm_rank == rank:
                pass #do transfer in receive
            else:
                orientation = blk.connectivity[face]['orientation']
                tag = int(f'{blk.nblki+1}2{neighbor+1}')
                comm.Isend([blk.array['x'][slices_s[face]], MPIDOUBLE], dest=comm_rank, tag=tag)

    comm.Barrier()

    #Post recieves
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            comm_rank   = blk.connectivity[face]['comm_rank']
            if neighbor is None:
                continue
            if comm_rank == rank:
                temp[face] = mb[mb.index_by_nblki(neighbor)].array["x"][slices_r[face]]
            else:
                orientation = blk.connectivity[face]['orientation']
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
