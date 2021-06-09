from .mpiutils import get_comm_rank_size
import numpy as np
from mpi4py.MPI import DOUBLE as MPIDOUBLE



def halo_grid(mb,config):

    comm,rank,size = get_comm_rank_size()
    block_list = mb.block_list
    for blk in mb:
        slices_s = {}
        slices_s['1'] = np.s_[1:1+blk.ngls,:,:]
        slices_s['2'] = np.s_[-(blk.ngls+1):-blk.ngls,:,:]

        slices_s['3'] = np.s_[:,1:1+blk.ngls,:]
        slices_s['4'] = np.s_[:,-(blk.ngls+1):-blk.ngls,:]

        slices_s['5'] = np.s_[:,:,1:1+blk.ngls]
        slices_s['6'] = np.s_[:,:,-(blk.ngls+1):-blk.ngls]

        slices_r = {}
        slices_r['1'] = np.s_[0:blk.ngls,:,:]
        slices_r['2'] = np.s_[-blk.ngls:,:,:]

        slices_r['3'] = np.s_[:,0:blk.ngls,:]
        slices_r['4'] = np.s_[:,-blk.ngls:,:]

        slices_r['5'] = np.s_[:,:,0:blk.ngls]
        slices_r['6'] = np.s_[:,:,-blk.ngls:]

        temp = {}
        for face in ['1','2','3','4','5','6']:
            temp[face] = np.empy(blk.array['x'][slices_s[face]].shape)

        #Post sends
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            if comm_rank == rank:
                pass #do something here
            else:
                orientation = blk.connectivity[face]['orientation']
                comm_rank   = blk.connectivity[face]['comm_rank']

                comm.Isend([blk.array['x'][slices_s[face]], MPIDOUBLE], dest=comm_rank, tag=f'{blk.nblki}_2_{neighbor}')

        #Post recieves
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            if comm_rank == rank:
                pass #do something here
            else:
                orientation = blk.connectivity[face]['orientation']
                comm_rank   = blk.connectivity[face]['comm_rank']
                comm.Irecv([temp, MPIDOUBLE], source=neighbor, tag=f'{neighbor}_2_{blk.nblki}')

        comm.Barrier()

        #Post recieves
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            if comm_rank == rank:
                pass #do something here
            else:
                orientation = blk.connectivity[face]['orientation']
                comm_rank   = blk.connectivity[face]['comm_rank']
                reshape(temp)

                blk.array['x'][slices_r] = temp
