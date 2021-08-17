from ..mpicomm import mpiutils
from mpi4py.MPI import DOUBLE as MPIDOUBLE

from . import generate_halo
from .. import mpicomm
from ..bcs import face_slice

fs = face_slice.fs

def unify_solver_grid(mb):
    generate_halo(mb)

    mpicomm.blockcomm.communicate(mb,['x','y','z'])

    comm,rank,size = mpiutils.get_comm_rank_size()
    #Take care of periodic BCs
    for var in ['x','y','z']:
        for _ in range(3):
            #Post sends
            for blk in mb:
                send = blk.sendbuffer3
                slice_s = blk.slice_s3
                for face in ['1','2','3','4','5','6']:
                    bc = blk.connectivity[face]['bc']
                    if bc != 'b1':
                        continue
                    neighbor = blk.connectivity[face]['neighbor']
                    orientation = blk.connectivity[face]['orientation']
                    nface = blk.connectivity[face]['nface']
                    comm_rank = blk.connectivity[face]['comm_rank']
                    tag = int(f'1{blk.nblki}2{neighbor}1{nface}')
                    send[face][:] = blk.orient[face]( blk.array[var][fs[face]['s2_']]
                                                     -blk.array[var][fs[face]['s1_']])
                    comm.Isend([send[face], MPIDOUBLE], dest=comm_rank, tag=tag)

            #Post recieves
            for blk in mb:
                recv = blk.recvbuffer3
                slice_r = blk.slice_r3
                for face in ['1','2','3','4','5','6']:
                    bc = blk.connectivity[face]['bc']
                    if bc != 'b1':
                        continue
                    neighbor = blk.connectivity[face]['neighbor']
                    orientation = blk.connectivity[face]['orientation']
                    comm_rank   = blk.connectivity[face]['comm_rank']
                    tag = int(f'1{neighbor}2{blk.nblki}1{face}')
                    comm.Recv([recv[face][:], MPIDOUBLE], source=comm_rank, tag=tag)
                    blk.array[var][fs[face]['s0_']] = blk.array[var][fs[face]['s1_']] + recv[face][:]

            comm.Barrier()
