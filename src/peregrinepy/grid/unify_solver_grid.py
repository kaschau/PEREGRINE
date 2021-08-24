from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request

from . import generate_halo
from .. import mpicomm
from ..bcs import face_slice

fs = face_slice.fs

def unify_solver_grid(mb):

    generate_halo(mb)

    # Lets just be clean and create the edges and corners
    for _ in range(3):
        mpicomm.blockcomm.communicate(mb,['x','y','z'])

    comm,rank,size = mpicomm.mpiutils.get_comm_rank_size()

    for var in ['x','y','z']:
        for _ in range(3):
            reqs = []
            #Post non-blocking recieves
            for blk in mb:
                for face in blk.faces:
                    bc = face.connectivity['bctype']
                    if bc != 'b1':
                        continue
                    neighbor = face.connectivity['neighbor']
                    orientation = face.connectivity['orientation']
                    comm_rank   = face.comm_rank
                    tag = int(f'1{neighbor}2{blk.nblki}1{face.nface}')

                    ssize = face.recvbuffer3.size
                    reqs.append(comm.Irecv([face.recvbuffer3[:], ssize, MPIDOUBLE], source=comm_rank, tag=tag))

            #Post non-blocking sends
            for blk in mb:
                for face in blk.faces:
                    bc = face.connectivity['bctype']
                    if bc != 'b1':
                        continue
                    neighbor = face.connectivity['neighbor']
                    orientation = face.connectivity['orientation']
                    comm_rank = face.comm_rank
                    tag = int(f'1{blk.nblki}2{neighbor}1{face.neighbor_face}')
                    face.sendbuffer3[:] = face.orient( blk.array[var][fs[face.nface]['s2_']]
                                                      -blk.array[var][fs[face.nface]['s1_']])
                    ssize = face.sendbuffer3.size
                    comm.Send([face.sendbuffer3, ssize, MPIDOUBLE], dest=comm_rank, tag=tag)

            #wait and assign
            Request.Waitall(reqs)
            for blk in mb:
                for face in blk.faces:
                    neighbor = face.connectivity['neighbor']
                    if neighbor is None:
                        continue
                    blk.array[var][fs[face.nface]['s0_']] = blk.array[var][fs[face.nface]['s1_']] + face.recvbuffer3[:]

    comm.Barrier()
