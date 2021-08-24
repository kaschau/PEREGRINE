from ..misc import FrozenDict


class face:

    def __init__(self,nface):
        assert (1<=nface<=6), 'nface must be between (1,6)'

        self.nface = nface
        self.connectivity = FrozenDict({'bcfam': None,
                                        'bctype': 'adiabatic_slip_wall',
                                        'neighbor': None,
                                        'orientation': None})
        self.connectivity._freeze()

        #Boundary condition dictionary
        self.bc = FrozenDict({})

        #MPI variables - only set for solver blocks, but we will store them
        # all the time for now
        self.comm_rank = None
        self.neighbor_face = None
        self.neighbor_orientation = None

        self.orient   = None
        self.slice_s3 = None
        self.slice_r3 = None
        self.slice_s4 = None
        self.slice_r4 = None

        self.sendbuffer3 = None
        self.recvbuffer3 = None
        self.sendbuffer4 = None
        self.recvbuffer4 = None
