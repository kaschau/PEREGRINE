import sys
import atexit


def getCommRankSize():

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    return comm, rank, size


def registerFinalizeHandler():
    import mpi4py.rc
    # Prevent mpi4py from calling MPI_Finalize
    mpi4py.rc.finalize = False
    from mpi4py import MPI

    # Intercept any uncaught exceptions
    class ExceptHook(object):
        def __init__(self):
            self.exception = None

            self._orig_excepthook = sys.excepthook
            sys.excepthook = self._excepthook

        def _excepthook(self, exc_type, exc, *args):
            self.exception = exc
            self._orig_excepthook(exc_type, exc, *args)

    # Register our exception hook
    excepthook = ExceptHook()

    def onexit():
        if not MPI.Is_initialized() or MPI.Is_finalized():
            return

        # Get the current exception (if any)
        exc = excepthook.exception

        # If we are exiting normally then call MPI_Finalize
        if (
            MPI.COMM_WORLD.size == 1
            or exc is None
            or isinstance(exc, KeyboardInterrupt)
            or (isinstance(exc, SystemExit) and exc.code == 0)
        ):
            MPI.Finalize()
        # Otherwise forcefully abort
        else:
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)

    # Register our exit handler
    atexit.register(onexit)
