from peregrinepy.misc import null


class coprocessor:
    def __init__(self, mb):
        config = mb.config
        # Trace points
        if config["coprocess"]["trace"]:
            from .tracePointCoprocesor import tracePointsCoprocessor

            self.trace = tracePointsCoprocessor(mb)
        else:
            self.trace = null

        # ParaView/Catalyst coprocessing
        if config["coprocess"]["catalyst"]:
            try:
                from .catalystCoprocessor import catalystCoprocessor

                self.catalyst = catalystCoprocessor(mb)
            except ImportError:
                raise ImportError("Could not import the coprocessing module.")
        else:
            self.catalyst = null

    def __call__(self, mb):
        self.trace(mb)
        self.catalyst(mb)

    def finalize(self):
        try:
            self.catalyst.finalize()
        except AttributeError:
            pass
