class coprocessor:
    """Whatever the config asks for alongside the run, each called every step
    and told when the run is over."""

    def __init__(self, mb):
        config = mb.config["coprocess"]
        self.parts = []
        if config["trace"]:
            from .tracePointCoprocessor import tracePointsCoprocessor

            self.parts.append(tracePointsCoprocessor(mb))
        if config["catalyst"]:
            from .catalystCoprocessor import catalystCoprocessor

            self.parts.append(catalystCoprocessor(mb))

    def __call__(self, mb):
        for part in self.parts:
            part(mb)

    def finalize(self):
        for part in self.parts:
            if hasattr(part, "finalize"):
                part.finalize()
