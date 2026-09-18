from .gridBlock import gridBlock


class restartBlock(gridBlock):
    """A block with a state on it: its primitive vector, as wide as the
    multiBlock's primitive variables."""

    def __init__(self, nblki, mb):
        super().__init__(nblki, mb)
        self.primVars = mb.primVars
        self.ne = mb.ne
