"""What a physics is, as a spec the solver interrogates: what it needs
declared -- metrics of the grid, block arrays, kernels by tag, boundary
conditions, what the jit bakes -- and recipes, taking the solver, for its
graphs and its initial state. A simulator owns nothing: no
solver, no arrays, no state, no graphs. A new physics is a new subclass;
the config names it by its `physics`, and getSimulator makes it."""


class BaseSimulator:
    """One physics: the declarations a solver builds for, and the graphs
    the solver runs."""

    # what the config's simulation section calls it
    name = None
    # whether the physics solves diffusion, which the time integration reads
    viscous = False
    # the grid metrics it needs
    metrics = ()
    # the base its boundaries are found under, and the hooks they have
    # bodies at
    bcBase = None
    bcHooks = ()

    def __init__(self, config):
        self.config = config
        self.validate(config)

    def validate(self, config):
        """Refuses what this physics does not describe yet, rather than run
        it."""

    @property
    def primVars(self):
        """Names the primitive variables, the components of the primitive
        vector: the least a result holds to start a case from."""
        raise NotImplementedError

    @property
    def exportVars(self):
        """Names what a result writes by default: the primitive variables
        and what this physics derives beside them."""
        return self.primVars

    def exportData(self, blk, names):
        """Gives the named variables of a block as host arrays over every
        cell, halos included, derived from the state."""
        raise NotImplementedError

    def arrays(self):
        """Gives the block arrays this physics needs: name -> what declArray
        takes, the kind a block array class, exchanged True for the whole
        halo or the planes to trade."""
        raise NotImplementedError

    def declKernels(self):
        """Makes the kernels this physics calls, by tag, and keeps them for
        its graphs."""
        self.kernels = {}
        return self.kernels

    def bakes(self):
        """Gives what the jit bakes into every kernel beyond the halo
        depth: the mixture's species data and the simulation section."""
        raise NotImplementedError

    def graphs(self, dt):
        """Gives this physics' graphs by stage, of graph.py's nodes over
        its kernels, each stage a list of graphs cut where a message is
        waited on; :dt: is the case's step where the kernels run, for a
        node that integrates over it."""
        raise NotImplementedError

    def initialState(self):
        """Gives the primitive vector every cell starts from, for the
        uniform state the config describes."""
        raise NotImplementedError

    def report(self):
        """Says what this physics is, for the banner."""
        return f"  Physics: {self.name}\n"
