from .basePartitioner import BasePartitioner
from .greedyPartitioner import GreedyPartitioner
from .metisPartitioner import MetisPartitioner


class AutoPartitioner(BasePartitioner):
    """Takes metis when every rank holds many blocks and greedy when it does
    not, per sub problem, since the node split and each node's fill are
    different shapes."""

    name = "auto"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metis = MetisPartitioner(**kwargs)
        self.greedy = GreedyPartitioner(**kwargs)

    def _assign(self, weights, edges, k):
        manyBlocksPerRank = weights.shape[0] >= 16 * k
        kind = self.metis if manyBlocksPerRank and edges else self.greedy
        return kind._assign(weights, edges, k)
