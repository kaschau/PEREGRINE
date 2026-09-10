import numpy as np

from .basePartitioner import BasePartitioner


class MetisPartitioner(BasePartitioner):
    """METIS multilevel k-way. Balances to a target rather than a cap, so it
    gives way once one block is a sizeable fraction of a rank's share."""

    partitionerName = "metis"

    def _assign(self, weights, edges, k):
        import pymetis

        adj = [[] for _ in range(weights.shape[0])]
        for (a, b), w in edges.items():
            adj[a].append((b, w))
            adj[b].append((a, w))
        xadj, adjncy, eweights = [0], [], []
        for a in range(weights.shape[0]):
            for b, w in adj[a]:
                adjncy.append(b)
                eweights.append(int(w))
            xadj.append(len(adjncy))

        opts = pymetis.Options()
        # metis states imbalance in thousandths, and its constraint is a target
        # rather than a cap
        opts.ufactor = max(1, int(self.tol * 1000))
        opts.seed = 0
        result = pymetis.part_graph(
            k,
            adjacency=pymetis.CSRAdjacency(xadj, adjncy),
            vweights=[int(w) for w in weights],
            eweights=eweights,
            options=opts,
        )
        return np.array(result.vertex_part, dtype=np.int64)
