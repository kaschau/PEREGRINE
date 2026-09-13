"""
Placing blocks on ranks.

A block's weight is its interior cells, the work a rank does for it. An edge
between two blocks is weighted by the cells of the face plane they share, the
halo traffic one exchange costs. Communication comes in tiers, cheapest first:
same rank is free, same node goes over shared memory, and across a node goes
over the network. Balancing the weights alone leaves every neighbor on a
different rank, so the placement has to spend both.

A partitioner cuts a grid down until its pieces fit, turns it into a weighted
graph, and places that graph on ranks. Only the placement differs between
kinds, so it is the only thing a subclass writes.
"""

import numpy as np

from .cutMixin import CutMixin
from .mergeMixin import MergeMixin


class BasePartitioner(CutMixin, MergeMixin):
    partitionerName = None

    def __init__(self, tol=0.05, refinePasses=10, nStarts=6):
        # how far past an even share a rank may be loaded
        self.tol = tol
        self.refinePasses = refinePasses
        self.nStarts = nStarts

    def partition(self, mb, nProcs, ranksPerNode, granularity=None):
        """Place :mb:'s blocks on :nProcs: ranks, :ranksPerNode: of which share
        a node. With :granularity:, cut the blocks first so that no piece holds
        more than an even share divided by it -- 1 cuts only what cannot fit a
        rank, 2 aims for two pieces per rank. Returns the blocks each rank owns."""
        if nProcs % ranksPerNode:
            raise ValueError(
                f"{nProcs} ranks is not a whole number of nodes at"
                f" {ranksPerNode} ranks per node."
            )

        if granularity is not None:
            self.cutToFit(mb, int(self.blockCells(mb).sum() / nProcs / granularity))

        weights, edges = self.cellWeights(mb), self.edgesFromMb(mb)
        assign = self.hierarchical(weights, edges, nProcs // ranksPerNode, ranksPerNode)

        return [[int(n) for n in np.flatnonzero(assign == r)] for r in range(nProcs)]

    def _assign(self, weights, edges, k):
        """Which of k parts owns each block. The one thing a kind writes."""
        raise NotImplementedError

    ###########################################################################
    # The grid as a weighted graph
    ###########################################################################
    @staticmethod
    def blockCells(mb):
        return np.array([b.nCells for b in mb.blocks])

    def cellWeights(self, mb):
        """The work each block is: its interior cells."""
        return self.blockCells(mb).astype(np.int64)

    @staticmethod
    def facePlaneCells(blk, face):
        """The traffic one exchange across a face costs: its plane of cells."""
        cells = [blk.ni - 1, blk.nj - 1, blk.nk - 1]
        del cells[face.myAxis]
        return cells[0] * cells[1]

    def edgesFromMb(self, mb):
        """Edge weights {(a, b): plane cells} from a multiBlock's connectivity."""
        edges = {}
        for blk, face in mb.faces():
            if face.neighbor is None:
                continue
            key = tuple(sorted((blk.nblki, face.neighbor)))
            # a connection shows up from both sides, count it once
            if blk.nblki > face.neighbor and key in edges:
                continue
            edges[key] = edges.get(key, 0) + self.facePlaneCells(blk, face)
        return edges

    @staticmethod
    def cutWeight(assign, edges):
        """The traffic an assignment leaves crossing ranks."""
        return sum(w for (a, b), w in edges.items() if assign[a] != assign[b])

    ###########################################################################
    # Cutting a grid down until it can be balanced
    ###########################################################################
    def cutToFit(self, mb, maxCells):
        """Cut blocks until none holds more than maxCells. A cut runs on through
        every block its plane meets, so the cheapest axis is the shortest path."""
        while True:
            sizes = self.blockCells(mb)
            worst = int(sizes.argmax())
            if sizes[worst] <= maxCells:
                return
            blk = mb.blocks[worst]

            best = None
            for axis, nNodes in zip("ijk", (blk.ni, blk.nj, blk.nk)):
                # every piece needs a cell, so n nodes take at most n-2 cuts
                nCuts = min(int(np.ceil(sizes[worst] / maxCells)) - 1, nNodes - 2)
                if nCuts < 1:
                    continue
                cost = len(self.cutPath(mb, blk.nblki, axis)) * nCuts
                if best is None or cost < best[0]:
                    best = (cost, axis, nCuts)
            if best is None:
                raise ValueError(
                    f"block {blk.nblki} holds {sizes[worst]} cells and cannot be cut"
                    f" under {maxCells}: it is only {blk.ni}x{blk.nj}x{blk.nk} nodes"
                )
            self.performCutOperations(mb, [[blk.nblki, best[1], best[2]]])

    @staticmethod
    def loadCeiling(sizes, nProcs):
        """The best efficiency any grouping of whole blocks can reach. A rank
        holding the largest block cannot finish before it does, so once the
        largest block is bigger than an even share it, and not the grouping, is
        what limits the result."""
        return min(1.0, np.sum(sizes) / nProcs / np.max(sizes)) * 100

    ###########################################################################
    # Placing the graph
    ###########################################################################
    def hierarchical(self, weights, edges, nNodes, ranksPerNode):
        """Split blocks over nodes first, so the network carries as little as
        possible, then over the ranks within each node where a neighbor is only
        a shared memory copy away. rank = node * ranksPerNode + localRank."""
        if nNodes <= 1:
            return self._assign(weights, edges, ranksPerNode)

        nodeAssign = self._assign(weights, edges, nNodes)
        assign = np.full(weights.shape[0], -1, dtype=np.int64)
        for node in range(nNodes):
            blockIds = np.where(nodeAssign == node)[0]
            if len(blockIds) == 0:
                continue
            subW, subE = self.subProblem(blockIds, weights, edges)
            assign[blockIds] = node * ranksPerNode + self._assign(
                subW, subE, ranksPerNode
            )
        return self.balancePolish(
            assign, weights, edges, nNodes * ranksPerNode, ranksPerNode
        )

    @staticmethod
    def subProblem(blockIds, weights, edges):
        """The graph restricted to blockIds, renumbered 0..m-1."""
        localOf = {int(b): i for i, b in enumerate(blockIds)}
        subE = {}
        for (a, b), w in edges.items():
            if a in localOf and b in localOf:
                subE[(localOf[a], localOf[b])] = w
        return weights[blockIds], subE

    def balancePolish(self, assign, weights, edges, nProcs, ranksPerNode):
        """Move pieces off the busiest rank until every rank is within tol of the
        mean. The node split and each node's fill carry their own slack and the
        step time sees the product, so this runs after both. A destination on the
        same node is preferred, then the rank the piece talks to most."""
        assign = assign.copy()
        load = np.bincount(assign, weights=weights, minlength=nProcs)
        cap = load.mean() * (1.0 + self.tol)
        aff = {}
        for (a, b), w in edges.items():
            aff.setdefault(a, []).append((b, w))
            aff.setdefault(b, []).append((a, w))

        for _ in range(len(weights)):
            hi = int(np.argmax(load))
            if load[hi] <= cap:
                break
            best = None
            for p in np.where(assign == hi)[0]:
                w = weights[p]
                talk = {}
                for nbr, e in aff.get(int(p), []):
                    talk[int(assign[nbr])] = talk.get(int(assign[nbr]), 0) + e
                for r in range(nProcs):
                    if r == hi or load[r] + w >= load[hi]:
                        continue
                    key = (
                        r // ranksPerNode != hi // ranksPerNode,
                        load[r] + w,
                        -talk.get(r, 0),
                    )
                    if best is None or key < best[0]:
                        best = (key, int(p), r)
            if best is not None:
                _, p, r = best
                assign[p] = r
                load[hi] -= weights[p]
                load[r] += weights[p]
                continue

            # no single move lowers the max, so swap for something lighter
            for p in np.where(assign == hi)[0]:
                w = weights[p]
                lighter = np.where((weights < w) & (assign != hi))[0]
                r = assign[lighter]
                ok = (load[r] - weights[lighter] + w < load[hi]) & (
                    load[hi] - w + weights[lighter] < load[hi]
                )
                for q in lighter[ok]:
                    key = (
                        assign[q] // ranksPerNode != hi // ranksPerNode,
                        max(
                            load[assign[q]] - weights[q] + w, load[hi] - w + weights[q]
                        ),
                    )
                    if best is None or key < best[0]:
                        best = (key, int(p), int(q))
            if best is None:
                break
            _, p, q = best
            r = assign[q]
            assign[p], assign[q] = r, hi
            load[hi] += weights[q] - weights[p]
            load[r] += weights[p] - weights[q]
        return assign

    def metrics(self, assign, weights, edges, k, ranksPerNode=None):
        """Balance and traffic of an assignment. intraFraction is the share of
        exchanged face cells that stays on a rank and costs nothing."""
        load = np.bincount(assign, weights=weights, minlength=k)
        total = sum(edges.values())
        rpn = ranksPerNode or k
        nNodes = (k + rpn - 1) // rpn

        cut = onNode = offNode = 0
        nodeTrunk = np.zeros(nNodes)
        for (a, b), w in edges.items():
            ra, rb = int(assign[a]), int(assign[b])
            if ra == rb:
                continue  # same rank, free
            cut += w
            na, nb = ra // rpn, rb // rpn
            if ranksPerNode and na == nb:
                onNode += w
            else:
                offNode += w
                nodeTrunk[na] += w
                nodeTrunk[nb] += w

        m = {
            "efficiency": load.mean() / load.max() * 100.0 if load.max() > 0 else 100.0,
            "blocks": len(weights),
            "maxBlocksOnRank": int(np.bincount(assign, minlength=k).max()),
            "maxLoad": float(load.max()),
            "totalEdge": total,
            "cut": cut,
            "intraFraction": 100.0 * (1.0 - cut / total) if total else 100.0,
        }
        if ranksPerNode:
            m["onNodeFraction"] = 100.0 * onNode / total if total else 0.0
            m["offNodeFraction"] = 100.0 * offNode / total if total else 0.0
            # the busiest node's network traffic is what paces a synchronous step
            m["maxNodeTrunk"] = float(nodeTrunk.max()) if nNodes else 0.0
        return m
