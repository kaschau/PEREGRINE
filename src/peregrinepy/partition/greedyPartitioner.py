import numpy as np

from .basePartitioner import BasePartitioner


class GreedyPartitioner(BasePartitioner):
    """Grow parts from seeds, then refine by moving and swapping blocks. Runs
    from several seeds and keeps the best, which costs nStarts passes and buys
    a placement that does not depend on which block happens to be first."""

    name = "greedy"

    def _assign(self, weights, edges, k):
        if k <= 1:
            return np.zeros(weights.shape[0], dtype=np.int64)

        nblks = weights.shape[0]
        # fewer blocks than ranks leaves ranks empty; only cutting can fill them
        if nblks <= k:
            return np.arange(nblks, dtype=np.int64)

        order = np.argsort(weights, kind="stable")[::-1]
        seeds = [None] + [
            int(order[(s * max(1, nblks // self.nStarts)) % nblks])
            for s in range(1, self.nStarts)
        ]

        best, bestKey = None, None
        for seedBlock in seeds:
            assign = self._once(weights, edges, k, seedBlock)
            load = np.bincount(assign, weights=weights, minlength=k)
            key = (self.cutWeight(assign, edges), load.max() - load.min())
            if bestKey is None or key < bestKey:
                best, bestKey = assign, key
        return best

    def _once(self, weights, edges, k, seedBlock):
        """One greedy-grow then refine partition of blocks 0..n-1 into k parts.
        seedBlock picks what seeds part 0, the multi start lever. Deterministic."""
        nblks = weights.shape[0]
        target = weights.sum() / k
        cap = target * (1.0 + self.tol)

        neighbors = [[] for _ in range(nblks)]
        for (a, b), w in edges.items():
            neighbors[a].append((b, w))
            neighbors[b].append((a, w))

        assign = np.full(nblks, -1, dtype=np.int64)
        load = np.zeros(k)
        # edge weight from each block to each part
        aff = np.zeros((nblks, k))
        order = np.argsort(weights, kind="stable")[::-1]

        def place(b, r):
            assign[b] = r
            load[r] += weights[b]
            for nbr, w in neighbors[b]:
                aff[nbr, r] += w

        def unplace(b):
            r = assign[b]
            assign[b] = -1
            load[r] -= weights[b]
            for nbr, w in neighbors[b]:
                aff[nbr, r] -= w
            return r

        # part 0 takes the seed, the rest spread out by least affinity to what is
        # already seeded
        place(int(seedBlock) if seedBlock is not None else order[0], 0)
        for r in range(1, k):
            pool = [b for b in order if assign[b] == -1]
            place(min(pool, key=lambda b: (aff[b, :r].sum(), -weights[b], b)), r)

        affScale = max(max((w for _, w in nb), default=0) for nb in neighbors) or 1.0
        for b in order:
            if assign[b] != -1:
                continue
            # the part we talk to most, discounted by how loaded it already is
            score = aff[b] / affScale - load / target
            score[load + weights[b] > cap] = -np.inf
            r = int(np.argmax(score))
            if np.isinf(score[r]):
                r = int(np.argmin(load))
            place(b, r)

        # moves alone cannot fix a balanced but wrong split, so swap as well
        for _ in range(self.refinePasses):
            moved = 0
            for b in range(nblks):
                r = assign[b]
                best, bestGain = r, 0.0
                for cand in set(int(assign[n]) for n, _ in neighbors[b]):
                    if cand == r or load[cand] + weights[b] > cap:
                        continue
                    gain = aff[b, cand] - aff[b, r]
                    if gain > bestGain:
                        best, bestGain = cand, gain
                if best != r:
                    unplace(b)
                    place(b, best)
                    moved += 1

            for (a, b), w in sorted(edges.items()):
                ra, rb = int(assign[a]), int(assign[b])
                if ra == rb:
                    continue
                gain = (aff[a, rb] - aff[a, ra]) + (aff[b, ra] - aff[b, rb]) - 2.0 * w
                if gain <= 0.0:
                    continue
                if (
                    load[rb] + weights[a] - weights[b] > cap
                    or load[ra] + weights[b] - weights[a] > cap
                ):
                    continue
                unplace(a)
                unplace(b)
                place(a, rb)
                place(b, ra)
                moved += 1
            if moved == 0:
                break

        # tighten the load spread, but never at the cost of more traffic
        for _ in range(self.refinePasses):
            hi, lo = int(np.argmax(load)), int(np.argmin(load))
            spread = load[hi] - load[lo]
            if spread <= 0:
                break
            bestPair, bestNew = None, spread
            for a in np.where(assign == hi)[0]:
                for b in np.where(assign == lo)[0]:
                    d = weights[a] - weights[b]
                    if d <= 0:
                        continue
                    newSpread = abs((load[hi] - d) - (load[lo] + d))
                    cutDelta = (
                        (aff[a, lo] - aff[a, hi])
                        + (aff[b, hi] - aff[b, lo])
                        - 2.0 * edges.get(tuple(sorted((int(a), int(b)))), 0.0)
                    )
                    if newSpread < bestNew and cutDelta >= 0.0:
                        bestPair, bestNew = (int(a), int(b)), newSpread
            if bestPair is None:
                break
            a, b = bestPair
            unplace(a)
            unplace(b)
            place(a, lo)
            place(b, hi)
        return assign
