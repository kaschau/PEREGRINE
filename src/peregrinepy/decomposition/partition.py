"""
Placing blocks on ranks.

A block's weight is its interior cells, the work a rank does for it. An edge
between two blocks is weighted by the cells of the face plane they share, the
halo traffic one exchange costs. Communication comes in tiers, cheapest first:
same rank is free, same node goes over shared memory, and across a node goes
over the network. Balancing the weights alone leaves every neighbor on a
different rank, so the placement has to spend both.
"""

import numpy as np


def cellWeights(mb):
    """The work each block is: its interior cells."""
    return np.array(
        [(blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1) for blk in mb], dtype=np.int64
    )


def facePlaneCells(blk, face):
    """The traffic one exchange across a face costs: its plane of cells."""
    cells = [blk.ni - 1, blk.nj - 1, blk.nk - 1]
    del cells[face.myAxis]
    return cells[0] * cells[1]


def edgesFromMb(mb):
    """Edge weights {(a, b): plane cells} from a multiBlock's connectivity."""
    edges = {}
    for blk, face in mb.connections():
        key = tuple(sorted((blk.nblki, face.neighbor)))
        # a connection shows up from both sides, count it once
        if blk.nblki > face.neighbor and key in edges:
            continue
        edges[key] = edges.get(key, 0) + facePlaneCells(blk, face)
    return edges


def cutWeight(assign, edges):
    """The traffic an assignment leaves crossing ranks."""
    return sum(w for (a, b), w in edges.items() if assign[a] != assign[b])


def _partitionOnce(weights, edges, k, tol, refinePasses, seedBlock):
    """One greedy-grow then refine partition of blocks 0..n-1 into k parts.
    seedBlock picks what seeds part 0, the multi start lever. Deterministic."""
    nblks = weights.shape[0]
    target = weights.sum() / k
    cap = target * (1.0 + tol)

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
    for _ in range(refinePasses):
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
    for _ in range(refinePasses):
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


def _partitionMetis(weights, edges, k, tol):
    """METIS multilevel k-way partition, deterministically seeded."""
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
    opts.ufactor = max(1, int(tol * 1000))
    opts.seed = 0
    result = pymetis.part_graph(
        k,
        adjacency=pymetis.CSRAdjacency(xadj, adjncy),
        vweights=[int(w) for w in weights],
        eweights=eweights,
        options=opts,
    )
    return np.array(result.vertex_part, dtype=np.int64)


def partition(weights, edges, k, tol=0.05, method="auto", refinePasses=10, nStarts=6):
    """Which of k ranks owns each block. "metis" is multilevel k-way, "greedy"
    the multi start fill below, "auto" takes metis when every rank holds many
    blocks: metis balances to a target, not a cap, so it gives way once one
    block is a sizeable fraction of a rank's share."""
    if k <= 1:
        return np.zeros(weights.shape[0], dtype=np.int64)

    nblks = weights.shape[0]
    # fewer blocks than ranks leaves ranks empty; only cutting can fill them
    if nblks <= k:
        return np.arange(nblks, dtype=np.int64)

    if method == "auto":
        method = "metis" if nblks >= 16 * k else "greedy"
    if method == "metis" and edges:
        return _partitionMetis(weights, edges, k, tol)

    order = np.argsort(weights, kind="stable")[::-1]
    seeds = [None] + [
        int(order[(s * max(1, nblks // nStarts)) % nblks]) for s in range(1, nStarts)
    ]

    best, bestKey = None, None
    for seedBlock in seeds:
        assign = _partitionOnce(weights, edges, k, tol, refinePasses, seedBlock)
        load = np.bincount(assign, weights=weights, minlength=k)
        key = (cutWeight(assign, edges), load.max() - load.min())
        if bestKey is None or key < bestKey:
            best, bestKey = assign, key
    return best


def _subProblem(blockIds, weights, edges):
    """The graph restricted to blockIds, renumbered 0..m-1."""
    localOf = {int(b): i for i, b in enumerate(blockIds)}
    subE = {}
    for (a, b), w in edges.items():
        if a in localOf and b in localOf:
            subE[(localOf[a], localOf[b])] = w
    return weights[blockIds], subE


def balancePolish(assign, weights, edges, nProcs, ranksPerNode, tol=0.05):
    """Move pieces off the busiest rank until every rank is within tol of the
    mean. The node split and each node's fill carry their own slack and the
    step time sees the product, so this runs after both. A destination on the
    same node is preferred, then the rank the piece talks to most."""
    assign = assign.copy()
    load = np.bincount(assign, weights=weights, minlength=nProcs)
    cap = load.mean() * (1.0 + tol)
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
                    max(load[assign[q]] - weights[q] + w, load[hi] - w + weights[q]),
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


def partitionHierarchical(
    weights, edges, nNodes, ranksPerNode, tol=0.05, method="auto", **kw
):
    """Split blocks over nodes first, so the network carries as little as
    possible, then over the ranks within each node where a neighbor is only a
    shared memory copy away. rank = node * ranksPerNode + localRank."""
    if nNodes <= 1:
        return partition(weights, edges, ranksPerNode, tol, method, **kw)

    nodeAssign = partition(weights, edges, nNodes, tol, method, **kw)
    assign = np.full(weights.shape[0], -1, dtype=np.int64)
    for node in range(nNodes):
        blockIds = np.where(nodeAssign == node)[0]
        if len(blockIds) == 0:
            continue
        subW, subE = _subProblem(blockIds, weights, edges)
        assign[blockIds] = node * ranksPerNode + partition(
            subW, subE, ranksPerNode, tol, method, **kw
        )
    return balancePolish(
        assign, weights, edges, nNodes * ranksPerNode, ranksPerNode, tol
    )


def metrics(assign, weights, edges, k, ranksPerNode=None):
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
