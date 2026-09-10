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


class BasePartitioner:
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
        return np.array([(b.ni - 1) * (b.nj - 1) * (b.nk - 1) for b in mb])

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
        for blk, face in mb.connections():
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
            blk = mb[worst]

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

    ###########################################################################
    # Cutting a grid into smaller blocks. A cut plane through one block has to
    # continue through every block it meets, so a cut is a path, not a single
    # split; the pieces it leaves are re-paired by matching face centers.
    ###########################################################################
    def _faceSlice(self, nface):
        """The index of the plane of nodes a face sits on."""
        return (slice(None),) * ((nface - 1) // 2) + (0 if nface % 2 else -1,)

    def _faceCenter(self, blk, face):
        """The center of a face, as the mean of its four corners."""
        center = np.empty(3)
        for n, var in enumerate(("x", "y", "z")):
            nodes = blk.array[var][self._faceSlice(face.nface)]
            center[n] = np.mean(
                [nodes[0, 0], nodes[0, -1], nodes[-1, 0], nodes[-1, -1]]
            )
        return center

    def _faceSearchPoint(self, face, center):
        """Where to go looking for a face's partner. Only a periodic sits
        somewhere other than on top of its partner, so only a periodic moves."""
        if face.bcType == "periodicTransLow":
            return center + face.periodicAxis * face.periodicSpan
        elif face.bcType == "periodicTransHigh":
            return center - face.periodicAxis * face.periodicSpan
        elif face.bcType == "periodicRotLow":
            return np.matmul(face.array["periodicRotMatrixUp"], center)
        elif face.bcType == "periodicRotHigh":
            return np.matmul(face.array["periodicRotMatrixDown"], center)
        return center

    def _pairCutFaces(self, openFaces):
        """Pair the faces a cut left open, by their centers. Every one of them
        has its partner somewhere in the set, so a face left over means the cut
        path did not close."""
        centers = [self._faceCenter(blk, face) for blk, face in openFaces]
        searchPoints = [
            self._faceSearchPoint(face, center)
            for (blk, face), center in zip(openFaces, centers)
        ]

        for index, (blk, face) in enumerate(openFaces):
            # an earlier face may already have claimed us
            if face.neighbor is not None:
                continue
            for testIndex, (testBlk, testFace) in enumerate(openFaces):
                if testIndex == index or testFace.neighbor is not None:
                    continue
                dist = np.linalg.norm(searchPoints[index] - centers[testIndex])
                if dist < 1e-9:
                    face.neighbor = testBlk.nblki
                    testFace.neighbor = blk.nblki
                    break
            else:
                raise ValueError(
                    f"Block {blk.nblki} face {face.nface} has nothing to pair with across the cut."
                )

    def cutBlock(self, mb, nblki, cutAxis, cutIndex):
        """Split a block in two at cutIndex along cutAxis. The low half stays as
        the block, the high half is appended to the multiBlock. Returns the
        (block, face) pairs the cut left open, to be paired up once every block
        on the path has been cut."""
        oldBlk = mb.getBlock(nblki)
        axis = "ijk".index(cutAxis)

        # Make sure we arent trying to split at an index greater than the number of grid points
        assert (
            cutIndex < getattr(oldBlk, f"n{cutAxis}") - 1
        ), f"Error, trying to cut block {nblki} along axis {cutAxis} at index {cutIndex} >= n{cutAxis} == {getattr(oldBlk, f'n{cutAxis}')-1}."
        mb.appendBlock()
        newBlk = mb[-1]

        newCutNface = 2 * axis + 1
        oldCutNface = 2 * axis + 2

        # Before we change the cut face info copy the info
        # from the oldFace to the new block's opposite face
        oldCutFace = oldBlk.getFace(oldCutNface)
        newOppFace = newBlk.getFace(oldCutNface)
        newOppFace.neighbor = oldCutFace.neighbor
        newOppFace.orientation = oldCutFace.orientation
        newOppFace.bcType = oldCutFace.bcType
        newOppFace.bcFam = oldCutFace.bcFam
        if oldCutFace.bcType.startswith("periodic"):
            newOppFace.periodicSpan = oldCutFace.periodicSpan
            newOppFace.periodicAxis = oldCutFace.periodicAxis

        # We also need to update the oppFace neighbor of the oldBlk
        if oldCutFace.neighbor is not None:
            neighborBlk = mb.getBlock(oldCutFace.neighbor)
            neighborBlk.getFace(oldCutFace.neighborNface).neighbor = newBlk.nblki

        # We know everything about the cut faces, so set it here
        newCutFace = newBlk.getFace(newCutNface)
        oldCutFace.neighbor = newBlk.nblki
        newCutFace.neighbor = oldBlk.nblki
        oldCutFace.orientation = "123"
        newCutFace.orientation = "123"
        oldCutFace.bcType = "interior"
        newCutFace.bcType = "interior"
        oldCutFace.bcFam = None
        newCutFace.bcFam = None

        # the four faces along the cut split in two; only the neighbor is unknown
        openFaces = []
        for nface in (n for n in range(1, 7) if (n - 1) // 2 != axis):
            oldSplitFace = oldBlk.getFace(nface)
            newSplitFace = newBlk.getFace(nface)
            newSplitFace.orientation = oldSplitFace.orientation
            newSplitFace.bcFam = oldSplitFace.bcFam
            newSplitFace.bcType = oldSplitFace.bcType
            # if the split face is a periodic, they need the perodic info
            if oldSplitFace.bcType.startswith("periodic"):
                newSplitFace.periodicSpan = oldSplitFace.periodicSpan
                newSplitFace.periodicAxis = oldSplitFace.periodicAxis
            if oldSplitFace.neighbor is None:
                newSplitFace.neighbor = None
                continue
            # our neighbor is cut too, so which halves meet waits for the path
            oldSplitFace.neighbor = None
            openFaces.append((oldBlk, oldSplitFace))
            openFaces.append((newBlk, newSplitFace))

        # cutIndex is local, so it lands that far along whatever slab we already are
        base = list(
            oldBlk.baseSlice or (0, oldBlk.ni - 1, 0, oldBlk.nj - 1, 0, oldBlk.nk - 1)
        )
        low, high = list(base), list(base)
        low[2 * axis + 1] = base[2 * axis] + cutIndex
        high[2 * axis] = base[2 * axis] + cutIndex
        newBlk.baseNblki = oldBlk.baseNblki
        oldBlk.baseSlice, newBlk.baseSlice = tuple(low), tuple(high)

        # Now transfer the coordinate arrays
        oldSlice, newSlice = [slice(None)] * 3, [slice(None)] * 3
        oldSlice[axis] = slice(0, cutIndex + 1)
        newSlice[axis] = slice(cutIndex, None)
        for var in ["x", "y", "z"]:
            newBlk.array[var] = np.copy(oldBlk.array[var][tuple(newSlice)])
            oldBlk.array[var] = np.copy(oldBlk.array[var][tuple(oldSlice)])

        oldBlk.ni, oldBlk.nj, oldBlk.nk = oldBlk.array["x"].shape
        newBlk.ni, newBlk.nj, newBlk.nk = newBlk.array["x"].shape

        return openFaces

    @staticmethod
    def cutTable(mb):
        """Which base block each block is a piece of, and which slab of it, as a
        (nblks, 7) table of baseNblki and inclusive node bounds i0, i1, j0, j1,
        k0, k1. A decomposition is this table plus which rank owns each row."""
        table = np.empty((len(mb), 7), dtype=np.int32)
        for n, blk in enumerate(mb):
            table[n, 0] = blk.baseNblki
            table[n, 1:] = blk.baseSlice or (
                0,
                blk.ni - 1,
                0,
                blk.nj - 1,
                0,
                blk.nk - 1,
            )
        return table

    def cutPath(self, mb, nblki, cutAxis):
        """Every block a cut runs on into, as [block, the axis it is cut on,
        whether its cut index counts from the far end]."""
        #              [ block, axis, switchBool ]
        blocksToCut = [[nblki, cutAxis, False]]
        blocksToCheck = [[nblki, cutAxis, False]]

        while blocksToCheck != []:
            checkNblki, checkAxis, checkSwitch = blocksToCheck.pop(0)
            checkBlk = mb.getBlock(checkNblki)
            checkAxisIndex = "ijk".index(checkAxis)

            # the cut runs out through the four faces it is parallel to
            for splitFace in (n for n in range(1, 7) if (n - 1) // 2 != checkAxisIndex):
                face = checkBlk.getFace(splitFace)
                neighbor = face.neighbor
                if neighbor is None or neighbor in [item[0] for item in blocksToCut]:
                    continue
                axis, counterAligned = face.signedAxis(face.orientation[checkAxisIndex])
                neighborSwitch = checkSwitch != counterAligned
                blocksToCheck.append([neighbor, "ijk"[axis], neighborSwitch])
                blocksToCut.append([neighbor, "ijk"[axis], neighborSwitch])

        return blocksToCut

    def performCutOperations(self, mb, cutOps):
        print("Performing cut/s...")
        for nblki, axis, nCuts in cutOps:
            cutBlk = mb.getBlock(nblki)
            ogNx = getattr(cutBlk, f"n{axis}")
            print(f"  Cutting Block {nblki}'s {axis} axis {nCuts} times.")

            for cut in range(nCuts):
                blocksToCut = self.cutPath(mb, nblki, axis)
                cutNx = getattr(cutBlk, f"n{axis}")

                cutIndex = int(ogNx * (nCuts - cut) / (nCuts + 1))
                switchCutIndex = cutNx - cutIndex - 1

                openFaces = []
                for cutNblki, cutAxis, switch in blocksToCut:
                    assert getattr(mb.getBlock(cutNblki), f"n{cutAxis}") == cutNx
                    index = switchCutIndex if switch else cutIndex
                    openFaces += self.cutBlock(mb, cutNblki, cutAxis, index)

                self._pairCutFaces(openFaces)

    ###########################################################################
    # Removing an interface, the inverse of a cut. The plane an interface lies on
    # runs on through whatever it meets, so every pair it passes through merges
    # at once, and only if the plane closes.
    ###########################################################################
    def _pairsOnPlane(self, mb, nblki, nface):
        """Every (block, face) pair the plane of this interface passes through, or
        None if the plane does not close."""
        start = mb.getBlock(nblki).getFace(nface)
        # a periodic names is a boundary
        if start.neighbor is None or start.bcType.startswith("periodic"):
            return None

        pairs, queue, seen = [], [(nblki, nface)], set()
        while queue:
            a, fa = queue.pop()
            A = mb.getBlock(a)
            faceA = A.getFace(fa)
            B = mb.getBlock(faceA.neighbor)
            key = frozenset({(a, fa), (B.nblki, faceA.neighborNface)})
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, fa))

            for sideA in A.faces:
                if sideA.myAxis == faceA.myAxis:
                    continue
                sideB = B.getFace(A.neighborNfaceOf(sideA.nface, across=fa))
                if sideA.neighbor is None or sideB.neighbor is None:
                    # the plane may end here, but on both sides and the same boundary
                    if (
                        sideA.neighbor is None
                        and sideB.neighbor is None
                        and sideA.bcType == sideB.bcType
                        and sideA.bcFam == sideB.bcFam
                    ):
                        continue
                    return None
                if sideA.bcType.startswith("periodic") or sideB.bcType.startswith(
                    "periodic"
                ):
                    return None
                # our side neighbor must meet the same block on the far side we do
                NA = mb.getBlock(sideA.neighbor)
                planeFace = A.neighborNfaceOf(fa, across=sideA.nface)
                if NA.getFace(planeFace).neighbor != sideB.neighbor:
                    return None
                queue.append((NA.nblki, planeFace))
        return pairs

    def removablePlanes(self, mb):
        """Every plane of the grid that can be removed, as its pairs. A plane is
        reported once."""
        planes, done = [], set()
        for blk in mb:
            for face in blk.faces:
                if (blk.nblki, face.nface) in done or face.neighbor is None:
                    continue
                pairs = self._pairsOnPlane(mb, blk.nblki, face.nface)
                if pairs is None:
                    done.add((blk.nblki, face.nface))
                    continue
                for a, fa in pairs:
                    far = mb.getBlock(a).getFace(fa)
                    done.add((a, fa))
                    done.add((far.neighbor, far.neighborNface))
                planes.append(pairs)
        return planes

    def mergePlane(self, mb, pairs):
        """Merge every pair on one plane. The near block of each pair keeps its
        id; the far block is removed."""
        merged = {}
        for a, fa in pairs:
            A = mb.getBlock(a)
            faceA = A.getFace(fa)
            B = mb.getBlock(faceA.neighbor)
            self.reorientBlock(mb, B, *faceA.alignsNeighborBy)

            axis = faceA.myAxis
            lower, upper = (B, A) if faceA.amILow else (A, B)
            dropShared = [slice(None)] * 3
            dropShared[axis] = slice(1, None)
            for name in ("x", "y", "z"):
                A.array[name] = np.concatenate(
                    [lower.array[name], upper.array[name][tuple(dropShared)]], axis=axis
                )
            dims = [A.ni, A.nj, A.nk]
            dims[axis] += (B.ni, B.nj, B.nk)[axis] - 1
            A.ni, A.nj, A.nk = dims

            # B is now in our frame, so its far face is ours on that side
            A.faces[fa - 1] = B.getFace(fa)
            merged[B.nblki] = A.nblki

        for blk in mb:
            for face in blk.faces:
                if face.neighbor in merged:
                    face.neighbor = merged[face.neighbor]
        self._compact(mb, set(merged))

    def _compact(self, mb, gone):
        """Drop the merged away blocks and number what is left from zero."""
        keep = [blk for blk in mb if blk.nblki not in gone]
        renumber = {blk.nblki: n for n, blk in enumerate(keep)}
        for blk in keep:
            blk.nblki = renumber[blk.nblki]
            # a merged block is its own base again
            blk.baseNblki, blk.baseSlice = blk.nblki, None
            for face in blk.faces:
                if face.neighbor is not None:
                    face.neighbor = renumber[face.neighbor]
        mb.data = keep
        mb.totalBlocks = len(keep)

    def mergeAll(self, mb):
        """Remove every removable plane, largest first, until none is left.
        Returns how many planes went."""
        removed = 0
        while True:
            planes = self.removablePlanes(mb)
            if not planes:
                return removed
            self.mergePlane(mb, max(planes, key=len))
            removed += 1

    ###########################################################################
    # Re-indexing a block's storage. A block's axes can be relabelled without
    # moving the grid it holds: new axis m runs along old axis perm[m], and runs
    # backwards when flips[m].
    ###########################################################################
    def _longestFirst(self, blk):
        """The relabelling that puts a block's longest extent on i and its second
        on j. k is reversed when the axis order is odd, so the block stays right
        handed."""
        perm = [int(a) for a in np.argsort([-blk.ni, -blk.nj, -blk.nk], kind="stable")]
        swaps = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b])
        return perm, [False, False, swaps % 2 == 1]

    def reorientBlock(self, mb, blk, perm, flips):
        """Relabel blk's axes by (perm, flips), in place."""
        if perm == [0, 1, 2] and not any(flips):
            return

        face = blk.faces[0]
        newAxis = {old: new for new, old in enumerate(perm)}

        # their strings name our axes, so each character becomes our new label
        for other in mb:
            if other is blk:
                continue
            for theirs in other.faces:
                if theirs.neighbor != blk.nblki or theirs.orientation is None:
                    continue
                theirs.orientation = "".join(
                    face.orientationCode(
                        newAxis[ours], counterAligned != flips[newAxis[ours]]
                    )
                    for ours, counterAligned in map(face.signedAxis, theirs.orientation)
                )

        # ours are indexed by our axis, so permuted not rewritten; a flip inverts
        for f in blk.faces:
            if f.orientation is None:
                continue
            old = f.orientation
            f.orientation = "".join(
                face.orientationCode(
                    *(lambda axis, counterAligned: (axis, counterAligned != flips[m]))(
                        *face.signedAxis(old[perm[m]])
                    )
                )
                for m in range(3)
            )

        # a face keeps its identity; its number follows its axis, a flip swaps ends
        reordered = [None] * 6
        for f in blk.faces:
            m = newAxis[f.myAxis]
            low = f.amILow != flips[m]
            f.nface = 2 * m + (1 if low else 2)
            reordered[f.nface - 1] = f
        blk.faces = reordered

        dims = (blk.ni, blk.nj, blk.nk)
        blk.ni, blk.nj, blk.nk = (dims[perm[m]] for m in range(3))
        for name, values in blk.array.items():
            if values is None:
                continue
            moved = np.moveaxis(values, perm, (0, 1, 2))
            for m in range(3):
                if flips[m]:
                    moved = np.flip(moved, axis=m)
            blk.array[name] = np.ascontiguousarray(moved)

    def longestAxisFirst(self, mb):
        """Relabel every block so its longest extent is i, the axis a launch walks
        innermost."""
        for blk in mb:
            self.reorientBlock(mb, blk, *self._longestFirst(blk))

    def condition(self, mb):
        """Ready a freshly translated grid: merge away every interface the grid
        does not need, then relabel every block so its longest extent is i."""
        print("Conditioning the grid...")
        before = len(mb)
        removed = self.mergeAll(mb)
        self.longestAxisFirst(mb)
        print(f"  merged away {removed} interface(s), {before} blocks -> {len(mb)}")
        print("  every block re-indexed so its longest extent is i")
