from .cut import (
    cutBlock,
    cutPath,
    cutTable,
    faceCenter,
    faceSearchPoint,
    faceSlice,
    pairCutFaces,
    performCutOperations,
)
from .partition import (
    cellWeights,
    edgesFromMb,
    metrics,
    partition,
    partitionHierarchical,
)
from .merge import compact, mergeAll, mergePlane, pairsOnPlane, removablePlanes
from .reorient import longestAxisFirst, longestFirst, reorientBlock


def condition(mb):
    """Ready a freshly translated grid: merge away every interface the grid
    does not need, then relabel every block so its longest extent is i."""
    print("Conditioning the grid...")
    before = len(mb)
    removed = mergeAll(mb)
    longestAxisFirst(mb)
    print(f"  merged away {removed} interface(s), {before} blocks -> {len(mb)}")
    print("  every block re-indexed so its longest extent is i")


__all__ = [
    "compact",
    "condition",
    "cutBlock",
    "cutPath",
    "cutTable",
    "faceCenter",
    "faceSearchPoint",
    "faceSlice",
    "longestAxisFirst",
    "longestFirst",
    "cellWeights",
    "edgesFromMb",
    "mergeAll",
    "metrics",
    "partition",
    "partitionHierarchical",
    "mergePlane",
    "pairCutFaces",
    "pairsOnPlane",
    "performCutOperations",
    "removablePlanes",
    "reorientBlock",
]
