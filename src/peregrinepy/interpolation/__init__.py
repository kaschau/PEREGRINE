from .baseInterpolator import BaseInterpolator
from .nearestInterpolator import NearestInterpolator
from .rbfInterpolator import RbfInterpolator


def getInterpolator(function="nearest", smooth=0.5, verboseSearch=False):
    """The interpolator :function: names. "nearest" is a nearest neighbour;
    anything else is one of scipy's radial basis functions, which is the one
    place that taxonomy is known."""
    if function == "nearest":
        return NearestInterpolator(verboseSearch=verboseSearch)
    return RbfInterpolator(
        function=function, smooth=smooth, verboseSearch=verboseSearch
    )


__all__ = [
    "BaseInterpolator",
    "NearestInterpolator",
    "RbfInterpolator",
    "getInterpolator",
]
