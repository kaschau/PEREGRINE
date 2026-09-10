from ..misc import subclassWhere
from .basePartitioner import BasePartitioner
from .greedyPartitioner import GreedyPartitioner
from .metisPartitioner import MetisPartitioner
from .autoPartitioner import AutoPartitioner


def getPartitioner(name="auto", **kwargs):
    return subclassWhere(BasePartitioner, partitionerName=name)(**kwargs)


__all__ = [
    "AutoPartitioner",
    "BasePartitioner",
    "GreedyPartitioner",
    "MetisPartitioner",
    "getPartitioner",
]
