from ..misc import subclassWhere
from .blockOpsMixin import BlockOpsMixin
from .conditioner import Conditioner
from .basePartitioner import BasePartitioner
from .greedyPartitioner import GreedyPartitioner
from .metisPartitioner import MetisPartitioner
from .autoPartitioner import AutoPartitioner


def getPartitioner(name="auto", **kwargs):
    return subclassWhere(BasePartitioner, partitionerName=name)(**kwargs)


__all__ = [
    "AutoPartitioner",
    "BasePartitioner",
    "BlockOpsMixin",
    "Conditioner",
    "GreedyPartitioner",
    "MetisPartitioner",
    "getPartitioner",
]
