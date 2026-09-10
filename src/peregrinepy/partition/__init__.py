from ..misc import subclassWhere
from .cutMixin import CutMixin
from .mergeMixin import MergeMixin
from .orientMixin import OrientMixin
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
    "CutMixin",
    "Conditioner",
    "GreedyPartitioner",
    "MergeMixin",
    "MetisPartitioner",
    "OrientMixin",
    "getPartitioner",
]
