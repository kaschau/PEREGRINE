from ..misc import subclassWhere
from .baseMesher import BaseMesher
from .cubeMesher import CubeMesher
from .annulusMesher import AnnulusMesher


def getMesher(name, **kwargs):
    return subclassWhere(BaseMesher, mesherName=name)(**kwargs)


__all__ = ["AnnulusMesher", "BaseMesher", "CubeMesher", "getMesher"]
