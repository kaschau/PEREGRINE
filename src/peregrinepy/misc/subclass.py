def subclasses(cls):
    """Every subclass of cls, at any depth."""
    direct = cls.__subclasses__()
    return direct + [g for s in direct for g in subclasses(s)]


def subclassWhere(cls, **kwargs):
    """The one subclass of cls whose attributes match."""
    for s in subclasses(cls):
        if all(getattr(s, k, None) == v for k, v in kwargs.items()):
            return s
    attrs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    raise KeyError(f"No subclass of {cls.__name__} with {attrs}.")
