"""How each step is sized, separate from how it is taken."""


class BaseController:
    # what the config calls it
    name = None

    def __init__(self, cfgsect):
        pass

    def dt(self, solver):
        """The size of the next step."""
        raise NotImplementedError


class Fixed(BaseController):
    """Every step the config's dt."""

    name = "fixed"

    def __init__(self, cfgsect):
        self._dt = cfgsect["dt"]

    def dt(self, solver):
        return self._dt


class CFL(BaseController):
    """Each step as large as the config's max CFL allows, up to its max dt."""

    name = "cfl"

    def __init__(self, cfgsect):
        self.maxCFL, self.maxDt = cfgsect["maxCFL"], cfgsect["maxDt"]

    def dt(self, solver):
        return min(self.maxCFL / solver.maxCFL()[2], self.maxDt)
