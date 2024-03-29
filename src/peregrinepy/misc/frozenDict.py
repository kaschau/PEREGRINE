class frozenDict(dict):
    """
    Defines a dictionary that can be frozen after its key/values pairs are set.

    We use this so users can't make mistakes and modify some dictionaries
    inappropriately.
    """

    __isfrozen = False

    def __setitem__(self, key, value):
        if self.__isfrozen and key not in self.keys():
            raise KeyError(
                f"{key} is not a valid input for this FrozenDict, check spelling and case"
            )
        super().__setitem__(key, value)

    def _freeze(self):
        self.__isfrozen = True
