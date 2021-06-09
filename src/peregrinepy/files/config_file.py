# -*- coding: utf-8 -*-

""" config.py

Authors:

Kyle Schau

This module holds a python dictionary version of a standard PEREGRINE config file.

"""

class FrozenDict(dict):
    __isfrozen = False

    def __setitem__(self, key, value):
        if self.__isfrozen and not key in self.keys():
            raise KeyError('{} is not a valid input file attribute, check spelling and case'.format(key))
        super(FrozenDict, self).__setitem__(key, value)

    def _freeze(self):
        self.__isfrozen = True


class config_file(FrozenDict):

    def __init__(self):
        self['io'] = FrozenDict({'griddir'   : './Grid',
                                 'inputdir'  : './Input',
                                 'outputdir' : './Output'})
        self['Kokkos'] = FrozenDict({'Space' : 'Default'})

        for key in self.keys():
            self[key]._freeze()

        # Freeze input file from adding new keys
        self._freeze()
