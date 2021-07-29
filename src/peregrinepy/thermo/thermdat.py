import cantera as ct
from ..compute import thermdat_
from pathlib import Path

class thermdat(thermdat_):

    def __init__(self, config):
        thermdat_.__init__(self)

        relpath = str(Path(__file__).parent)
        ct.add_directory(relpath)
        gas = ct.Solution(config['thermochem']['ctfile'])

        self.ns = gas.n_species
        self.Ru = ct.gas_constant

        # Species names string
        self.species_names = list(gas.species_names)

        # Species MW
        self.MW = list(gas.molecular_weights)

        #Set gas to STP
        gas.TP = 293.15,101325.0
        # Values for constant Cp
        # J/(kg.K)
        self.cp0 = list(gas.standard_cp_R*ct.gas_constant/gas.molecular_weights)
