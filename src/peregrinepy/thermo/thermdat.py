import cantera as ct
from ..compute import thermdat_

class thermdat(thermdat_):

    def __init__(self, config):

        gas = ct.Solution(ct)
        self.ns = gas.n_species

        self.species_names = list(gas.species_names)

        #Set gas to STP
        gas.TP = 293.15,101325.0

        # Values for constant Cp
        # J/(kg.K)
        self.cp0 = list(gas.standard_cp_R*ct.gas_constant/gas.molecular_weight)
