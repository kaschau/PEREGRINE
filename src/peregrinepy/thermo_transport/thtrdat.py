import cantera as ct
from ..compute import thtrdat_
from pathlib import Path

class ththdat(thtrdat_):

    def __init__(self, config):
        thtrdat_.__init__(self)

        relpath = str(Path(__file__).parent)
        ct.add_directory(relpath)
        gas = ct.Solution(config['thermochem']['ctfile'])

        self.ns = gas.n_species
        self.Ru = ct.gas_constant
        self.kb = ct.boltzmann

        # Species names string
        self.species_names = list(gas.species_names)

        # Species MW
        self.MW = list(gas.molecular_weights)


        ##############################################################
        ####### Thermodynamic properties
        ##############################################################
        #Set either constant cp or NASA7 polynomial coefficients
        if config['thermochem']['eos'] == 'cpg':
            #Set gas to STP
            gas.TP = 293.15,101325.0
            # Values for constant Cp
            # J/(kg.K)
            cp0 = list(gas.standard_cp_R*ct.gas_constant/gas.molecular_weights)
            if len(cp0) != gas.n_species:
                raise ValueError('PEREGRINE ERROR: CPG info for all species (check cp0)')
            self.cp0 = cp0
        elif config['thermochem']['eos'] == 'tpg':
            N7 = []
            for n in range(gas.n_species):
                N7.append(list(gas.species()[n].thermo.coeffs))
            self.N7 = N7
        else:
            raise KeyError(f'PEREGRINE ERROR: Unknown EOS {config["thermochem"]["eos"]}')

        ##############################################################
        ####### Transport properties
        ##############################################################
