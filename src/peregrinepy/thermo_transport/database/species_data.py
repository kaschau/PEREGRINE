import yaml
import cantera as ct

class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

gas = ct.Solution('ab-tpg.yaml')
gas.TP = 298.0, 101325.0

spl = {}
spl['avogadro'] = ct.avogadro
spl['Ru'] = ct.gas_constant
spl['kb'] = ct.boltzmann
spl['epsilon_0'] = ct.epsilon_0

for i,sp in enumerate(gas.species()):

    spl[sp.name] = {}
    s = spl[sp.name]

    s['comp'] = sp.composition

    s['MW'] = float(gas.molecular_weights[i])

    s['cp0'] = float(gas.standard_cp_R[i]*ct.gas_constant/gas.molecular_weights[i])
    s['NASA7'] = [float(j) for j in list(sp.thermo.coeffs)]

    s['well']     = float(sp.transport.well_depth)
    s['diam']     = float(sp.transport.diameter)
    s['dipole']   = float(sp.transport.dipole)
    s['polarize'] = float(sp.transport.polarizability)
    s['zrot']     = float(sp.transport.rotational_relaxation)
    s['acentric'] = float(sp.transport.acentric_factor)
    s['geometry'] = str(sp.transport.geometry)


with open(f"test.yaml", 'w') as f:
    yaml.dump(spl, f, Dumper=MyDumper, sort_keys=False)
