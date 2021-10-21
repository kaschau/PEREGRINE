#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This utility converts a cantera input yaml into
a PEREGRINE species data input file.
"""

import sys
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
        return super().increase_indent(flow, False)


ctfile = sys.argv[1]

gas = ct.Solution(ctfile)
gas.TP = 298.0, 101325.0

spl = {}

for i, sp in enumerate(gas.species()):
    temp = [0.0 for _ in range(gas.n_species)]
    temp[i] = 1.0
    gas.X = temp

    spl[sp.name] = {}
    s = spl[sp.name]

    s["comp"] = sp.composition

    s["MW"] = float(gas.molecular_weights[i])

    s["cp0"] = float(gas.standard_cp_R[i] * ct.gas_constant / gas.molecular_weights[i])

    s["mu0"] = float(gas.species_viscosities[i])

    s["kappa0"] = float(gas.thermal_conductivity)

    s["NASA7"] = [float(j) for j in list(sp.thermo.coeffs)]

    try:
        s["well"] = float(sp.transport.well_depth)
        s["diam"] = float(sp.transport.diameter)
        s["dipole"] = float(sp.transport.dipole)
        s["polarize"] = float(sp.transport.polarizability)
        s["zrot"] = float(sp.transport.rotational_relaxation)
        s["acentric"] = float(sp.transport.acentric_factor)
        s["geometry"] = str(sp.transport.geometry)
    except AttributeError:
        print("Transport data incomplete, leaving out of thtr species data.")


output = "thtr_" + ctfile
with open(output, "w") as f:
    yaml.dump(spl, f, Dumper=MyDumper, sort_keys=False)
