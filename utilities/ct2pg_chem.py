#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''This utility converts a Cantera input yaml format file into a hard coded PEREGRINE chem finite rate source file.

Requires a cantera installation.

Default behavior is to use the name of the .yaml file for the output c++ file name.

Example
-------
ct2pg_chem.py <cantera.yaml>

'''

import cantera as ct
import numpy as np
import argparse

class UnknownReactionType(Exception):
    def __init__(self,rtype,num,eq):
            message = f'Unknown Reaction type: "{rtype}", for reacion #{num}: {eq}'
            super().__init__(message)

class UnknownFalloffType(Exception):
    def __init__(self,ftype,num,eq):
            message = f'Unknown Falloff type: "{ftype}", for reacion #{num}: {eq}'
            super().__init__(message)

def int_or_float(val):
    ival = int(val)

    if abs(val-ival) > 1e-16:
        return val
    else:
        return ival

def ct2pg_chem(ctyaml, cpp):

    gas = ct.Solution(ctyaml)
    ns = gas.n_species
    nr = gas.n_reactions
    l_tbc = []
    Ea_f = []
    m_f = []
    A_f = []
    Ea_o = []
    m_o = []
    A_o = []
    aij = []
    nu_f = gas.reactant_stoich_coeffs()
    nu_b = gas.product_stoich_coeffs()

    Ru = ct.gas_constant #J/kmol.K

    for i,r in enumerate(gas.reactions()):
        if r.reaction_type in ['three-body','falloff']: #ThreeBodyReaction or FallOffReactions
            l_tbc.append(i)
            efficiencies = []
            for j in range(ns):
                efficiencies.append(r.efficiency(gas.species_names[j]))
            aij.append(efficiencies)

            if r.reaction_type == 'falloff':
                Ea_f.append(r.high_rate.activation_energy/Ru)
                m_f.append(r.high_rate.temperature_exponent)
                A_f.append(r.high_rate.pre_exponential_factor)
                Ea_o.append(r.low_rate.activation_energy/Ru)
                m_o.append(r.low_rate.temperature_exponent)
                A_o.append(r.low_rate.pre_exponential_factor)
            else: #three-body
                Ea_f.append(r.rate.activation_energy/Ru)
                m_f.append(r.rate.temperature_exponent)
                A_f.append(r.rate.pre_exponential_factor)
                Ea_o.append(0.0)
                m_o.append(0.0)
                A_o.append(0.0)
        elif r.reaction_type == 'elementary': #elementry
            Ea_f.append(r.rate.activation_energy/Ru)
            m_f.append(int_or_float(r.rate.temperature_exponent))
            A_f.append(r.rate.pre_exponential_factor)
        else:
            raise UnknownReactionType(r.reaction_type,i+1,r.equation)


    nl_tbc = len(l_tbc)

    pg_mech = open(cpp, 'w')

    # WRITE OUT SPECIES ORDER
    pg_mech.write('// ==================================================================== //\n')
    for i,sp in enumerate(gas.species_names):
        pg_mech.write(f'// Y({i:>3d}) = {sp}\n')
    pg_mech.write('// ==================================================================== //\n\n')

    # WRITE OUT INITIALIZATION BLOCK UP TO CHEMICAL SOURCE TERMS
    out_string = (
                  '#include "Kokkos_Core.hpp"\n'
                  '#include "kokkos_types.hpp"\n'
                  '#include "block_.hpp"\n'
                  '#include "thtrdat_.hpp"\n'
                  '#include "compute.hpp"\n'
                  '#include <math.h>\n'
                  '#include <vector>\n'
                  '\n'
                 f'void {cpp.replace(".cpp","")}(std::vector<block_> mb, thtrdat_ th) {{\n'
                  'for(block_ b : mb){\n'
                  '\n'
                  '//-------------------------------------------------------------------------------------------|\n'
                  '// cc range\n'
                  '//-------------------------------------------------------------------------------------------|\n'
                  '  MDRange3 range = MDRange3({1,1,1},{b.ni,b.nj,b.nk});\n'
                  '\n'
                  '  Kokkos::parallel_for("Compute chemical source terms",\n'
                  '                       range,\n'
                  '                       KOKKOS_LAMBDA(const int i,\n'
                  '                                     const int j,\n'
                  '                                     const int k) {\n'
                  '\n'
                 f'  const int ns={ns};\n'
                 f'  const int nr={nr};\n'
                 f'  const int l_tbc={nl_tbc};\n'
                  '  double T,logT,prefRuT;\n'
                  '  double Y[ns],cs[ns];\n'
                  '\n'
                  '  double rho;\n'
                  '\n'
                  '  T = b.q(i,j,k,4);\n'
                  '  logT = log(T);\n'
                  '  prefRuT = 101325.0/(th.Ru*T);\n'
                  '  rho = b.Q(i,j,k,0);\n'
                  '\n'
                  '  // Compute nth species Y\n'
                  '  Y[ns-1] = 1.0;\n'
                  '  for (int n=0; n<ns-1; n++)\n'
                  '  {\n'
                  '    Y[n] = b.q(i,j,k,5+n);\n'
                  '    Y[ns-1] -= Y[n];\n'
                  '  }\n'
                  '  Y[ns-1] = std::max(0.0,Y[ns-1]);\n'
                  '\n'
                  '  // Conecntrations\n'
                  '  for (int n=0; n<=ns-1; n++)\n'
                  '  {\n'
                  '    cs[n] = rho*Y[n]/th.MW[n];\n'
                  '  }\n'
                  '\n'
                  )

    pg_mech.write(out_string)

    #-----------------------------------------------------------------------------
    # WRITE Chaperone Efficiencies
    #-----------------------------------------------------------------------------

    out_string = (
                  '  // -------------------------------------------------------------- >\n'
                  '  // Chaperon efficiencies. --------------------------------------- >\n'
                  '  // -------------------------------------------------------------- >\n'
                  '\n'
                  '  std::array<double, nr> S_tbc;\n'
                  '  S_tbc.fill(1.0);\n\n'
                  )

    pg_mech.write(out_string)

    tbc_count = 0
    for i,r in enumerate(gas.reactions()):
        if r.reaction_type in ['three-body', 'falloff']: #ThreeBodyReaction and FallOffReactions
            out_string = []
            for j in range(ns):
                eff = aij[tbc_count][j]
                if eff > 0.0:
                    if eff != 1.0:
                        out_string.append(f' + {eff}*cs[{j}]')
                    else:
                        out_string.append(f' + cs[{j}]')
            out_string[0] = out_string[0].replace(' + ', '')
            pg_mech.write(f'  S_tbc[{i}] = ')
            for item in out_string:
                pg_mech.write(item)
            pg_mech.write(';\n\n')
            tbc_count += 1

    out_string = (
                  '  // -------------------------------------------------------------- >\n'
                  '  // Gibbs energy. ------------------------------------------------ >\n'
                  '  // -------------------------------------------------------------- >\n'
                  '\n'
                  '  int m;\n'
                  '  double hi,scs;\n'
                  '  double gbs[ns];\n'
                  '\n'
                  '  for (int n=0; n<=ns-1; n++)\n'
                  '  {\n'
                  '    m = ( T <= th.NASA7[n][0] ) ? 8 : 1;\n'
                  '\n'
                  '    hi     = th.NASA7[n][m+0]                  +\n'
                  '             th.NASA7[n][m+1]*    T      / 2.0 +\n'
                  '             th.NASA7[n][m+2]*pow(T,2.0) / 3.0 +\n'
                  '             th.NASA7[n][m+3]*pow(T,3.0) / 4.0 +\n'
                  '             th.NASA7[n][m+4]*pow(T,4.0) / 5.0 +\n'
                  '             th.NASA7[n][m+5]/    T            ;\n'
                  '    scs    = th.NASA7[n][m+0]*log(T)           +\n'
                  '             th.NASA7[n][m+1]*    T            +\n'
                  '             th.NASA7[n][m+2]*pow(T,2.0) / 2.0 +\n'
                  '             th.NASA7[n][m+3]*pow(T,3.0) / 3.0 +\n'
                  '             th.NASA7[n][m+4]*pow(T,4.0) / 4.0 +\n'
                  '             th.NASA7[n][m+6]                  ;\n'
                  '\n'
                  '    gbs[n] = hi-scs                         ;\n'
                  '  }\n'
                  '\n'
                  )

    pg_mech.write(out_string)

    #-----------------------------------------------------------------------------
    #WRITE OUT HARD CODED k_f dG and K_c
    #-----------------------------------------------------------------------------
    out_string = '''  // -------------------------------------------------------------- >
  // Rate Constants. ---------------------------------------------- >
  // -------------------------------------------------------------- >

  double q_f[nr],k_f[nr];
  double q_b[nr],k_b[nr];

  double dG[nr],K_c[nr],q[nr]; \n\n'''

    pg_mech.write(out_string)

    for i,r in enumerate(gas.reactions()):
        if m_f[i] == 0.0 and Ea_f[i] == 0.0:
            out_string = f'  k_f[{i}] = {A_f[i]};\n'
        elif m_f[i] == 0.0 and Ea_f[i] != 0.0:
            out_string = f'  k_f[{i}] = exp(log({A_f[i]})-({Ea_f[i]}/T));\n'
        elif isinstance(m_f[i],float) and Ea_f[i] == 0.0:
            out_string = f'  k_f[{i}] = exp(log({A_f[i]}){ m_f[i]:+}*logT);\n'
        elif isinstance(m_f[i],int) and Ea_f[i] == 0.0:
            if m_f[i] < 0:
                out_string = f'  k_f[{i}] = {A_f[i]}'+''.join("/T" for _ in range(m_f[i]))+');\n'
            elif m_f[i] > 0:
                out_string = f'  k_f[{i}] = {A_f[i]}'+''.join("*T" for _ in range(m_f[i]))+';\n'
            else:
                raise ValueError('Huh?')
        elif Ea_f[i] != 0.0:
            out_string = f'  k_f[{i}] = exp(log({A_f[i]}){ m_f[i]:+}*logT-({Ea_f[i]}/T));\n'
        else:
            raise ValueError(f'Something is wrong here, m_f = {m_f[i]}  Ea_f={Ea_f[i]} ')

        pg_mech.write(out_string)
        nu_sum = nu_b[:,i] - nu_f[:,i]
        out_string = []
        for j,s in enumerate(nu_sum):
            if s == 1:
                out_string.append(f' + gbs[{j}]')
            elif s == -1:
                out_string.append(f' - gbs[{j}]')
            elif s != 0:
                out_string.append(f' {s:+}*gbs[{j}]')
        out_string[0] = out_string[0].replace('+', '')
        pg_mech.write(f'   dG[{i}] = ')
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(';\n')

        sum_nu_sum = np.sum(nu_sum)
        if sum_nu_sum != 0.0:
            if sum_nu_sum == 1.0:
                out_string = f'  K_c[{i}] = prefRuT*exp(-dG[{i}]);'
            elif sum_nu_sum == -1.0:
                out_string = f'  K_c[{i}] = exp(-dG[{i}])/prefRuT;'
            else:
                out_string = f'  K_c[{i}] = pow(prefRuT,{sum_nu_sum})*exp(-dG[{i}]);'
        else:
            out_string = f'  K_c[{i}] = exp(-dG[{i}]);'
        pg_mech.write(out_string)
        pg_mech.write('\n\n')

    #-----------------------------------------------------------------------------
    #WRITE FallOff Calculations
    #-----------------------------------------------------------------------------

    out_string = f'''  // -------------------------------------------------------------- >
  // FallOff Calculations. ---------------------------------------- >
  // -------------------------------------------------------------- >

  double Fcent[{nl_tbc}];
  double Pr_pdr;
  double B_pdr,C_pdr,F_pdr;
  double Ccent,Ncent;
\n'''

    pg_mech.write(out_string)

    tbc_count = 0
    for i,r in enumerate(gas.reactions()):
        if r.reaction_type == 'three-body': #ThreeBodyReaction
            pg_mech.write(f'  //  Three Body Reaction #{i+1}\n')
            tbc_count += 1
        elif r.reaction_type == 'falloff': # FallOff Reactions
            if r.falloff.type in ['Simple','Lindemann']:
                pg_mech.write(f'  //  Lindeman Reaction #{i+1}\n')
                out_string = f'''  Fcent[{tbc_count}] = 1.0;
  Pr_pdr = S_tbc[{i}]*( {A_o[tbc_count]}*pow(T,{m_o[tbc_count]})*exp(-({Ea_o[tbc_count]})/T) )/k_f[{i}];
  k_f[{i}] = k_f[{i}]*( Pr_pdr/(1.0 + Pr_pdr) );
  S_tbc[{i}] = 1.0;\n'''
                pg_mech.write(out_string)

            elif r.falloff.type == 'Troe':
                alpha = r.falloff.parameters[0]
                Tsss = r.falloff.parameters[1]
                Ts = r.falloff.parameters[2]
                pg_mech.write(f'  //  Troe Reaction #{i+1}\n')
                tp = r.falloff.parameters
                if tp[-1] == 0: # Three Parameter Troe form
                    out_string = f'''  Fcent[{tbc_count}] =   (1.0 - ({alpha}))*exp(-T/({Tsss}))
                        + ({alpha}) *exp(-T/({Ts}));\n'''
                    pg_mech.write(out_string)
                elif tp[-1] != 0: # Four Parameter Troe form
                    Tss = r.falloff.parameters[3]
                    out_string = f'''  Fcent[{tbc_count}] =   (1.0 - ({alpha}))*exp(-T/({Tsss}))
                             + ({alpha}) *exp(-T/({Ts})) + exp(-({Tss})/T);\n'''
                    pg_mech.write(out_string)

                out_string = f'''  Ccent = - 0.4 - 0.67*log10(Fcent[{tbc_count}]);
  Ncent =   0.75 - 1.27*log10(Fcent[{tbc_count}]);

  Pr_pdr = S_tbc[{i}]*( ({A_o[tbc_count]})*pow(T,{m_o[tbc_count]})*exp(-({Ea_o[tbc_count]})/T) )/k_f[{i}];

  B_pdr = log10(Pr_pdr) + Ccent;
  C_pdr = 1.0/(1.0 + pow(B_pdr/(Ncent - 0.14*B_pdr),2.0));

  F_pdr = pow(10.0,log10(Fcent[{tbc_count}])*C_pdr);

  k_f[{i}] = k_f[{i}]*( Pr_pdr/(1.0 + Pr_pdr) )*F_pdr;
  S_tbc[{i}] = 1.0; \n'''

                pg_mech.write(out_string)
            elif r.falloff.type == 'SRI': # SRI Form
                raise ValueError(' Warning, this utility cant handle SRI type reactions yet... so add it now')
            else:
                raise UnknownFalloffType(r.falloff.type,i+1,r.equation)
            tbc_count += 1
            pg_mech.write('\n')

    pg_mech.write('\n\n')

    #-----------------------------------------------------------------------------
    #WRITE OUT RATES OF PROGRESS AND SOURCE TERMS
    #-----------------------------------------------------------------------------

    out_string = '''  // -------------------------------------------------------------- >
  // Rates of Progress. ---------------------------------------- >
  // -------------------------------------------------------------- >\n\n'''

    pg_mech.write(out_string)
    for i,r in enumerate(gas.reactions()):
        out_string = []
        for j,s in enumerate(nu_f[:,i]):
            if s == 1:
                out_string.append(f' * cs[{j}]')
            elif s > 0.0:
                out_string.append(f' * pow(cs[{j}],{float(s)})')
        pg_mech.write(f'  q_f[{i}] =   S_tbc[{i}] * k_f[{i}]')
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(';\n')

        out_string = []
        for j,s in enumerate(nu_b[:,i]):
            if s == 1:
                out_string.append(f' * cs[{j}]')
            elif s > 0.0:
                out_string.append(f' * pow(cs[{j}],{float(s)})')
        pg_mech.write(f'  q_b[{i}] = - S_tbc[{i}] * k_f[{i}]/K_c[{i}]')
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(';\n')
        if r.reversible:
            pg_mech.write(f'  q[  {i}] =   q_f[{i}] + q_b[{i}];\n\n')
        else:
            pg_mech.write(f'  q[  {i}] =   q_f[{i}];\n\n')

    pg_mech.write('  // -------------------------------------------------------------->\n')

    pg_mech.write('  double omega[ns-1];\n\n')

    for i in range(gas.n_species-1):
        out_string = []
        nu_sum = nu_b[i,:] - nu_f[i,:]
        for j,s in enumerate(nu_sum):
            if s == 1:
                out_string.append(f' +q[{j}]')
            elif s == -1:
                out_string.append(f' -q[{j}]')
            elif s != 0:
                out_string.append(f' {s:+}*q[{j}]')

        if len(out_string) == 0:
            pg_mech.write(f'  omega[{i}] = th.MW[{i}] * (0.0')
        else:
            out_string[0] = out_string[0].replace('+', '')
            pg_mech.write(f'  omega[{i}] = th.MW[{i}] * (')
            for item in out_string:
                pg_mech.write(item)
        pg_mech.write(');\n')

    out_string = '''
  // Add source terms to RHS
  for (int n=0; n<th.ns-1; n++)
  {
    b.dQ(i,j,k,5+n) += omega[n];
  }

  });\n}}'''

    pg_mech.write(out_string)

    pg_mech.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert a cantera .yaml file to hard coded finite rate chemical source term c++ source code used by PEREGRINE')
    parser.add_argument('ct_file_name', metavar='<ct_file>', help='Cantera .yaml file to convert into hard coded PEREGRINE chemical source term.', type=str)
    args = parser.parse_args()

    ct_file_name = args.ct_file_name

    cpp_file_name = f'chem_{ct_file_name.replace(".yaml",".cpp")}'.replace('-','_')

    ct2pg_chem(ct_file_name, cpp_file_name)
