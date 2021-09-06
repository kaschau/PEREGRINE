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
            l_tbc.append(i+1)
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
            else: #thre-body
                Ea_f.append(r.rate.activation_energy/Ru)
                m_f.append(r.rate.temperature_exponent)
                A_f.append(r.rate.pre_exponential_factor)
                Ea_o.append(0.0)
                m_o.append(0.0)
                A_o.append(0.0)
        else: #elementry
            Ea_f.append(r.rate.activation_energy/Ru)
            m_f.append(r.rate.temperature_exponent)
            A_f.append(r.rate.pre_exponential_factor)

    nl_tbc = len(l_tbc)

    pg_mech = open(cpp, 'w')

    # WRITE OUT SPECIES ORDER
    pg_mech.write('// ==================================================================== //\n')
    for i,sp in enumerate(gas.species_names):
        pg_mech.write(f'// Y({i:>3d}) = {sp}\n')
    pg_mech.write('// ==================================================================== //\n')

    print(cpp.replace(".cpp",""))

    # WRITE OUT INITIALIZATION BLOCK UP TO CHEMICAL SOURCE TERMS
    out_string = f'''
#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <vector>

void {cpp.replace(".cpp","")}(std::vector<block_> mb, thtrdat_ th) {{
for(block_ b : mb){{

//-------------------------------------------------------------------------------------------|
// cc range
//-------------------------------------------------------------------------------------------|
  MDRange3 range = MDRange3({{1,1,1}},{{b.ni,b.nj,b.nk}});

  Kokkos::parallel_for("Compute chemical source terms",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {{

  const int ns={ns};
  const int nr={nr};
  const int l_tbc={nl_tbc};
  double p;
  double T;
  double Y[ns],cs[ns];

  double rho,rhoinv;

  p = b.q(i,j,k,0);
  T = b.q(i,j,k,4);
  rho = b.Q(i,j,k,0);
  rhoinv = 1.0/rho;
  // Compute nth species Y
  Y[ns-1] = 1.0;
  for (int n=0; n<ns-1; n++)
  {{
    Y[n] = b.q(i,j,k,5+n);
    Y[ns-1] -= Y[n];
  }}
  Y[ns-1] = std::max(0.0,Y[ns-1]);

  // Conecntrations
  for (int n=0; n<=ns-1; n++)
  {{
    cs[n] = rho*Y[n]/th.MW[n];
  }}
\n'''

    pg_mech.write(out_string)

    #-----------------------------------------------------------------------------
    # WRITE Chaperone Efficiencies
    #-----------------------------------------------------------------------------

    out_string = '''  // -------------------------------------------------------------- >
  // Chaperon efficiencies. --------------------------------------- >
  // -------------------------------------------------------------- >

  double S_tbc[nr] = {1.e0};
  double J_tbc[nr] = {0.e0};\n\n'''

    pg_mech.write(out_string)

    tbc_count = 0
    for i,r in enumerate(gas.reactions()):
        if r.reaction_type in ['three-body', 'falloff']: #ThreeBodyReaction and FallOffReactions
            out_string = []
            tbc_count += 1
            for j in range(ns):
                eff = aij[tbc_count-1][j]
                if eff > 0.0:
                    if eff != 1.0:
                        out_string.append(f' + {eff}*cs[{j}]')
                    else:
                        out_string.append(f' + cs[{j}]')
            out_string[0] = out_string[0].replace(' + ', '')
            pg_mech.write(f'  S_tbc[{i}] = ')
            for item in out_string:
                pg_mech.write(item)
            pg_mech.write(';\n')
            pg_mech.write(f'  J_tbc[{i}] = 1.e0;')
            pg_mech.write('\n\n')

    out_string = '''  // -------------------------------------------------------------- >
  // Gibbs energy. ------------------------------------------------ >
  // -------------------------------------------------------------- >

  int m;
  double hi,scs;
  double gbs[ns];

  for (int n=0; n<=ns-1; n++)
  {
    m = ( T <= th.NASA7[n][0] ) ? 8 : 1;

    hi     = th.NASA7[n][m+0]                  +
             th.NASA7[n][m+1]*    T      / 2.0 +
             th.NASA7[n][m+2]*pow(T,2.0) / 3.0 +
             th.NASA7[n][m+3]*pow(T,3.0) / 4.0 +
             th.NASA7[n][m+4]*pow(T,4.0) / 5.0 +
             th.NASA7[n][m+5]/    T            ;
    scs    = th.NASA7[n][m+0]*log(T)           +
             th.NASA7[n][m+1]*    T            +
             th.NASA7[n][m+2]*pow(T,2.0) / 2.0 +
             th.NASA7[n][m+3]*pow(T,3.0) / 3.0 +
             th.NASA7[n][m+4]*pow(T,4.0) / 4.0 +
             th.NASA7[n][m+6]                  ;

    gbs[n] = hi-scs                         ;
  }
\n'''

    pg_mech.write(out_string)

    #-----------------------------------------------------------------------------
    #WRITE OUT HARD CODED k_f dG and K_c
    #-----------------------------------------------------------------------------
    out_string = '''  // -------------------------------------------------------------- >
  // Rate Constants. --------------------------------------- >
  // -------------------------------------------------------------- >

  double q_f[nr],k_f[nr],c_f[nr];
  double q_b[nr],k_b[nr],c_b[nr];

  double dG[nr],K_c[nr],q[nr]; \n\n'''

    pg_mech.write(out_string)

    for i,r in enumerate(gas.reactions()):
        out_string = f'  k_f[{i}] = {A_f[i]}*pow(T,{m_f[i]})*exp(-({Ea_f[i]})/T);\n'
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
        pg_mech.write(f'  dG[{i}]  = ')
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(';\n')

        if np.sum(nu_sum) != 0:
            out_string = f'  K_c[{i}] = pow(101325.0/(th.Ru*T),{np.sum(nu_sum)})*exp(-dG[{i}]);'
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

  double Fcent[{nl_tbc}],dFcent[{nl_tbc}];
  double Pr_pdr;
  double A_pdr[{nl_tbc}];
  double B_pdr,C_pdr,D_pdr,E_pdr,F_pdr;
  double Ccent,Ncent;
\n'''

    pg_mech.write(out_string)

    tbc_count = 0
    for i,r in enumerate(gas.reactions()):
        if r.reaction_type == 'three-body': #ThreeBodyReaction
            tbc_count += 1
            pg_mech.write(f'  //  Three Body Reaction #{i+1}\n')
            pg_mech.write(f'  A_pdr[{tbc_count}] = 0.0;\n')
        elif r.reaction_type == 'falloff': # FallOff Reactions
            tbc_count += 1
            if r.falloff.type in ['Simple','Lindemann']:
                pg_mech.write(f'  //  Lindeman Reaction #{i+1}\n')
                out_string = f'''  Fcent[{tbc_count}] = 1.0;
  dFcent[{tbc_count}] = 0.0;
  Pr_pdr = S_tbc[{i}]*( {A_o[tbc_count-1]}*pow(T,{m_o[tbc_count-1]})*exp(-({Ea_o[tbc_count-1]})/T) )/k_f[{i}];
  A_pdr[{tbc_count}]  = 1.e0/(1.e0 + Pr_pdr);
  k_f[{i}] = k_f[{1}]*( Pr_pdr/(1.e0 + Pr_pdr) );
  S_tbc[{i}] = 1.0;\n'''
                pg_mech.write(out_string)

            elif r.falloff.type == 'Troe':
                alpha = r.falloff.parameters[0]
                Tsss = r.falloff.parameters[1]
                Ts = r.falloff.parameters[2]
                pg_mech.write(f'  //  Troe Reaction #{i+1}\n')
                tp = r.falloff.parameters
                if tp[-1] == 0: # Three Parameter Troe form
                    out_string = f'''  Fcent[{tbc_count}] =   (1.e0 - ({alpha}))*exp(-T/({Tsss}))
                        + ({alpha}) *exp(-T/({Ts}));
  dFcent[{tbc_count}] = - (1.e0 - ({alpha}))*exp(-T/({alpha}))/({alpha})
                             - ({alpha}) *exp(-T/({Ts}))/({Ts}); \n'''
                    pg_mech.write(out_string)
                elif tp[-1] != 0: # Four Parameter Troe form
                    Tss = r.falloff.parameters[3]
                    out_string = f'''  Fcent[{tbc_count}] =   (1.e0 - ({alpha}))*exp(-T/({Tsss}))
                             + ({alpha}) *exp(-T/({Ts})) + exp(-({Tss})/T);
  dFcent[{tbc_count}]= - (1.e0 - ({alpha}))*exp(-T/({Tsss}))/({Tsss})
                             - ({alpha}) *exp(-T/({Ts}))/({Ts}) + exp(-({Tss})/T)*({Tss})/pow(T,2); \n'''
                    pg_mech.write(out_string)
                else:
                    raise ValueError('Unknown Falloff type: {}, for reacion {}'.format(r.falloff.type, r.equation))

                out_string = f'''  Ccent = - 0.4e+0 - 0.67e+0*log10(Fcent[{tbc_count}]);
  Ncent =   0.75e+0 - 1.27e+0*log10(Fcent[{tbc_count}]);

  Pr_pdr = S_tbc[i]*( ({A_o[tbc_count-1]})*pow(T,{m_o[tbc_count-1]})*exp(-({Ea_o[tbc_count-1]})/T) )/k_f[i];

  B_pdr = log10(Pr_pdr) + Ccent;
  C_pdr = 1.e0/(1.e0 + pow(B_pdr/(Ncent - 0.14e+0*B_pdr),2.0));

  F_pdr = pow(10.e0,log10(Fcent[{tbc_count}])*C_pdr);

  D_pdr = 2.e0*B_pdr*log10(F_pdr)/pow(Ncent - 0.14e+0*B_pdr,3.0);
  E_pdr = C_pdr*(1.e0 + D_pdr*(1.27e+0*B_pdr - 0.67e+0*Ncent));

  dFcent[{tbc_count}]  = E_pdr*dFcent[{tbc_count}];

  A_pdr[{tbc_count}]  = 1.e0/(1.e0 + Pr_pdr) - Ncent*C_pdr*D_pdr;
  k_f[{i}] = k_f[{i}]*( Pr_pdr/(1.e0 + Pr_pdr) )*F_pdr;
  S_tbc[{i}] = 1.0; \n'''

                pg_mech.write(out_string)
            elif r.falloff.type == 'SRI': # SRI Form
                raise TypeError(' Warning, this utility cant handle SRI type reactions yet... so add it now')
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
    b.dQ(i,j,k,5+n) = omega[n]*b.dQ(i,j,k,5+n)*b.J(i,j,k);
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
