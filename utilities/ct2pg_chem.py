#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This utility converts a Cantera input yaml format file into a hard coded
PEREGRINE chem finite rate source file.

Requires Cantera >2.6 installation.

Default behavior is to use the name of the .yaml file for the output c++
file name.

Example
-------
ct2pg_chem.py <cantera.yaml>

"""

import cantera as ct
import numpy as np


class UnknownReactionType(Exception):
    def __init__(self, rtype, num, eq):
        message = f'Unknown Reaction type: "{rtype}", for reacion #{num}: {eq}'
        super().__init__(message)


class UnknownFalloffType(Exception):
    def __init__(self, ftype, num, eq):
        message = f'Unknown Falloff type: "{ftype}", for reacion #{num}: {eq}'
        super().__init__(message)


def int_or_float(val):
    ival = int(val)

    if abs(val - ival) > 1e-16:
        return val
    else:
        return ival


def rate_const_string(A, m, Ea):
    if m == 0.0 and Ea == 0.0:
        string = f"{A}"
    elif m == 0.0 and Ea != 0.0:
        string = f"exp(log({A})-({Ea}/T))"
    elif isinstance(m, float) and Ea == 0.0:
        string = f"exp(log({A}){ m:+}*logT)"
    elif isinstance(m, int) and Ea == 0.0:
        if m < 0:
            string = f"{A}" + "".join("/T" for _ in range(m)) + ")"
        elif m > 0:
            string = f"{A}" + "".join("*T" for _ in range(m)) + ""
        else:
            raise ValueError("Huh?")
    elif Ea != 0.0:
        string = f"exp(log({A}){ m:+}*logT-({Ea}/T))"
    else:
        raise ValueError(f"Something is wrong here, m = {m}  Ea={Ea} ")

    return string


def ct2pg_chem(ctyaml, cpp):

    gas = ct.Solution(ctyaml)
    ns = gas.n_species
    nr = gas.n_reactions
    l_tbc = []  # list of all three body reactions

    Ea_f = []
    m_f = []
    A_f = []
    Ea_o = []
    m_o = []
    A_o = []
    aij = []
    nu_f = gas.reactant_stoich_coeffs3
    nu_b = gas.product_stoich_coeffs3

    Ru = ct.gas_constant  # J/kmol.K

    for i, r in enumerate(gas.reactions()):
        if r.reaction_type in [
            "three-body",
            "falloff",
        ]:  # ThreeBodyReaction or FallOffReactions
            l_tbc.append(i)
            efficiencies = []
            for j in range(ns):
                efficiencies.append(r.efficiency(gas.species_names[j]))
            aij.append(efficiencies)

            if r.reaction_type == "falloff":
                Ea_f.append(r.high_rate.activation_energy / Ru)
                m_f.append(r.high_rate.temperature_exponent)
                A_f.append(r.high_rate.pre_exponential_factor)
                Ea_o.append(r.low_rate.activation_energy / Ru)
                m_o.append(r.low_rate.temperature_exponent)
                A_o.append(r.low_rate.pre_exponential_factor)
            else:  # three-body
                Ea_f.append(r.rate.activation_energy / Ru)
                m_f.append(r.rate.temperature_exponent)
                A_f.append(r.rate.pre_exponential_factor)
                Ea_o.append(0.0)
                m_o.append(0.0)
                A_o.append(0.0)
        elif r.reaction_type == "elementary":  # elementry
            Ea_f.append(r.rate.activation_energy / Ru)
            m_f.append(int_or_float(r.rate.temperature_exponent))
            A_f.append(r.rate.pre_exponential_factor)
        else:
            raise UnknownReactionType(r.reaction_type, i, r.equation)
    nl_tbc = len(l_tbc)

    pg_mech = open(cpp, "w")

    # --------------------------------
    # HEADER
    # --------------------------------
    # WRITE OUT SPECIES ORDER
    pg_mech.write("// ========================================================== //\n")
    for i, sp in enumerate(gas.species_names):
        pg_mech.write(f"// Y({i:>3d}) = {sp}\n")
    pg_mech.write(
        "// ========================================================== //\n\n"
    )

    out_string = (
        '#include "Kokkos_Core.hpp"\n'
        '#include "kokkos_types.hpp"\n'
        '#include "block_.hpp"\n'
        '#include "thtrdat_.hpp"\n'
        '#include "compute.hpp"\n'
        "#include <math.h>\n"
        "\n"
        f'void {cpp.replace(".cpp","")}(block_ b, thtrdat_ th, int face/*=0*/, int i/*=0*/, int j/*=0*/, int k/*=0*/) {{\n'
        "\n"
        "// --------------------------------------------------------------|\n"
        "// cc range\n"
        "// --------------------------------------------------------------|\n"
        "  MDRange3 range = get_range3(b, face, i, j, k);\n"
        "\n"
        '  Kokkos::parallel_for("Compute chemical source terms",\n'
        "                       range,\n"
        "                       KOKKOS_LAMBDA(const int i,\n"
        "                                     const int j,\n"
        "                                     const int k) {\n"
        "\n"
        f"  const int ns={ns};\n"
        f"  const int nr={nr};\n"
        f"  const int l_tbc={nl_tbc};\n"
        "  double T,logT,prefRuT;\n"
        "  double Y[ns],cs[ns];\n"
        "\n"
        "  double rho;\n"
        "\n"
        "  T = b.q(i,j,k,4);\n"
        "  logT = log(T);\n"
        "  prefRuT = 101325.0/(th.Ru*T);\n"
        "  rho = b.Q(i,j,k,0);\n"
        "\n"
        "  // Compute nth species Y\n"
        "  Y[ns-1] = 1.0;\n"
        "  for (int n=0; n<ns-1; n++)\n"
        "  {\n"
        "    Y[n] = b.q(i,j,k,5+n);\n"
        "    Y[ns-1] -= Y[n];\n"
        "  }\n"
        "  Y[ns-1] = std::max(0.0,Y[ns-1]);\n"
        "\n"
        "  // Conecntrations\n"
        "  for (int n=0; n<=ns-1; n++)\n"
        "  {\n"
        "    cs[n] = rho*Y[n]/th.MW[n];\n"
        "  }\n"
        "\n"
    )

    pg_mech.write(out_string)

    # -----------------------------------------------------------------------------
    # Chaperone Efficiencies
    # -----------------------------------------------------------------------------

    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // Chaperon efficiencies. ------------------------------------ >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        "  std::array<double, nr> S_tbc;\n"
        "  S_tbc.fill(1.0);\n\n"
    )
    pg_mech.write(out_string)

    # ThreeBodyReaction and FallOffReactions
    for tbc, i in enumerate(l_tbc):
        out_string = []
        for j in range(ns):
            eff = aij[tbc][j]
            if eff > 0.0:
                if eff != 1.0:
                    out_string.append(f" + {eff}*cs[{j}]")
                else:
                    out_string.append(f" + cs[{j}]")
        out_string[0] = out_string[0].replace(" + ", "")
        pg_mech.write(f"  S_tbc[{i}] = ")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n\n")

    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // Gibbs energy. --------------------------------------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        "  int m;\n"
        "  double hi,scs;\n"
        "  double gbs[ns];\n"
        "\n"
        "  for (int n=0; n<=ns-1; n++)\n"
        "  {\n"
        "    m = ( T <= th.NASA7[n][0] ) ? 8 : 1;\n"
        "\n"
        "    hi     = th.NASA7[n][m+0]                  +\n"
        "             th.NASA7[n][m+1]*    T      / 2.0 +\n"
        "             th.NASA7[n][m+2]*pow(T,2.0) / 3.0 +\n"
        "             th.NASA7[n][m+3]*pow(T,3.0) / 4.0 +\n"
        "             th.NASA7[n][m+4]*pow(T,4.0) / 5.0 +\n"
        "             th.NASA7[n][m+5]/    T            ;\n"
        "    scs    = th.NASA7[n][m+0]*log(T)           +\n"
        "             th.NASA7[n][m+1]*    T            +\n"
        "             th.NASA7[n][m+2]*pow(T,2.0) / 2.0 +\n"
        "             th.NASA7[n][m+3]*pow(T,3.0) / 3.0 +\n"
        "             th.NASA7[n][m+4]*pow(T,4.0) / 4.0 +\n"
        "             th.NASA7[n][m+6]                  ;\n"
        "\n"
        "    gbs[n] = hi-scs                         ;\n"
        "  }\n"
        "\n"
    )
    pg_mech.write(out_string)

    # -----------------------------------------------------------------------------
    # HARD CODED forward rate constants k_f,  dG and K_c
    # -----------------------------------------------------------------------------
    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // Rate Constants. ------------------------------------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        "  double q_f[nr],k_f[nr];\n"
        "  double q_b[nr],k_b[nr];\n"
        "\n"
        "  double dG[nr],K_c[nr],q[nr]; \n\n"
    )
    pg_mech.write(out_string)

    for i in range(nr):
        out_string = (
            f"  k_f[{i}] = " + rate_const_string(A_f[i], m_f[i], Ea_f[i]) + ";\n"
        )

        pg_mech.write(out_string)
        nu_sum = nu_b[:, i] - nu_f[:, i]
        out_string = []
        for j, s in enumerate(nu_sum):
            if s == 1:
                out_string.append(f" + gbs[{j}]")
            elif s == -1:
                out_string.append(f" - gbs[{j}]")
            elif s != 0:
                out_string.append(f" {s:+}*gbs[{j}]")
        out_string[0] = out_string[0].replace("+", "")
        pg_mech.write(f"   dG[{i}] = ")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n")

        sum_nu_sum = np.sum(nu_sum)
        if sum_nu_sum != 0.0:
            if sum_nu_sum == 1.0:
                out_string = f"  K_c[{i}] = prefRuT*exp(-dG[{i}]);"
            elif sum_nu_sum == -1.0:
                out_string = f"  K_c[{i}] = exp(-dG[{i}])/prefRuT;"
            else:
                out_string = f"  K_c[{i}] = pow(prefRuT,{sum_nu_sum})*exp(-dG[{i}]);"
        else:
            out_string = f"  K_c[{i}] = exp(-dG[{i}]);"
        pg_mech.write(out_string)
        pg_mech.write("\n\n")

    # -----------------------------------------------------------------------------
    # FallOff Modifications
    # -----------------------------------------------------------------------------

    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // FallOff Modifications. ------------------------------------ >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        f"  double Fcent[{nl_tbc}];\n"
        f"  double pmod[{nl_tbc}];\n"
        "  double Pr,k0;\n"
        "  double A,f1,F_pdr;\n"
        "  double C,N;\n"
        "\n"
    )
    pg_mech.write(out_string)

    for i, r in enumerate([gas.reaction(j) for j in l_tbc]):
        if r.reaction_type == "three-body":  # ThreeBodyReaction
            pg_mech.write(f"  //  Three Body Reaction #{l_tbc[i]}\n")
        elif r.reaction_type == "falloff":  # FallOff Reactions
            if r.falloff.type in ["Simple", "Lindemann"]:
                pg_mech.write(f"  //  Lindeman Reaction #{l_tbc[i]}\n")
                pg_mech.write(f"  Fcent[{i}] = 1.0;\n")
                pg_mech.write(
                    "  k0 = " + rate_const_string(A_o[i], m_o[i], Ea_o[i]) + ";\n"
                )
                out_string = (
                    f"  Pr = S_tbc[{l_tbc[i]}]*k0/k_f[{l_tbc[i]}];\n"
                    f"  pmod[{i}] = Pr/(1.0 + Pr);\n"
                    f"  k_f[{l_tbc[i]}] = k_f[{l_tbc[i]}]*pmod[{i}];\n"
                )
                pg_mech.write(out_string)

            elif r.falloff.type == "Troe":
                alpha = r.falloff.parameters[0]
                Tsss = r.falloff.parameters[1]
                Ts = r.falloff.parameters[2]
                pg_mech.write(f"  //  Troe Reaction #{l_tbc[i]}\n")
                tp = r.falloff.parameters
                if tp[-1] == 0:  # Three Parameter Troe form
                    out_string = f"  Fcent[{i}] = (1.0 - ({alpha}))*exp(-T/({Tsss})) + ({alpha}) *exp(-T/({Ts}));\n"
                    pg_mech.write(out_string)
                elif tp[-1] != 0:  # Four Parameter Troe form
                    Tss = r.falloff.parameters[3]
                    out_string = f"  Fcent[{i}] = (1.0 - ({alpha}))*exp(-T/({Tsss})) + ({alpha}) *exp(-T/({Ts})) + exp(-({Tss})/T);\n"
                    pg_mech.write(out_string)

                out_string = (
                    f"  C = - 0.4 - 0.67*log10(Fcent[{i}]);\n"
                    f"  N =   0.75 - 1.27*log10(Fcent[{i}]);\n"
                )
                pg_mech.write(out_string)
                pg_mech.write(
                    "  k0 = " + rate_const_string(A_o[i], m_o[i], Ea_o[i]) + ";\n"
                )
                out_string = (
                    f"  Pr = S_tbc[{l_tbc[i]}]*k0/k_f[{l_tbc[i]}];\n"
                    "  A = log10(Pr) + C;\n"
                    "  f1 = A/(N - 0.14*A);\n"
                    f"  F_pdr = pow(10.0,log10(Fcent[{i}])/(1.0+f1*f1));\n"
                    "\n"
                    f"  pmod[{i}] =  Pr/(1.0 + Pr) * F_pdr;\n"
                    f"  k_f[{l_tbc[i]}] = k_f[{l_tbc[i]}]*pmod[{i}];\n"
                )

                pg_mech.write(out_string)
            elif r.falloff.type == "SRI":  # SRI Form
                raise NotImplementedError(
                    " Warning, this utility cant handle SRI type reactions yet... so add it now"
                )
            else:
                raise UnknownFalloffType(r.falloff.type, i, r.equation)
            pg_mech.write("\n")

    pg_mech.write("\n\n")

    # -----------------------------------------------------------------------------
    # Rates of progress
    # -----------------------------------------------------------------------------

    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // Forward, backward, net rates of progress. ----------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
    )
    pg_mech.write(out_string)

    for i, r in enumerate(gas.reactions()):
        out_string = []
        for j, s in enumerate(nu_f[:, i]):
            if s == 1.0:
                out_string.append(f" * cs[{j}]")
            elif s > 0.0:
                out_string.append(f" * pow(cs[{j}],{float(s)})")
        # S_tbc has already been applied to falloffs above!!!
        if r.reaction_type == "falloff":
            pg_mech.write(f"  q_f[{i}] =   k_f[{i}]")
        else:
            pg_mech.write(f"  q_f[{i}] =   S_tbc[{i}] * k_f[{i}]")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n")

        out_string = []
        for j, s in enumerate(nu_b[:, i]):
            if s == 1:
                out_string.append(f" * cs[{j}]")
            elif s > 0.0:
                out_string.append(f" * pow(cs[{j}],{float(s)})")
        # S_tbc has already been applied to falloffs above!!!
        if r.reaction_type == "falloff":
            pg_mech.write(f"  q_b[{i}] = - k_f[{i}]/K_c[{i}]")
        else:
            pg_mech.write(f"  q_b[{i}] = - S_tbc[{i}] * k_f[{i}]/K_c[{i}]")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n")
        if r.reversible:
            pg_mech.write(f"  q[  {i}] =   q_f[{i}] + q_b[{i}];\n\n")
        else:
            pg_mech.write(f"  q[  {i}] =   q_f[{i}];\n\n")

    # -------------------------------------------------------------------
    # SOURCE TERMS
    # -------------------------------------------------------------------
    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // Source terms. --------------------------------------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
    )
    pg_mech.write(out_string)

    for i in range(gas.n_species):
        out_string = []
        nu_sum = nu_b[i, :] - nu_f[i, :]
        for j, s in enumerate(nu_sum):
            if s == 1:
                out_string.append(f" +q[{j}]")
            elif s == -1:
                out_string.append(f" -q[{j}]")
            elif s != 0:
                out_string.append(f" {s:+}*q[{j}]")

        if len(out_string) == 0:
            pg_mech.write(f"  b.omega(i,j,k,{i+1}) = th.MW[{i}] * (0.0")
        else:
            out_string[0] = out_string[0].replace("+", "")
            pg_mech.write(f"  b.omega(i,j,k,{i+1}) = th.MW[{i}] * (")
            for item in out_string:
                pg_mech.write(item)
        pg_mech.write(");\n")

    out_string = (
        "\n"
        "  // Add source terms to RHS\n"
        "  for (int n=0; n<th.ns-1; n++)\n"
        "  {\n"
        "    b.dQ(i,j,k,5+n) += b.omega(i,j,k,n+1);\n"
        "  }\n"
        "  // Compute constant pressure dTdt dYdt (for implicit chem integration)\n"
        "  double dTdt = 0.0;\n"
        "  for (int n=0; n<=th.ns-1; n++)\n"
        "  {\n"
        "    dTdt -= b.qh(i,j,k,5+n) * b.omega(i,j,k,n+1);\n"
        "    b.omega(i,j,k,n+1) /= b.Q(i,j,k,0);\n"
        "  }\n"
        "  dTdt /= b.qh(i,j,k,1) * b.Q(i,j,k,0);\n"
        "  b.omega(i,j,k,0) = dTdt;\n"
        "\n"
    )
    pg_mech.write(out_string)

    # END
    out_string = "  });\n" "}"
    pg_mech.write(out_string)
    pg_mech.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="""Convert a cantera .yaml file to hard coded finite rate
        chemical source term c++ source code used by PEREGRINE"""
    )
    parser.add_argument(
        "ct_file_name",
        metavar="<ct_file>",
        help="""Cantera .yaml file to convert into hard coded PEREGRINE
        chemical source term.""",
        type=str,
    )
    args = parser.parse_args()

    ct_file_name = args.ct_file_name

    cpp_file_name = f'chem_{ct_file_name.replace(".yaml",".cpp")}'.replace("-", "_")

    ct2pg_chem(ct_file_name, cpp_file_name)
