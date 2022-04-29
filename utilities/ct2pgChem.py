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
    rTBC = []  # list of all three body reactions

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
        rate = r.rate
        if r.reaction_type in [
            "three-body",
            "falloff",
        ]:  # ThreeBodyReaction or FallOffReactions
            rTBC.append(i)
            efficiencies = []
            for j in range(ns):
                efficiencies.append(r.efficiency(gas.species_names[j]))
            aij.append(efficiencies)

            if r.reaction_type == "falloff":
                Ea_f.append(rate.high_rate.activation_energy / Ru)
                m_f.append(rate.high_rate.temperature_exponent)
                A_f.append(rate.high_rate.pre_exponential_factor)
                Ea_o.append(rate.low_rate.activation_energy / Ru)
                m_o.append(rate.low_rate.temperature_exponent)
                A_o.append(rate.low_rate.pre_exponential_factor)
            else:  # three-body
                Ea_f.append(rate.activation_energy / Ru)
                m_f.append(rate.temperature_exponent)
                A_f.append(rate.pre_exponential_factor)
                Ea_o.append(0.0)
                m_o.append(0.0)
                A_o.append(0.0)
        elif r.reaction_type == "reaction":
            Ea_f.append(rate.activation_energy / Ru)
            m_f.append(int_or_float(rate.temperature_exponent))
            A_f.append(rate.pre_exponential_factor)
        else:
            raise UnknownReactionType(r.reaction_type, i, r.equation)

    nTBC = len(rTBC)  # number of third body collision reaction
    pg_mech = open(cpp, "w")

    # --------------------------------
    # HEADER
    # --------------------------------
    # WRITE OUT SPECIES ORDER
    pg_mech.write("// ========================================================== //\n")
    for i, sp in enumerate(gas.species_names):
        pg_mech.write(f"// Y({i:>3d}) = {sp}\n")
    pg_mech.write(
        "\n"
        f"// {nr} reactions.\n"
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
        f'void {cpp.replace(".cpp","")}(block_ b, thtrdat_ th, int face/*=0*/, int indxI/*=0*/, int indxJ/*=0*/, int indxK/*=0*/, int nChemSubSteps/*=1*/, double dt/*1.0*/) {{\n'
        "\n"
        "// --------------------------------------------------------------|\n"
        "// cc range\n"
        "// --------------------------------------------------------------|\n"
        "  MDRange3 range = get_range3(b, face, indxI, indxJ, indxK);\n"
        "\n"
        '  Kokkos::parallel_for("Compute chemical source terms",\n'
        "                       range,\n"
        "                       KOKKOS_LAMBDA(const int i,\n"
        "                                     const int j,\n"
        "                                     const int k) {\n"
        "\n"
        "  double T = b.q(i,j,k,4);\n"
        "  double& rho = b.Q(i,j,k,0);\n"
        f"  double Y[{ns}];\n"
        f"  double dYdt[{ns-1}];\n"
        "  double dTdt=0.0;\n"
        "  const double logT = log(T);\n"
        "  const double prefRuT = 101325.0/(th.Ru*T);\n"
        "\n"
        "  //Set the initial values of Y array\n"
        f"  for (int n=0; n<{ns-1}; n++)\n"
        "  {\n"
        "    Y[n] = b.q(i,j,k,5+n);\n"
        "  }\n"
        "\n"
        "  for (int nSub=0; nSub<nChemSubSteps; nSub++){\n"
        "\n"
        "  // Compute nth species Y\n"
        f"  Y[{ns-1}] = 1.0;\n"
        "  double testSum = 0.0;\n"
        f"  for (int n=0; n<{ns-1}; n++)\n"
        "  {\n"
        "    Y[n] = fmax(fmin(Y[n],1.0),0.0);\n"
        f"    Y[{ns-1}] -= Y[n];\n"
        "    testSum += Y[n];\n"
        "  }\n"
        "  if (testSum > 1.0){\n"
        f"    Y[{ns-1}] = 0.0;\n"
        f"    for (int n=0; n<{ns-1}; n++)\n"
        "    {\n"
        "      Y[n] /= testSum;\n"
        "    }\n"
        "  }\n"
        "\n"
        "  // Conecntrations\n"
        f"  double cs[{ns}];\n"
        f"  for (int n=0; n<={ns-1}; n++)\n"
        "  {\n"
        "    cs[n] = rho*Y[n]/th.MW(n);\n"
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
        f"  double cTBC[{nTBC}];\n\n"
    )
    pg_mech.write(out_string)

    # ThreeBodyReaction and FallOffReactions
    for i in range(nTBC):
        out_string = []
        for j in range(ns):
            eff = aij[i][j]
            if eff > 0.0:
                if eff != 1.0:
                    out_string.append(f" + {eff}*cs[{j}]")
                else:
                    out_string.append(f" + cs[{j}]")
        out_string[0] = out_string[0].replace(" + ", "")
        pg_mech.write(f"  cTBC[{i}] = ")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n\n")

    out_string = (
        "  // ----------------------------------------------------------- >\n"
        "  // Gibbs energy. --------------------------------------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        "  int m;\n"
        f"  double hi[{ns}];\n"
        f"  double gbs[{ns}];\n"
        f"  double cp = 0.0;\n"
        "\n"
        f"  for (int n=0; n<={ns-1}; n++)\n"
        "  {\n"
        "    m = ( T <= th.NASA7(n,0) ) ? 8 : 1;\n"
        "    double cps =(th.NASA7(n,m+0)            +\n"
        "                 th.NASA7(n,m+1)*    T      +\n"
        "                 th.NASA7(n,m+2)*pow(T,2.0) +\n"
        "                 th.NASA7(n,m+3)*pow(T,3.0) +\n"
        "                 th.NASA7(n,m+4)*pow(T,4.0) )*th.Ru/th.MW(n);\n"
        "\n"
        "    hi[n]      = th.NASA7(n,m+0)                  +\n"
        "                 th.NASA7(n,m+1)*    T      / 2.0 +\n"
        "                 th.NASA7(n,m+2)*pow(T,2.0) / 3.0 +\n"
        "                 th.NASA7(n,m+3)*pow(T,3.0) / 4.0 +\n"
        "                 th.NASA7(n,m+4)*pow(T,4.0) / 5.0 +\n"
        "                 th.NASA7(n,m+5)/    T            ;\n"
        "\n"
        "    double scs = th.NASA7(n,m+0)*log(T)           +\n"
        "                 th.NASA7(n,m+1)*    T            +\n"
        "                 th.NASA7(n,m+2)*pow(T,2.0) / 2.0 +\n"
        "                 th.NASA7(n,m+3)*pow(T,3.0) / 3.0 +\n"
        "                 th.NASA7(n,m+4)*pow(T,4.0) / 4.0 +\n"
        "                 th.NASA7(n,m+6)                  ;\n"
        "\n"
        "    cp += cps  *Y[n];\n"
        "    gbs[n] = hi[n]-scs;\n"
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
        "  // FallOff Modifications. ------------------------------------ >\n"
        "  // Forward, backward, net rates of progress. ----------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        "  double k_f, dG, K_c; \n\n"
    )
    pg_mech.write(out_string)

    out_string = (
        "  double Fcent;\n"
        "  double pmod;\n"
        "  double Pr,k0;\n"
        "  double A,f1,F_pdr;\n"
        "  double C,N;\n"
        "\n"
    )
    pg_mech.write(out_string)

    out_string = "  double q_f, q_b;\n" + f"  double q[{nr}];\n\n"
    pg_mech.write(out_string)

    for i in range(nr):
        pg_mech.write(f"  // Reaction #{i}\n")
        out_string = "  k_f = " + rate_const_string(A_f[i], m_f[i], Ea_f[i]) + ";\n"

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
        pg_mech.write("   dG = ")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n")

        sum_nu_sum = np.sum(nu_sum)
        if sum_nu_sum != 0.0:
            if sum_nu_sum == 1.0:
                out_string = "  K_c = prefRuT*exp(-dG);"
            elif sum_nu_sum == -1.0:
                out_string = "  K_c = exp(-dG)/prefRuT;"
            else:
                out_string = f"  K_c = pow(prefRuT,{sum_nu_sum})*exp(-dG);"
        else:
            out_string = "  K_c = exp(-dG);"
        pg_mech.write(out_string)
        pg_mech.write("\n")

        # -----------------------------------------------------------------------------
        # FallOff Modifications
        # -----------------------------------------------------------------------------
        r = gas.reactions()[i]
        if i in rTBC:
            j = rTBC.index(i)
        if r.reaction_type == "three-body":  # ThreeBodyReaction
            pg_mech.write(f"  //  Three Body Reaction #{i}\n")
            out_string = f"  k_f *= cTBC[{j}];\n"
            pg_mech.write(out_string)
        elif r.reaction_type == "falloff":  # FallOff Reactions
            rate = r.rate
            if rate.type == "Lindemann":
                pg_mech.write(f"  //  Lindeman Reaction #{i}\n")
                pg_mech.write("  Fcent = 1.0;\n")
                pg_mech.write(
                    "  k0 = " + rate_const_string(A_o[j], m_o[j], Ea_o[j]) + ";\n"
                )
                out_string = (
                    f"  Pr = cTBC[{j}]*k0/k_f;\n"
                    f"  pmod = Pr/(1.0 + Pr);\n"
                    f"  k_f *= pmod;\n"
                )
                pg_mech.write(out_string)

            elif rate.type == "Troe":
                alpha = rate.falloff_coeffs[0]
                Tsss = rate.falloff_coeffs[1]
                Ts = rate.falloff_coeffs[2]
                pg_mech.write(f"  //  Troe Reaction #{i}\n")
                tp = rate.falloff_coeffs
                if tp[-1] == 0:  # Three Parameter Troe form
                    out_string = f"  Fcent = (1.0 - ({alpha}))*exp(-T/({Tsss})) + ({alpha}) *exp(-T/({Ts}));\n"
                    pg_mech.write(out_string)
                elif tp[-1] != 0:  # Four Parameter Troe form
                    Tss = rate.falloff_coeffs[3]
                    out_string = f"  Fcent = (1.0 - ({alpha}))*exp(-T/({Tsss})) + ({alpha}) *exp(-T/({Ts})) + exp(-({Tss})/T);\n"
                    pg_mech.write(out_string)

                out_string = (
                    "  C = - 0.4 - 0.67*log10(Fcent);\n"
                    "  N =   0.75 - 1.27*log10(Fcent);\n"
                )
                pg_mech.write(out_string)
                pg_mech.write(
                    "  k0 = " + rate_const_string(A_o[j], m_o[j], Ea_o[j]) + ";\n"
                )
                out_string = (
                    f"  Pr = cTBC[{j}]*k0/k_f;\n"
                    "  A = log10(Pr) + C;\n"
                    "  f1 = A/(N - 0.14*A);\n"
                    "  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));\n"
                    "  pmod = Pr/(1.0 + Pr) * F_pdr;\n"
                    "  k_f *= pmod;\n"
                )

                pg_mech.write(out_string)
            elif rate.type == "SRI":  # SRI Form
                raise NotImplementedError(
                    " Warning, this utility cant handle SRI type reactions yet... so add it now"
                )
            else:
                raise UnknownFalloffType(rate.type, i, r.equation)

        # -----------------------------------------------------------------------------
        # Rates of progress
        # -----------------------------------------------------------------------------
        out_string = []
        for j, s in enumerate(nu_f[:, i]):
            if s == 1.0:
                out_string.append(f" * cs[{j}]")
            elif s > 0.0:
                out_string.append(f" * pow(cs[{j}],{float(s)})")
        # cTBC has already been applied to falloffs above!!!
        pg_mech.write("  q_f =  k_f")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n")

        out_string = []
        for j, s in enumerate(nu_b[:, i]):
            if s == 1:
                out_string.append(f" * cs[{j}]")
            elif s > 0.0:
                out_string.append(f" * pow(cs[{j}],{float(s)})")
        # cTBC has already been applied to falloffs above!!!
        pg_mech.write("  q_b = -k_f/K_c")
        for item in out_string:
            pg_mech.write(item)
        pg_mech.write(";\n")
        if r.reversible:
            pg_mech.write(f"  q[{i}] = q_f + q_b;\n\n")
        else:
            pg_mech.write(f"  q[{i}] = q_f;\n\n")

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

    for i in range(gas.n_species - 1):
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
            pg_mech.write(f"  dYdt[{i}] = th.MW({i}) * (0.0")
        else:
            out_string[0] = out_string[0].replace("+", "")
            pg_mech.write(f"  dYdt[{i}] = th.MW({i}) * (")
            for item in out_string:
                pg_mech.write(item)
        pg_mech.write(");\n")

    out_string = (
        "\n"
        "  dTdt = 0.0;\n"
        f"  for (int n=0; n<{ns-1}; n++)\n"
        "  {\n"
        "    dTdt -= hi[n] * dYdt[n];\n"
        "    Y[n] += dYdt[n]/rho*(dt/nChemSubSteps);\n"
        "  }\n"
        "  dTdt /= cp * rho;\n"
        "  T += dTdt*dt/(nChemSubSteps);\n"
        "\n"
        "  }// End of chem sub step for loop\n"
        "\n"
        "  // Compute d(rhoYi)/dt based on where we end up\n"
        "  // Add source terms to RHS\n"
        f"  for (int n=0; n<{ns-1}; n++)\n"
        "  {\n"
        "    b.dQ(i,j,k,5+n) += (Y[n]*rho - b.Q(i,j,k,5+n))/dt;\n"
        "  }\n"
        "\n"
        "  // Store dTdt and dYdt (for implicit chem integration)\n"
        f"  for (int n=0; n<{ns-1}; n++)\n"
        "  {\n"
        "    b.omega(i,j,k,n+1) = dYdt[n] / b.Q(i,j,k,0);\n"
        "  }\n"
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

    if float(ct.__version__[0:3]) < 2.6:
        raise ImportError("Cantera version > 2.6 required.")

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
