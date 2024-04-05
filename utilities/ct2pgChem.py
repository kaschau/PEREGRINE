#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This utility converts a Cantera input yaml format file into a hard coded
PEREGRINE chem finite rate source file.

Requires Cantera >3.0 installation.

Default behavior is to use the name of the .yaml file for the output c++
file name.

Example
-------
ct2pgChem.py <cantera.yaml>

"""

import subprocess

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


def intOrFloat(val):
    ival = int(val)

    if abs(val - ival) > 1e-16:
        return val
    else:
        return ival


def rateConstString(A, m, Ea):
    if m == 0.0 and Ea == 0.0:
        string = f"{A}"
    elif m == 0.0 and Ea != 0.0:
        string = f"exp(log({A})-({Ea}*Tinv))"
    elif isinstance(m, float) and Ea == 0.0:
        string = f"exp(log({A}){ m:+}*logT)"
    elif isinstance(m, int) and Ea == 0.0:
        if m < 0:
            string = f"{A}" + "".join("*Tinv" for _ in range(m)) + ")"
        elif m > 0:
            string = f"{A}" + "".join("*T" for _ in range(m)) + ""
        else:
            raise ValueError("Huh?")
    elif Ea != 0.0:
        string = f"exp(log({A}){ m:+}*logT-({Ea}*Tinv))"
    else:
        raise ValueError(f"Something is wrong here, m = {m}  Ea={Ea} ")

    return string


def ct2pgChem(ctyaml, cpp):
    gas = ct.Solution(ctyaml)
    ns = gas.n_species
    nr = gas.n_reactions

    Ea_f = np.zeros(nr)
    m_f = np.zeros(nr)
    A_f = np.zeros(nr)
    Ea_o = np.zeros(nr)
    m_o = np.zeros(nr)
    A_o = np.zeros(nr)
    aij = np.zeros((nr, ns))
    nu_f = gas.reactant_stoich_coeffs
    nu_b = gas.product_stoich_coeffs

    Ru = ct.gas_constant  # J/kmol.K

    for i, r in enumerate(gas.reactions()):
        rate = r.rate
        if r.reaction_type.startswith(
            tuple(["three-body", "falloff"])
        ):  # ThreeBodyReaction or FallOffReactions
            for j in range(ns):
                aij[i, j] = r.third_body.efficiency(gas.species_names[j])

            if r.reaction_type.startswith("falloff"):
                Ea_f[i] = rate.high_rate.activation_energy / Ru
                m_f[i] = rate.high_rate.temperature_exponent
                A_f[i] = rate.high_rate.pre_exponential_factor
                Ea_o[i] = rate.low_rate.activation_energy / Ru
                m_o[i] = rate.low_rate.temperature_exponent
                A_o[i] = rate.low_rate.pre_exponential_factor
            else:  # three-body
                Ea_f[i] = rate.activation_energy / Ru
                m_f[i] = rate.temperature_exponent
                A_f[i] = rate.pre_exponential_factor
                Ea_o[i] = 0.0
                m_o[i] = 0.0
                A_o[i] = 0.0
        elif r.reaction_type == "Arrhenius":
            Ea_f[i] = rate.activation_energy / Ru
            m_f[i] = intOrFloat(rate.temperature_exponent)
            A_f[i] = rate.pre_exponential_factor
        else:
            raise UnknownReactionType(r.reaction_type, i, r.equation)

    pgMech = open(cpp, "w")

    # --------------------------------
    # HEADER
    # --------------------------------
    # WRITE OUT SPECIES ORDER
    for line in gas.input_header["description"].split("\n"):
        pgMech.write("// " + line + "\n")
    pgMech.write("")
    pgMech.write("// ========================================================== //\n")
    for i, sp in enumerate(gas.species_names):
        pgMech.write(f"// Y({i:>3d}) = {sp}\n")
    pgMech.write(
        "\n"
        f"// {nr} reactions.\n"
        "// ========================================================== //\n\n"
    )

    outString = (
        '#include "kokkosTypes.hpp"\n'
        '#include "block_.hpp"\n'
        '#include "thtrdat_.hpp"\n'
        '#include "compute.hpp"\n'
        "#include <Kokkos_Core.hpp>\n"
        "#include <math.h>\n"
        "\n"
        f"void {cpp.replace('.cpp','')}(block_ &b,\n"
        "                               const thtrdat_ &th,\n"
        "                               const int &rface/*=0*/,\n"
        "                               const int &indxI/*=0*/, const int &indxJ/*=0*/, const int &indxK/*=0*/,\n"
        "                               const int &nChemSubSteps/*=1*/, const double &dt/*=1.0*/) {\n"
        "\n"
        "// --------------------------------------------------------------|\n"
        "// cc range\n"
        "// --------------------------------------------------------------|\n"
        "  MDRange3 range = getRange3(b, rface, indxI, indxJ, indxK);\n"
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
        f"  double dYdt[{ns}];\n"
        "  double dTdt=0.0;\n"
        "  double tSub=dt/nChemSubSteps;\n"
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
        "  // Concentrations\n"
        f"  double cs[{ns}];\n"
        f"  for (int n=0; n<={ns-1}; n++)\n"
        "  {\n"
        "    cs[n] = rho*Y[n]/th.MW(n);\n"
        "  }\n"
        "\n"
    )

    pgMech.write(outString)

    outString = (
        "  // ----------------------------------------------------------- >\n"
        "  // Gibbs energy. --------------------------------------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
        f"  double hi[{ns}];\n"
        f"  double gbs[{ns}];\n"
        f"  double cp = 0.0;\n"
        "// start scope of precomputed T**\n"
        "{\n"
        "  double logT = log(T);\n"
        "  double Tinv = 1.0/T;\n"
        "  double To2 = T/2.0;\n"
        "  double T2 = pow(T,2);\n"
        "  double T3 = pow(T,3);\n"
        "  double T4 = pow(T,4);\n"
        "  double T2o2 = T2/2.0;\n"
        "  double T3o3 = T3/3.0;\n"
        "  double T4o4 = T4/4.0;\n"
        "  double T2o3 = T2/3.0;\n"
        "  double T3o4 = T3/4.0;\n"
        "  double T4o5 = T4/5.0;\n"
        "\n"
        f"  for (int n=0; n<={ns-1}; n++)\n"
        "  {\n"
        "    int m = ( T <= th.NASA7(n,0) ) ? 8 : 1;\n"
        "    double cps =(th.NASA7(n,m+0)    +\n"
        "                 th.NASA7(n,m+1)*T  +\n"
        "                 th.NASA7(n,m+2)*T2 +\n"
        "                 th.NASA7(n,m+3)*T3 +\n"
        "                 th.NASA7(n,m+4)*T4 )*th.Ru/th.MW(n);\n"
        "\n"
        "    hi[n]      = th.NASA7(n,m+0)      +\n"
        "                 th.NASA7(n,m+1)*To2  +\n"
        "                 th.NASA7(n,m+2)*T2o3 +\n"
        "                 th.NASA7(n,m+3)*T3o4 +\n"
        "                 th.NASA7(n,m+4)*T4o5 +\n"
        "                 th.NASA7(n,m+5)*Tinv ;\n"
        "\n"
        "    double scs = th.NASA7(n,m+0)*logT +\n"
        "                 th.NASA7(n,m+1)*   T +\n"
        "                 th.NASA7(n,m+2)*T2o2 +\n"
        "                 th.NASA7(n,m+3)*T3o3 +\n"
        "                 th.NASA7(n,m+4)*T4o4 +\n"
        "                 th.NASA7(n,m+6)      ;\n"
        "\n"
        "    cp += cps  *Y[n];\n"
        "    gbs[n] = hi[n]-scs;\n"
        "  }\n"
        "// ends scope of precomputed T**\n"
        "}\n"
    )
    pgMech.write(outString)

    # -----------------------------------------------------------------------------
    # HARD CODED forward rate constants k_f,  dG and K_c
    # -----------------------------------------------------------------------------
    outString = (
        "  // ----------------------------------------------------------- >\n"
        "  // Rate Constants. ------------------------------------------- >\n"
        "  // FallOff Modifications. ------------------------------------ >\n"
        "  // Forward, backward, net rates of progress. ----------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
    )
    pgMech.write(outString)

    outString = (
        f"  double q[{nr}];\n\n"
        "// start scope of these temp vars\n"
        "{\n"
        "  double cTBC, k_f, dG, K_c; \n\n"
        "  double Fcent;\n"
        "  double pmod;\n"
        "  double Pr,k0;\n"
        "  double A,f1,F_pdr;\n"
        "  double C,N;\n"
        "  double q_f, q_b;\n"
        "  double logT = log(T);\n"
        "  double Tinv = 1.0/T;\n"
        "  double prefRuT = 101325.0/(th.Ru*T);\n"
    )

    pgMech.write(outString)

    for i in range(nr):
        pgMech.write(f"  // Reaction #{i}\n")
        outString = "  k_f = " + rateConstString(A_f[i], m_f[i], Ea_f[i]) + ";\n"

        pgMech.write(outString)
        nu_sum = nu_b[:, i] - nu_f[:, i]
        outString = []
        for j, s in enumerate(nu_sum):
            if s == 1:
                outString.append(f" + gbs[{j}]")
            elif s == -1:
                outString.append(f" - gbs[{j}]")
            elif s != 0:
                outString.append(f" {s:+}*gbs[{j}]")
        outString[0] = outString[0].replace("+", "")
        pgMech.write("   dG = ")
        for item in outString:
            pgMech.write(item)
        pgMech.write(";\n")

        sum_nu_sum = np.sum(nu_sum)
        if sum_nu_sum != 0.0:
            if sum_nu_sum == 1.0:
                outString = "  K_c = prefRuT*exp(-dG);"
            elif sum_nu_sum == -1.0:
                outString = "  K_c = exp(-dG)/prefRuT;"
            else:
                outString = f"  K_c = pow(prefRuT,{sum_nu_sum})*exp(-dG);"
        else:
            outString = "  K_c = exp(-dG);"
        pgMech.write(outString)
        pgMech.write("\n")

        # -----------------------------------------------------------------------------
        # FallOff Modifications
        # -----------------------------------------------------------------------------
        r = gas.reactions()[i]
        rate = r.rate
        if np.sum(aij[i]) > 0.0:  # then this is a third body raction
            pgMech.write(f"  //  Three Body Reaction #{i}\n")
            outString = []
            for j in range(ns):
                eff = aij[i][j]
                if eff > 0.0:
                    if eff != 1.0:
                        outString.append(f" + {eff}*cs[{j}]")
                    else:
                        outString.append(f" + cs[{j}]")
            outString[0] = outString[0].replace(" + ", "")
            pgMech.write("  cTBC = ")
            for item in outString:
                pgMech.write(item)
            pgMech.write(";\n")

        if r.reaction_type == "three-body-Arrhenius":  # ThreeBodyReaction
            outString = "  k_f *= cTBC;\n"
            pgMech.write(outString)
        elif r.reaction_type == "falloff-Lindemann":
            pgMech.write(f"  //  Lindeman Reaction #{i}\n")
            pgMech.write("  Fcent = 1.0;\n")
            pgMech.write("  k0 = " + rateConstString(A_o[i], m_o[i], Ea_o[i]) + ";\n")
            outString = (
                "  Pr = cTBC*k0/k_f;\n" "  pmod = Pr/(1.0 + Pr);\n" "  k_f *= pmod;\n"
            )
            pgMech.write(outString)

        elif r.reaction_type == "falloff-Troe":
            alpha = rate.falloff_coeffs[0]
            Tsss = rate.falloff_coeffs[1]
            Ts = rate.falloff_coeffs[2]
            pgMech.write(f"  //  Troe Reaction #{i}\n")
            tp = rate.falloff_coeffs
            if tp[-1] == 0:  # Three Parameter Troe form
                outString = f"  Fcent = (1.0 - ({alpha}))*exp(-T/({Tsss})) + ({alpha}) *exp(-T/({Ts}));\n"
                pgMech.write(outString)
            elif tp[-1] != 0:  # Four Parameter Troe form
                Tss = rate.falloff_coeffs[3]
                outString = f"  Fcent = (1.0 - ({alpha}))*exp(-T/({Tsss})) + ({alpha}) *exp(-T/({Ts})) + exp(-({Tss})*Tinv);\n"
                pgMech.write(outString)

            outString = (
                "  C = - 0.4 - 0.67*log10(Fcent);\n"
                "  N =   0.75 - 1.27*log10(Fcent);\n"
            )
            pgMech.write(outString)
            pgMech.write("  k0 = " + rateConstString(A_o[i], m_o[i], Ea_o[i]) + ";\n")
            outString = (
                "  Pr = cTBC*k0/k_f;\n"
                "  A = log10(Pr) + C;\n"
                "  f1 = A/(N - 0.14*A);\n"
                "  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));\n"
                "  pmod = Pr/(1.0 + Pr) * F_pdr;\n"
                "  k_f *= pmod;\n"
            )

            pgMech.write(outString)
        elif r.reaction_type == "SRI":  # SRI Form
            raise NotImplementedError(
                " Warning, this utility cant handle SRI type reactions yet... so add it now"
            )
        else:
            if r.reaction_type not in ["Arrhenius"]:
                raise UnknownReactionType(r.reaction_type, i, r.equation)

        # -----------------------------------------------------------------------------
        # Rates of progress
        # -----------------------------------------------------------------------------
        outString = []
        for j, s in enumerate(nu_f[:, i]):
            if s == 1.0:
                outString.append(f" * cs[{j}]")
            elif s > 0.0:
                outString.append(f" * pow(cs[{j}],{float(s)})")
        # cTBC has already been applied to falloffs above!!!
        pgMech.write("  q_f =  k_f")
        for item in outString:
            pgMech.write(item)
        pgMech.write(";\n")

        outString = []
        for j, s in enumerate(nu_b[:, i]):
            if s == 1:
                outString.append(f" * cs[{j}]")
            elif s > 0.0:
                outString.append(f" * pow(cs[{j}],{float(s)})")
        # cTBC has already been applied to falloffs above!!!
        pgMech.write("  q_b = -k_f/K_c")
        for item in outString:
            pgMech.write(item)
        pgMech.write(";\n")
        if r.reversible:
            pgMech.write(f"  q[{i}] = q_f + q_b;\n\n")
        else:
            pgMech.write(f"  q[{i}] = q_f;\n\n")

    pgMech.write("}\n")
    pgMech.write("// ends scope of logT and prefRuT\n")
    # -------------------------------------------------------------------
    # SOURCE TERMS
    # -------------------------------------------------------------------
    outString = (
        "  // ----------------------------------------------------------- >\n"
        "  // Source terms. --------------------------------------------- >\n"
        "  // ----------------------------------------------------------- >\n"
        "\n"
    )
    pgMech.write(outString)

    for i in range(gas.n_species):
        outString = []
        nu_sum = nu_b[i, :] - nu_f[i, :]
        for j, s in enumerate(nu_sum):
            if s == 1:
                outString.append(f" +q[{j}]")
            elif s == -1:
                outString.append(f" -q[{j}]")
            elif s != 0:
                outString.append(f" {s:+}*q[{j}]")

        if len(outString) == 0:
            pgMech.write(f"  dYdt[{i}] = th.MW({i}) * (0.0")
        else:
            outString[0] = outString[0].replace("+", "")
            pgMech.write(f"  dYdt[{i}] = th.MW({i}) * (")
            for item in outString:
                pgMech.write(item)
        pgMech.write(");\n")

    outString = (
        "\n"
        "  dTdt = 0.0;\n"
        f"  for (int n=0; n<{ns}; n++)\n"
        "  {\n"
        "    dTdt -= hi[n] * dYdt[n];\n"
        "    Y[n] += dYdt[n]/rho*tSub;\n"
        "  }\n"
        "  dTdt /= cp * rho;\n"
        "  T += dTdt*tSub;\n"
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
    pgMech.write(outString)

    # END
    outString = "  });\n" "}"
    pgMech.write(outString)
    pgMech.close()


if __name__ == "__main__":
    import argparse

    if float(ct.__version__[0:3]) < 3.0:
        raise ImportError("Cantera version > 3.0 required.")

    parser = argparse.ArgumentParser(
        description="""Convert a cantera .yaml file to hard coded finite rate
        chemical source term c++ source code used by PEREGRINE"""
    )
    parser.add_argument(
        "ctFileName",
        metavar="<ct_file>",
        help="""Cantera .yaml file to convert into hard coded PEREGRINE
        chemical source term.""",
        type=str,
    )
    args = parser.parse_args()

    ctFileName = args.ctFileName

    cppFileName = f'chem_{ctFileName.replace(".yaml",".cpp")}'.replace("-", "_")

    ct2pgChem(ctFileName, cppFileName)

    proc = subprocess.run(["clang-format", "-i", cppFileName])
    if proc.returncode != 0:
        print(proc.returncode)
