// A real gas on a cubic equation of state (Peng-Robinson), the ideal parts of
// cp and h from the thermally perfect polynomials plus the departures.
#ifndef __realGas_H__
#define __realGas_H__

#include "conserved.hpp"
#include "cubic.hpp"
#include "kernel.hpp"
#include "species.hpp"

namespace realGas {

// the mixture state at one point, as the eos works it out
struct state {
  double p, T, rho, e, h, cp, gamma, c;
};

// what a cell keeps of the eos beyond p and T: gamma, cp, rho h, c, rho e,
// then each species' enthalpy, departure included, which no polynomial in T
// gives back
constexpr int qhComponents = 5 + ns;

// each species' enthalpy, as a callable of the species: read out of the
// cell's qh, where fromPrims and fromCons left them
template <class Qh>
KOKKOS_INLINE_FUNCTION auto enthalpies(const double, const Qh &qh) {
  return [=](const int n) { return qh(5 + n); };
}

// -----------------------------------------------------------------------//
// Peng-Robinson
constexpr double uRG = 2.0, wRG = -1.0, biConst = 0.077796, aiConst = 0.457240,
                 fw0 = 0.37464, fw1 = 1.54226, fw2 = -0.26992;
// -----------------------------------------------------------------------//
// Soave-Redlich-Kwong
// constexpr double uRG=1.0, wRG= 0.0, biConst=0.0866403,
// aiConst=0.4274802, fw0=0.480  , fw1=1.574  , fw2=-0.176  ;
// -----------------------------------------------------------------------//

// the cubic at T for the mixture X: its mixing coefficients and their
// temperature derivative, and, given p, the compressibility Z and dZ/dT
struct cubic {
  double am, bm, dam, Astar, Bstar, Z, z1, z2, dZdT;
};

KOKKOS_INLINE_FUNCTION void coefficients(const double T, const double *X,
                                         double *ai, cubic &c) {
  c.am = 0.0;
  c.bm = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    double Tr = T / Tcrit(n);
    double fOmega = fw0 + fw1 * acentric(n) + fw2 * pow(acentric(n), 2.0);
    double alpha = pow(1.0 + fOmega * (1 - sqrt(Tr)), 2.0);
    ai[n] = aiConst * (pow(Ru * Tcrit(n), 2.0) * alpha) / pcrit(n);
    double bi = biConst * (Ru * Tcrit(n)) / pcrit(n);
    c.bm += X[n] * bi;
  }
  for (int n = 0; n <= ns - 1; n++) {
    for (int n2 = 0; n2 <= ns - 1; n2++) {
      c.am += X[n] * X[n2] *
              sqrt(ai[n] * ai[n2]); // - (1 - kij)  <- For now we ignore binary
                                    // interaciton coeff, i.e. assume kij=1
    }
  }
}

KOKKOS_INLINE_FUNCTION void compressibility(const double p, const double T,
                                            cubic &c) {
  c.Astar = c.am * p / pow(Ru * T, 2.0);
  c.Bstar = c.bm * p / (Ru * T);

  // Solve cubic EOS for Z
  // https://www.e-education.psu.edu/png520/m11_p6.html
  double z0;
  double Bstar2 = pow(c.Bstar, 2.0);
  z0 = -(c.Astar * c.Bstar + wRG * Bstar2 + wRG * Bstar2 * c.Bstar);
  c.z1 = c.Astar + wRG * Bstar2 - uRG * c.Bstar - uRG * Bstar2;
  c.z2 = -(1.0 + c.Bstar - uRG * c.Bstar);

  double cardanoQ, RR, M;
  cardanoQ = (pow(c.z2, 2.0) - 3.0 * c.z1) / 9.0;
  RR = (2.0 * pow(c.z2, 3.0) - 9.0 * c.z2 * c.z1 + 27.0 * z0) / 54.0;
  M = pow(RR, 2.0) - pow(cardanoQ, 3.0);

  double z2o3 = c.z2 / 3.0;
  if (M > 0.0) {
    double S = -RR / abs(RR) * pow(abs(RR) + sqrt(M), (1.0 / 3.0));
    c.Z = S + cardanoQ / S - z2o3;
  } else {
    double q1p5 = pow(cardanoQ, 1.5);
    double sqQ = sqrt(cardanoQ);
    double theta = acos(RR / q1p5);
    double x1 = -(2.0 * sqQ * cos(theta / 3.0)) - z2o3;
    double x2 =
        -(2.0 * sqQ * cos((theta + 2 * 3.14159265358979323846) / 3.0)) - z2o3;
    double x3 =
        -(2.0 * sqQ * cos((theta - 2 * 3.14159265358979323846) / 3.0)) - z2o3;

    c.Z = stableRoot(x1, x2, x3, c.Astar, c.Bstar, uRG, wRG);
  }
}

// the temperature derivative of the mixing coefficient and of Z, and the
// departures of h and cp; the ideal parts are added by the caller
KOKKOS_INLINE_FUNCTION void departures(const double T, const double *X,
                                       const double *ai, cubic &c, double &hDep,
                                       double &cpDep) {
  c.dam = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    for (int n2 = 0; n2 <= ns - 1; n2++) {
      double fOmegaN = fw0 + fw1 * acentric(n) + fw2 * pow(acentric(n), 2.0);
      double fOmegaN2 = fw0 + fw1 * acentric(n2) + fw2 * pow(acentric(n2), 2.0);
      c.dam += X[n2] * X[n] * 1.0 *
               (fOmegaN2 * sqrt(ai[n] * Tcrit(n2) / pcrit(n2)) +
                fOmegaN * sqrt(ai[n2] * Tcrit(n) / pcrit(n)));
    }
  }
  c.dam *= -0.5 * Ru * sqrt(aiConst / T);
  double dAstardT = -2.0 * (c.Astar / T) * (1.0 - 0.5 * (T / c.am) * c.dam);
  double dBstardT = -c.Bstar / T;

  double dz0 =
      -(c.Bstar * dAstardT +
        (c.Astar + (2.0 * c.Bstar + 3.0 * pow(c.Bstar, 2.0)) * wRG) * dBstardT);
  double dz1 = (dAstardT + (2.0 * c.Bstar * (wRG - uRG) - uRG) * dBstardT);
  double dz2 = -(1.0 - uRG) * dBstardT;

  c.dZdT = -(dz2 * pow(c.Z, 2.0) + dz1 * c.Z + dz0) /
           (3.0 * pow(c.Z, 2.0) + 2.0 * c.Z * c.z2 + c.z1);

  double Cuw = 1.0 / (c.bm * Ru * sqrt(pow(uRG, 2.0) - 4.0 * wRG));
  double ZoB = c.Z / c.Bstar;
  double logZoB = log((2.0 * ZoB + (uRG - sqrt(pow(uRG, 2.0) - 4.0 * wRG))) /
                      (2.0 * ZoB + (uRG + sqrt(pow(uRG, 2.0) - 4.0 * wRG))));

  cpDep =
      Cuw * (c.am / T - c.dam) * logZoB * 0.5 * T / c.am * c.dam +
      pow((pow(ZoB, 2.0) + uRG * ZoB + wRG) -
              (c.dam / (c.bm * Ru)) * (ZoB - 1.0),
          2.0) /
          (pow(pow(ZoB, 2.0) + uRG * ZoB + wRG, 2.0) -
           (c.am / (c.bm * Ru * T)) * (2.0 * ZoB + uRG) * pow(ZoB - 1.0, 2.0)) -
      1.0;

  hDep = Cuw * (c.am / T - c.dam) * logZoB + (c.Z - 1.0);
}

// the mixture h and cp at T, departures first, each species' enthalpy with
// its departure into hi[]
template <class Yf>
KOKKOS_INLINE_FUNCTION void
properties(const double T, const Yf &Y, const double MWmix, const double hDep,
           const double cpDep, double &h, double &cp, double *hi) {
  h = Ru * T * hDep / MWmix;
  cp = Ru * cpDep / MWmix;
  const double u = log(T), Tinv = 1.0 / T;
  for (int n = 0; n <= ns - 1; n++) {
    // ideal cp/R and h/(RT), Horner in u
    double cpR = 0.0, hRT = 0.0;
    for (int m = cpPolyTerms(n) - 1; m >= 0; m--)
      cpR = cpR * u + cpPoly(n, m);
    for (int m = hPolyTerms(n) - 1; m >= 0; m--)
      hRT = hRT * u + hPoly(n, m);
    hRT += hRef(n) * Tinv;
    const double Rn = Ru * MWinv(n);
    hi[n] = hRT * T * Rn;
    cp += cpR * Rn * Y(n);
    h += hi[n] * Y(n);
    // Add departure to individual hi
    hi[n] += Ru * T * hDep * MWinv(n);
  }
}

// gamma and the speed of sound from Z's pressure derivative
KOKKOS_INLINE_FUNCTION void ratios(const double p, const double T,
                                   const double rho, const double cp,
                                   const cubic &c, double &gamma, double &cs) {
  double dAstardp = c.Astar / p;
  double dBstardp = c.Bstar / p;

  double dz0 =
      -(c.Bstar * dAstardp +
        (c.Astar + (2.0 * c.Bstar + 3.0 * pow(c.Bstar, 2.0)) * wRG) * dBstardp);
  double dz1 = (dAstardp + (2.0 * c.Bstar * (wRG - uRG) - uRG) * dBstardp);
  double dz2 = -(1.0 - uRG) * dBstardp;

  double dZdp = -(dz2 * pow(c.Z, 2.0) + dz1 * c.Z + dz0) /
                (3.0 * pow(c.Z, 2.0) + 2.0 * c.Z * c.z2 + c.z1);

  double drhodp = (rho / p) * (1.e0 - p * dZdp / c.Z);
  double drhodt = -(rho / T) * (1.e0 + T * c.dZdT / c.Z);
  gamma = drhodp / (drhodp - (T / cp) * pow(drhodt / rho, 2.0));
  cs = sqrt(abs(gamma / drhodp));
}

// the state from pressure, temperature and mass fractions; hi(n, value)
// receives each species' enthalpy, departure included
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromPrims(const double p, const double T,
                                       const Yf &Y, const Hf &hi) {
  state s;
  double X[ns], ai[ns], his[ns];
  const double MWmix = moleFractions(Y, X);
  const double Rmix = Ru / MWmix;
  cubic c;
  coefficients(T, X, ai, c);
  compressibility(p, T, c);
  double hDep, cpDep;
  departures(T, X, ai, c, hDep, cpDep);
  properties(T, Y, MWmix, hDep, cpDep, s.h, s.cp, his);
  s.p = p;
  s.T = T;
  s.rho = p / (c.Z * Rmix * T);
  ratios(p, T, s.rho, s.cp, c, s.gamma, s.c);
  s.e = s.h - p / s.rho;
  for (int n = 0; n <= ns - 1; n++) {
    hi(n, his[n]);
  }
  return s;
}

// the state from density, internal energy and mass fractions: Newton on T
// from the guess, the cubic solved for p at each T
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromCons(const double rho, const double e,
                                      const Yf &Y, const double Tguess,
                                      const Hf &hi) {
  state s;
  double X[ns], ai[ns], his[ns];
  const double MWmix = moleFractions(Y, X);
  const double Rmix = Ru / MWmix;
  // molar volume
  double Vm = MWmix / rho;
  int nitr = 0, maxitr = 100;
  double tol = 1e-8;
  double error = 1e100;
  cubic c;
  double p = 0.0;
  double T = (Tguess < 1.0) ? 300.0 : Tguess;
  while ((abs(error) > tol) && (nitr < maxitr)) {
    // With a T, we can compute p
    coefficients(T, X, ai, c);
    // PR
    double Cc = c.bm;
    // SRK
    // double Cc = 0.0;
    p = Ru * T / (Vm - c.bm) - c.am / (Vm * (Vm + c.bm) + Cc * (Vm - c.bm));
    compressibility(p, T, c);
    double hDep, cpDep;
    departures(T, X, ai, c, hDep, cpDep);
    properties(T, Y, MWmix, hDep, cpDep, s.h, s.cp, his);
    error = e - (s.h - c.Z * Rmix * T);
    T = T - error / (-s.cp + Rmix * (c.Z * c.dZdT));
    nitr += 1;
  }
  s.rho = rho;
  s.e = e;
  s.p = p;
  s.T = T;
  ratios(p, T, rho, s.cp, c, s.gamma, s.c);
  for (int n = 0; n <= ns - 1; n++) {
    hi(n, his[n]);
  }
  return s;
}

// the density's derivatives at (p, T, Y) for the dual time preconditioning:
// not yet; the solver refuses dual time on a real gas until the cubic's
// derivatives are written here
template <class Yf>
KOKKOS_INLINE_FUNCTION void densityDerivatives(const double, const double,
                                               const double, const Yf &,
                                               double &, double &, double *) {
  Kokkos::abort("realGas: the dual time density derivatives are not written");
}

} // namespace realGas

#endif
