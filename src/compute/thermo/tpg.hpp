// A thermally perfect gas: every species' cp and h one polynomial in ln T.
#ifndef __tpg_H__
#define __tpg_H__

#include "kernel.hpp"
#include "species.hpp"

namespace tpg {

// the mixture state at one point, as the eos works it out
struct state {
  double p, T, rho, e, h, cp, gamma, c;
};

// what a cell keeps of the eos beyond p and T: gamma, cp, rho h, c, rho e;
// a species enthalpy is one polynomial in T, so none are kept
constexpr int qhComponents = 5;

// the mixture gas constant from the mass fractions Y(n), n over every species
template <class Yf> KOKKOS_INLINE_FUNCTION double gasConstant(const Yf &Y) {
  double Rmix = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    Rmix += Y(n) * MWinv(n);
  }
  return Rmix * Ru;
}

// species n's enthalpy at T, u = ln T: h/(RT), Horner in u
KOKKOS_INLINE_FUNCTION double enthalpy(const int n, const double T,
                                       const double u, const double Tinv) {
  double hRT = 0.0;
  for (int m = hPolyTerms(n) - 1; m >= 0; m--)
    hRT = hRT * u + hPoly(n, m);
  hRT += hRef(n) * Tinv;
  return hRT * T * Ru * MWinv(n);
}

// each species' enthalpy at T, as a callable of the species; the cell's qh
// is what a cubic eos reads its own from
template <class Qh>
KOKKOS_INLINE_FUNCTION auto enthalpies(const double T, const Qh &) {
  const double u = log(T), Tinv = 1.0 / T;
  return [=](const int n) { return enthalpy(n, T, u, Tinv); };
}

// the mixture enthalpy and cp at T
template <class Yf>
KOKKOS_INLINE_FUNCTION void properties(const double T, const Yf &Y, double &h,
                                       double &cp) {
  h = 0.0;
  cp = 0.0;
  const double u = log(T), Tinv = 1.0 / T;
  for (int n = 0; n <= ns - 1; n++) {
    // cp/R, Horner in u
    double cpR = 0.0;
    for (int m = cpPolyTerms(n) - 1; m >= 0; m--)
      cpR = cpR * u + cpPoly(n, m);
    cp += cpR * Ru * MWinv(n) * Y(n);
    h += enthalpy(n, T, u, Tinv) * Y(n);
  }
}

// the state from pressure, temperature and mass fractions; hi(n, value)
// receives each species' enthalpy an eos keeps in qh, which this one does not
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromPrims(const double p, const double T,
                                       const Yf &Y, const Hf &) {
  state s;
  const double Rmix = gasConstant(Y);
  properties(T, Y, s.h, s.cp);
  s.p = p;
  s.T = T;
  s.gamma = s.cp / (s.cp - Rmix);
  s.c = sqrt(s.gamma * Rmix * T);
  s.rho = p / (Rmix * T);
  s.e = s.h - Rmix * T;
  return s;
}

// the state from density, internal energy and mass fractions: Newton on T
// from the guess
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromCons(const double rho, const double e,
                                      const Yf &Y, const double Tguess,
                                      const Hf &) {
  state s;
  const double Rmix = gasConstant(Y);
  int nitr = 0, maxitr = 100;
  double tol = 1e-8;
  double error = 1e100;
  double T = (Tguess < 1.0) ? 300.0 : Tguess;
  while ((abs(error) > tol) && (nitr < maxitr)) {
    properties(T, Y, s.h, s.cp);
    error = e - (s.h - Rmix * T);
    T = T - error / (-s.cp + Rmix);
    nitr += 1;
  }
  s.rho = rho;
  s.e = e;
  s.T = T;
  s.p = rho * Rmix * T;
  s.gamma = s.cp / (s.cp - Rmix);
  s.c = sqrt(s.gamma * Rmix * T);
  return s;
}

// the density's derivatives at (p, T, Y): by p, by T, and by each of the
// first ns - 1 mass fractions with the last taking up the change; what the
// dual time preconditioning linearizes with
template <class Yf>
KOKKOS_INLINE_FUNCTION void
densityDerivatives(const double p, const double T, const double rho,
                   const Yf &Y, double &rho_p, double &rho_T, double *rho_Y) {
  double denom = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    denom += Y(n) * MWinv(n);
  }
  // the mean molecular weight: Y sums to one, so sum(X MW) is 1 / sum(Y / MW)
  const double MWmix = 1.0 / denom;
  rho_p = rho / p;
  rho_T = -rho / T;
  for (int n = 0; n < ns - 1; n++) {
    rho_Y[n] = -rho * MWmix * (MWinv(n) - MWinv(ns - 1));
  }
}

} // namespace tpg

#endif
