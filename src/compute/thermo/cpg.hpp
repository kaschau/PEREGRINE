// A calorically perfect gas: every species a constant cp.
#ifndef __cpg_H__
#define __cpg_H__

#include "kernel.hpp"
#include "species.hpp"

namespace cpg {

// the mixture state at one point, as the eos works it out
struct state {
  double p, T, rho, e, h, cp, gamma, c;
};

// what a cell keeps of the eos beyond p and T: gamma, cp, rho h, c, rho e;
// a species enthalpy is T cp0, so none are kept
constexpr int qhComponents = 5;

// what any (p, T, Y) or (rho, e, Y) shares: the mixture gas constant and cp
// from the mass fractions Y(n), n over every species
template <class Yf>
KOKKOS_INLINE_FUNCTION void mixture(const Yf &Y, double &Rmix, double &cp) {
  Rmix = 0.0;
  cp = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    Rmix += Y(n) * MWinv(n);
    cp += Y(n) * cp0(n);
  }
  Rmix *= Ru;
}

// each species' enthalpy at T, as a callable of the species; the cell's qh
// is what a cubic eos reads its own from
template <class Qh>
KOKKOS_INLINE_FUNCTION auto enthalpies(const double T, const Qh &) {
  return [=](const int n) { return T * cp0(n); };
}

// the state from pressure, temperature and mass fractions; hi(n, value)
// receives each species' enthalpy an eos keeps in qh, which this one does not
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromPrims(const double p, const double T,
                                       const Yf &Y, const Hf &) {
  state s;
  double Rmix, cp;
  mixture(Y, Rmix, cp);
  s.p = p;
  s.T = T;
  s.h = cp * T;
  s.cp = cp;
  s.gamma = cp / (cp - Rmix);
  s.c = sqrt(s.gamma * Rmix * T);
  s.rho = p / (Rmix * T);
  s.e = s.h - Rmix * T;
  return s;
}

// the state from density, internal energy and mass fractions; the guess is
// what an iterative eos starts its temperature from
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromCons(const double rho, const double e,
                                      const Yf &Y, const double, const Hf &) {
  state s;
  double Rmix, cp;
  mixture(Y, Rmix, cp);
  s.rho = rho;
  s.e = e;
  s.T = e / (cp - Rmix);
  s.p = rho * Rmix * s.T;
  s.h = e + Rmix * s.T;
  s.cp = cp;
  s.gamma = cp / (cp - Rmix);
  s.c = sqrt(s.gamma * Rmix * s.T);
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

} // namespace cpg

#endif
