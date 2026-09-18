// Finite-rate chemistry composed from the case's reactions: every
// reaction is one body, chosen by its baked type at compile time, over the
// state of a cell; the net production rate of every species is their
// fold, in line. In log space throughout -- ln k, ln c, the affinity -- with
// the floors and clamps that keep an absent species and a hot cell finite in
// any precision (PyFR's finite-rate kernels, ported). Nothing here loops
// over reactions at run time or branches on one's type.
#ifndef __rates_H__
#define __rates_H__

#include "chemistry/reactions.hpp"
#include "conserved.hpp"
#include "kernel.hpp"
#include "species.hpp"
#include <limits>
#include <utility>

namespace chemistry {

using namespace reactions;

// what every reaction reads of a cell, made once
struct rateState {
  fpdtype T, logT, Tinv, logPrefRuT, ctot, expClamp;
  // each species' concentration, its log floored for an absent species,
  // and its g / (Ru T)
  fpdtype c[ns], logc[ns], gbs[ns];
};

// a species' g / (Ru T) at u = ln T: one Horner over the refit h less s
// the eos works with, and the reference enthalpy over T
KOKKOS_INLINE_FUNCTION fpdtype gibbs(const int n, const fpdtype u,
                                     const fpdtype Tinv) {
  fpdtype g = 0.0;
  for (int m = gPolyTerms - 1; m >= 0; m--)
    g = g * u + gPoly(n, m);
  return g + hRef(n) * Tinv;
}

// the state of a cell of density rho, mass fractions Y(n) and temperature T
template <class Yf>
KOKKOS_INLINE_FUNCTION rateState stateOf(const fpdtype rho, const Yf &Y,
                                         const fpdtype T) {
  rateState s;
  s.T = T;
  s.logT = log(T);
  s.Tinv = 1.0 / T;
  s.logPrefRuT = logPrefRu - s.logT;
  // the largest rate of progress whose production rates stay finite
  s.expClamp =
      log(std::numeric_limits<fpdtype>::max()) - log(2.0 * maxOmegaGain);
  s.ctot = 0.0;
  for (int n = 0; n < ns; n++) {
    s.c[n] = rho * Y(n) * MWinv(n);
    s.ctot += s.c[n];
    // log of nothing, or of a negative, is the floor: fmax takes the number
    s.logc[n] = fmax(logCFloor, log(s.c[n]));
    s.gbs[n] = gibbs(n, s.logT, s.Tinv);
  }
  return s;
}

// ln of an Arrhenius rate, only the terms the reaction has
template <int R> KOKKOS_INLINE_FUNCTION fpdtype arrhenius(const rateState &s) {
  fpdtype lk = logA(R);
  if constexpr (b(R) != 0.0)
    lk += b(R) * s.logT;
  if constexpr (EaR(R) != 0.0)
    lk -= EaR(R) * s.Tinv;
  return lk;
}
template <int R>
KOKKOS_INLINE_FUNCTION fpdtype pressureRatio(const rateState &s) {
  fpdtype lk = logPrA(R);
  if constexpr (prB(R) != 0.0)
    lk += prB(R) * s.logT;
  if constexpr (prEaR(R) != 0.0)
    lk -= prEaR(R) * s.Tinv;
  return lk;
}

// the third body's concentration: the default efficiency of every species
// and the deviations from it
template <int R> KOKKOS_INLINE_FUNCTION fpdtype thirdBody(const rateState &s) {
  fpdtype cTBC = defaultEfficiency(R) * s.ctot;
  for (int k = 0; k < effSpeciesTerms; k++)
    cTBC += effDeviation(R, k) * s.c[effSpecies(R, k)];
  return cTBC;
}

// ln(Pr / (1 + Pr)) in softplus form: exact and finite for any ln Pr
KOKKOS_INLINE_FUNCTION fpdtype logFalloff(const fpdtype logPr) {
  return fmin(logPr, 0.0) - log1p(exp(-fabs(logPr)));
}

// log10 of Troe's Fcent by a signed log-sum-exp about its largest term;
// a mechanism whose terms cancel is floored as Cantera floors it
template <int R> KOKKOS_INLINE_FUNCTION fpdtype log10Fcent(const rateState &s) {
  constexpr fpdtype log10e = 0.4342944819032518;
  constexpr int terms = fcentTerms(R);
  if constexpr (terms == 0) {
    return log(std::numeric_limits<fpdtype>::min()) * log10e;
  } else {
    fpdtype e[terms], ref = -std::numeric_limits<fpdtype>::max();
    for (int k = 0; k < terms; k++) {
      e[k] = fcentC0(R, k) + fcentCT(R, k) * s.T + fcentCTinv(R, k) * s.Tinv;
      ref = fmax(ref, e[k]);
    }
    fpdtype sum = 0.0;
    for (int k = 0; k < terms; k++)
      sum += fcentSign(R, k) * exp(e[k] - ref);
    return (ref + log(fmax(std::numeric_limits<fpdtype>::min(), sum))) * log10e;
  }
}

// one reaction's net rate of progress, added to the production rates
template <int R>
KOKKOS_INLINE_FUNCTION void reaction(const rateState &s, fpdtype *omega) {
  fpdtype lk = arrhenius<R>(s);
  if constexpr (type(R) == threeBody) {
    lk += log(fmax(std::numeric_limits<fpdtype>::min(), thirdBody<R>(s)));
  }
  if constexpr (type(R) == lindemann || type(R) == troe) {
    const fpdtype logPr =
        log(fmax(std::numeric_limits<fpdtype>::min(), thirdBody<R>(s))) +
        pressureRatio<R>(s);
    lk += logFalloff(logPr);
    if constexpr (type(R) == troe) {
      constexpr fpdtype ln10 = 2.302585092994046;
      const fpdtype lfc = log10Fcent<R>(s);
      const fpdtype C = -0.4 - 0.67 * lfc, N = 0.75 - 1.27 * lfc;
      const fpdtype A = logPr / ln10 + C;
      const fpdtype f1 = A / (N - 0.14 * A);
      lk += lfc / (1.0 + f1 * f1) * ln10;
    }
  }
  // the forward rate of progress, ln
  fpdtype lrp = lk;
  for (int k = 0; k < fwdSpeciesTerms; k++)
    lrp += fwdExponent(R, k) * s.logc[fwdSpecies(R, k)];
  fpdtype rp;
  if constexpr (reversible(R)) {
    // the affinity ln(forward / reverse); the surviving direction is
    // taken, and expm1 keeps the net rate near equilibrium
    fpdtype diff = nuTotal(R) * s.logPrefRuT;
    for (int k = 0; k < netSpeciesTerms; k++)
      diff -=
          netNu(R, k) * (s.gbs[netSpecies(R, k)] + s.logc[netSpecies(R, k)]);
    rp = copysign(exp(fmin(lrp - fmin(diff, 0.0), s.expClamp)), diff) *
         (-expm1(-fabs(diff)));
  } else {
    rp = exp(fmin(lrp, s.expClamp));
  }
  for (int k = 0; k < netSpeciesTerms; k++)
    omega[netSpecies(R, k)] += netNuMW(R, k) * rp;
}

// the reactions, one each
template <int... R>
KOKKOS_INLINE_FUNCTION void reactions(const rateState &s, fpdtype *omega,
                                      std::integer_sequence<int, R...>) {
  (reaction<R>(s, omega), ...);
}

// every species' net production rate, kg / m^3 / s, from a cell's state
KOKKOS_INLINE_FUNCTION void netProduction(const rateState &s, fpdtype *omega) {
  for (int n = 0; n < ns; n++)
    omega[n] = 0.0;
  reactions(s, omega, std::make_integer_sequence<int, nr>{});
}

// the mass fractions of a cell clipped to [0, 1], the last what the rest
// leave, renormalized when they overshoot: what the state kernels do
template <class Q>
KOKKOS_INLINE_FUNCTION void fractionsOf(const Q &Qc, const fpdtype rhoinv,
                                        fpdtype *Y) {
  Y[ns - 1] = 1.0;
  fpdtype sum = 0.0;
  for (int n = 0; n < ns - 1; n++) {
    Y[n] = fmax(fmin(Qc(5 + n) * rhoinv, 1.0), 0.0);
    Y[ns - 1] -= Y[n];
    sum += Y[n];
  }
  // a sum past one is scaled back; the last species is then none
  const fpdtype scale = 1.0 / fmax(sum, 1.0);
  for (int n = 0; n < ns - 1; n++)
    Y[n] *= scale;
  Y[ns - 1] = fmax(Y[ns - 1], 0.0);
}

} // namespace chemistry

#endif
