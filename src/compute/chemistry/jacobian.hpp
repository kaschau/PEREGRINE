// The species source's Jacobian composed from the case's reactions: the
// derivatives of the log-space rates of rates.hpp, one reaction at a time
// over the pieces its rate of progress is made of, folded as the rates
// are. A rung names the entries evaluated; every one goes to a sink, none
// is stored. What the pseudo system of dual time is preconditioned with.
#ifndef __jacobian_H__
#define __jacobian_H__

#include "chemistry/rates.hpp"

namespace chemistry {

// the entries of d omega / d q a rung evaluates: each carried species'
// own diagonal entry and its temperature entry, both at fixed density
enum rung { diagonal };

// d(g / (Ru T)) / dT of a species at u = ln T: the refit polynomial's
// derivative over T, less the reference enthalpy over T^2
KOKKOS_INLINE_FUNCTION fpdtype dGibbsdT(const int n, const fpdtype u,
                                        const fpdtype Tinv) {
  fpdtype d = 0.0;
  for (int m = gPolyTerms - 1; m >= 1; m--)
    d = d * u + m * gPoly(n, m);
  return d * Tinv - hRef(n) * Tinv * Tinv;
}

// d ln k / d ln cTBC of a reaction with a third body: all of it for a
// three-body reaction, the falloff's 1 / (1 + Pr) for a pressure-dependent
// one, and Troe's broadening's share on top of that
template <int R>
KOKKOS_INLINE_FUNCTION fpdtype thirdBodyShare(const rateOfProgress &pr) {
  if constexpr (type(R) == threeBody) {
    return 1.0;
  } else {
    fpdtype share = 1.0 / (1.0 + exp(pr.logPr));
    if constexpr (type(R) == troe) {
      const fpdtype g = pr.N - 0.14 * pr.A, h = 1.0 + pr.f1 * pr.f1;
      share -= 2.0 * pr.lfc * pr.f1 * pr.N / (h * h * g * g);
    }
    return share;
  }
}

// a species' third-body efficiency in a reaction: the default, and its
// deviation from it if it has one
template <int R> KOKKOS_INLINE_FUNCTION fpdtype efficiency(const int m) {
  fpdtype e = defaultEfficiency(R);
  for (int k = 0; k < effSpeciesTerms; k++)
    e += effSpecies(R, k) == m ? effDeviation(R, k) : 0.0;
  return e;
}

// d rp / d c_m of one reaction from its rate of progress: the forward
// order's term and the reverse order's, each the direction's rate over c_m
// in log form so it is finite at the concentration floor and under the
// clamp, and the third body's, what a unit of its concentration passes to
// the rate times the species' efficiency
template <int R>
KOKKOS_INLINE_FUNCTION fpdtype dProgress(const rateState &s,
                                         const rateOfProgress &pr,
                                         const fpdtype throughThirdBody,
                                         const int m) {
  fpdtype d = 0.0;
  if constexpr (type(R) != elementary)
    d = efficiency<R>(m) * throughThirdBody;
  for (int k = 0; k < fwdSpeciesTerms; k++)
    if (fwdSpecies(R, k) == m)
      d += fwdExponent(R, k) * exp(fmin(pr.lrp - s.logc[m], s.expClamp));
  if constexpr (reversible(R)) {
    for (int k = 0; k < revSpeciesTerms; k++)
      if (revSpecies(R, k) == m)
        d -= revExponent(R, k) *
             exp(fmin(pr.lrp - pr.diff - s.logc[m], s.expClamp));
  }
  return d;
}

// d log10 Fcent / dT of a Troe reaction: its terms' weights over their
// sum; a floored Fcent is constant
template <int R>
KOKKOS_INLINE_FUNCTION fpdtype dLog10FcentdT(const rateState &s) {
  constexpr fpdtype log10e = 0.4342944819032518;
  constexpr int terms = fcentTerms(R);
  if constexpr (terms == 0) {
    return 0.0;
  } else {
    fpdtype e[terms], ref = -std::numeric_limits<fpdtype>::max();
    for (int k = 0; k < terms; k++) {
      e[k] = fcentC0(R, k) + fcentCT(R, k) * s.T + fcentCTinv(R, k) * s.Tinv;
      ref = fmax(ref, e[k]);
    }
    fpdtype sum = 0.0, dsum = 0.0;
    for (int k = 0; k < terms; k++) {
      const fpdtype w = fcentSign(R, k) * exp(e[k] - ref);
      sum += w;
      dsum += w * (fcentCT(R, k) - fcentCTinv(R, k) * s.Tinv * s.Tinv);
    }
    return sum > std::numeric_limits<fpdtype>::min() ? log10e * dsum / sum
                                                     : 0.0;
  }
}

// d ln F / d log10 Fcent of Troe's broadening: the centering moves the
// constants C and N as well as the exponent
KOKKOS_INLINE_FUNCTION fpdtype dLogFdlfc(const rateOfProgress &pr) {
  constexpr fpdtype ln10 = 2.302585092994046, dC = -0.67, dN = -1.27;
  const fpdtype g = pr.N - 0.14 * pr.A, h = 1.0 + pr.f1 * pr.f1;
  const fpdtype df1 = (dC * g - pr.A * (dN - 0.14 * dC)) / (g * g);
  return ln10 * (1.0 / h - 2.0 * pr.lfc * pr.f1 * df1 / (h * h));
}

// d rp / dT of one reaction at fixed concentrations: the rate constant's
// Arrhenius sensitivity, a falloff's through its pressure ratio and Troe's
// centering, which the net rate carries; and a reversible reaction's
// affinity through the Gibbs energies and the pressure reference, which
// the reverse rate alone carries
template <int R>
KOKKOS_INLINE_FUNCTION fpdtype dProgressdT(const rateState &s,
                                           const rateOfProgress &pr,
                                           const fpdtype *dgbs) {
  fpdtype dlk = (b(R) + EaR(R) * s.Tinv) * s.Tinv;
  if constexpr (type(R) == lindemann || type(R) == troe) {
    dlk += thirdBodyShare<R>(pr) * (prB(R) + prEaR(R) * s.Tinv) * s.Tinv;
    if constexpr (type(R) == troe)
      dlk += dLogFdlfc(pr) * dLog10FcentdT<R>(s);
  }
  if constexpr (reversible(R)) {
    fpdtype ddiff = -nuTotal(R) * s.Tinv;
    for (int k = 0; k < netSpeciesTerms; k++)
      ddiff -= netNu(R, k) * dgbs[netSpecies(R, k)];
    return pr.rp * dlk + exp(fmin(pr.lrp - pr.diff, s.expClamp)) * ddiff;
  } else {
    return pr.rp * dlk;
  }
}

// one reaction's entries: for every carried species it makes or takes,
// its own diagonal entry, d omega_j / d Y_j at fixed density and
// temperature with the last species taking up the change, and its
// temperature entry, d omega_j / dT at fixed density and composition
template <int R, rung G, class Add>
KOKKOS_INLINE_FUNCTION void
reactionJacobian(const rateState &s, const fpdtype rho, const fpdtype *dgbs,
                 const Add &add) {
  static_assert(G == diagonal, "the diagonal is the rung there is");
  const rateOfProgress pr = progress<R>(s);
  fpdtype throughThirdBody = 0.0;
  if constexpr (type(R) != elementary)
    throughThirdBody = thirdBodyShare<R>(pr) * pr.rp /
                       fmax(std::numeric_limits<fpdtype>::min(), pr.cTBC);
  const fpdtype last =
      MWinv(ns - 1) * dProgress<R>(s, pr, throughThirdBody, ns - 1);
  const fpdtype byT = dProgressdT<R>(s, pr, dgbs);
  for (int k = 0; k < netSpeciesTerms; k++) {
    const int j = netSpecies(R, k);
    if (netNu(R, k) == 0.0 || j == ns - 1)
      continue;
    const fpdtype makes = netNuMW(R, k);
    add(j, 5 + j,
        makes * rho *
            (MWinv(j) * dProgress<R>(s, pr, throughThirdBody, j) - last));
    add(j, 4, makes * byT);
  }
}

// the reactions, one each
template <rung G, class Add, int... R>
KOKKOS_INLINE_FUNCTION void
reactionJacobians(const rateState &s, const fpdtype rho, const fpdtype *dgbs,
                  const Add &add, std::integer_sequence<int, R...>) {
  (reactionJacobian<R, G>(s, rho, dgbs, add), ...);
}

// the entries of the source's Jacobian a rung evaluates from a cell's
// conserved state and temperature, each to add(j, col, value): the row a
// carried species j, the column the primitive's, 4 for the temperature
// and 5 + k for species k, the value d omega_j / d(that primitive) at
// fixed density and the rest, the last species taking up a change of
// composition
template <rung G, class Conserved, class State, class Add>
KOKKOS_INLINE_FUNCTION void jacobianOf(const Conserved &Q, const State &q,
                                       const Add &add) {
  const fpdtype rho = Q(0);
  fpdtype Y[ns], dgbs[ns];
  fractionsOf(Q, 1.0 / rho, Y);
  const auto s = stateOf(rho, [&](const int n) { return Y[n]; }, q(1));
  for (int n = 0; n < ns; n++)
    dgbs[n] = dGibbsdT(n, s.logT, s.Tinv);
  reactionJacobians<G>(s, rho, dgbs, add,
                       std::make_integer_sequence<int, nr>{});
}

} // namespace chemistry

#endif
