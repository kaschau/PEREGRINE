#include "chemistry/rates.hpp"
#include "kernel.hpp"
#include <limits>

// The species source over the step by substeps sized on the spot: each
// takes the state as far as the fastest species allows before it leaves
// [0, 1], and no further than the mixture's entropy keeps rising along it,
// up to PG_CHEMISTRY_MAX_SUBSTEPS of them, the temperature following at
// constant volume; the source is the mean rate over the step, which lands
// the state the substeps reached; dQ is begun with it. PyFR's finite-rate
// auto scheme with the entropy cap of Schau, Witherden and Jameson ("A
// Limiter in Time"): along a substep's ray the entropy is concave, so it
// has one maximizer, located by PG_CHEMISTRY_ENTROPY_BISECTIONS bisections
// (none: no cap); a substep past it is what pins a fast species to its
// bound and burns substeps.
#ifndef PG_CHEMISTRY_MAX_SUBSTEPS
#error "the substepped chemistry takes chemistryMaxSubSteps from the config"
#endif
#ifndef PG_CHEMISTRY_ENTROPY_BISECTIONS
#error                                                                         \
    "the substepped chemistry takes chemistryEntropyBisections from the config"
#endif

namespace chemistry {

// the mixture's entropy production along a move (dTdt, dYdt) from (T, Y)
// at fixed density: grad s . the move, s the entropy per mass at the
// actual partial pressures; positive along the rates (the second law),
// negative once a substep's ray has passed the entropy's maximizer
KOKKOS_INLINE_FUNCTION fpdtype entropyRate(const fpdtype rho, const fpdtype T,
                                           const fpdtype *Y, const fpdtype dTdt,
                                           const fpdtype *dYdt) {
  constexpr fpdtype tiny = std::numeric_limits<fpdtype>::min();
  const fpdtype logT = log(T), Tinv = 1.0 / T;
  const fpdtype logRuTPref = logT - logPrefRu;
  fpdtype cv = 0.0, dS = 0.0;
  for (int n = 0; n < ns; n++) {
    fpdtype cpR = 0.0, hRT = 0.0;
    for (int m = cpPolyTerms - 1; m >= 0; m--)
      cpR = cpR * logT + cpPoly(n, m);
    for (int m = hPolyTerms - 1; m >= 0; m--)
      hRT = hRT * logT + hPoly(n, m);
    // s / R at the species' partial pressure, less the mixing term's own
    // derivative (d ln c / dY times Y is one)
    const fpdtype sR = hRT + hRef(n) * Tinv - gibbs(n, logT, Tinv) -
                       log(rho * fmax(Y[n], tiny) * MWinv(n)) - logRuTPref -
                       1.0;
    cv += Y[n] * (cpR - 1.0) * Ru * MWinv(n);
    dS += Ru * MWinv(n) * sR * dYdt[n];
  }
  return dS + cv * Tinv * dTdt;
}

// the same at the point a time t along the ray from (T, Y)
KOKKOS_INLINE_FUNCTION fpdtype
entropyRateAlong(const fpdtype rho, const fpdtype T, const fpdtype *Y,
                 const fpdtype dTdt, const fpdtype *dYdt, const fpdtype t) {
  fpdtype Yt[ns];
  for (int n = 0; n < ns; n++)
    Yt[n] = Y[n] + t * dYdt[n];
  return entropyRate(rho, T + t * dTdt, Yt, dTdt, dYdt);
}

} // namespace chemistry

PG_RANGE(cellCenters)
struct finiteRateSubstep {
  cellVecIn Q, q;
  cellVecOut dQ;
  caseIn dt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    constexpr fpdtype eps = std::numeric_limits<fpdtype>::epsilon();
    constexpr fpdtype tiny = std::numeric_limits<fpdtype>::min();
    constexpr int bisections = PG_CHEMISTRY_ENTROPY_BISECTIONS;
    // a substep below this share of the step is none: the rates point out
    // of what the entropy allows, the chemistry is done for the step
    constexpr fpdtype stepFrac = 1e-12;
    const fpdtype rho = Q(0), rhoinv = 1.0 / rho, step = dt();
    fpdtype Y[ns], src[ns], rate[ns], dYdt[ns];
    chemistry::fractionsOf(Q, rhoinv, Y);
    fpdtype T = q(1);
    for (int n = 0; n < ns; n++)
      src[n] = 0.0;
    fpdtype remaining = 1.0;
    for (int sub = 0; sub < PG_CHEMISTRY_MAX_SUBSTEPS && remaining > stepFrac;
         sub++) {
      const auto s =
          chemistry::stateOf(rho, [&](const int n) { return Y[n]; }, T);
      chemistry::netProduction(s, rate);
      // the substep, as a share of the step: the fastest species' way to
      // its bound, never past what remains
      fpdtype share = remaining;
      for (int n = 0; n < ns; n++) {
        const fpdtype sign = copysign(1.0, rate[n]);
        const fpdtype headroom =
            0.5 * ((1.0 - sign) * Y[n] + (1.0 + sign) * (1.0 - Y[n]));
        const fpdtype most =
            rho * fmax(headroom, eps) / fmax(fabs(rate[n]), tiny);
        share = fmin(share, most / step);
      }
      // the temperature's rate at constant volume: cv, and the internal
      // energy each species' rate carries, h less Ru T / MW
      fpdtype cv = 0.0, dTdt = 0.0;
      for (int n = 0; n < ns; n++) {
        fpdtype cpR = 0.0, hRT = 0.0;
        for (int m = cpPolyTerms - 1; m >= 0; m--)
          cpR = cpR * s.logT + cpPoly(n, m);
        for (int m = hPolyTerms - 1; m >= 0; m--)
          hRT = hRT * s.logT + hPoly(n, m);
        cv += (cpR - 1.0) * Ru * MWinv(n) * Y[n];
        dTdt -= (hRT + hRef(n) * s.Tinv - 1.0) * T * Ru * MWinv(n) * rate[n];
        dYdt[n] = rate[n] * rhoinv;
      }
      dTdt /= cv * rho;
      // the entropy cap: no further along the ray than the entropy rises;
      // the end of the ray first, the common case, and the sign at its
      // start, the second law's, guards only against roundoff
      if constexpr (bisections > 0) {
        fpdtype high = share * step;
        if (chemistry::entropyRateAlong(rho, T, Y, dTdt, dYdt, high) < 0.0 &&
            chemistry::entropyRateAlong(rho, T, Y, dTdt, dYdt, 0.0) > 0.0) {
          fpdtype low = 0.0;
          for (int it = 0; it < bisections; it++) {
            const fpdtype mid = 0.5 * (low + high);
            if (chemistry::entropyRateAlong(rho, T, Y, dTdt, dYdt, mid) >= 0.0)
              low = mid;
            else
              high = mid;
          }
          share = low / step;
        }
      }
      const fpdtype tSub = share * step;
      for (int n = 0; n < ns; n++) {
        Y[n] += dYdt[n] * tSub;
        src[n] += rate[n] * share;
      }
      T += dTdt * tSub;
      remaining -= share;
      if (share <= stepFrac)
        break;
    }
    chemistry::beginWithSource(dQ, src);
  }
};

PG_ABI void pgFiniteRateSubstep(const finiteRateSubstep &k, const pgTiling &t) {
  forCells("finite rate chemistry, substepped", t, k);
}
