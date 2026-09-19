#include "chemistry/rates.hpp"
#include "kernel.hpp"
#include <limits>

// The species source over the step by substeps sized on the spot: each
// takes the state as far as the fastest species allows before it leaves
// [0, 1], up to PG_CHEMISTRY_MAX_SUBSTEPS of them, the temperature
// following at constant density; the source is the mean rate over the
// step, which lands the state the substeps reached; dQ is begun with it.
// PyFR's finite-rate auto scheme.
#ifndef PG_CHEMISTRY_MAX_SUBSTEPS
#error "the substepped chemistry takes chemistryMaxSubSteps from the config"
#endif

PG_RANGE(cellCenters)
struct finiteRateSubstep {
  cellVecIn Q, q;
  cellVecOut dQ;
  caseIn dt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    constexpr fpdtype eps = std::numeric_limits<fpdtype>::epsilon();
    constexpr fpdtype tiny = std::numeric_limits<fpdtype>::min();
    const fpdtype rho = Q(0), rhoinv = 1.0 / rho, step = dt();
    fpdtype Y[ns], src[ns], rate[ns];
    chemistry::fractionsOf(Q, rhoinv, Y);
    fpdtype T = q(1);
    for (int n = 0; n < ns; n++)
      src[n] = 0.0;
    fpdtype remaining = 1.0;
    for (int sub = 0; sub < PG_CHEMISTRY_MAX_SUBSTEPS && remaining > eps;
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
      const fpdtype tSub = share * step;
      // the temperature at constant density: cp and each species' enthalpy
      fpdtype cp = 0.0, dTdt = 0.0;
      for (int n = 0; n < ns; n++) {
        fpdtype cpR = 0.0;
        for (int m = cpPolyTerms - 1; m >= 0; m--)
          cpR = cpR * s.logT + cpPoly(n, m);
        cp += cpR * Ru * MWinv(n) * Y[n];
        fpdtype hRT = 0.0;
        for (int m = hPolyTerms - 1; m >= 0; m--)
          hRT = hRT * s.logT + hPoly(n, m);
        dTdt -= (hRT + hRef(n) * s.Tinv) * T * Ru * MWinv(n) * rate[n];
        Y[n] += rate[n] * rhoinv * tSub;
        src[n] += rate[n] * share;
      }
      T += dTdt / (cp * rho) * tSub;
      remaining -= share;
    }
    chemistry::beginWithSource(dQ, src);
  }
};

PG_ABI void pgFiniteRateSubstep(const finiteRateSubstep &k, const pgTiling &t) {
  forCells("finite rate chemistry, substepped", t, k);
}
