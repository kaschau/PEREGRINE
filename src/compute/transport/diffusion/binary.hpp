// Mixture-averaged diffusion from the binary coefficients, each pair's a
// fit in ln T from the collision integrals.
#ifndef __binary_H__
#define __binary_H__

#include "transport/mixing.hpp"

namespace binary {

// each species' D from the binary fits, Horner in u = ln T at unit pressure
// and scaled by T^(3/2): D[n] = 1 / (p sum_m X_m / D_nm + p X_n / (MWmix -
// MW_n X_n) sum_m X_m MW_m / D_nm), the sums over m != n. Each pair's fit
// is evaluated twice, once from each side, so a species' two sums are
// scalars: a per-species array of sums is indexed by the inner loop, which
// puts it in memory, and that cost three times the arithmetic it saved. A
// pure fluid has no diffusion coefficient: every D is zero.
KOKKOS_INLINE_FUNCTION void coefficients(const mixtureState &s, double *D) {
  const double p = s.p, u = s.u, MWmix = s.MWmix;
  const double *X = s.X;
  for (int n = 0; n <= ns - 1; n++) {
    if (X[n] == 1.0) {
      for (int m = 0; m <= ns - 1; m++) {
        D[m] = 0.0;
      }
      return;
    }
  }
  const double T_3o2 = s.T * sqrt(s.T);
  for (int n = 0; n <= ns - 1; n++) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int m = 0; m <= ns - 1; m++) {
      if (m == n) {
        continue;
      }
      double Dnm = 0.0;
      for (int k = dijTerms - 1; k >= 0; k--)
        Dnm = Dnm * u + dij(n, m, k);
      const double Dinv = 1.0 / (Dnm * T_3o2);
      sum1 += X[m] * Dinv;
      sum2 += X[m] * MW(m) * Dinv;
    }
    D[n] = 1.0 / (p * sum1 + p * X[n] / (MWmix - MW(n) * X[n]) * sum2);
  }
}

} // namespace binary

#endif
