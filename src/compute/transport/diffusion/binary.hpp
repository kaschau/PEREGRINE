// Mixture-averaged diffusion from the binary coefficients, each pair's a
// fit in ln T from the collision integrals.
#ifndef __binary_H__
#define __binary_H__

#include "transport/mixing.hpp"

namespace binary {

// the mixture-averaged diffusion coefficient of each species from the
// binary fits, Horner in u = ln T at unit pressure and scaled by T^(3/2):
// each pair's D_nm once, its reciprocal into both species' sums; D[n] is
// 1 / (p sum_m X_m / D_nm + p X_n / (MWmix - MW_n X_n) sum_m X_m MW_m /
// D_nm), the sums over m != n. A pure fluid has no diffusion coefficient:
// every D is zero.
KOKKOS_INLINE_FUNCTION void coefficients(const mixtureState &s, double *D) {
  const double p = s.p, T = s.T, u = s.u, MWmix = s.MWmix;
  const double *X = s.X;
  for (int n = 0; n <= ns - 1; n++) {
    if (X[n] == 1.0) {
      for (int m = 0; m <= ns - 1; m++) {
        D[m] = 0.0;
      }
      return;
    }
  }
  double sum1[ns], sum2[ns];
  for (int n = 0; n <= ns - 1; n++) {
    sum1[n] = 0.0;
    sum2[n] = 0.0;
  }
  const double T_3o2 = T * sqrt(T);
  for (int n = 0; n <= ns - 1; n++) {
    for (int m = n + 1; m <= ns - 1; m++) {
      double Dnm = 0.0;
      for (int k = dijTerms(n, m) - 1; k >= 0; k--)
        Dnm = Dnm * u + dij(n, m, k);
      const double Dinv = 1.0 / (Dnm * T_3o2);
      sum1[n] += X[m] * Dinv;
      sum2[n] += X[m] * MW(m) * Dinv;
      sum1[m] += X[n] * Dinv;
      sum2[m] += X[n] * MW(n) * Dinv;
    }
  }
  for (int n = 0; n <= ns - 1; n++) {
    D[n] = 1.0 / (p * sum1[n] + p * X[n] / (MWmix - MW(n) * X[n]) * sum2[n]);
  }
}

} // namespace binary

#endif
