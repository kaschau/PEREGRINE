// Wilke's rule: mu = sum_n mu_n X_n / sum_m phi_nm X_m, with
// phi_nm = (1 + sqrt(mu_n / mu_m) (MW_m / MW_n)^(1/4))^2 / sqrt(8 (1 + MW_n
// / MW_m)). With s_n = sqrt(mu_n) / MW_n^(1/4) the first factor is (1 + s_n
// / s_m)^2 and the second a baked pair constant, so the pair loop is
// multiplies and adds. sqrtMu[n] is sqrt(mu_n), what the viscosity fits
// give up to a factor of T^(1/4) that cancels in the ratio.
#ifndef __wilke_H__
#define __wilke_H__

#include "transport/mixing.hpp"

namespace wilke {

KOKKOS_INLINE_FUNCTION double viscosity(const double *X, const double *sqrtMu) {
  double sInv[ns];
  for (int n = 0; n <= ns - 1; n++) {
    sInv[n] = 1.0 / (sqrtMu[n] * MWqInv(n));
  }
  double mu = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    const double sn = sqrtMu[n] * MWqInv(n);
    double phi = 0.0;
    for (int m = 0; m <= ns - 1; m++) {
      const double r = 1.0 + sn * sInv[m];
      phi += r * r * wilkePair(n, m) * X[m];
    }
    mu += sqrtMu[n] * sqrtMu[n] * X[n] / phi;
  }
  return mu;
}

} // namespace wilke

#endif
