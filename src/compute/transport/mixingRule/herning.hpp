// Herning and Zipperer's rule: mu = sum_n X_n mu_n sqrt(MW_n) / sum_n X_n
// sqrt(MW_n), one pass over the species. Within a few percent of Wilke for
// mixtures of like molecules; the light species (H2, H) are where it
// strays. sqrtMu[n] is sqrt(mu_n), what the viscosity fits give up to a
// factor of T^(1/4).
#ifndef __herning_H__
#define __herning_H__

#include "transport/mixing.hpp"

namespace herning {

KOKKOS_INLINE_FUNCTION double viscosity(const double *X, const double *sqrtMu) {
  double num = 0.0, den = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    const double w = X[n] * sqrtMW(n);
    num += w * sqrtMu[n] * sqrtMu[n];
    den += w;
  }
  return num / den;
}

} // namespace herning

#endif
