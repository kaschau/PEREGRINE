// Diffusion at each species' Lewis number: D = kappa / (rho cp Le).
#ifndef __lewis_H__
#define __lewis_H__

#include "transport/mixing.hpp"

namespace lewis {

KOKKOS_INLINE_FUNCTION void coefficients(const mixtureState &s, double *D) {
  const double kappaOverRhoCp = s.kappa * s.rhoinv / s.cp;
  for (int n = 0; n <= ns - 1; n++) {
    D[n] = kappaOverRhoCp / lewisNumber(n);
  }
}

} // namespace lewis

#endif
