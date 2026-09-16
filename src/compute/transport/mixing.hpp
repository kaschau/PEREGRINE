// What the transport kernels share, on per-species values a kernel has in
// hand: the mean of the series and parallel sums for the conductivity, and
// what a kernel hands the species diffusion piece. The viscosity mixing
// rule is a piece of its own, transport/mixingRule.hpp.
#ifndef __mixing_H__
#define __mixing_H__

#include "kernel.hpp"
#include "species.hpp"

// the mixture conductivity: the mean of the mole-weighted series and
// parallel sums of the species'
KOKKOS_INLINE_FUNCTION double mixtureConductivity(const double *X,
                                                  const double *kappaSp) {
  double series = 0.0, parallel = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    series += X[n] * kappaSp[n];
    parallel += X[n] / kappaSp[n];
  }
  return 0.5 * (series + 1.0 / parallel);
}

// what a transport kernel has in hand when the species diffusion piece
// runs: the state, the mole fractions, and the mixture it just made
struct mixtureState {
  double p, T, u, rhoinv, cp, kappa, MWmix;
  const double *X;
};

#endif
