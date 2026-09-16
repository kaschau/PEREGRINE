// The mixture rules the transport kernels share, on per-species values a
// kernel has in hand: Wilke's rule for the viscosity and the mean of the
// series and parallel sums for the conductivity; and what a kernel hands
// the species diffusion piece. Nothing here keeps a species-by-species
// array: the pair rule runs over each pair once.
#ifndef __mixing_H__
#define __mixing_H__

#include "kernel.hpp"
#include "species.hpp"

// Wilke's rule: mu = sum_n mu_n X_n / sum_m phi_nm X_m, with
// phi_nm = (1 + sqrt(mu_n / mu_m) (MW_m / MW_n)^(1/4))^2 / sqrt(8 (1 + MW_n
// / MW_m)). With s_n = sqrt(mu_n) / MW_n^(1/4) the first factor is (1 + s_n
// / s_m)^2 and the second a baked pair constant, so the pair loop is
// multiplies and adds. sqrtMu[n] is sqrt(mu_n), what the viscosity fits
// give up to a factor of T^(1/4) that cancels in the ratio.
KOKKOS_INLINE_FUNCTION double wilkeViscosity(const double *X,
                                             const double *sqrtMu) {
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
      phi += r * r * wilke(n, m) * X[m];
    }
    mu += sqrtMu[n] * sqrtMu[n] * X[n] / phi;
  }
  return mu;
}

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
