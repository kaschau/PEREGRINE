#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

PG_ABI void pgApplyFlux(const pgView *J_, const pgView *dQ_, const pgView *iF_,
                        const pgView *jF_, const pgView *kF_, const pgDims *d) {
  auto J = as3(*J_);
  auto dQ = as4(*dQ_);
  auto iF = as4(*iF_);
  auto jF = as4(*jF_);
  auto kF = as4(*kF_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;

  //-------------------------------------------------------------------------------------------|
  // Apply fluxes to cc range
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "Apply current fluxes to RHS", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // Add fluxes to RHS
        dQ(i, j, k, l) +=
            (iF(i, j, k, l) + jF(i, j, k, l) + kF(i, j, k, l)) / J(i, j, k);

        dQ(i, j, k, l) -=
            (iF(i + 1, j, k, l) + jF(i, j + 1, k, l) + kF(i, j, k + 1, l)) /
            J(i, j, k);
      });
}

PG_ABI void pgApplyHybridFlux(const pgView *J_, const pgView *dQ_,
                              const pgView *iF_, const pgView *jF_,
                              const pgView *kF_, const pgView *phi_,
                              const pgDims *d, double primary) {
  auto J = as3(*J_);
  auto dQ = as4(*dQ_);
  auto iF = as4(*iF_);
  auto jF = as4(*jF_);
  auto kF = as4(*kF_);
  auto phi = as4(*phi_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;

  //-------------------------------------------------------------------------------------------|
  // Apply fluxes to cc range
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "Apply hybrid fluxes to RHS", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // Compute switch on face
        double iFphi = fmax(phi(i, j, k, 0), phi(i - 1, j, k, 0));
        double iFphi1 = fmax(phi(i, j, k, 0), phi(i + 1, j, k, 0));
        double jFphi = fmax(phi(i, j, k, 1), phi(i, j - 1, k, 1));
        double jFphi1 = fmax(phi(i, j, k, 1), phi(i, j + 1, k, 1));
        double kFphi = fmax(phi(i, j, k, 2), phi(i, j, k - 1, 2));
        double kFphi1 = fmax(phi(i, j, k, 2), phi(i, j, k + 1, 2));

        double dPrimary = 2.0 * primary - 1.0;

        // Add fluxes to RHS
        // format is F_primary*(1-switch) + F_secondary*(switch)
        // so when switch == 0, we dont switch from primary
        //    when switch == 1 we completely switch
        dQ(i, j, k, l) += (iF(i, j, k, l) * (primary - iFphi * dPrimary) +
                           jF(i, j, k, l) * (primary - jFphi * dPrimary) +
                           kF(i, j, k, l) * (primary - kFphi * dPrimary)) /
                          J(i, j, k);

        dQ(i, j, k, l) -= (iF(i + 1, j, k, l) * (primary - iFphi1 * dPrimary) +
                           jF(i, j + 1, k, l) * (primary - jFphi1 * dPrimary) +
                           kF(i, j, k + 1, l) * (primary - kFphi1 * dPrimary)) /
                          J(i, j, k);
      });
}

PG_ABI void pgApplyDissipationFlux(const pgView *J_, const pgView *dQ_,
                                   const pgView *iF_, const pgView *jF_,
                                   const pgView *kF_, const pgDims *d) {
  auto J = as3(*J_);
  auto dQ = as4(*dQ_);
  auto iF = as4(*iF_);
  auto jF = as4(*jF_);
  auto kF = as4(*kF_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;

  //-------------------------------------------------------------------------------------------|
  // Apply fluxes to cc range
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "Apply dissipaiton fluxes to RHS", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // Add fluxes to RHS
        dQ(i, j, k, l) -=
            (iF(i, j, k, l) + jF(i, j, k, l) + kF(i, j, k, l)) / J(i, j, k);

        dQ(i, j, k, l) +=
            (iF(i + 1, j, k, l) + jF(i, j + 1, k, l) + kF(i, j, k + 1, l)) /
            J(i, j, k);
      });
}
