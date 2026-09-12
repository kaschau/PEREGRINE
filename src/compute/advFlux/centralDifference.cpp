#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const unmanaged<double ****> &Q,
                        const unmanaged<double ****> &q, const pgDims &d,
                        unmanaged<double ****> &iF,
                        const unmanaged<double ****> &iS, const int iMod,
                        const int jMod, const int kMod) {

  const int ni = d.ni, nj = d.nj, nk = d.nk;
  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});

  Kokkos::parallel_for(
      "2nd order central difference face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uR = q(i, j, k, 1);
        double uL = q(i - iMod, j - jMod, k - kMod, 1);
        double vR = q(i, j, k, 2);
        double vL = q(i - iMod, j - jMod, k - kMod, 2);
        double wR = q(i, j, k, 3);
        double wL = q(i - iMod, j - jMod, k - kMod, 3);

        double UfR =
            uR * iS(i, j, k, 0) + vR * iS(i, j, k, 1) + wR * iS(i, j, k, 2);
        double UfL =
            uL * iS(i, j, k, 0) + vL * iS(i, j, k, 1) + wL * iS(i, j, k, 2);

        double pR = q(i, j, k, 0);
        double pL = q(i - iMod, j - jMod, k - kMod, 0);
        // Compute fluxes

        // Continuity rho*Ui
        double rhouR = Q(i, j, k, 1);
        double rhovR = Q(i, j, k, 2);
        double rhowR = Q(i, j, k, 3);
        double rhouL = Q(i - iMod, j - jMod, k - kMod, 1);
        double rhovL = Q(i - iMod, j - jMod, k - kMod, 2);
        double rhowL = Q(i - iMod, j - jMod, k - kMod, 3);
        double CjR = iS(i, j, k, 0) * rhouR + iS(i, j, k, 1) * rhovR +
                     iS(i, j, k, 2) * rhowR;
        double CjL = iS(i, j, k, 0) * rhouL + iS(i, j, k, 1) * rhovL +
                     iS(i, j, k, 2) * rhowL;
        iF(i, j, k, 0) = 0.5 * (CjR + CjL);

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = 0.5 * (rhouR * UfR + pR * iS(i, j, k, 0) +
                                rhouL * UfL + pL * iS(i, j, k, 0));

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = 0.5 * (rhovR * UfR + pR * iS(i, j, k, 1) +
                                rhovL * UfL + pL * iS(i, j, k, 1));

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = 0.5 * (rhowR * UfR + pR * iS(i, j, k, 2) +
                                rhowL * UfL + pL * iS(i, j, k, 2));

        // Total energy (rhoE+ p)*Ui)
        double rhoER = Q(i, j, k, 4);
        double rhoEL = Q(i - iMod, j - jMod, k - kMod, 4);

        iF(i, j, k, 4) = 0.5 * ((rhoER + pR) * UfR + (rhoEL + pL) * UfL);

        // Species
        for (int n = 0; n < ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              0.5 * (Q(i, j, k, 5 + n) * UfR +
                     Q(i - iMod, j - jMod, k - kMod, 5 + n) * UfL);
        }
      });
}

PG_ABI void pgCentralDifference(int count, const pgView *Q_, const pgView *iF_,
                                const pgView *iS_, const pgView *jF_,
                                const pgView *jS_, const pgView *kF_,
                                const pgView *kS_, const pgView *q_,
                                const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto iF = as4(iF_[e]);
    auto iS = as4(iS_[e]);
    auto jF = as4(jF_[e]);
    auto jS = as4(jS_[e]);
    auto kF = as4(kF_[e]);
    auto kS = as4(kS_[e]);
    auto q = as4(q_[e]);
    computeFlux(Q, q, d[e], iF, iS, 1, 0, 0);
    computeFlux(Q, q, d[e], jF, jS, 0, 1, 0);
    computeFlux(Q, q, d[e], kF, kS, 0, 0, 1);
  }
}
