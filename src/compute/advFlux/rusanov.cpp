#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const unmanaged<double ****> &Q,
                        const unmanaged<double ****> &q,
                        const unmanaged<double ****> &qh, const int ne,
                        const int ng, const int ni, const int nj, const int nk,
                        unmanaged<double ****> &iF,
                        const unmanaged<double ****> &iS, const int iMod,
                        const int jMod, const int kMod) {

  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});
  Kokkos::parallel_for(
      "rusanov face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        double UR;
        double UL;

        double &ufR = q(i, j, k, 1);
        double &vfR = q(i, j, k, 2);
        double &wfR = q(i, j, k, 3);

        double &ufL = q(i - iMod, j - jMod, k - kMod, 1);
        double &vfL = q(i - iMod, j - jMod, k - kMod, 2);
        double &wfL = q(i - iMod, j - jMod, k - kMod, 3);

        UR = nx * ufR + ny * vfR + nz * wfR;
        UL = nx * ufL + ny * vfL + nz * wfL;

        double &rhoR = Q(i, j, k, 0);
        double &rhoL = Q(i - iMod, j - jMod, k - kMod, 0);

        double &pR = q(i, j, k, 0);
        double &pL = q(i - iMod, j - jMod, k - kMod, 0);

        double &ER = Q(i, j, k, 4);
        double &EL = Q(i - iMod, j - jMod, k - kMod, 4);

        // wave speed estimate
        double lam = fmax(abs(UL) + qh(i, j, k, 3),
                          abs(UR) + qh(i - iMod, j - jMod, k - kMod, 3)) *
                     S;
        UR *= S;
        UL *= S;

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        iF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * iS(i, j, k, 0);
        FUL = UL * ufL * rhoL + pL * iS(i, j, k, 0);
        iF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * iS(i, j, k, 1);
        FUL = UL * vfL * rhoL + pL * iS(i, j, k, 1);
        iF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * iS(i, j, k, 2);
        FUL = UL * wfL * rhoL + pL * iS(i, j, k, 2);
        iF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        iF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        double FYiR, FYiL;
        double YiR, YiL;
        for (int n = 0; n < ne - 5; n++) {
          FYiR = Q(i, j, k, 5 + n) * UR;
          FYiL = Q(i - iMod, j - jMod, k - kMod, 5 + n) * UL;
          YiR = Q(i, j, k, 5 + n);
          YiL = Q(i - iMod, j - jMod, k - kMod, 5 + n);
          iF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (YiR - YiL));
        }
      });
}

PG_ABI void pgRusanov(const pgView *Q_, const pgView *iF_, const pgView *iS_,
                      const pgView *jF_, const pgView *jS_, const pgView *kF_,
                      const pgView *kS_, const pgView *q_, const pgView *qh_,
                      const pgDims *d) {
  auto Q = as4(*Q_);
  auto iF = as4(*iF_);
  auto iS = as4(*iS_);
  auto jF = as4(*jF_);
  auto jS = as4(*jS_);
  auto kF = as4(*kF_);
  auto kS = as4(*kS_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = Q.extent(3);
  computeFlux(Q, q, qh, ne, ng, ni, nj, nk, iF, iS, 1, 0, 0);
  computeFlux(Q, q, qh, ne, ng, ni, nj, nk, jF, jS, 0, 1, 0);
  computeFlux(Q, q, qh, ne, ng, ni, nj, nk, kF, kS, 0, 0, 1);
}
