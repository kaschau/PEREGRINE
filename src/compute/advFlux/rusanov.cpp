#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const in4 &Q, const in4 &q, const in4 &qh,
                        const pgDims &d, const out4 &iF, const in4 &iS,
                        const int iMod, const int jMod, const int kMod) {

  const int ni = d.ni, nj = d.nj, nk = d.nk;
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

        const double &ufR = q(i, j, k, 1);
        const double &vfR = q(i, j, k, 2);
        const double &wfR = q(i, j, k, 3);

        const double &ufL = q(i - iMod, j - jMod, k - kMod, 1);
        const double &vfL = q(i - iMod, j - jMod, k - kMod, 2);
        const double &wfL = q(i - iMod, j - jMod, k - kMod, 3);

        UR = nx * ufR + ny * vfR + nz * wfR;
        UL = nx * ufL + ny * vfL + nz * wfL;

        const double &rhoR = Q(i, j, k, 0);
        const double &rhoL = Q(i - iMod, j - jMod, k - kMod, 0);

        const double &pR = q(i, j, k, 0);
        const double &pL = q(i - iMod, j - jMod, k - kMod, 0);

        const double &ER = Q(i, j, k, 4);
        const double &EL = Q(i - iMod, j - jMod, k - kMod, 4);

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

PG_ABI void pgRusanov(int count, pgIn *Q_, pgOut *iF_, pgIn *iS_, pgOut *jF_,
                      pgIn *jS_, pgOut *kF_, pgIn *kS_, pgIn *q_, pgIn *qh_,
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
    auto qh = as4(qh_[e]);
    computeFlux(Q, q, qh, d[e], iF, iS, 1, 0, 0);
    computeFlux(Q, q, qh, d[e], jF, jS, 0, 1, 0);
    computeFlux(Q, q, qh, d[e], kF, kS, 0, 0, 1);
  }
}
