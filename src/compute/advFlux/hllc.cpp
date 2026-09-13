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
      "hllc face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        const double &ufR = q(i, j, k, 1);
        const double &vfR = q(i, j, k, 2);
        const double &wfR = q(i, j, k, 3);

        const double &ufL = q(i - iMod, j - jMod, k - kMod, 1);
        const double &vfL = q(i - iMod, j - jMod, k - kMod, 2);
        const double &wfL = q(i - iMod, j - jMod, k - kMod, 3);

        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        const double &rhoR = Q(i, j, k, 0);
        const double &rhoL = Q(i - iMod, j - jMod, k - kMod, 0);

        const double &rhouR = Q(i, j, k, 1);
        const double &rhouL = Q(i - iMod, j - jMod, k - kMod, 1);
        const double &rhovR = Q(i, j, k, 2);
        const double &rhovL = Q(i - iMod, j - jMod, k - kMod, 2);
        const double &rhowR = Q(i, j, k, 3);
        const double &rhowL = Q(i - iMod, j - jMod, k - kMod, 3);

        const double &pR = q(i, j, k, 0);
        const double &pL = q(i - iMod, j - jMod, k - kMod, 0);

        const double &ER = Q(i, j, k, 4);
        const double &EL = Q(i - iMod, j - jMod, k - kMod, 4);

        const double &cR = qh(i, j, k, 3);
        const double &cL = qh(i - iMod, j - jMod, k - kMod, 3);

        double pstar = 0.5 * (pL + pR) -
                       0.5 * (UR - UL) * 0.5 * (rhoL + rhoR) * 0.5 * (cL + cR);
        pstar = fmax(0.0, pstar);

        // wave speed estimate
        double SL = UL - cL;
        double SR = UR + cR;
        double Sstar =
            (pR - pL + rhoL * UL * (SL - UL) - rhoR * UR * (SR - UR)) /
            (rhoL * (SL - UL) - rhoR * (SR - UR));

        if (SL >= 0.0) {
          iF(i, j, k, 0) = UL * rhoL * S;
          iF(i, j, k, 1) = UL * rhouL * S + pL * iS(i, j, k, 0);
          iF(i, j, k, 2) = UL * rhovL * S + pL * iS(i, j, k, 1);
          iF(i, j, k, 3) = UL * rhowL * S + pL * iS(i, j, k, 2);
          iF(i, j, k, 4) = UL * (EL + pL) * S;
          for (int n = 0; n < ne - 5; n++) {
            double rhoYiL = Q(i - iMod, j - jMod, k - kMod, 5 + n);
            iF(i, j, k, 5 + n) = UL * rhoYiL * S;
          }
        } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
          double FrhoL, FUL, FVL, FWL, FEL, UstarL;
          FrhoL = UL * rhoL * S;
          FUL = UL * rhouL * S + pL * iS(i, j, k, 0);
          FVL = UL * rhovL * S + pL * iS(i, j, k, 1);
          FWL = UL * rhowL * S + pL * iS(i, j, k, 2);
          FEL = UL * (EL + pL) * S;
          UstarL = rhoL * (SL - UL) / (SL - Sstar);

          iF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * S;
          iF(i, j, k, 1) = FUL + SL * (UstarL * Sstar * nx - rhouL) * S;
          iF(i, j, k, 2) = FVL + SL * (UstarL * Sstar * ny - rhovL) * S;
          iF(i, j, k, 3) = FWL + SL * (UstarL * Sstar * nz - rhowL) * S;
          iF(i, j, k, 4) =
              FEL +
              SL *
                  (UstarL * (EL / rhoL +
                             (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                   EL) *
                  S;
          for (int n = 0; n < ne - 5; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = Q(i - iMod, j - jMod, k - kMod, 5 + n) * UL * S;
            YiL = q(i - iMod, j - jMod, k - kMod, 5 + n);
            rhoYiL = Q(i - iMod, j - jMod, k - kMod, 5 + n);
            iF(i, j, k, 5 + n) = FYiL + SL * (UstarL * YiL - rhoYiL) * S;
          }
        } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
          double FrhoR, FUR, FVR, FWR, FER, UstarR;
          FrhoR = UR * rhoR * S;
          FUR = UR * rhouR * S + pR * iS(i, j, k, 0);
          FVR = UR * rhovR * S + pR * iS(i, j, k, 1);
          FWR = UR * rhowR * S + pR * iS(i, j, k, 2);
          FER = UR * (ER + pR) * S;
          UstarR = rhoR * (SR - UR) / (SR - Sstar);

          iF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * S;
          iF(i, j, k, 1) = FUR + SR * (UstarR * Sstar * nx - rhouR) * S;
          iF(i, j, k, 2) = FVR + SR * (UstarR * Sstar * ny - rhovR) * S;
          iF(i, j, k, 3) = FWR + SR * (UstarR * Sstar * nz - rhowR) * S;
          iF(i, j, k, 4) =
              FER +
              SR *
                  (UstarR * (ER / rhoR +
                             (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                   ER) *
                  S;
          for (int n = 0; n < ne - 5; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = Q(i, j, k, 5 + n) * UR * S;
            YiR = q(i, j, k, 5 + n);
            rhoYiR = Q(i, j, k, 5 + n);
            iF(i, j, k, 5 + n) = FYiR + SR * (UstarR * YiR - rhoYiR) * S;
          }
        } else if (SR <= 0.0) {
          iF(i, j, k, 0) = UR * rhoR * S;
          iF(i, j, k, 1) = UR * rhouR * S + pR * iS(i, j, k, 0);
          iF(i, j, k, 2) = UR * rhovR * S + pR * iS(i, j, k, 1);
          iF(i, j, k, 3) = UR * rhowR * S + pR * iS(i, j, k, 2);
          iF(i, j, k, 4) = UR * (ER + pR) * S;
          for (int n = 0; n < ne - 5; n++) {
            double rhoYiR = Q(i, j, k, 5 + n);
            iF(i, j, k, 5 + n) = UR * rhoYiR * S;
          }
        }
      });
}

PG_ABI void pgHllc(int count, pgIn *Q_, pgOut *iF_, pgIn *iS_, pgOut *jF_,
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
