#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

static void computeFlux(const unmanaged<double ****> &Q,
                        const unmanaged<double ****> &q,
                        const unmanaged<double ****> &qh, const pgDims &d,
                        unmanaged<double ****> &iF,
                        const unmanaged<double ****> &iS, const int iMod,
                        const int jMod, const int kMod) {

  const int ni = d.ni, nj = d.nj, nk = d.nk;
  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});
  Kokkos::parallel_for(
      "AUSM+UP face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        double &ufR = q(i, j, k, 1);
        double &vfR = q(i, j, k, 2);
        double &wfR = q(i, j, k, 3);

        double ufL = q(i - iMod, j - jMod, k - kMod, 1);
        double vfL = q(i - iMod, j - jMod, k - kMod, 2);
        double wfL = q(i - iMod, j - jMod, k - kMod, 3);

        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        double &rhoR = Q(i, j, k, 0);
        double rhoL = Q(i - iMod, j - jMod, k - kMod, 0);

        double &pR = q(i, j, k, 0);
        double pL = q(i - iMod, j - jMod, k - kMod, 0);

        double a12 =
            0.5 * (qh(i, j, k, 3) + qh(i - iMod, j - jMod, k - kMod, 3));
        double ML = UL / a12;
        double MR = UR / a12;

        double MbarSQ = (pow(UR, 2.0) + pow(UL, 2.0)) / (2.0 * pow(a12, 2.0));

        const double MinfSQ = 0.1;
        double Mo = sqrt(fmin(1.0, fmax(MbarSQ, MinfSQ)));
        double fa = Mo * (2.0 - Mo);

        const double beta = 1.0 / 8.0;

        double M1Plus = 0.5 * (ML + abs(ML));
        double M1Minus = 0.5 * (MR - abs(MR));
        double M2Plus = 0.25 * pow(ML + 1.0, 2.0);
        double M2Minus = -0.25 * pow(MR - 1.0, 2.0);
        double M4Plus =
            (abs(ML) >= 1.0) ? M1Plus : M2Plus * (1.0 - 16.0 * beta * M2Minus);
        double M4Minus =
            (abs(MR) >= 1.0) ? M1Minus : M2Minus * (1.0 + 16.0 * beta * M2Plus);

        const double Kp = 0.25;
        const double sigma = 1.0;
        double rho12 = 0.5 * (rhoR + rhoL);

        double M12 = M4Plus + M4Minus -
                     Kp / fa * (fmax(1.0 - sigma * MbarSQ, 0.0)) * (pR - pL) /
                         (rho12 * pow(a12, 2.0));

        double mDot12 = (M12 > 0.0) ? a12 * M12 * rhoL : a12 * M12 * rhoR;

        const double alpha = 3.0 / 16.0 * (-4.0 + 5.0 * pow(fa, 2.0));
        double p5Plus =
            (abs(ML) >= 1.0)
                ? 1.0 / ML * M1Plus
                : M2Plus * ((2.0 - ML) - 16.0 * alpha * ML * M2Minus);
        double p5Minus =
            (abs(MR) >= 1.0)
                ? 1.0 / MR * M1Minus
                : M2Minus * ((-2.0 - MR) + 16.0 * alpha * MR * M2Plus);

        const double Ku = 0.75;
        double p12 =
            p5Plus * pL + p5Minus * pR -
            Ku * p5Plus * p5Minus * (rhoR + rhoL) * (fa * a12) * (UR - UL);

        // Upwind the flux
        const int indx = (mDot12 > 0.0) ? -1 : 0;
        const int iIndx = i + indx * iMod;
        const int jIndx = j + indx * jMod;
        const int kIndx = k + indx * kMod;
        // Continuity rho*Ui
        iF(i, j, k, 0) = mDot12 * S;

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) =
            mDot12 * q(iIndx, jIndx, kIndx, 1) * S + p12 * iS(i, j, k, 0);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) =
            mDot12 * q(iIndx, jIndx, kIndx, 2) * S + p12 * iS(i, j, k, 1);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) =
            mDot12 * q(iIndx, jIndx, kIndx, 3) * S + p12 * iS(i, j, k, 2);

        // Total energy (rhoE+ p)*Ui)
        iF(i, j, k, 4) =
            mDot12 * (Q(iIndx, jIndx, kIndx, 4) + q(iIndx, jIndx, kIndx, 0)) /
            Q(iIndx, jIndx, kIndx, 0) * S;

        // Species
        for (int n = 0; n < ne - 5; n++) {
          iF(i, j, k, 5 + n) = mDot12 * q(iIndx, jIndx, kIndx, 5 + n) * S;
        }
      });
}

PG_ABI void pgAusmPlusUp(int count, const pgView *Q_, const pgView *iF_,
                         const pgView *iS_, const pgView *jF_,
                         const pgView *jS_, const pgView *kF_,
                         const pgView *kS_, const pgView *q_, const pgView *qh_,
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
