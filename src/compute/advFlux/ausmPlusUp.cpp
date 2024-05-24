#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const threeDview &iS,
                        const threeDview &isx, const threeDview &isy,
                        const threeDview &isz, const threeDview &inx,
                        const threeDview &iny, const threeDview &inz,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});
  Kokkos::parallel_for(
      "AUSM+UP face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double ufL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double vfL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double wfL = b.q(i - iMod, j - jMod, k - kMod, 3);

        double UR =
            inx(i, j, k) * ufR + iny(i, j, k) * vfR + inz(i, j, k) * wfR;
        double UL =
            inx(i, j, k) * ufL + iny(i, j, k) * vfL + inz(i, j, k) * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double rhoL = b.Q(i - iMod, j - jMod, k - kMod, 0);

        double &pR = b.q(i, j, k, 0);
        double pL = b.q(i - iMod, j - jMod, k - kMod, 0);

        double a12 =
            0.5 * (b.qh(i, j, k, 3) + b.qh(i - iMod, j - jMod, k - kMod, 3));
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
        iF(i, j, k, 0) = mDot12 * iS(i, j, k);

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = mDot12 * b.q(iIndx, jIndx, kIndx, 1) * iS(i, j, k) +
                         p12 * isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = mDot12 * b.q(iIndx, jIndx, kIndx, 2) * iS(i, j, k) +
                         p12 * isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = mDot12 * b.q(iIndx, jIndx, kIndx, 3) * iS(i, j, k) +
                         p12 * isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        iF(i, j, k, 4) =
            mDot12 *
            (b.Q(iIndx, jIndx, kIndx, 4) + b.q(iIndx, jIndx, kIndx, 0)) /
            b.Q(iIndx, jIndx, kIndx, 0) * iS(i, j, k);

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              mDot12 * b.q(iIndx, jIndx, kIndx, 5 + n) * iS(i, j, k);
        }
      });
}

void ausmPlusUp(block_ &b) {
  computeFlux(b, b.iF, b.iS, b.isx, b.isy, b.isz, b.inx, b.iny, b.inz, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, b.jsx, b.jsy, b.jsz, b.jnx, b.jny, b.jnz, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, b.ksx, b.ksy, b.ksz, b.knx, b.kny, b.knz, 0, 0, 1);
}
