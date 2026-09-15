#include "faces.hpp"
#include "math.h"

PG_RANGE(faces)
struct ausmPlusUp {
  inL QL, qL, qhL;
  inR QR, qR, qhR;
  out F;
  in A;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    const double &ufR = qR(1);
    const double &vfR = qR(2);
    const double &wfR = qR(3);

    double ufL = qL(1);
    double vfL = qL(2);
    double wfL = qL(3);

    double UR = nx * ufR + ny * vfR + nz * wfR;
    double UL = nx * ufL + ny * vfL + nz * wfL;

    const double &rhoR = QR(0);
    double rhoL = QL(0);

    const double &pR = qR(0);
    double pL = qL(0);

    double a12 = 0.5 * (qhR(3) + qhL(3));
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
    double p5Plus = (abs(ML) >= 1.0)
                        ? 1.0 / ML * M1Plus
                        : M2Plus * ((2.0 - ML) - 16.0 * alpha * ML * M2Minus);
    double p5Minus = (abs(MR) >= 1.0)
                         ? 1.0 / MR * M1Minus
                         : M2Minus * ((-2.0 - MR) + 16.0 * alpha * MR * M2Plus);

    const double Ku = 0.75;
    double p12 = p5Plus * pL + p5Minus * pR -
                 Ku * p5Plus * p5Minus * (rhoR + rhoL) * (fa * a12) * (UR - UL);

    // Upwind the flux
    const int indx = (mDot12 > 0.0) ? -1 : 0;
    // Continuity rho*Ui
    F(0) = mDot12 * S;

    // x momentum rho*u*Ui+ p*Ax
    F(1) = mDot12 * qR(indx * N, 1) * S + p12 * A(0);

    // y momentum rho*v*Ui+ p*Ay
    F(2) = mDot12 * qR(indx * N, 2) * S + p12 * A(1);

    // w momentum rho*w*Ui+ p*Az
    F(3) = mDot12 * qR(indx * N, 3) * S + p12 * A(2);

    // Total energy (rhoE+ p)*Ui)
    F(4) = mDot12 * (QR(indx * N, 4) + qR(indx * N, 0)) / QR(indx * N, 0) * S;

    // Species
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) = mDot12 * qR(indx * N, 5 + n) * S;
    }
  }
};

PG_ABI void pgAusmPlusUp(const ausmPlusUp &k, const pgTiling &t) {
  forCells("AUSM+UP face conv fluxes", t, k);
}
