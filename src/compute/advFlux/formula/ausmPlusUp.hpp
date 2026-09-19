// AUSM+-up: a mass flux from the two sides' Mach numbers, a pressure flux
// from their pressures, and the upwind side carrying the rest.
#ifndef __formulaAusmPlusUp_H__
#define __formulaAusmPlusUp_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"

struct ausmPlusUp {
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const faceVecIn &A,
                                          const Out &F) {
    fpdtype S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;
    const fpdtype UR = nx * R.u + ny * R.v + nz * R.w;
    const fpdtype UL = nx * L.u + ny * L.v + nz * L.w;

    const fpdtype a12 = 0.5 * (R.c + L.c);
    const fpdtype ML = UL / a12;
    const fpdtype MR = UR / a12;

    const fpdtype MbarSQ =
        (pow(UR, 2.0) + pow(UL, 2.0)) / (2.0 * pow(a12, 2.0));

    const fpdtype MinfSQ = 0.1;
    const fpdtype Mo = sqrt(fmin(1.0, fmax(MbarSQ, MinfSQ)));
    const fpdtype fa = Mo * (2.0 - Mo);

    const fpdtype beta = 1.0 / 8.0;

    const fpdtype M1Plus = 0.5 * (ML + fabs(ML));
    const fpdtype M1Minus = 0.5 * (MR - fabs(MR));
    const fpdtype M2Plus = 0.25 * pow(ML + 1.0, 2.0);
    const fpdtype M2Minus = -0.25 * pow(MR - 1.0, 2.0);
    const fpdtype M4Plus =
        (fabs(ML) >= 1.0) ? M1Plus : M2Plus * (1.0 - 16.0 * beta * M2Minus);
    const fpdtype M4Minus =
        (fabs(MR) >= 1.0) ? M1Minus : M2Minus * (1.0 + 16.0 * beta * M2Plus);

    const fpdtype Kp = 0.25;
    const fpdtype sigma = 1.0;
    const fpdtype rho12 = 0.5 * (R.rho + L.rho);

    const fpdtype M12 = M4Plus + M4Minus -
                        Kp / fa * (fmax(1.0 - sigma * MbarSQ, 0.0)) *
                            (R.p - L.p) / (rho12 * pow(a12, 2.0));

    const fpdtype mDot12 = (M12 > 0.0) ? a12 * M12 * L.rho : a12 * M12 * R.rho;

    const fpdtype alpha = 3.0 / 16.0 * (-4.0 + 5.0 * pow(fa, 2.0));
    const fpdtype p5Plus =
        (fabs(ML) >= 1.0) ? 1.0 / ML * M1Plus
                          : M2Plus * ((2.0 - ML) - 16.0 * alpha * ML * M2Minus);
    const fpdtype p5Minus =
        (fabs(MR) >= 1.0)
            ? 1.0 / MR * M1Minus
            : M2Minus * ((-2.0 - MR) + 16.0 * alpha * MR * M2Plus);

    const fpdtype Ku = 0.75;
    const fpdtype p12 =
        p5Plus * L.p + p5Minus * R.p -
        Ku * p5Plus * p5Minus * (R.rho + L.rho) * (fa * a12) * (UR - UL);

    // the upwind side carries the rest
    const bool fromL = mDot12 > 0.0;
    const faceState &U = fromL ? L : R;
    const fpdtype rhoinvU = 1.0 / U.rho;
    // Continuity rho*Ui
    F(0) = mDot12 * S;
    // momentum rho*u*Ui + p*A
    F(1) = mDot12 * U.u * S + p12 * A(0);
    F(2) = mDot12 * U.v * S + p12 * A(1);
    F(3) = mDot12 * U.w * S + p12 * A(2);
    // Total energy (rhoE + p)*Ui
    F(4) = mDot12 * (U.E + U.p) * rhoinvU * S;
    // Species
    for (int n = 0; n < ne - 5; n++) {
      fpdtype rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      F(5 + n) = mDot12 * (fromL ? rhoYL : rhoYR) * rhoinvU * S;
    }
  }
};

#endif
