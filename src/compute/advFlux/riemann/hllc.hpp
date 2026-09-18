// HLLC: the flux of whichever of the left, left star, right star or right
// states the face sits in, by the three wave speeds.
#ifndef __riemannHllc_H__
#define __riemannHllc_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"

struct hllc {
  // one side's own flux, and its star state's correction to it
  template <class Recon>
  static KOKKOS_INLINE_FUNCTION void
  own(const Recon &r, const typename Recon::sides &s, const faceState &X,
      double U, double S, const cellFaceIn &A, const cellFaceOut &F,
      bool left) {
    F(0) = U * X.rho * S;
    F(1) = U * X.rhou * S + X.p * A(0);
    F(2) = U * X.rhov * S + X.p * A(1);
    F(3) = U * X.rhow * S + X.p * A(2);
    F(4) = U * (X.E + X.p) * S;
    for (int n = 0; n < ne - 5; n++) {
      double rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      F(5 + n) = U * (left ? rhoYL : rhoYR) * S;
    }
  }
  template <class Recon>
  static KOKKOS_INLINE_FUNCTION void
  star(const Recon &r, const typename Recon::sides &s, const faceState &X,
       double U, double SX, double Sstar, double S, double nx, double ny,
       double nz, const cellFaceIn &A, const cellFaceOut &F, bool left) {
    const double rhoinv = 1.0 / X.rho;
    const double Frho = U * X.rho * S;
    const double FU = U * X.rhou * S + X.p * A(0);
    const double FV = U * X.rhov * S + X.p * A(1);
    const double FW = U * X.rhow * S + X.p * A(2);
    const double FE = U * (X.E + X.p) * S;
    const double Ustar = X.rho * (SX - U) / (SX - Sstar);
    F(0) = Frho + SX * (Ustar - X.rho) * S;
    F(1) = FU + SX * (Ustar * Sstar * nx - X.rhou) * S;
    F(2) = FV + SX * (Ustar * Sstar * ny - X.rhov) * S;
    F(3) = FW + SX * (Ustar * Sstar * nz - X.rhow) * S;
    F(4) =
        FE + SX *
                 (Ustar * (X.E * rhoinv +
                           (Sstar - U) * (Sstar + X.p / (X.rho * (SX - U)))) -
                  X.E) *
                 S;
    for (int n = 0; n < ne - 5; n++) {
      double rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      const double rhoY = left ? rhoYL : rhoYR;
      const double FY = rhoY * U * S;
      const double Y = rhoY * rhoinv;
      F(5 + n) = FY + SX * (Ustar * Y - rhoY) * S;
    }
  }

  template <class Recon>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const cellFaceIn &A,
                                          const cellFaceOut &F) {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;
    const double UR = nx * R.u + ny * R.v + nz * R.w;
    const double UL = nx * L.u + ny * L.v + nz * L.w;

    // wave speed estimate
    const double SL = UL - L.c;
    const double SR = UR + R.c;
    const double Sstar =
        (R.p - L.p + L.rho * UL * (SL - UL) - R.rho * UR * (SR - UR)) /
        (L.rho * (SL - UL) - R.rho * (SR - UR));

    if (SL >= 0.0)
      own(r, s, L, UL, S, A, F, true);
    else if ((SL <= 0.0) && (Sstar >= 0.0))
      star(r, s, L, UL, SL, Sstar, S, nx, ny, nz, A, F, true);
    else if ((SR >= 0.0) && (Sstar <= 0.0))
      star(r, s, R, UR, SR, Sstar, S, nx, ny, nz, A, F, false);
    else if (SR <= 0.0)
      own(r, s, R, UR, S, A, F, false);
  }
};

#endif
