// Rusanov: the average of the two sides' fluxes, less the largest wave
// speed times the jump.
#ifndef __riemannRusanov_H__
#define __riemannRusanov_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"

struct rusanov {
  template <class Recon>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const cellFaceIn &A,
                                          const cellFaceOut &F) {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;
    double UR = nx * R.u + ny * R.v + nz * R.w;
    double UL = nx * L.u + ny * L.v + nz * L.w;

    // wave speed estimate
    const double lam = fmax(fabs(UL) + R.c, fabs(UR) + L.c) * S;
    UR *= S;
    UL *= S;

    // Continuity rho*Ui
    F(0) = 0.5 * (UR * R.rho + UL * L.rho - lam * (R.rho - L.rho));
    // momentum rho*u*Ui + p*A, each side's own then the average
    const double *uL[] = {&L.u, &L.v, &L.w}, *uR[] = {&R.u, &R.v, &R.w};
    for (int d = 0; d < 3; d++) {
      const double FUR = UR * *uR[d] * R.rho + R.p * A(d);
      const double FUL = UL * *uL[d] * L.rho + L.p * A(d);
      F(1 + d) = 0.5 * (FUR + FUL - lam * (R.rho * *uR[d] - L.rho * *uL[d]));
    }
    // Total energy (rhoE + p)*Ui
    F(4) = 0.5 * (UR * (R.E + R.p) + UL * (L.E + L.p) - lam * (R.E - L.E));
    // Species
    for (int n = 0; n < ne - 5; n++) {
      double rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      F(5 + n) = 0.5 * (rhoYR * UR + rhoYL * UL - lam * (rhoYR - rhoYL));
    }
  }
};

#endif
