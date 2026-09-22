// Rusanov: the average of the two sides' fluxes, less the largest wave
// speed times the jump.
#ifndef __formulaRusanov_H__
#define __formulaRusanov_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"
#include "utils/normal.hpp"

struct rusanov {
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const faceVecIn &A,
                                          const Out &F) {
    fpdtype S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;
    fpdtype UR = normalVelocity(R.rho, R.rhou, R.rhov, R.rhow, nx, ny, nz);
    fpdtype UL = normalVelocity(L.rho, L.rhou, L.rhov, L.rhow, nx, ny, nz);

    // wave speed estimate
    const fpdtype lam = fmax(fabs(UL) + R.c, fabs(UR) + L.c) * S;
    UR *= S;
    UL *= S;

    // Continuity rho*Ui
    F(0) = 0.5 * (UR * R.rho + UL * L.rho - lam * (R.rho - L.rho));
    // momentum rho*u*Ui + p*A, each side's own then the average
    const fpdtype *mL[] = {&L.rhou, &L.rhov, &L.rhow},
                  *mR[] = {&R.rhou, &R.rhov, &R.rhow};
    for (int d = 0; d < 3; d++) {
      const fpdtype FUR = UR * *mR[d] + R.p * A(d);
      const fpdtype FUL = UL * *mL[d] + L.p * A(d);
      F(1 + d) = 0.5 * (FUR + FUL - lam * (*mR[d] - *mL[d]));
    }
    // Total energy (rhoE + p)*Ui
    F(4) = 0.5 * (UR * (R.E + R.p) + UL * (L.E + L.p) - lam * (R.E - L.E));
    // Species
    for (int n = 0; n < ne - 5; n++) {
      fpdtype rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      F(5 + n) = 0.5 * (rhoYR * UR + rhoYL * UL - lam * (rhoYR - rhoYL));
    }
  }
};

#endif
