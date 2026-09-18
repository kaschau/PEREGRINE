// KEPaEC: kinetic energy and pressure equilibrium preserving, entropy
// consistent -- the second order central scheme of Shima et al. (2021),
// on the two cells as they are.
#ifndef __formulaKEPaEC_H__
#define __formulaKEPaEC_H__

#include "advFlux/reconstruct/piecewiseConstant.hpp"
#include "faces.hpp"

struct KEPaEC {
  // the flux of two cells, a and b, each its conserved state, its p and T,
  // and what the eos keeps
  template <class Ca, class Cb, class Out>
  static KOKKOS_INLINE_FUNCTION void
  twoPoint(const Ca &Qa, const Ca &qa, const Ca &qha, const Cb &Qb,
           const Cb &qb, const Cb &qhb, const cellFaceIn &A, const Out &F) {
    // each side's velocity, off its conserved state
    const fpdtype rhoinvL = 1.0 / Qa(0), rhoinvR = 1.0 / Qb(0);
    const fpdtype uL = Qa(1) * rhoinvL, vL = Qa(2) * rhoinvL,
                  wL = Qa(3) * rhoinvL;
    const fpdtype uR = Qb(1) * rhoinvR, vR = Qb(2) * rhoinvR,
                  wR = Qb(3) * rhoinvR;

    // Compute face normal volume flux vector
    fpdtype uf = 0.5 * (uR + uL);
    fpdtype vf = 0.5 * (vR + vL);
    fpdtype wf = 0.5 * (wR + wL);

    fpdtype U = A(0) * uf + A(1) * vf + A(2) * wf;

    fpdtype pf = 0.5 * (qb(0) + qa(0));

    // Compute fluxes
    fpdtype rho = 0.5 * (Qb(0) + Qa(0));

    // Continuity rho*Ui
    fpdtype C = rho * U;
    F(0) = C;

    // x momentum rho*u*Ui+ p*Ax
    F(1) = C * uf + pf * A(0);

    // y momentum rho*v*Ui+ p*Ay
    F(2) = C * vf + pf * A(1);

    // w momentum rho*w*Ui+ p*Az
    F(3) = C * wf + pf * A(2);

    // Total energy (rhoE+ p)*Ui)
    fpdtype Kj = C * 0.5 * (uR * uL + vR * vL + wR * wL);

    fpdtype Pj = 0.5 * (qa(0) * (uR * A(0) + vR * A(1) + wR * A(2)) +
                        qb(0) * (uL * A(0) + vL * A(1) + wL * A(2)));

    // solve for internal energy flux
    fpdtype eR = qhb(4) * rhoinvR;
    fpdtype eL = qha(4) * rhoinvL;
    fpdtype Ij = 2.0 * (eL * eR) / (eL + eR) * C;

    F(4) = Ij + Kj + Pj;

    // Species
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) = 0.5 * (Qb(5 + n) * rhoinvR + Qa(5 + n) * rhoinvL) * C;
    }
  }
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const cellFaceIn &A,
                                          const Out &F) {
    static_assert(std::is_base_of_v<piecewiseConstant, Recon>,
                  "KEPaEC is a central scheme: it takes no reconstruction");
    twoPoint(r.QL, r.qL, r.qhL, r.QR, r.qR, r.qhR, A, F);
  }
};

#endif
