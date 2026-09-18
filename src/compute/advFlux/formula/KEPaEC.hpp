// KEPaEC: kinetic energy and pressure equilibrium preserving, entropy
// consistent -- the second order central scheme of Shima et al. (2021),
// on the two cells as they are.
#ifndef __formulaKEPaEC_H__
#define __formulaKEPaEC_H__

#include "advFlux/reconstruct/piecewiseConstant.hpp"
#include "faces.hpp"

struct KEPaEC {
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const cellFaceIn &A,
                                          const Out &F) {
    static_assert(std::is_base_of_v<piecewiseConstant, Recon>,
                  "KEPaEC is a central scheme: it takes no reconstruction");
    // each side's velocity, off its conserved state
    const double rhoinvL = 1.0 / r.QL(0), rhoinvR = 1.0 / r.QR(0);
    const double uL = r.QL(1) * rhoinvL, vL = r.QL(2) * rhoinvL,
                 wL = r.QL(3) * rhoinvL;
    const double uR = r.QR(1) * rhoinvR, vR = r.QR(2) * rhoinvR,
                 wR = r.QR(3) * rhoinvR;

    // Compute face normal volume flux vector
    double uf = 0.5 * (uR + uL);
    double vf = 0.5 * (vR + vL);
    double wf = 0.5 * (wR + wL);

    double U = A(0) * uf + A(1) * vf + A(2) * wf;

    double pf = 0.5 * (r.qR(0) + r.qL(0));

    // Compute fluxes
    double rho = 0.5 * (r.QR(0) + r.QL(0));

    // Continuity rho*Ui
    double C = rho * U;
    F(0) = C;

    // x momentum rho*u*Ui+ p*Ax
    F(1) = C * uf + pf * A(0);

    // y momentum rho*v*Ui+ p*Ay
    F(2) = C * vf + pf * A(1);

    // w momentum rho*w*Ui+ p*Az
    F(3) = C * wf + pf * A(2);

    // Total energy (rhoE+ p)*Ui)
    double Kj = C * 0.5 * (uR * uL + vR * vL + wR * wL);

    double Pj = 0.5 * (r.qL(0) * (uR * A(0) + vR * A(1) + wR * A(2)) +
                       r.qR(0) * (uL * A(0) + vL * A(1) + wL * A(2)));

    // solve for internal energy flux
    double eR = r.qhR(4) * rhoinvR;
    double eL = r.qhL(4) * rhoinvL;
    double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

    F(4) = Ij + Kj + Pj;

    // Species
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) = 0.5 * (r.QR(5 + n) * rhoinvR + r.QL(5 + n) * rhoinvL) * C;
    }
  }
};

#endif
