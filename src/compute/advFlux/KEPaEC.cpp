#include "faces.hpp"

PG_RANGE(cellFaces)
struct KEPaEC {
  cellCenterL QL, qL, qhL;
  cellCenterR QR, qR, qhR;
  cellFaceOut F;
  cellFaceIn A;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // each side's velocity, off its conserved state
    const double rhoinvL = 1.0 / QL(0), rhoinvR = 1.0 / QR(0);
    const double uL = QL(1) * rhoinvL, vL = QL(2) * rhoinvL,
                 wL = QL(3) * rhoinvL;
    const double uR = QR(1) * rhoinvR, vR = QR(2) * rhoinvR,
                 wR = QR(3) * rhoinvR;

    // Compute face normal volume flux vector
    double uf = 0.5 * (uR + uL);
    double vf = 0.5 * (vR + vL);
    double wf = 0.5 * (wR + wL);

    double U = A(0) * uf + A(1) * vf + A(2) * wf;

    double pf = 0.5 * (qR(0) + qL(0));

    // Compute fluxes
    double rho = 0.5 * (QR(0) + QL(0));

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

    double Pj = 0.5 * (qL(0) * (uR * A(0) + vR * A(1) + wR * A(2)) +
                       qR(0) * (uL * A(0) + vL * A(1) + wL * A(2)));

    // solve for internal energy flux
    double eR = qhR(4) * rhoinvR;
    double eL = qhL(4) * rhoinvL;
    double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

    F(4) = Ij + Kj + Pj;

    // Species
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) = 0.5 * (QR(5 + n) * rhoinvR + QL(5 + n) * rhoinvL) * C;
    }
  }
};

PG_ABI void pgKEPaEC(const KEPaEC &k, const pgTiling &t) {
  forCells("2nd order KEPaEC conv fluxes", t, k);
}
