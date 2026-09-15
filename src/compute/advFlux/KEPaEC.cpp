#include "faces.hpp"

PG_RANGE(faces)
struct KEPaEC {
  inL QL, qL, qhL;
  inR QR, qR, qhR;
  out F;
  in A;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // Compute face normal volume flux vector
    double uf = 0.5 * (qR(1) + qL(1));
    double vf = 0.5 * (qR(2) + qL(2));
    double wf = 0.5 * (qR(3) + qL(3));

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
    double Kj = C * 0.5 * (qR(1) * qL(1) + qR(2) * qL(2) + qR(3) * qL(3));

    double Pj = 0.5 * (qL(0) * (qR(1) * A(0) + qR(2) * A(1) + qR(3) * A(2)) +
                       qR(0) * (qL(1) * A(0) + qL(2) * A(1) + qL(3) * A(2)));

    // solve for internal energy flux
    double eR = qhR(4) / QR(0);
    double eL = qhL(4) / QL(0);
    double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

    F(4) = Ij + Kj + Pj;

    // Species
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) = 0.5 * (qR(5 + n) + qL(5 + n)) * C;
    }
  }
};

PG_ABI void pgKEPaEC(const KEPaEC &k, const pgTiling &t) {
  forCells("2nd order KEPaEC conv fluxes", t, k);
}
