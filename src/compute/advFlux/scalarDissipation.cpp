#include "faces.hpp"

PG_STENCIL(2);

PG_RANGE(faces)
struct scalarDissipation {
  inL QL, phiL, qL, qhL;
  inR QR, phiR, qR, qhR;
  inLL QLL;
  inRR QRR;
  out F;
  in A;
  static constexpr double kappa2 = 0.5;
  static constexpr double kappa4 = 0.005;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    // The weird mod indexing math is so we grab the correct last
    // phi index for each dimension
    const int phiIndex = (iMod - 1) * iMod + (jMod)*jMod + (kMod + 1) * kMod;
    const double eps2 = kappa2 * fmax(phiR(phiIndex), phiL(phiIndex));
    const double eps4 = fmax(0.0, kappa4 - eps2);

    // Compute face normal volume flux vector
    const double uf = 0.5 * (qR(1) + qL(1));
    const double vf = 0.5 * (qR(2) + qL(2));
    const double wf = 0.5 * (qR(3) + qL(3));

    const double U = nx * uf + ny * vf + nz * wf;

    // negative: this flux leaves the state, and is applied like any
    // other
    const double a = -(abs(U) + 0.5 * (qhR(3) + qhL(3))) * S;

    double rho2, rho4;
    rho2 = QR(0) - QL(0);
    rho4 = QRR(0) - 3.0 * QR(0) + 3.0 * QL(0) - QLL(0);

    // Continuity dissipation
    F(0) = a * (eps2 * rho2 - eps4 * rho4);

    // u momentum dissipation
    double u2, u4;
    u2 = QR(1) - QL(1);
    u4 = QRR(1) - 3.0 * QR(1) + 3.0 * QL(1) - QLL(1);

    F(1) = a * (eps2 * u2 - eps4 * u4);

    // v momentum dissipation
    double v2, v4;
    v2 = QR(2) - QL(2);
    v4 = QRR(2) - 3.0 * QR(2) + 3.0 * QL(2) - QLL(2);

    F(2) = a * (eps2 * v2 - eps4 * v4);

    // w momentum dissipation
    double w2, w4;
    w2 = QR(3) - QL(3);
    w4 = QRR(3) - 3.0 * QR(3) + 3.0 * QL(3) - QLL(3);

    F(3) = a * (eps2 * w2 - eps4 * w4);

    // total energy dissipation
    double e2, e4;
    e2 = QR(4) - QL(4);
    e4 = QRR(4) - 3.0 * QR(4) + 3.0 * QL(4) - QLL(4);

    F(4) = a * (eps2 * e2 - eps4 * e4);

    // Species
    for (int n = 0; n < ne - 5; n++) {
      double Y2, Y4;
      Y2 = QR(5 + n) - QL(5 + n);
      Y4 = QRR(5 + n) - 3.0 * QR(5 + n) + 3.0 * QL(5 + n) - QLL(5 + n);
      F(5 + n) = a * (eps2 * Y2 - eps4 * Y4);
    }
  }
};

PG_ABI void pgScalarDissipation(const scalarDissipation &k, const pgTiling &t) {
  forCells("Scalar Dissipation face conv fluxes", t, k);
}
