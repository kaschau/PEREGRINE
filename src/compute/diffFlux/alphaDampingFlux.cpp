#include "faces.hpp"

// References
//
// Robust numerical fluxes for unrealizable states
//
// Hiroaki Nishikawa
// Journal of Computational Physics
// 408 (2020)

PG_RANGE(faces)
struct alphaDampingFlux {
  inL QL, cellsL, gradsL, qL, qhL, qtL;
  inR QR, cellsR, gradsR, qR, qhR, qtR;
  inout F;
  in A, Faces;
  static constexpr double bulkVisc = 0.0;
  static constexpr double alpha = 1.0;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    double mu = 0.5 * (qtR(0) + qtL(0));
    double kappa = 0.5 * (qtR(1) + qtL(1));
    double lambda = bulkVisc - 2.0 / 3.0 * mu;

    // no mass diffuses, so the continuity flux is left alone

    // Geometric terms
    double e[3] = {cellsR(0) - cellsL(0), cellsR(1) - cellsL(1),
                   cellsR(2) - cellsL(2)};
    double njk[3] = {nx, ny, nz};

    double MAGeDOTn = sqrt(pow(e[0] * njk[0], 2.0) + pow(e[1] * njk[1], 2.0) +
                           pow(e[2] * njk[2], 2.0));

    double damp[3] = {alpha / MAGeDOTn * njk[0], alpha / MAGeDOTn * njk[1],
                      alpha / MAGeDOTn * njk[2]};
    double xcim1[3] = {Faces(0) - cellsL(0), Faces(1) - cellsL(1),
                       Faces(2) - cellsL(2)};
    double xci[3] = {Faces(0) - cellsR(0), Faces(1) - cellsR(1),
                     Faces(2) - cellsR(2)};

    // Face derivatives = consistent + damping
    double wDOTxc = (gradsL(1, 0) * xcim1[0] + gradsL(1, 1) * xcim1[1] +
                     gradsL(1, 2) * xcim1[2]);
    double wL = qL(1) + wDOTxc;
    wDOTxc =
        (gradsR(1, 0) * xci[0] + gradsR(1, 1) * xci[1] + gradsR(1, 2) * xci[2]);
    double wR = qR(1) + wDOTxc;
    double dudx = 0.5 * (gradsR(1, 0) + gradsL(1, 0)) + damp[0] * (wR - wL);
    double dudy = 0.5 * (gradsR(1, 1) + gradsL(1, 1)) + damp[1] * (wR - wL);
    double dudz = 0.5 * (gradsR(1, 2) + gradsL(1, 2)) + damp[2] * (wR - wL);

    wDOTxc = (gradsL(2, 0) * xcim1[0] + gradsL(2, 1) * xcim1[1] +
              gradsL(2, 2) * xcim1[2]);
    wL = qL(2) + wDOTxc;
    wDOTxc =
        (gradsR(2, 0) * xci[0] + gradsR(2, 1) * xci[1] + gradsR(2, 2) * xci[2]);
    wR = qR(2) + wDOTxc;
    double dvdx = 0.5 * (gradsR(2, 0) + gradsL(2, 0)) + damp[0] * (wR - wL);
    double dvdy = 0.5 * (gradsR(2, 1) + gradsL(2, 1)) + damp[1] * (wR - wL);
    double dvdz = 0.5 * (gradsR(2, 2) + gradsL(2, 2)) + damp[2] * (wR - wL);

    wDOTxc = (gradsL(3, 0) * xcim1[0] + gradsL(3, 1) * xcim1[1] +
              gradsL(3, 2) * xcim1[2]);
    wL = qL(3) + wDOTxc;
    wDOTxc =
        (gradsR(3, 0) * xci[0] + gradsR(3, 1) * xci[1] + gradsR(3, 2) * xci[2]);
    wR = qR(3) + wDOTxc;
    double dwdx = 0.5 * (gradsR(3, 0) + gradsL(3, 0)) + damp[0] * (wR - wL);
    double dwdy = 0.5 * (gradsR(3, 1) + gradsL(3, 1)) + damp[1] * (wR - wL);
    double dwdz = 0.5 * (gradsR(3, 2) + gradsL(3, 2)) + damp[2] * (wR - wL);

    double div = dudx + dvdy + dwdz;

    // x momentum
    double txx = -2.0 * mu * dudx - lambda * div;
    double txy = -mu * (dvdx + dudy);
    double txz = -mu * (dwdx + dudz);

    F(1) += txx * A(0) + txy * A(1) + txz * A(2);

    // y momentum
    double &tyx = txy;
    double tyy = -2.0 * mu * dvdy - lambda * div;
    double tyz = -mu * (dwdy + dvdz);

    F(2) += tyx * A(0) + tyy * A(1) + tyz * A(2);

    // z momentum
    double &tzx = txz;
    double &tzy = tyz;
    double tzz = -2.0 * mu * dwdz - lambda * div;

    F(3) += tzx * A(0) + tzy * A(1) + tzz * A(2);

    // energy
    //   heat conduction
    wDOTxc = (gradsL(4, 0) * xcim1[0] + gradsL(4, 1) * xcim1[1] +
              gradsL(4, 2) * xcim1[2]);
    wL = qL(4) + wDOTxc;
    wDOTxc =
        (gradsR(4, 0) * xci[0] + gradsR(4, 1) * xci[1] + gradsR(4, 2) * xci[2]);
    wR = qR(4) + wDOTxc;
    double dTdx = 0.5 * (gradsR(4, 0) + gradsL(4, 0)) + damp[0] * (wR - wL);
    double dTdy = 0.5 * (gradsR(4, 1) + gradsL(4, 1)) + damp[1] * (wR - wL);
    double dTdz = 0.5 * (gradsR(4, 2) + gradsL(4, 2)) + damp[2] * (wR - wL);

    double heatFlux = -kappa * (dTdx * A(0) + dTdy * A(1) + dTdz * A(2));

    // flow work
    // Compute face normal volume flux vector
    double uf = 0.5 * (qR(1) + qL(1));
    double vf = 0.5 * (qR(2) + qL(2));
    double wf = 0.5 * (qR(3) + qL(3));

    F(4) += -(uf * txx + vf * txy + wf * txz) * A(0) -
            (uf * tyx + vf * tyy + wf * tyz) * A(1) -
            (uf * tzx + vf * tzy + wf * tzz) * A(2) + heatFlux;

    // Species
    double Dk, Vc = 0.0;
    double gradYns = 0.0;
    double rho = 0.5 * (QR(0) + QL(0));
    // Compute the species flux and correction term \sum(k=1,ns)
    // Dk*gradYk
    for (int n = 0; n < ne - 5; n++) {
      Dk = 0.5 * (qtR(2 + n) + qtL(2 + n));
      wDOTxc = (gradsL(5 + n, 0) * xcim1[0] + gradsL(5 + n, 1) * xcim1[1] +
                gradsL(5 + n, 2) * xcim1[2]);
      wL = qL(5 + n) + wDOTxc;
      wDOTxc = (gradsR(5 + n, 0) * xci[0] + gradsR(5 + n, 1) * xci[1] +
                gradsR(5 + n, 2) * xci[2]);
      wR = qR(5 + n) + wDOTxc;
      double dYdx =
          0.5 * (gradsR(5 + n, 0) + gradsL(5 + n, 0)) + damp[0] * (wR - wL);
      double dYdy =
          0.5 * (gradsR(5 + n, 1) + gradsL(5 + n, 1)) + damp[1] * (wR - wL);
      double dYdz =
          0.5 * (gradsR(5 + n, 2) + gradsL(5 + n, 2)) + damp[2] * (wR - wL);

      double gradYk = (dYdx * A(0) + dYdy * A(1) + dYdz * A(2));
      gradYns -= gradYk;
      Vc += Dk * gradYk;
      // the species flux before its correction, and its enthalpy with
      // it
      double Jk = -rho * Dk * gradYk;
      double hk = 0.5 * (qhR(5 + n) + qhL(5 + n));
      F(5 + n) += Jk;
      F(4) += Jk * hk;
    }
    // Apply n=ns species to correction
    Dk = 0.5 * (qtR(2 + ne - 5) + qtL(2 + ne - 5));
    Vc += Dk * gradYns;

    // Apply correction and species thermal flux
    double hk;
    double Yns = 1.0;
    for (int n = 0; n < ne - 5; n++) {
      double Yk = 0.5 * (qR(5 + n) + qL(5 + n));
      Yns -= Yk;
      // the correction, and its enthalpy with it
      double corr = Yk * rho * Vc;
      hk = 0.5 * (qhR(5 + n) + qhL(5 + n));
      F(5 + n) += corr;
      F(4) += corr * hk;
    }
    // Apply the n=ns species to thermal diffusion
    // If we are single species, this should be zero,
    // so Yns will = 1.0, but gradYns will == 0.0 and
    // Vc will == 0.0 (bc gradYns==0.0) from above
    // so this is zero for ns=1
    hk = 0.5 * (qhR(ne) + qhL(ne));
    F(4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
  }
};

PG_ABI void pgAlphaDampingFlux(const alphaDampingFlux &k, const pgTiling &t) {
  forCells("face visc fluxes", t, k);
}
