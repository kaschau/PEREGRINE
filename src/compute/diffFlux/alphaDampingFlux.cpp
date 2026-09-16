#include "faces.hpp"
#include "thermo/eos.hpp"

// References
//
// Robust numerical fluxes for unrealizable states
//
// Hiroaki Nishikawa
// Journal of Computational Physics
// 408 (2020)

// grads holds the velocity in slots 0 .. 2, T in 3 and Y(n) in 4 + n; the
// velocity and Y at each side come off its conserved state, the species
// enthalpies from the eos at its temperature.
PG_RANGE(cellFaces)
struct alphaDampingFlux {
  cellCenterL QL, cellsL, gradsL, qL, qhL, qtL;
  cellCenterR QR, cellsR, gradsR, qR, qhR, qtR;
  cellFaceInOut F;
  cellFaceIn A, Faces;
  static constexpr double bulkVisc = 0.0;
  static constexpr double alpha = 1.0;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    double mu = 0.5 * (qtR(0) + qtL(0));
    double kappa = 0.5 * (qtR(1) + qtL(1));
    double lambda = bulkVisc - 2.0 / 3.0 * mu;

    // each side's velocity, off its conserved state
    const double rhoinvL = 1.0 / QL(0), rhoinvR = 1.0 / QR(0);
    const double uL = QL(1) * rhoinvL, vL = QL(2) * rhoinvL,
                 wL = QL(3) * rhoinvL;
    const double uR = QR(1) * rhoinvR, vR = QR(2) * rhoinvR,
                 wR = QR(3) * rhoinvR;

    // Compute face normal gradient with alpha damping
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

    // the face gradient of slot l of grads: the sides' average, damped
    // toward the difference of the values each side extrapolates to the face
    const auto faceGrad = [&](const int l, const double phiL, const double phiR,
                              double *g) {
      double wDOTxc = (gradsL(l, 0) * xcim1[0] + gradsL(l, 1) * xcim1[1] +
                       gradsL(l, 2) * xcim1[2]);
      const double wl = phiL + wDOTxc;
      wDOTxc = (gradsR(l, 0) * xci[0] + gradsR(l, 1) * xci[1] +
                gradsR(l, 2) * xci[2]);
      const double wr = phiR + wDOTxc;
      for (int d = 0; d < 3; d++) {
        g[d] = 0.5 * (gradsR(l, d) + gradsL(l, d)) + damp[d] * (wr - wl);
      }
    };

    double du[3], dv[3], dw[3];
    faceGrad(0, uL, uR, du);
    faceGrad(1, vL, vR, dv);
    faceGrad(2, wL, wR, dw);
    const double dudx = du[0], dudy = du[1], dudz = du[2];
    const double dvdx = dv[0], dvdy = dv[1], dvdz = dv[2];
    const double dwdx = dw[0], dwdy = dw[1], dwdz = dw[2];

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
    double dT[3];
    faceGrad(3, qL(1), qR(1), dT);
    double heatFlux = -kappa * (dT[0] * A(0) + dT[1] * A(1) + dT[2] * A(2));

    double uf = 0.5 * (uR + uL);
    double vf = 0.5 * (vR + vL);
    double wf = 0.5 * (wR + wL);

    F(4) += -(uf * txx + vf * txy + wf * txz) * A(0) -
            (uf * tyx + vf * tyy + wf * tyz) * A(1) -
            (uf * tzx + vf * tzy + wf * tzz) * A(2) + heatFlux;

    // species, with the correction that keeps the diffusion fluxes summing
    // to zero; the enthalpies from each side's eos
    const auto hL = eos::enthalpies(qL(1), qhL);
    const auto hR = eos::enthalpies(qR(1), qhR);
    double Dk, Vc = 0.0;
    double gradYns = 0.0;
    double rho = 0.5 * (QR(0) + QL(0));
    for (int n = 0; n < ne - 5; n++) {
      Dk = 0.5 * (qtR(2 + n) + qtL(2 + n));
      double dY[3];
      faceGrad(4 + n, QL(5 + n) * rhoinvL, QR(5 + n) * rhoinvR, dY);
      double gradYk = (dY[0] * A(0) + dY[1] * A(1) + dY[2] * A(2));
      gradYns -= gradYk;
      Vc += Dk * gradYk;
      double Jk = -rho * Dk * gradYk;
      double hk = 0.5 * (hR(n) + hL(n));
      F(5 + n) += Jk;
      F(4) += Jk * hk;
    }
    Dk = 0.5 * (qtR(2 + ne - 5) + qtL(2 + ne - 5));
    Vc += Dk * gradYns;

    double hk;
    double Yns = 1.0;
    for (int n = 0; n < ne - 5; n++) {
      double Yk = 0.5 * (QR(5 + n) * rhoinvR + QL(5 + n) * rhoinvL);
      Yns -= Yk;
      double corr = Yk * rho * Vc;
      hk = 0.5 * (hR(n) + hL(n));
      F(5 + n) += corr;
      F(4) += corr * hk;
    }
    hk = 0.5 * (hR(ns - 1) + hL(ns - 1));
    F(4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
  }
};

PG_ABI void pgAlphaDampingFlux(const alphaDampingFlux &k, const pgTiling &t) {
  forCells("face visc fluxes", t, k);
}
