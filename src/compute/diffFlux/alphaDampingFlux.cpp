#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

// References
//
// Robust numerical fluxes for unrealizable states
//
// Hiroaki Nishikawa
// Journal of Computational Physics
// 408 (2020)

static void computeFlux(const in4 &Q, const in4 &cells, const in5 &grads,
                        const in4 &q, const in4 &qh, const in4 &qt,
                        const pgDims &d, const out4 &iF, const in4 &iS,
                        const in4 &iFaces, const int iMod, const int jMod,
                        const int kMod) {

  // Stokes hypothesis
  double const bulkVisc = 0.0;

  // damping parameter
  double const alpha = 1.0;

  const int ni = d.ni, nj = d.nj, nk = d.nk;
  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});

  Kokkos::parallel_for(
      "face visc fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        double mu =
            0.5 * (qt(i, j, k, 0) + qt(i - iMod, j - jMod, k - kMod, 0));
        double kappa =
            0.5 * (qt(i, j, k, 1) + qt(i - iMod, j - jMod, k - kMod, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // continuity
        iF(i, j, k, 0) = 0.0;

        // Geometric terms
        double e[3] = {
            cells(i, j, k, 0) - cells(i - iMod, j - jMod, k - kMod, 0),
            cells(i, j, k, 1) - cells(i - iMod, j - jMod, k - kMod, 1),
            cells(i, j, k, 2) - cells(i - iMod, j - jMod, k - kMod, 2)};
        double njk[3] = {nx, ny, nz};

        double MAGeDOTn =
            sqrt(pow(e[0] * njk[0], 2.0) + pow(e[1] * njk[1], 2.0) +
                 pow(e[2] * njk[2], 2.0));

        double damp[3] = {alpha / MAGeDOTn * njk[0], alpha / MAGeDOTn * njk[1],
                          alpha / MAGeDOTn * njk[2]};
        double xcim1[3] = {
            iFaces(i, j, k, 0) - cells(i - iMod, j - jMod, k - kMod, 0),
            iFaces(i, j, k, 1) - cells(i - iMod, j - jMod, k - kMod, 1),
            iFaces(i, j, k, 2) - cells(i - iMod, j - jMod, k - kMod, 2)};
        double xci[3] = {iFaces(i, j, k, 0) - cells(i, j, k, 0),
                         iFaces(i, j, k, 1) - cells(i, j, k, 1),
                         iFaces(i, j, k, 2) - cells(i, j, k, 2)};

        // Face derivatives = consistent + damping
        double wDOTxc = (grads(i - iMod, j - jMod, k - kMod, 1, 0) * xcim1[0] +
                         grads(i - iMod, j - jMod, k - kMod, 1, 1) * xcim1[1] +
                         grads(i - iMod, j - jMod, k - kMod, 1, 2) * xcim1[2]);
        double wL = q(i - iMod, j - jMod, k - kMod, 1) + wDOTxc;
        wDOTxc =
            (grads(i, j, k, 1, 0) * xci[0] + grads(i, j, k, 1, 1) * xci[1] +
             grads(i, j, k, 1, 2) * xci[2]);
        double wR = q(i, j, k, 1) + wDOTxc;
        double dudx = 0.5 * (grads(i, j, k, 1, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 1, 0)) +
                      damp[0] * (wR - wL);
        double dudy = 0.5 * (grads(i, j, k, 1, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 1, 1)) +
                      damp[1] * (wR - wL);
        double dudz = 0.5 * (grads(i, j, k, 1, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 1, 2)) +
                      damp[2] * (wR - wL);

        wDOTxc = (grads(i - iMod, j - jMod, k - kMod, 2, 0) * xcim1[0] +
                  grads(i - iMod, j - jMod, k - kMod, 2, 1) * xcim1[1] +
                  grads(i - iMod, j - jMod, k - kMod, 2, 2) * xcim1[2]);
        wL = q(i - iMod, j - jMod, k - kMod, 2) + wDOTxc;
        wDOTxc =
            (grads(i, j, k, 2, 0) * xci[0] + grads(i, j, k, 2, 1) * xci[1] +
             grads(i, j, k, 2, 2) * xci[2]);
        wR = q(i, j, k, 2) + wDOTxc;
        double dvdx = 0.5 * (grads(i, j, k, 2, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 2, 0)) +
                      damp[0] * (wR - wL);
        double dvdy = 0.5 * (grads(i, j, k, 2, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 2, 1)) +
                      damp[1] * (wR - wL);
        double dvdz = 0.5 * (grads(i, j, k, 2, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 2, 2)) +
                      damp[2] * (wR - wL);

        wDOTxc = (grads(i - iMod, j - jMod, k - kMod, 3, 0) * xcim1[0] +
                  grads(i - iMod, j - jMod, k - kMod, 3, 1) * xcim1[1] +
                  grads(i - iMod, j - jMod, k - kMod, 3, 2) * xcim1[2]);
        wL = q(i - iMod, j - jMod, k - kMod, 3) + wDOTxc;
        wDOTxc =
            (grads(i, j, k, 3, 0) * xci[0] + grads(i, j, k, 3, 1) * xci[1] +
             grads(i, j, k, 3, 2) * xci[2]);
        wR = q(i, j, k, 3) + wDOTxc;
        double dwdx = 0.5 * (grads(i, j, k, 3, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 3, 0)) +
                      damp[0] * (wR - wL);
        double dwdy = 0.5 * (grads(i, j, k, 3, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 3, 1)) +
                      damp[1] * (wR - wL);
        double dwdz = 0.5 * (grads(i, j, k, 3, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 3, 2)) +
                      damp[2] * (wR - wL);

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        iF(i, j, k, 1) =
            txx * iS(i, j, k, 0) + txy * iS(i, j, k, 1) + txz * iS(i, j, k, 2);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        iF(i, j, k, 2) =
            tyx * iS(i, j, k, 0) + tyy * iS(i, j, k, 1) + tyz * iS(i, j, k, 2);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        iF(i, j, k, 3) =
            tzx * iS(i, j, k, 0) + tzy * iS(i, j, k, 1) + tzz * iS(i, j, k, 2);

        // energy
        //   heat conduction
        wDOTxc = (grads(i - iMod, j - jMod, k - kMod, 4, 0) * xcim1[0] +
                  grads(i - iMod, j - jMod, k - kMod, 4, 1) * xcim1[1] +
                  grads(i - iMod, j - jMod, k - kMod, 4, 2) * xcim1[2]);
        wL = q(i - iMod, j - jMod, k - kMod, 4) + wDOTxc;
        wDOTxc =
            (grads(i, j, k, 4, 0) * xci[0] + grads(i, j, k, 4, 1) * xci[1] +
             grads(i, j, k, 4, 2) * xci[2]);
        wR = q(i, j, k, 4) + wDOTxc;
        double dTdx = 0.5 * (grads(i, j, k, 4, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 4, 0)) +
                      damp[0] * (wR - wL);
        double dTdy = 0.5 * (grads(i, j, k, 4, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 4, 1)) +
                      damp[1] * (wR - wL);
        double dTdz = 0.5 * (grads(i, j, k, 4, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 4, 2)) +
                      damp[2] * (wR - wL);

        double heatFlux =
            -kappa * (dTdx * iS(i, j, k, 0) + dTdy * iS(i, j, k, 1) +
                      dTdz * iS(i, j, k, 2));

        // flow work
        // Compute face normal volume flux vector
        double uf = 0.5 * (q(i, j, k, 1) + q(i - iMod, j - jMod, k - kMod, 1));
        double vf = 0.5 * (q(i, j, k, 2) + q(i - iMod, j - jMod, k - kMod, 2));
        double wf = 0.5 * (q(i, j, k, 3) + q(i - iMod, j - jMod, k - kMod, 3));

        iF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * iS(i, j, k, 0) -
                         (uf * tyx + vf * tyy + wf * tyz) * iS(i, j, k, 1) -
                         (uf * tzx + vf * tzy + wf * tzz) * iS(i, j, k, 2) +
                         heatFlux;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (Q(i, j, k, 0) + Q(i - iMod, j - jMod, k - kMod, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < ne - 5; n++) {
          Dk = 0.5 *
               (qt(i, j, k, 2 + n) + qt(i - iMod, j - jMod, k - kMod, 2 + n));
          wDOTxc = (grads(i - iMod, j - jMod, k - kMod, 5 + n, 0) * xcim1[0] +
                    grads(i - iMod, j - jMod, k - kMod, 5 + n, 1) * xcim1[1] +
                    grads(i - iMod, j - jMod, k - kMod, 5 + n, 2) * xcim1[2]);
          wL = q(i - iMod, j - jMod, k - kMod, 5 + n) + wDOTxc;
          wDOTxc = (grads(i, j, k, 5 + n, 0) * xci[0] +
                    grads(i, j, k, 5 + n, 1) * xci[1] +
                    grads(i, j, k, 5 + n, 2) * xci[2]);
          wR = q(i, j, k, 5 + n) + wDOTxc;
          double dYdx = 0.5 * (grads(i, j, k, 5 + n, 0) +
                               grads(i - iMod, j - jMod, k - kMod, 5 + n, 0)) +
                        damp[0] * (wR - wL);
          double dYdy = 0.5 * (grads(i, j, k, 5 + n, 1) +
                               grads(i - iMod, j - jMod, k - kMod, 5 + n, 1)) +
                        damp[1] * (wR - wL);
          double dYdz = 0.5 * (grads(i, j, k, 5 + n, 2) +
                               grads(i - iMod, j - jMod, k - kMod, 5 + n, 2)) +
                        damp[2] * (wR - wL);

          double gradYk = (dYdx * iS(i, j, k, 0) + dYdy * iS(i, j, k, 1) +
                           dYdz * iS(i, j, k, 2));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          iF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 * (qt(i, j, k, 2 + ne - 5) +
                    qt(i - iMod, j - jMod, k - kMod, 2 + ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double hk;
        double Yns = 1.0;
        for (int n = 0; n < ne - 5; n++) {
          double Yk = 0.5 * (q(i, j, k, 5 + n) +
                             q(i - iMod, j - jMod, k - kMod, 5 + n));
          Yns -= Yk;
          iF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 *
               (qh(i, j, k, 5 + n) + qh(i - iMod, j - jMod, k - kMod, 5 + n));
          iF(i, j, k, 4) += iF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        // If we are single species, this should be zero,
        // so Yns will = 1.0, but gradYns will == 0.0 and
        // Vc will == 0.0 (bc gradYns==0.0) from above
        // so this is zero for ns=1
        hk = 0.5 * (qh(i, j, k, ne) + qh(i - iMod, j - jMod, k - kMod, ne));
        iF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });
}

PG_ABI void pgAlphaDampingFlux(int count, pgIn *Q_, pgIn *cells_, pgIn *grads_,
                               pgOut *iF_, pgIn *iFaces_, pgIn *iS_, pgOut *jF_,
                               pgIn *jFaces_, pgIn *jS_, pgOut *kF_,
                               pgIn *kFaces_, pgIn *kS_, pgIn *q_, pgIn *qh_,
                               pgIn *qt_, const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto cells = as4(cells_[e]);
    auto grads = as5(grads_[e]);
    auto iF = as4(iF_[e]);
    auto iFaces = as4(iFaces_[e]);
    auto iS = as4(iS_[e]);
    auto jF = as4(jF_[e]);
    auto jFaces = as4(jFaces_[e]);
    auto jS = as4(jS_[e]);
    auto kF = as4(kF_[e]);
    auto kFaces = as4(kFaces_[e]);
    auto kS = as4(kS_[e]);
    auto q = as4(q_[e]);
    auto qh = as4(qh_[e]);
    auto qt = as4(qt_[e]);

    computeFlux(Q, cells, grads, q, qh, qt, d[e], iF, iS, iFaces, 1, 0, 0);
    computeFlux(Q, cells, grads, q, qh, qt, d[e], jF, jS, jFaces, 0, 1, 0);
    computeFlux(Q, cells, grads, q, qh, qt, d[e], kF, kS, kFaces, 0, 0, 1);
  }
}
