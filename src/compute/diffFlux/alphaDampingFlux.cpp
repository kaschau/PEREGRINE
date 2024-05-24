#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

// References
//
// Robust numerical fluxes for unrealizable states
//
// Hiroaki Nishikawa
// Journal of Computational Physics
// 408 (2020)

static void computeFlux(const block_ &b, fourDview &F, const threeDview &isx,
                        const threeDview &isy, const threeDview &isz,
                        const threeDview &inx, const threeDview &iny,
                        const threeDview &inz, const threeDview &ixc,
                        const threeDview &iyc, const threeDview &izc,
                        const int iMod, const int jMod, const int kMod) {

  // Stokes hypothesis
  double const bulkVisc = 0.0;

  // damping parameter
  double const alpha = 1.0;

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "face visc fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double mu =
            0.5 * (b.qt(i, j, k, 0) + b.qt(i - iMod, j - jMod, k - kMod, 0));
        double kappa =
            0.5 * (b.qt(i, j, k, 1) + b.qt(i - iMod, j - jMod, k - kMod, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // continuity
        F(i, j, k, 0) = 0.0;

        // Geometric terms
        double e[3] = {b.xc(i, j, k) - b.xc(i - iMod, j - jMod, k - kMod),
                       b.yc(i, j, k) - b.yc(i - iMod, j - jMod, k - kMod),
                       b.zc(i, j, k) - b.zc(i - iMod, j - jMod, k - kMod)};
        double njk[3] = {inx(i, j, k), iny(i, j, k), inz(i, j, k)};

        double MAGeDOTn =
            sqrt(pow(e[0] * njk[0], 2.0) + pow(e[1] * njk[1], 2.0) +
                 pow(e[2] * njk[2], 2.0));

        double damp[3] = {alpha / MAGeDOTn * njk[0], alpha / MAGeDOTn * njk[1],
                          alpha / MAGeDOTn * njk[2]};
        double xcim1[3] = {ixc(i, j, k) - b.xc(i - iMod, j - jMod, k - kMod),
                           iyc(i, j, k) - b.yc(i - iMod, j - jMod, k - kMod),
                           izc(i, j, k) - b.zc(i - iMod, j - jMod, k - kMod)};
        double xci[3] = {ixc(i, j, k) - b.xc(i, j, k),
                         iyc(i, j, k) - b.yc(i, j, k),
                         izc(i, j, k) - b.zc(i, j, k)};

        // Face derivatives = consistent + damping
        double wDOTxc = (b.dqdx(i - iMod, j - jMod, k - kMod, 1) * xcim1[0] +
                         b.dqdy(i - iMod, j - jMod, k - kMod, 1) * xcim1[1] +
                         b.dqdz(i - iMod, j - jMod, k - kMod, 1) * xcim1[2]);
        double wL = b.q(i - iMod, j - jMod, k - kMod, 1) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 1) * xci[0] + b.dqdy(i, j, k, 1) * xci[1] +
                  b.dqdz(i, j, k, 1) * xci[2]);
        double wR = b.q(i, j, k, 1) + wDOTxc;
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) +
                             b.dqdx(i - iMod, j - jMod, k - kMod, 1)) +
                      damp[0] * (wR - wL);
        double dudy = 0.5 * (b.dqdy(i, j, k, 1) +
                             b.dqdy(i - iMod, j - jMod, k - kMod, 1)) +
                      damp[1] * (wR - wL);
        double dudz = 0.5 * (b.dqdz(i, j, k, 1) +
                             b.dqdz(i - iMod, j - jMod, k - kMod, 1)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i - iMod, j - jMod, k - kMod, 2) * xcim1[0] +
                  b.dqdy(i - iMod, j - jMod, k - kMod, 2) * xcim1[1] +
                  b.dqdz(i - iMod, j - jMod, k - kMod, 2) * xcim1[2]);
        wL = b.q(i - iMod, j - jMod, k - kMod, 2) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 2) * xci[0] + b.dqdy(i, j, k, 2) * xci[1] +
                  b.dqdz(i, j, k, 2) * xci[2]);
        wR = b.q(i, j, k, 2) + wDOTxc;
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) +
                             b.dqdx(i - iMod, j - jMod, k - kMod, 2)) +
                      damp[0] * (wR - wL);
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) +
                             b.dqdy(i - iMod, j - jMod, k - kMod, 2)) +
                      damp[1] * (wR - wL);
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) +
                             b.dqdz(i - iMod, j - jMod, k - kMod, 2)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i - iMod, j - jMod, k - kMod, 3) * xcim1[0] +
                  b.dqdy(i - iMod, j - jMod, k - kMod, 3) * xcim1[1] +
                  b.dqdz(i - iMod, j - jMod, k - kMod, 3) * xcim1[2]);
        wL = b.q(i - iMod, j - jMod, k - kMod, 3) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 3) * xci[0] + b.dqdy(i, j, k, 3) * xci[1] +
                  b.dqdz(i, j, k, 3) * xci[2]);
        wR = b.q(i, j, k, 3) + wDOTxc;
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) +
                             b.dqdx(i - iMod, j - jMod, k - kMod, 3)) +
                      damp[0] * (wR - wL);
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) +
                             b.dqdy(i - iMod, j - jMod, k - kMod, 3)) +
                      damp[1] * (wR - wL);
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) +
                             b.dqdz(i - iMod, j - jMod, k - kMod, 3)) +
                      damp[2] * (wR - wL);

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        F(i, j, k, 1) =
            txx * isx(i, j, k) + txy * isy(i, j, k) + txz * isz(i, j, k);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        F(i, j, k, 2) =
            tyx * isx(i, j, k) + tyy * isy(i, j, k) + tyz * isz(i, j, k);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        F(i, j, k, 3) =
            tzx * isx(i, j, k) + tzy * isy(i, j, k) + tzz * isz(i, j, k);

        // energy
        //   heat conduction
        wDOTxc = (b.dqdx(i - iMod, j - jMod, k - kMod, 4) * xcim1[0] +
                  b.dqdy(i - iMod, j - jMod, k - kMod, 4) * xcim1[1] +
                  b.dqdz(i - iMod, j - jMod, k - kMod, 4) * xcim1[2]);
        wL = b.q(i - iMod, j - jMod, k - kMod, 4) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 4) * xci[0] + b.dqdy(i, j, k, 4) * xci[1] +
                  b.dqdz(i, j, k, 4) * xci[2]);
        wR = b.q(i, j, k, 4) + wDOTxc;
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) +
                             b.dqdx(i - iMod, j - jMod, k - kMod, 4)) +
                      damp[0] * (wR - wL);
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) +
                             b.dqdy(i - iMod, j - jMod, k - kMod, 4)) +
                      damp[1] * (wR - wL);
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) +
                             b.dqdz(i - iMod, j - jMod, k - kMod, 4)) +
                      damp[2] * (wR - wL);

        double q = -kappa * (dTdx * isx(i, j, k) + dTdy * isy(i, j, k) +
                             dTdz * isz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        double uf =
            0.5 * (b.q(i, j, k, 1) + b.q(i - iMod, j - jMod, k - kMod, 1));
        double vf =
            0.5 * (b.q(i, j, k, 2) + b.q(i - iMod, j - jMod, k - kMod, 2));
        double wf =
            0.5 * (b.q(i, j, k, 3) + b.q(i - iMod, j - jMod, k - kMod, 3));

        F(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * isx(i, j, k) -
                        (uf * tyx + vf * tyy + wf * tyz) * isy(i, j, k) -
                        (uf * tzx + vf * tzy + wf * tzz) * isz(i, j, k) + q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho =
            0.5 * (b.Q(i, j, k, 0) + b.Q(i - iMod, j - jMod, k - kMod, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) +
                      b.qt(i - iMod, j - jMod, k - kMod, 2 + n));
          wDOTxc = (b.dqdx(i - iMod, j - jMod, k - kMod, 5 + n) * xcim1[0] +
                    b.dqdy(i - iMod, j - jMod, k - kMod, 5 + n) * xcim1[1] +
                    b.dqdz(i - iMod, j - jMod, k - kMod, 5 + n) * xcim1[2]);
          wL = b.q(i - iMod, j - jMod, k - kMod, 5 + n) + wDOTxc;
          wDOTxc = (b.dqdx(i, j, k, 5 + n) * xci[0] +
                    b.dqdy(i, j, k, 5 + n) * xci[1] +
                    b.dqdz(i, j, k, 5 + n) * xci[2]);
          wR = b.q(i, j, k, 5 + n) + wDOTxc;
          double dYdx = 0.5 * (b.dqdx(i, j, k, 5 + n) +
                               b.dqdx(i - iMod, j - jMod, k - kMod, 5 + n)) +
                        damp[0] * (wR - wL);
          double dYdy = 0.5 * (b.dqdy(i, j, k, 5 + n) +
                               b.dqdy(i - iMod, j - jMod, k - kMod, 5 + n)) +
                        damp[1] * (wR - wL);
          double dYdz = 0.5 * (b.dqdz(i, j, k, 5 + n) +
                               b.dqdz(i - iMod, j - jMod, k - kMod, 5 + n)) +
                        damp[2] * (wR - wL);

          double gradYk =
              (dYdx * isx(i, j, k) + dYdy * isy(i, j, k) + dYdz * isz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          F(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 * (b.qt(i, j, k, 2 + b.ne - 5) +
                    b.qt(i - iMod, j - jMod, k - kMod, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 *
               (b.q(i, j, k, 5 + n) + b.q(i - iMod, j - jMod, k - kMod, 5 + n));
          Yns -= Yk;
          F(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) +
                      b.qh(i - iMod, j - jMod, k - kMod, 5 + n));
          F(i, j, k, 4) += F(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 *
             (b.qh(i, j, k, b.ne) + b.qh(i - iMod, j - jMod, k - kMod, b.ne));
        F(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });
}

void alphaDampingFlux(block_ &b) {

  computeFlux(b, b.iF, b.isx, b.isy, b.isz, b.inx, b.iny, b.inz, b.ixc, b.iyc,
              b.izc, 1, 0, 0);
  computeFlux(b, b.jF, b.jsx, b.jsy, b.jsz, b.jnx, b.jny, b.jnz, b.jxc, b.jyc,
              b.jzc, 0, 1, 0);
  computeFlux(b, b.kF, b.ksx, b.ksy, b.ksz, b.knx, b.kny, b.knz, b.kxc, b.kyc,
              b.kzc, 0, 0, 1);
}
