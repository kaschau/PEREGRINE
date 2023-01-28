#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"

void alphaDampingFlux(block_ &b) {

  // Stokes hypothesis
  double const bulkVisc = 0.0;

  // damping parameter
  double const alpha = 1.0;

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "i face visc fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i - 1, j, k, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i - 1, j, k, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // continuity
        b.iF(i, j, k, 0) = 0.0;

        // Geometric terms
        double e[3] = {b.xc(i, j, k) - b.xc(i - 1, j, k),
                       b.yc(i, j, k) - b.yc(i - 1, j, k),
                       b.zc(i, j, k) - b.zc(i - 1, j, k)};
        double njk[3] = {b.inx(i, j, k), b.iny(i, j, k), b.inz(i, j, k)};

        double MAGeDOTn =
            sqrt(pow(e[0] * njk[0], 2.0) + pow(e[1] * njk[1], 2.0) +
                 pow(e[2] * njk[2], 2.0));

        double damp[3] = {alpha / MAGeDOTn * njk[0], alpha / MAGeDOTn * njk[1],
                          alpha / MAGeDOTn * njk[2]};
        double xcim1[3] = {b.ixc(i, j, k) - b.xc(i - 1, j, k),
                           b.iyc(i, j, k) - b.yc(i - 1, j, k),
                           b.izc(i, j, k) - b.zc(i - 1, j, k)};
        double xci[3] = {b.ixc(i, j, k) - b.xc(i, j, k),
                         b.iyc(i, j, k) - b.yc(i, j, k),
                         b.izc(i, j, k) - b.zc(i, j, k)};

        double wDOTxc, wL, wR;

        // Face derivatives = consistent + damping
        wDOTxc = (b.dqdx(i - 1, j, k, 1) * xcim1[0] +
                  b.dqdy(i - 1, j, k, 1) * xcim1[1] +
                  b.dqdz(i - 1, j, k, 1) * xcim1[2]);
        wL = b.q(i - 1, j, k, 1) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 1) * xci[0] + b.dqdy(i, j, k, 1) * xci[1] +
                  b.dqdz(i, j, k, 1) * xci[2]);
        wR = b.q(i, j, k, 1) + wDOTxc;
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) + b.dqdx(i - 1, j, k, 1)) +
                      damp[0] * (wR - wL);
        double dudy = 0.5 * (b.dqdy(i, j, k, 1) + b.dqdy(i - 1, j, k, 1)) +
                      damp[1] * (wR - wL);
        double dudz = 0.5 * (b.dqdz(i, j, k, 1) + b.dqdz(i - 1, j, k, 1)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i - 1, j, k, 2) * xcim1[0] +
                  b.dqdy(i - 1, j, k, 2) * xcim1[1] +
                  b.dqdz(i - 1, j, k, 2) * xcim1[2]);
        wL = b.q(i - 1, j, k, 2) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 2) * xci[0] + b.dqdy(i, j, k, 2) * xci[1] +
                  b.dqdz(i, j, k, 2) * xci[2]);
        wR = b.q(i, j, k, 2) + wDOTxc;
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) + b.dqdx(i - 1, j, k, 2)) +
                      damp[0] * (wR - wL);
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) + b.dqdy(i - 1, j, k, 2)) +
                      damp[1] * (wR - wL);
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) + b.dqdz(i - 1, j, k, 2)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i - 1, j, k, 3) * xcim1[0] +
                  b.dqdy(i - 1, j, k, 3) * xcim1[1] +
                  b.dqdz(i - 1, j, k, 3) * xcim1[2]);
        wL = b.q(i - 1, j, k, 3) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 3) * xci[0] + b.dqdy(i, j, k, 3) * xci[1] +
                  b.dqdz(i, j, k, 3) * xci[2]);
        wR = b.q(i, j, k, 3) + wDOTxc;
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) + b.dqdx(i - 1, j, k, 3)) +
                      damp[0] * (wR - wL);
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) + b.dqdy(i - 1, j, k, 3)) +
                      damp[1] * (wR - wL);
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) + b.dqdz(i - 1, j, k, 3)) +
                      damp[2] * (wR - wL);

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        b.iF(i, j, k, 1) =
            txx * b.isx(i, j, k) + txy * b.isy(i, j, k) + txz * b.isz(i, j, k);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        b.iF(i, j, k, 2) =
            tyx * b.isx(i, j, k) + tyy * b.isy(i, j, k) + tyz * b.isz(i, j, k);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        b.iF(i, j, k, 3) =
            tzx * b.isx(i, j, k) + tzy * b.isy(i, j, k) + tzz * b.isz(i, j, k);

        // energy
        //   heat conduction
        wDOTxc = (b.dqdx(i - 1, j, k, 4) * xcim1[0] +
                  b.dqdy(i - 1, j, k, 4) * xcim1[1] +
                  b.dqdz(i - 1, j, k, 4) * xcim1[2]);
        wL = b.q(i - 1, j, k, 4) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 4) * xci[0] + b.dqdy(i, j, k, 4) * xci[1] +
                  b.dqdz(i, j, k, 4) * xci[2]);
        wR = b.q(i, j, k, 4) + wDOTxc;
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) + b.dqdx(i - 1, j, k, 4)) +
                      damp[0] * (wR - wL);
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) + b.dqdy(i - 1, j, k, 4)) +
                      damp[1] * (wR - wL);
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) + b.dqdz(i - 1, j, k, 4)) +
                      damp[2] * (wR - wL);

        double q = -kappa * (dTdx * b.isx(i, j, k) + dTdy * b.isy(i, j, k) +
                             dTdz * b.isz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        b.iF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * b.isx(i, j, k) -
                           (uf * tyx + vf * tyy + wf * tyz) * b.isy(i, j, k) -
                           (uf * tzx + vf * tzy + wf * tzz) * b.isz(i, j, k) +
                           q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i - 1, j, k, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i - 1, j, k, 2 + n));
          wDOTxc = (b.dqdx(i - 1, j, k, 5 + n) * xcim1[0] +
                    b.dqdy(i - 1, j, k, 5 + n) * xcim1[1] +
                    b.dqdz(i - 1, j, k, 5 + n) * xcim1[2]);
          wL = b.q(i - 1, j, k, 5 + n) + wDOTxc;
          wDOTxc = (b.dqdx(i, j, k, 5 + n) * xci[0] +
                    b.dqdy(i, j, k, 5 + n) * xci[1] +
                    b.dqdz(i, j, k, 5 + n) * xci[2]);
          wR = b.q(i, j, k, 5 + n) + wDOTxc;
          double dYdx =
              0.5 * (b.dqdx(i, j, k, 5 + n) + b.dqdx(i - 1, j, k, 5 + n)) +
              damp[0] * (wR - wL);
          double dYdy =
              0.5 * (b.dqdy(i, j, k, 5 + n) + b.dqdy(i - 1, j, k, 5 + n)) +
              damp[1] * (wR - wL);
          double dYdz =
              0.5 * (b.dqdz(i, j, k, 5 + n) + b.dqdz(i - 1, j, k, 5 + n)) +
              damp[2] * (wR - wL);

          double gradYk = (dYdx * b.isx(i, j, k) + dYdy * b.isy(i, j, k) +
                           dYdz * b.isz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          b.iF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 *
             (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i - 1, j, k, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i - 1, j, k, 5 + n));
          Yns -= Yk;
          b.iF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i - 1, j, k, 5 + n));
          b.iF(i, j, k, 4) += b.iF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i - 1, j, k, b.ne));
        b.iF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "j face visc fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i, j - 1, k, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i, j - 1, k, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // continuity
        b.jF(i, j, k, 0) = 0.0;

        // Geometric terms
        double e[3] = {b.xc(i, j, k) - b.xc(i, j - 1, k),
                       b.yc(i, j, k) - b.yc(i, j - 1, k),
                       b.zc(i, j, k) - b.zc(i, j - 1, k)};
        double njk[3] = {b.jnx(i, j, k), b.jny(i, j, k), b.jnz(i, j, k)};

        double MAGeDOTn =
            sqrt(pow(e[0] * njk[0], 2.0) + pow(e[1] * njk[1], 2.0) +
                 pow(e[2] * njk[2], 2.0));

        double damp[3] = {alpha / MAGeDOTn * njk[0], alpha / MAGeDOTn * njk[1],
                          alpha / MAGeDOTn * njk[2]};
        double xcim1[3] = {b.jxc(i, j, k) - b.xc(i, j - 1, k),
                           b.jyc(i, j, k) - b.yc(i, j - 1, k),
                           b.jzc(i, j, k) - b.zc(i, j - 1, k)};
        double xci[3] = {b.jxc(i, j, k) - b.xc(i, j, k),
                         b.jyc(i, j, k) - b.yc(i, j, k),
                         b.jzc(i, j, k) - b.zc(i, j, k)};

        double wDOTxc, wL, wR;

        // Face derivatives = consistent + damping
        wDOTxc = (b.dqdx(i, j - 1, k, 1) * xcim1[0] +
                  b.dqdy(i, j - 1, k, 1) * xcim1[1] +
                  b.dqdz(i, j - 1, k, 1) * xcim1[2]);
        wL = b.q(i, j - 1, k, 1) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 1) * xci[0] + b.dqdy(i, j, k, 1) * xci[1] +
                  b.dqdz(i, j, k, 1) * xci[2]);
        wR = b.q(i, j, k, 1) + wDOTxc;
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) + b.dqdx(i, j - 1, k, 1)) +
                      damp[0] * (wR - wL);
        double dudy = 0.5 * (b.dqdy(i, j, k, 1) + b.dqdy(i, j - 1, k, 1)) +
                      damp[1] * (wR - wL);
        double dudz = 0.5 * (b.dqdz(i, j, k, 1) + b.dqdz(i, j - 1, k, 1)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i, j - 1, k, 2) * xcim1[0] +
                  b.dqdy(i, j - 1, k, 2) * xcim1[1] +
                  b.dqdz(i, j - 1, k, 2) * xcim1[2]);
        wL = b.q(i, j - 1, k, 2) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 2) * xci[0] + b.dqdy(i, j, k, 2) * xci[1] +
                  b.dqdz(i, j, k, 2) * xci[2]);
        wR = b.q(i, j, k, 2) + wDOTxc;
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) + b.dqdx(i, j - 1, k, 2)) +
                      damp[0] * (wR - wL);
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) + b.dqdy(i, j - 1, k, 2)) +
                      damp[1] * (wR - wL);
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) + b.dqdz(i, j - 1, k, 2)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i, j - 1, k, 3) * xcim1[0] +
                  b.dqdy(i, j - 1, k, 3) * xcim1[1] +
                  b.dqdz(i, j - 1, k, 3) * xcim1[2]);
        wL = b.q(i, j - 1, k, 3) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 3) * xci[0] + b.dqdy(i, j, k, 3) * xci[1] +
                  b.dqdz(i, j, k, 3) * xci[2]);
        wR = b.q(i, j, k, 3) + wDOTxc;
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) + b.dqdx(i, j - 1, k, 3)) +
                      damp[0] * (wR - wL);
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) + b.dqdy(i, j - 1, k, 3)) +
                      damp[1] * (wR - wL);
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) + b.dqdz(i, j - 1, k, 3)) +
                      damp[2] * (wR - wL);

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        b.jF(i, j, k, 1) =
            txx * b.jsx(i, j, k) + txy * b.jsy(i, j, k) + txz * b.jsz(i, j, k);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        b.jF(i, j, k, 2) =
            tyx * b.jsx(i, j, k) + tyy * b.jsy(i, j, k) + tyz * b.jsz(i, j, k);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        b.jF(i, j, k, 3) =
            tzx * b.jsx(i, j, k) + tzy * b.jsy(i, j, k) + tzz * b.jsz(i, j, k);

        // energy
        //   heat conduction
        wDOTxc = (b.dqdx(i, j - 1, k, 4) * xcim1[0] +
                  b.dqdy(i, j - 1, k, 4) * xcim1[1] +
                  b.dqdz(i, j - 1, k, 4) * xcim1[2]);
        wL = b.q(i, j - 1, k, 4) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 4) * xci[0] + b.dqdy(i, j, k, 4) * xci[1] +
                  b.dqdz(i, j, k, 4) * xci[2]);
        wR = b.q(i, j, k, 4) + wDOTxc;
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) + b.dqdx(i, j - 1, k, 4)) +
                      damp[0] * (wR - wL);
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) + b.dqdy(i, j - 1, k, 4)) +
                      damp[1] * (wR - wL);
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) + b.dqdz(i, j - 1, k, 4)) +
                      damp[2] * (wR - wL);

        double q = -kappa * (dTdx * b.jsx(i, j, k) + dTdy * b.jsy(i, j, k) +
                             dTdz * b.jsz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j - 1, k, 1));
        double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j - 1, k, 2));
        double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j - 1, k, 3));

        b.jF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * b.jsx(i, j, k) -
                           (uf * tyx + vf * tyy + wf * tyz) * b.jsy(i, j, k) -
                           (uf * tzx + vf * tzy + wf * tzz) * b.jsz(i, j, k) +
                           q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j - 1, k, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i, j - 1, k, 2 + n));
          wDOTxc = (b.dqdx(i, j - 1, k, 5 + n) * xcim1[0] +
                    b.dqdy(i, j - 1, k, 5 + n) * xcim1[1] +
                    b.dqdz(i, j - 1, k, 5 + n) * xcim1[2]);
          wL = b.q(i, j - 1, k, 5 + n) + wDOTxc;
          wDOTxc = (b.dqdx(i, j, k, 5 + n) * xci[0] +
                    b.dqdy(i, j, k, 5 + n) * xci[1] +
                    b.dqdz(i, j, k, 5 + n) * xci[2]);
          wR = b.q(i, j, k, 5 + n) + wDOTxc;
          double dYdx =
              0.5 * (b.dqdx(i, j, k, 5 + n) + b.dqdx(i, j - 1, k, 5 + n)) +
              damp[0] * (wR - wL);
          double dYdy =
              0.5 * (b.dqdy(i, j, k, 5 + n) + b.dqdy(i, j - 1, k, 5 + n)) +
              damp[1] * (wR - wL);
          double dYdz =
              0.5 * (b.dqdz(i, j, k, 5 + n) + b.dqdz(i, j - 1, k, 5 + n)) +
              damp[2] * (wR - wL);

          double gradYk = (dYdx * b.jsx(i, j, k) + dYdy * b.jsy(i, j, k) +
                           dYdz * b.jsz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          b.jF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 *
             (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i, j - 1, k, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j - 1, k, 5 + n));
          Yns -= Yk;
          b.jF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i, j - 1, k, 5 + n));
          b.jF(i, j, k, 4) += b.jF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i, j - 1, k, b.ne));
        b.jF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "k face visc fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i, j, k - 1, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i, j, k - 1, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // continuity
        b.kF(i, j, k, 0) = 0.0;

        // Geometric terms
        double e[3] = {b.xc(i, j, k) - b.xc(i, j, k - 1),
                       b.yc(i, j, k) - b.yc(i, j, k - 1),
                       b.zc(i, j, k) - b.zc(i, j, k - 1)};
        double njk[3] = {b.knx(i, j, k), b.kny(i, j, k), b.knz(i, j, k)};

        double MAGeDOTn =
            sqrt(pow(e[0] * njk[0], 2.0) + pow(e[1] * njk[1], 2.0) +
                 pow(e[2] * njk[2], 2.0));

        double damp[3] = {alpha / MAGeDOTn * njk[0], alpha / MAGeDOTn * njk[1],
                          alpha / MAGeDOTn * njk[2]};
        double xcim1[3] = {b.kxc(i, j, k) - b.xc(i, j, k - 1),
                           b.kyc(i, j, k) - b.yc(i, j, k - 1),
                           b.kzc(i, j, k) - b.zc(i, j, k - 1)};
        double xci[3] = {b.kxc(i, j, k) - b.xc(i, j, k),
                         b.kyc(i, j, k) - b.yc(i, j, k),
                         b.kzc(i, j, k) - b.zc(i, j, k)};

        double wDOTxc, wL, wR;

        // Face derivatives = consistent + damping
        wDOTxc = (b.dqdx(i, j, k - 1, 1) * xcim1[0] +
                  b.dqdy(i, j, k - 1, 1) * xcim1[1] +
                  b.dqdz(i, j, k - 1, 1) * xcim1[2]);
        wL = b.q(i, j, k - 1, 1) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 1) * xci[0] + b.dqdy(i, j, k, 1) * xci[1] +
                  b.dqdz(i, j, k, 1) * xci[2]);
        wR = b.q(i, j, k, 1) + wDOTxc;
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) + b.dqdx(i, j, k - 1, 1)) +
                      damp[0] * (wR - wL);
        double dudy = 0.5 * (b.dqdy(i, j, k, 1) + b.dqdy(i, j, k - 1, 1)) +
                      damp[1] * (wR - wL);
        double dudz = 0.5 * (b.dqdz(i, j, k, 1) + b.dqdz(i, j, k - 1, 1)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i, j, k - 1, 2) * xcim1[0] +
                  b.dqdy(i, j, k - 1, 2) * xcim1[1] +
                  b.dqdz(i, j, k - 1, 2) * xcim1[2]);
        wL = b.q(i, j, k - 1, 2) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 2) * xci[0] + b.dqdy(i, j, k, 2) * xci[1] +
                  b.dqdz(i, j, k, 2) * xci[2]);
        wR = b.q(i, j, k, 2) + wDOTxc;
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) + b.dqdx(i, j, k - 1, 2)) +
                      damp[0] * (wR - wL);
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) + b.dqdy(i, j, k - 1, 2)) +
                      damp[1] * (wR - wL);
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) + b.dqdz(i, j, k - 1, 2)) +
                      damp[2] * (wR - wL);

        wDOTxc = (b.dqdx(i, j, k - 1, 3) * xcim1[0] +
                  b.dqdy(i, j, k - 1, 3) * xcim1[1] +
                  b.dqdz(i, j, k - 1, 3) * xcim1[2]);
        wL = b.q(i, j, k - 1, 3) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 3) * xci[0] + b.dqdy(i, j, k, 3) * xci[1] +
                  b.dqdz(i, j, k, 3) * xci[2]);
        wR = b.q(i, j, k, 3) + wDOTxc;
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) + b.dqdx(i, j, k - 1, 3)) +
                      damp[0] * (wR - wL);
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) + b.dqdy(i, j, k - 1, 3)) +
                      damp[1] * (wR - wL);
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) + b.dqdz(i, j, k - 1, 3)) +
                      damp[2] * (wR - wL);

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        b.kF(i, j, k, 1) =
            txx * b.ksx(i, j, k) + txy * b.ksy(i, j, k) + txz * b.ksz(i, j, k);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        b.kF(i, j, k, 2) =
            tyx * b.ksx(i, j, k) + tyy * b.ksy(i, j, k) + tyz * b.ksz(i, j, k);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        b.kF(i, j, k, 3) =
            tzx * b.ksx(i, j, k) + tzy * b.ksy(i, j, k) + tzz * b.ksz(i, j, k);

        // energy
        //   heat conduction
        wDOTxc = (b.dqdx(i, j, k - 1, 4) * xcim1[0] +
                  b.dqdy(i, j, k - 1, 4) * xcim1[1] +
                  b.dqdz(i, j, k - 1, 4) * xcim1[2]);
        wL = b.q(i, j, k - 1, 4) + wDOTxc;
        wDOTxc = (b.dqdx(i, j, k, 4) * xci[0] + b.dqdy(i, j, k, 4) * xci[1] +
                  b.dqdz(i, j, k, 4) * xci[2]);
        wR = b.q(i, j, k, 4) + wDOTxc;
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) + b.dqdx(i, j, k - 1, 4)) +
                      damp[0] * (wR - wL);
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) + b.dqdy(i, j, k - 1, 4)) +
                      damp[1] * (wR - wL);
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) + b.dqdz(i, j, k - 1, 4)) +
                      damp[2] * (wR - wL);

        double q = -kappa * (dTdx * b.ksx(i, j, k) + dTdy * b.ksy(i, j, k) +
                             dTdz * b.ksz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j, k - 1, 1));
        double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j, k - 1, 2));
        double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j, k - 1, 3));

        b.kF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * b.ksx(i, j, k) -
                           (uf * tyx + vf * tyy + wf * tyz) * b.ksy(i, j, k) -
                           (uf * tzx + vf * tzy + wf * tzz) * b.ksz(i, j, k) +
                           q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j, k - 1, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i, j, k - 1, 2 + n));
          wDOTxc = (b.dqdx(i, j, k - 1, 5 + n) * xcim1[0] +
                    b.dqdy(i, j, k - 1, 5 + n) * xcim1[1] +
                    b.dqdz(i, j, k - 1, 5 + n) * xcim1[2]);
          wL = b.q(i, j, k - 1, 5 + n) + wDOTxc;
          wDOTxc = (b.dqdx(i, j, k, 5 + n) * xci[0] +
                    b.dqdy(i, j, k, 5 + n) * xci[1] +
                    b.dqdz(i, j, k, 5 + n) * xci[2]);
          wR = b.q(i, j, k, 5 + n) + wDOTxc;
          double dYdx =
              0.5 * (b.dqdx(i, j, k, 5 + n) + b.dqdx(i, j, k - 1, 5 + n)) +
              damp[0] * (wR - wL);
          double dYdy =
              0.5 * (b.dqdy(i, j, k, 5 + n) + b.dqdy(i, j, k - 1, 5 + n)) +
              damp[1] * (wR - wL);
          double dYdz =
              0.5 * (b.dqdz(i, j, k, 5 + n) + b.dqdz(i, j, k - 1, 5 + n)) +
              damp[2] * (wR - wL);
          double gradYk = (dYdx * b.ksx(i, j, k) + dYdy * b.ksy(i, j, k) +
                           dYdz * b.ksz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          b.kF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 *
             (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i, j, k - 1, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j, k - 1, 5 + n));
          Yns -= Yk;
          b.kF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i, j, k - 1, 5 + n));
          b.kF(i, j, k, 4) += b.kF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i, j, k - 1, b.ne));
        b.kF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });
}
