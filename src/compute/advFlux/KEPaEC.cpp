#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

void KEPaEC(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        double U =
            b.isx(i, j, k) * uf + b.isy(i, j, k) * vf + b.isz(i, j, k) * wf;

        double pf = 0.5 * (b.q(i, j, k, 0) + b.q(i - 1, j, k, 0));

        // Compute fluxes
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i - 1, j, k, 0));

        // Continuity rho*Ui
        double C = rho * U;
        b.iF(i, j, k, 0) = C;

        // x momentum rho*u*Ui+ p*Ax
        b.iF(i, j, k, 1) = C * uf + pf * b.isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.iF(i, j, k, 2) = C * vf + pf * b.isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.iF(i, j, k, 3) = C * wf + pf * b.isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        double Kj = C * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i - 1, j, k, 1) +
                     b.q(i, j, k, 2) * b.q(i - 1, j, k, 2) +
                     b.q(i, j, k, 3) * b.q(i - 1, j, k, 3));

        double Pj =
            0.5 * (b.q(i - 1, j, k, 0) * (b.q(i, j, k, 1) * b.isx(i, j, k) +
                                          b.q(i, j, k, 2) * b.isy(i, j, k) +
                                          b.q(i, j, k, 3) * b.isz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i - 1, j, k, 1) * b.isx(i, j, k) +
                                      b.q(i - 1, j, k, 2) * b.isy(i, j, k) +
                                      b.q(i - 1, j, k, 3) * b.isz(i, j, k)));

        // solve for internal energy flux
        double eR = b.qh(i, j, k, 4) / b.Q(i, j, k, 0);
        double eL = b.qh(i - 1, j, k, 4) / b.Q(i - 1, j, k, 0);
        double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

        b.iF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.iF(i, j, k, 5 + n) =
              0.5 * (b.q(i, j, k, 5 + n) + b.q(i - 1, j, k, 5 + n)) * C;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j - 1, k, 1));
        double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j - 1, k, 2));
        double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j - 1, k, 3));

        double V =
            b.jsx(i, j, k) * uf + b.jsy(i, j, k) * vf + b.jsz(i, j, k) * wf;

        double pf = 0.5 * (b.q(i, j, k, 0) + b.q(i, j - 1, k, 0));

        // Compute fluxes
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j - 1, k, 0));

        // Continuity rho*Vj
        double C = rho * V;
        b.jF(i, j, k, 0) = C;

        // x momentum rho*u*Vj+ pAx
        b.jF(i, j, k, 1) = C * uf + pf * b.jsx(i, j, k);

        // y momentum rho*v*Vj+ pAy
        b.jF(i, j, k, 2) = C * vf + pf * b.jsy(i, j, k);

        // w momentum rho*w*Vj+ pAz
        b.jF(i, j, k, 3) = C * wf + pf * b.jsz(i, j, k);

        // Total energy (rhoE+P)*Vj)
        double Kj = C * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i, j - 1, k, 1) +
                     b.q(i, j, k, 2) * b.q(i, j - 1, k, 2) +
                     b.q(i, j, k, 3) * b.q(i, j - 1, k, 3));

        double Pj =
            0.5 * (b.q(i, j - 1, k, 0) * (b.q(i, j, k, 1) * b.jsx(i, j, k) +
                                          b.q(i, j, k, 2) * b.jsy(i, j, k) +
                                          b.q(i, j, k, 3) * b.jsz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i, j - 1, k, 1) * b.jsx(i, j, k) +
                                      b.q(i, j - 1, k, 2) * b.jsy(i, j, k) +
                                      b.q(i, j - 1, k, 3) * b.jsz(i, j, k)));

        // solve for internal energy flux
        double eR = b.qh(i, j, k, 4) / b.Q(i, j, k, 0);
        double eL = b.qh(i, j - 1, k, 4) / b.Q(i, j - 1, k, 0);
        double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

        b.jF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.jF(i, j, k, 5 + n) =
              0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j - 1, k, 5 + n)) * C;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "2nd order KEEP k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j, k - 1, 1));
        double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j, k - 1, 2));
        double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j, k - 1, 3));

        double W =
            b.ksx(i, j, k) * uf + b.ksy(i, j, k) * vf + b.ksz(i, j, k) * wf;

        double pf = 0.5 * (b.q(i, j, k, 0) + b.q(i, j, k - 1, 0));

        // Compute fluxes
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j, k - 1, 0));

        // Continuity rho*Wk
        double C = rho * W;
        b.kF(i, j, k, 0) = C;

        // x momentum rho*u*Wk+ pAx
        b.kF(i, j, k, 1) = C * uf + pf * b.ksx(i, j, k);

        // y momentum rho*v*Wk+ pAy
        b.kF(i, j, k, 2) = C * vf + pf * b.ksy(i, j, k);

        // w momentum rho*w*Wk+ pAz
        b.kF(i, j, k, 3) = C * wf + pf * b.ksz(i, j, k);

        // Total energy (rhoE+P)*Wk)
        double Kj = C * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i, j, k - 1, 1) +
                     b.q(i, j, k, 2) * b.q(i, j, k - 1, 2) +
                     b.q(i, j, k, 3) * b.q(i, j, k - 1, 3));

        double Pj =
            0.5 * (b.q(i, j, k - 1, 0) * (b.q(i, j, k, 1) * b.ksx(i, j, k) +
                                          b.q(i, j, k, 2) * b.ksy(i, j, k) +
                                          b.q(i, j, k, 3) * b.ksz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i, j, k - 1, 1) * b.ksx(i, j, k) +
                                      b.q(i, j, k - 1, 2) * b.ksy(i, j, k) +
                                      b.q(i, j, k - 1, 3) * b.ksz(i, j, k)));

        // solve for internal energy flux
        double eR = b.qh(i, j, k, 4) / b.Q(i, j, k, 0);
        double eL = b.qh(i, j, k - 1, 4) / b.Q(i, j, k - 1, 0);
        double Ij = (eL * eR) / (eL + eR) * C;

        b.kF(i, j, k, 4) = Ij + Kj + Pj;
        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.kF(i, j, k, 5 + n) =
              0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j, k - 1, 5 + n)) * C;
        }
      });
}
